"""자유 문장 접수의 실행과 기록 보관.

접수 1회는 판단 모델 요청 1회다. 기록은 `<root>/<접수 ID>/`에 둔다.
모델 호출, 주문 검증과 기록 형식은 `pixel2place.agent.intake`가 소유한다.
실행 시작은 `JobManager`가 소유한다.
"""

from pathlib import Path
from threading import Lock
from typing import Protocol
from uuid import uuid4

from pydantic import ValidationError

from pixel2place.agent.inference import REQUEST_RETRY
from pixel2place.agent.intake import INTAKE_FILE, IntakeOutcome, IntakeRecord, run_intake
from pixel2place.config import DEFAULT_JUDGMENT_MODEL, TokenFactorySettings
from pixel2place.demo.contracts import IntakeView
from pixel2place.demo.orders import preview_order
from pixel2place.order import Order


class IntakeBusyError(Exception):
    """접수 1건이 실행 중이다. 모델 요청을 겹쳐 보내지 않는다."""


class IntakeUnavailableError(Exception):
    """서버에 Token Factory 설정이 없거나 설정이 틀렸다."""


class IntakeMismatchError(Exception):
    """실행 요청의 주문이 수락한 접수의 주문이 아니다."""


class OrderReader(Protocol):
    """문장 1개를 읽고 새 디렉터리 `out_dir`에 접수 기록을 쓴다."""

    def __call__(self, text: str, *, out_dir: Path, intake_id: str) -> IntakeRecord: ...


def read_with_token_factory(text: str, *, out_dir: Path, intake_id: str) -> IntakeRecord:
    """작업 디렉터리의 Token Factory 설정으로 판단 모델에 요청한다.

    일시 오류가 난 요청은 다시 보낸다.

    Raises:
        IntakeUnavailableError: 설정이 없거나 틀렸을 때. 기록을 만들지 않는다.
        OSError: 기록을 쓸 수 없을 때.
    """
    try:
        settings = TokenFactorySettings(  # pyright: ignore[reportCallIssue]
            nemotron_model=DEFAULT_JUDGMENT_MODEL
        )
    except ValidationError:
        raise IntakeUnavailableError from None
    return run_intake(
        text, out_dir=out_dir, settings=settings, intake_id=intake_id, retry=REQUEST_RETRY
    )


class IntakeDesk:
    """접수를 1건씩 실행하고 기록을 보관한다.

    주문을 고치지 않는다. 실행을 시작하지 않는다.
    """

    def __init__(self, *, root: Path, read: OrderReader = read_with_token_factory) -> None:
        self._root = root
        self._read = read
        self._lock = Lock()

    def read(self, text: str) -> IntakeView:
        """문장 1개를 주문으로 읽는다. 모델 요청은 유료다. 1건씩 보낸다.

        Raises:
            IntakeBusyError: 다른 접수가 실행 중일 때.
            IntakeUnavailableError: 서버에 모델 설정이 없을 때.
            OSError: 기록을 쓸 수 없을 때.
        """
        if not self._lock.acquire(blocking=False):
            raise IntakeBusyError
        try:
            intake_id = uuid4().hex
            record = self._read(text, out_dir=self._root / intake_id, intake_id=intake_id)
        finally:
            self._lock.release()
        order = record.reading.order
        return IntakeView(
            intake_id=record.intake_id,
            outcome=record.reading.outcome,
            model=record.model,
            order=order,
            preview=preview_order(order) if order is not None else None,
            reject_reason=record.reading.reject_reason,
        )

    def accepted_record(self, intake_id: str, *, order: Order) -> Path:
        """수락한 접수의 기록 파일을 돌려준다. `order`는 그 접수의 주문이어야 한다.

        `intake_id`는 검증한 ID다. 경로 구분자가 없다.

        Raises:
            IntakeMismatchError: 접수가 없을 때, 수락한 접수가 아닐 때 또는 주문이 다를 때.
            OSError: 기록을 읽을 수 없을 때.
        """
        path = self._root / intake_id / INTAKE_FILE
        if not path.is_file():
            raise IntakeMismatchError
        record = IntakeRecord.model_validate_json(path.read_bytes())
        reading = record.reading
        if (
            record.intake_id != intake_id
            or reading.outcome != IntakeOutcome.ACCEPTED
            or reading.order != order
        ):
            raise IntakeMismatchError
        return path
