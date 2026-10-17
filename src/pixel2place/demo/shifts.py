"""교대의 주문 큐와 기록.

Owns:
- 교대 기록(`<실행 루트>/shifts/<교대 ID>/shift.json`).
- 주문의 실행 순서, 다음 주문을 시작하는 때와 교대를 닫는 규칙.
- 교대의 화면용 응답.

Does not own:
- 실행 1건의 프로세스, 입력과 산출물(`pixel2place.demo.jobs`).
- 주문별 지표와 합계의 계산(`pixel2place.agent.shift`).
- 주문 계약과 지시문(`pixel2place.order`).
"""

import logging
from datetime import UTC, datetime
from threading import RLock
from uuid import uuid4

from pixel2place.agent.artifacts import write_record
from pixel2place.agent.shift import shift_report
from pixel2place.demo.contracts import (
    ShiftEntry,
    ShiftItemView,
    ShiftRecord,
    ShiftRequest,
    ShiftStatus,
    ShiftView,
)
from pixel2place.demo.jobs import JobManager, RunBusyError
from pixel2place.order import order_instruction

SHIFTS_DIR = "shifts"
SHIFT_FILE = "shift.json"
MAX_SHIFT_HISTORY = 10
logger = logging.getLogger(__name__)


def _with_entry(record: ShiftRecord, index: int, **fields: object) -> ShiftRecord:
    """주문 1건의 값을 바꾼 교대 기록."""
    items = list(record.items)
    items[index] = items[index].model_copy(update=fields)
    return record.model_copy(update={"items": tuple(items)})


def _current(record: ShiftRecord) -> tuple[int, str] | None:
    """마지막으로 시작한 주문의 순서와 실행 ID. 시작한 주문이 없으면 None이다."""
    started = [
        (index, entry.run_id)
        for index, entry in enumerate(record.items)
        if entry.run_id is not None
    ]
    return started[-1] if started else None


class ShiftDesk:
    """교대를 1개씩 실행한다. 주문은 `JobManager`의 실행으로 1건씩 시작한다.

    교대는 셀을 예약한다. 교대가 열린 동안 `JobManager`는 다른 실행을 거부한다.
    프로세스를 직접 만들거나 멈추지 않는다. 닫힌 교대를 다시 시작하지 않는다.
    """

    def __init__(self, *, jobs: JobManager) -> None:
        self._jobs = jobs
        # 디렉터리 이름이 실행 ID의 형식이 아니다. 실행 기록의 복원이 건너뛴다.
        self._root = jobs.root / SHIFTS_DIR
        self._lock = RLock()
        self._records: dict[str, ShiftRecord] = {}
        self._restore()

    def _restore(self) -> None:
        """저장한 교대를 읽는다. 열려 있던 교대는 `interrupted`로 닫는다."""
        for path in self._root.glob(f"*/{SHIFT_FILE}"):
            record = ShiftRecord.model_validate_json(path.read_text())
            if record.id != path.parent.name:
                raise ValueError("Shift identity does not match its directory.")
            if record.status == "running":
                record = record.model_copy(
                    update={"status": "interrupted", "finished_at": datetime.now(UTC)}
                )
                write_record(path, record)
            self._records[record.id] = record

    def _save(self, record: ShiftRecord) -> None:
        write_record(self._root / record.id / SHIFT_FILE, record)
        self._records[record.id] = record

    def _close(self, record: ShiftRecord, *, status: ShiftStatus) -> None:
        """교대를 닫고 셀의 예약을 푼다. 기록을 쓰지 못해도 예약은 푼다."""
        try:
            self._save(
                record.model_copy(update={"status": status, "finished_at": datetime.now(UTC)})
            )
        finally:
            self._jobs.release()
        logger.info("Closed a shift.", extra={"shift_id": record.id, "status": status})

    def _start_next(self, record: ShiftRecord) -> None:
        """시작하지 않은 첫 주문을 시작한다. 남은 주문이 없으면 교대를 닫는다."""
        index = next(
            (index for index, entry in enumerate(record.items) if entry.run_id is None), None
        )
        if index is None:
            self._close(record, status="finished")
            return
        try:
            run = self._jobs.start(record.items[index].request.run_request(), queued=True)
        except (OSError, RunBusyError):
            # 워커를 시작하지 못했다. 남은 주문도 같은 설정을 쓴다.
            # 셀이 바쁘면 교대 기록에 없는 실행이 있다. 교대는 그 실행을 기다리지 않는다.
            self._close(record, status="aborted")
            raise
        self._save(_with_entry(record, index, run_id=run.id))

    def start(self, request: ShiftRequest) -> ShiftView:
        """교대를 저장하고 첫 주문을 시작한다. 다음 주문은 `advance`가 시작한다.

        Raises:
            OrderNotFeasibleError: 주문이 배치 규칙을 어길 때. 교대 기록을 만들지 않는다.
            RunBusyError: 실행 또는 다른 교대가 셀을 쓰고 있을 때. 교대 기록을 만들지 않는다.
            OSError: 기록을 쓸 수 없거나 첫 워커를 시작할 수 없을 때.
        """
        for item in request.items:
            order_instruction(item.order)
        with self._lock:
            self._jobs.reserve()
            try:
                record = ShiftRecord(
                    id=uuid4().hex,
                    status="running",
                    created_at=datetime.now(UTC),
                    items=tuple(ShiftEntry(request=item) for item in request.items),
                )
                (self._root / record.id).mkdir(parents=True)
                self._save(record)
            except OSError:
                self._jobs.release()
                raise
            logger.info(
                "Started a shift.", extra={"shift_id": record.id, "orders": len(record.items)}
            )
            self._start_next(record)
            return self._view(self._records[record.id])

    def advance(self) -> None:
        """열린 교대의 현재 주문을 확인한다. 끝났으면 지표를 저장하고 다음 주문을 시작한다.

        불량인 주문과 모델 오류로 끝난 주문 뒤에는 다음 주문을 시작한다.
        사용자가 중단한 주문 뒤의 주문은 시작하지 않는다.
        실행 보고서 없이 끝난 주문 뒤의 주문도 시작하지 않는다. 그 워커는 실행의 앞에서 끝났다.

        Raises:
            OSError: 기록을 읽거나 쓸 수 없을 때, 다음 워커를 시작할 수 없을 때.
            RunBusyError: 교대 기록에 없는 실행이 셀을 쓰고 있을 때. 교대는 닫힌다.
        """
        with self._lock:
            record = next(
                (item for item in self._records.values() if item.status == "running"), None
            )
            if record is None:
                return
            current = _current(record)
            if current is not None:
                index, run_id = current
                run = self._jobs.summary(run_id)
                if run.status in ("running", "cancelling"):
                    return
                record = self._settled(record, index=index, run_id=run_id)
                if run.status == "cancelled":
                    self._close(record, status="cancelled")
                    return
                if record.items[index].report is None:
                    self._close(record, status="aborted")
                    return
            self._start_next(record)

    def cancel(self, shift_id: str) -> ShiftView:
        """현재 주문을 중단하고 교대를 닫는다. 남은 주문은 시작하지 않는다.

        닫힌 교대는 바꾸지 않는다.
        """
        with self._lock:
            record = self._records[shift_id]
            if record.status == "running":
                current = _current(record)
                if current is not None:
                    index, run_id = current
                    self._jobs.cancel(run_id)
                    record = self._settled(record, index=index, run_id=run_id)
                self._close(record, status="cancelled")
            return self._view(self._records[shift_id])

    def close(self) -> None:
        """서버가 멈출 때 열린 교대를 `interrupted`로 닫는다. 실행은 `JobManager`가 멈춘다."""
        with self._lock:
            for record in tuple(self._records.values()):
                if record.status == "running":
                    self._close(record, status="interrupted")

    def get(self, shift_id: str) -> ShiftView:
        """교대 1개의 주문별 상태와 지표. 실행을 시작하지 않는다."""
        with self._lock:
            return self._view(self._records[shift_id])

    def recent(self) -> tuple[ShiftView, ...]:
        """최근 교대를 반환한다. 새 교대가 앞이다. 실행을 시작하지 않는다."""
        with self._lock:
            records = sorted(self._records.values(), key=lambda item: item.created_at, reverse=True)
            return tuple(self._view(record) for record in records[:MAX_SHIFT_HISTORY])

    def _settled(self, record: ShiftRecord, *, index: int, run_id: str) -> ShiftRecord:
        """주문 1건에 끝난 실행의 지표를 넣은 기록. 저장하지 않는다."""
        report = self._jobs.order_report(run_id, name=record.items[index].request.name)
        return _with_entry(record, index, report=report)

    def _view(self, record: ShiftRecord) -> ShiftView:
        items = []
        for entry in record.items:
            request = entry.request
            run = None if entry.run_id is None else self._jobs.summary(entry.run_id)
            faults = request.faults
            items.append(
                ShiftItemView(
                    name=request.name,
                    order_kind=request.order.kind,
                    ordered_parts=request.order.part_count,
                    fixture=request.fixture,
                    # 빈 결함은 결함이 없는 주문이다.
                    faults=faults if faults is not None and faults.sequences else None,
                    run_id=entry.run_id,
                    status=run.status
                    if run is not None
                    else "queued"
                    if record.status == "running"
                    else "skipped",
                    disposition=None if run is None else run.disposition,
                    report=entry.report,
                )
            )
        reports = tuple(entry.report for entry in record.items if entry.report is not None)
        return ShiftView(
            id=record.id,
            status=record.status,
            created_at=record.created_at,
            finished_at=record.finished_at,
            items=tuple(items),
            totals=shift_report(reports, orders=len(record.items)).totals,
        )
