"""주문 여러 개의 연속 실행과 교대 보고서.

Owns:
- 교대 파일의 형식과 주문 파일의 해석.
- 주문의 실행 순서, 출력 디렉터리의 이름과 교대를 멈추는 규칙.
- 실행 기록에서 읽는 주문별 지표, 처분과 교대의 합계.

Does not own:
- 작업 실행 1회와 그 기록(`pixel2place.agent.cell`, `pixel2place.agent.runtime`).
- 주문 계약과 지시문(`pixel2place.order`).
- 판정과 실행기의 결과 형식(`pixel2place.cell`).
"""

from collections.abc import Callable, Sequence
from contextlib import suppress
from dataclasses import dataclass
from itertools import pairwise
from pathlib import Path
from time import perf_counter
from typing import Annotated, Literal

from pydantic import Field, ValidationError

from pixel2place.agent.artifacts import write_record
from pixel2place.agent.cell import run_agent_cell
from pixel2place.agent.contracts import (
    Contract,
    Count,
    ErrorCode,
    ExecutionResult,
    Measure,
    ToolState,
)
from pixel2place.agent.fixture import FixtureOffset, PartFaults, fixture_scene
from pixel2place.agent.runtime import EVENTS_FILE
from pixel2place.agent.runtime_contracts import AgentEvent, AgentReport, Disposition
from pixel2place.agent.tools import EXECUTION_DIR, REWORK_DIR
from pixel2place.cell.config import CellConfig
from pixel2place.cell.result import SimResult
from pixel2place.cell.sim import RESULT_FILE
from pixel2place.config import Settings, SimSettings, TokenFactorySettings
from pixel2place.order import Order, order_instruction, parse_order
from pixel2place.pipeline import SIM_DIR

SHIFT_REPORT_FILE = "shift_report.json"
# 출력 디렉터리의 번호는 2자리다.
MAX_SHIFT_ORDERS = 99
ItemName = Annotated[str, Field(pattern=r"^[a-z0-9][a-z0-9_-]{0,31}$")]
Ratio = Annotated[float, Field(ge=0, le=1)]


class ShiftItem(Contract):
    """교대 파일의 주문 1건."""

    name: ItemName
    """출력 디렉터리 이름의 뒷부분이다."""
    order: Annotated[str, Field(min_length=1)]
    """주문 JSON 파일의 경로. 상대 경로는 교대 파일의 디렉터리에서 읽는다."""
    fixture: FixtureOffset = FixtureOffset()
    """지그 오차. 범위는 데모의 지그 오차와 같다."""
    faults: PartFaults | None = None
    """부품의 결함. 부품은 배치 순서로 지정한다. 순서는 주문의 부품 수 이하다."""


class ShiftFile(Contract):
    """교대 파일. 항목의 순서가 실행 순서다."""

    schema_version: Literal[1] = 1
    items: Annotated[tuple[ShiftItem, ...], Field(min_length=1, max_length=MAX_SHIFT_ORDERS)]


@dataclass(frozen=True)
class ShiftOrder:
    """실행할 주문 1건. 교대 파일의 항목에서 주문 파일을 읽고 지시문을 만든 값이다."""

    name: str
    order: Order
    fixture: FixtureOffset
    prompt: str
    faults: PartFaults | None = None


class ShiftItemReport(Contract):
    """주문 1건의 실행 결과. 시간의 단위는 초다."""

    name: str
    directory: str
    """실행 디렉터리의 이름. `agent-shift`는 교대의 출력 디렉터리 안에 둔다.

    데모 서버는 실행 루트 안에 둔다. 이름은 실행 ID다.
    """
    order: Order
    fixture: FixtureOffset
    faults: PartFaults | None = None
    """실행에 넣은 부품의 결함. 없으면 None이다."""
    state: Literal[ToolState.COMPLETED, ToolState.ABORTED]
    error_code: ErrorCode | None
    disposition: Disposition
    reworked: bool
    """재작업 실행을 했다."""
    part_count: Count
    """주문의 부품 수."""
    passed_parts: Count | None
    """마지막 참값 판정을 통과한 부품 수. 재작업을 한 실행은 재작업 뒤의 수다.

    판정이 없는 실행은 None이다.
    """
    robot_time_s: Measure | None
    """실행기의 `sim_time_s`. 시뮬레이션 안의 팔 동작 시간이다. 재작업 실행의 시간을 더한다.

    판정이 없는 실행은 None이다.
    """
    wall_time_s: Measure
    """실행 1건의 시작부터 끝까지의 시간. 물리 계산의 시간이 대부분이다."""
    model_time_s: Measure
    """응답을 받은 Nemotron 요청의 지연의 합."""
    nemotron_requests: Count
    request_retries: Count


class ShiftTotals(Contract):
    """교대의 합계."""

    orders: Count
    """교대 파일의 주문 수."""
    completed_orders: Count
    """출하한 주문 수. 재작업 뒤에 출하한 주문도 센다."""
    first_pass_orders: Count
    """재작업 없이 출하한 주문 수."""
    rejected_orders: Count
    """불량으로 끝난 주문 수."""
    first_pass_yield: Ratio | None
    """1차 통과율. `first_pass_orders`를 실행한 주문 수로 나눈 값이다.

    실행한 주문이 없으면 None이다.
    """
    final_yield: Ratio | None
    """최종 통과율. `completed_orders`를 실행한 주문 수로 나눈 값이다."""
    changeovers: Count
    """주문 전환 수. 앞 실행의 주문과 다른 주문으로 시작한 실행의 수다."""
    wall_time_s: Measure


class ShiftReport(Contract):
    """교대 보고서. 항목은 실행한 주문만 있다. 순서는 실행 순서다."""

    schema_version: Literal[1] = 1
    items: tuple[ShiftItemReport, ...]
    totals: ShiftTotals


def load_shift(path: Path) -> tuple[ShiftOrder, ...]:
    """교대 파일과 모든 주문 파일을 읽는다. 실행의 앞에서 입력 오류를 찾는다.

    Raises:
        OSError: 교대 파일 또는 주문 파일을 읽을 수 없을 때.
        pydantic.ValidationError: 교대 파일 또는 주문이 계약과 다를 때.
        OrderNotFeasibleError: 보드 주문이 배치 규칙을 어길 때.
        ValueError: 결함의 부품 순서가 주문의 부품 수보다 클 때.
    """
    shift = ShiftFile.model_validate_json(path.read_bytes())
    orders: list[ShiftOrder] = []
    for item in shift.items:
        order = parse_order((path.parent / item.order).read_bytes())
        sequences = () if item.faults is None else item.faults.sequences
        if any(sequence > order.part_count for sequence in sequences):
            raise ValueError(
                f"Item {item.name} has a fault for a part that the order does not have."
            )
        orders.append(
            ShiftOrder(
                name=item.name,
                order=order,
                fixture=item.fixture,
                prompt=order_instruction(order),
                faults=item.faults,
            )
        )
    return tuple(orders)


def run_shift(
    orders: Sequence[ShiftOrder],
    *,
    out_dir: Path,
    cell: CellConfig,
    token_settings: TokenFactorySettings,
    cosmos_settings: Settings,
    sim_settings: SimSettings,
    review: bool = True,
    disposition: bool = True,
    on_item: Callable[[ShiftItemReport], None] | None = None,
) -> ShiftReport:
    """주문을 순서대로 1건씩 실행한다. 주문이 끝날 때마다 교대 보고서를 다시 쓴다.

    실행의 출력은 `out_dir/NN-이름/`이다. `NN`은 실행 순서다.
    완료하지 못한 주문 뒤에도 다음 주문을 실행한다. 불량인 주문 뒤에도 계속한다.
    사용자 인터럽트로 끝난 주문 뒤의 주문은 실행하지 않는다.
    실행이 보고를 만들기 전에 인터럽트를 받으면 그 주문은 보고서에 없다.

    Args:
        orders: 실행할 주문. `load_shift`가 만든다.
        out_dir: 새 출력 디렉터리.
        cell: 모든 주문이 쓰는 공칭 셀.
        token_settings: Nemotron 요청의 설정.
        cosmos_settings: Cosmos planner의 설정.
        sim_settings: Isaac 실행의 설정.
        review: 계획 검토를 켠다.
        disposition: 처분을 켠다. 모델이 검사 결과로 출하, 재작업과 불량을 고른다.
        on_item: 주문 1건이 끝날 때마다 그 결과로 부른다.

    Raises:
        FileExistsError: `out_dir`이 이미 있을 때.
        SimSetupError: Isaac 실행 파일 또는 USD가 없을 때.
        ValueError: 실행 설정이 틀렸을 때.
        OSError: 실행 기록 또는 교대 보고서를 읽거나 쓸 수 없을 때.
    """
    if out_dir.exists() or out_dir.is_symlink():
        raise FileExistsError("Shift output already exists.")
    shift = shift_report((), orders=len(orders))
    for index, planned in enumerate(orders, start=1):
        directory = f"{index:02d}-{planned.name}"
        started = perf_counter()
        try:
            # 첫 실행이 `out_dir`을 만든다. 설정 오류로 끝난 교대는 빈 디렉터리를 남기지 않는다.
            report = run_agent_cell(
                planned.prompt,
                out_dir=out_dir / directory,
                cell=cell,
                scene=fixture_scene(cell, planned.fixture, faults=planned.faults),
                token_settings=token_settings,
                cosmos_settings=cosmos_settings,
                sim_settings=sim_settings,
                review=review,
                disposition=disposition,
            )
            item = item_report(
                planned,
                directory=directory,
                root=out_dir / directory,
                report=report,
                wall_time_s=perf_counter() - started,
            )
        except KeyboardInterrupt:
            # 실행의 밖에서 받은 인터럽트다. 이 주문의 보고는 없다. 앞 주문의 보고서는 남아 있다.
            break
        shift = shift_report((*shift.items, item), orders=len(orders))
        write_record(out_dir / SHIFT_REPORT_FILE, shift)
        if on_item is not None:
            on_item(item)
        if _interrupted(report):
            break
    return shift


def _interrupted(report: AgentReport) -> bool:
    """사용자 인터럽트로 끝난 실행인지. 모델의 자발적 중단에는 최종 판단이 있다."""
    return report.error_code == ErrorCode.AGENT_ABORTED and report.final_decision is None


def item_report(
    planned: ShiftOrder, *, directory: str, root: Path, report: AgentReport, wall_time_s: float
) -> ShiftItemReport:
    """실행 보고와 실행 기록에서 주문 1건의 지표를 읽는다.

    Args:
        planned: 실행한 주문.
        directory: 보고서에 쓰는 실행 디렉터리의 이름.
        root: `agent-cell`의 출력 디렉터리. 이벤트와 실행기의 결과를 여기서 읽는다.
        report: 그 실행의 보고.
        wall_time_s: 실행의 시작부터 끝까지의 시간(초).
    """
    events = _read_events(root)
    robot_time_s = None
    # 판정 ID가 있는 실행만 검증한 배치 결과를 갖는다.
    if report.check_id is not None:
        run_dirs = [root / EXECUTION_DIR]
        if report.reworked:
            run_dirs.append(root / EXECUTION_DIR / REWORK_DIR)
        # 읽지 못한 결과는 지표를 비운다. 교대는 다음 주문으로 간다.
        with suppress(OSError, ValidationError):
            robot_time_s = sum(
                SimResult.model_validate_json(
                    (run_dir / SIM_DIR / RESULT_FILE).read_bytes()
                ).sim_time_s
                for run_dir in run_dirs
            )
    # 재작업을 한 실행은 판정 결과가 2개다. 마지막 결과가 처분의 근거다.
    passed_parts = next(
        (
            event.result.summary.passed_count
            for event in reversed(events)
            if isinstance(event.result, ExecutionResult)
        ),
        None,
    )
    return ShiftItemReport(
        name=planned.name,
        directory=directory,
        order=planned.order,
        fixture=planned.fixture,
        faults=planned.faults,
        state=report.state,
        error_code=report.error_code,
        disposition=report.disposition,
        reworked=report.reworked,
        part_count=planned.order.part_count,
        passed_parts=passed_parts,
        robot_time_s=robot_time_s,
        wall_time_s=wall_time_s,
        model_time_s=sum(
            event.evidence.latency_s
            for event in events
            if event.kind == "model_result" and event.evidence is not None
        ),
        nemotron_requests=report.calls.nemotron,
        request_retries=report.calls.retries,
    )


def _read_events(root: Path) -> tuple[AgentEvent, ...]:
    """실행의 이벤트를 읽는다. 읽지 못한 줄은 건너뛴다.

    기록 오류로 끝난 실행은 파일이 없거나 마지막 줄이 잘려 있다.
    그 실행의 지표는 읽은 줄에서만 나온다. 교대는 다음 주문으로 간다.
    """
    path = root / EVENTS_FILE
    if not path.is_file():
        return ()
    events = []
    for line in path.read_text(encoding="utf-8", errors="replace").splitlines():
        with suppress(ValidationError):
            events.append(AgentEvent.model_validate_json(line))
    return tuple(events)


def shift_report(items: tuple[ShiftItemReport, ...], *, orders: int) -> ShiftReport:
    """실행한 주문의 결과에서 합계를 계산한다. `orders`는 교대의 주문 수다."""
    completed = sum(item.state == ToolState.COMPLETED for item in items)
    first_pass = sum(item.disposition == Disposition.RELEASED for item in items)
    totals = ShiftTotals(
        orders=orders,
        completed_orders=completed,
        first_pass_orders=first_pass,
        rejected_orders=sum(item.disposition == Disposition.REJECTED for item in items),
        first_pass_yield=first_pass / len(items) if items else None,
        final_yield=completed / len(items) if items else None,
        changeovers=sum(previous.order != current.order for previous, current in pairwise(items)),
        wall_time_s=sum(item.wall_time_s for item in items),
    )
    return ShiftReport(items=items, totals=totals)
