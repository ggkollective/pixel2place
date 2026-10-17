"""웹 실행 입력과 화면 응답 계약. 모델·로봇 계약은 기존 실행 프로그램이 소유한다."""

from datetime import datetime
from typing import Annotated, Literal, Self

from pydantic import Field, model_validator

from pixel2place.agent.contracts import Contract, Count, OpaqueId, ToolState
from pixel2place.agent.fixture import FixtureOffset, PartFaults
from pixel2place.agent.intake import IntakeOutcome, IntakeText, RejectReason
from pixel2place.agent.runtime_contracts import AgentEvent, AgentReport, Disposition
from pixel2place.agent.shift import ItemName, ShiftItemReport, ShiftTotals
from pixel2place.cell.check import CheckReport
from pixel2place.cell.image_check import ImageCheckReport
from pixel2place.feasibility.checker import Issue
from pixel2place.layout.schema import Board, Cell, Color, Layout, PartType, Rotation
from pixel2place.order import Arrangement, Order

ScenePreset = Literal["nominal", "board_offset"]
JobStatus = Literal["running", "cancelling", "completed", "aborted", "cancelled", "interrupted"]
AssetName = Literal[
    "layout", "capture", "calibration", "video", "result", "rework_video", "rework_result"
]
OrderKind = Literal["board", "kit"]
ShiftStatus = Literal["running", "finished", "cancelled", "aborted", "interrupted"]
ShiftItemStatus = JobStatus | Literal["queued", "skipped"]
# 최근 작업 목록이 보여 주는 실행의 수다. 교대 1개의 실행은 그 목록에 모두 보인다.
MAX_SHIFT_REQUEST_ORDERS = 8


class RunRequest(Contract):
    """사용자는 지시문 원문과 주문 중 하나, 그리고 지그 오차와 부품의 결함을 준다.

    `intake_id`는 주문을 자유 문장에서 읽은 접수다. 서버는 그 접수의 주문과 비교한다.
    `faults`는 주문과 함께만 쓴다. 부품의 순서는 주문의 부품 수 이하다.
    """

    prompt: Annotated[str, Field(min_length=1, max_length=4000, pattern=r"\S")] | None = None
    order: Order | None = None
    intake_id: OpaqueId | None = None
    fixture: FixtureOffset = FixtureOffset()
    faults: PartFaults | None = None

    @model_validator(mode="after")
    def _require_one_goal(self) -> Self:
        if (self.prompt is None) == (self.order is None):
            raise ValueError("Give instruction text or a work order. Do not give both.")
        if self.intake_id is not None and self.order is None:
            raise ValueError("An intake applies only to an order.")
        return self

    @model_validator(mode="after")
    def _faults_name_ordered_parts(self) -> Self:
        if self.faults is None:
            return self
        if self.order is None:
            raise ValueError("Part faults apply only to an order.")
        if any(sequence > self.order.part_count for sequence in self.faults.sequences):
            raise ValueError("A part fault names a part that the order does not have.")
        return self


class JobRecord(Contract):
    """서버가 보존하는 실행 입력과 프로세스 상태. 자동 재개에는 쓰지 않는다."""

    id: OpaqueId
    prompt: str
    scene: ScenePreset
    status: JobStatus
    created_at: datetime
    finished_at: datetime | None = None
    worker_pid: int | None = None
    exit_code: int | None = None
    fixture: FixtureOffset | None = None
    """장면 프리셋만 저장한 기록에는 없다."""
    ordered_parts: Count | None = None
    """주문으로 시작한 실행의 부품 수. 지시문 원문으로 시작한 실행에는 없다."""
    order_kind: OrderKind | None = None
    """주문의 종류. 종류를 저장하지 않은 주문 기록은 보드 주문이다."""
    intake_id: OpaqueId | None = None
    """주문을 자유 문장에서 읽은 접수. 폼 주문과 지시문 원문의 실행에는 없다."""
    faults: PartFaults | None = None
    """실행에 넣은 부품의 결함. 없으면 None이다."""
    disposition: Disposition | None = None
    """끝난 실행의 처분. 실행 보고서에서 읽는다.

    실행 중인 실행, 사용자가 중단한 실행과 보고서가 없는 실행에는 없다.
    """


class RunSummary(Contract):
    """프로세스 ID와 파일 경로가 없는 실행 목록."""

    id: OpaqueId
    prompt: str
    scene: ScenePreset
    status: JobStatus
    created_at: datetime
    finished_at: datetime | None
    fixture: FixtureOffset
    ordered_parts: Count | None
    order_kind: OrderKind | None
    intake_id: OpaqueId | None
    faults: PartFaults | None
    disposition: Disposition | None


class OrderMatch(Contract):
    """주문과 준비한 레이아웃의 비교. 서버는 차이를 고치지 않는다.

    보드 주문은 종류, 색과 셀을 비교한다. 키트 주문은 종류와 색별 수량만 비교한다.
    """

    matches: bool
    missing: Count
    """주문에 있고 레이아웃에 없는 부품 수."""
    extra: Count
    """레이아웃에 있고 주문에 없는 부품 수."""


class RunView(RunSummary):
    """실제 CLI 이벤트와 산출물만 화면에 전달한다."""

    runtime_state: ToolState = ToolState.CREATED
    error_code: str | None = None
    events: tuple[AgentEvent, ...] = ()
    report: AgentReport | None = None
    layout: Layout | None = None
    check: CheckReport | None = None
    """마지막 배치의 부품 판정. 재작업을 한 실행은 재작업 뒤의 판정이다."""
    image_check: ImageCheckReport | None = None
    """마지막 배치의 카메라 판정. 화면 표시에만 쓴다. 기록이 없거나 읽지 못하면 없다."""
    first_check: CheckReport | None = None
    """재작업 앞의 부품 판정. 재작업의 판정이 있는 실행에만 있다."""
    first_image_check: ImageCheckReport | None = None
    """재작업 앞의 카메라 판정. 조건은 `first_check`와 같다."""
    assets: tuple[AssetName, ...] = ()
    order: Order | None = None
    order_match: OrderMatch | None = None
    """주문과 준비한 레이아웃이 모두 있을 때만 있다."""


class ShiftOrderRequest(Contract):
    """교대의 주문 1건. 뜻은 교대 파일의 항목과 같다. 주문은 파일 경로가 아니고 값이다."""

    name: ItemName
    order: Order
    fixture: FixtureOffset = FixtureOffset()
    faults: PartFaults | None = None

    def run_request(self) -> RunRequest:
        """이 주문의 실행 요청."""
        return RunRequest(order=self.order, fixture=self.fixture, faults=self.faults)

    @model_validator(mode="after")
    def _is_a_run_request(self) -> Self:
        # 주문에 없는 부품의 결함은 실행 요청의 계약이 거부한다.
        self.run_request()
        return self


class ShiftRequest(Contract):
    """주문의 큐 1개. 항목의 순서가 실행 순서다."""

    items: Annotated[
        tuple[ShiftOrderRequest, ...], Field(min_length=1, max_length=MAX_SHIFT_REQUEST_ORDERS)
    ]


class ShiftEntry(Contract):
    """교대 기록의 주문 1건."""

    request: ShiftOrderRequest
    run_id: OpaqueId | None = None
    """이 주문의 실행. 시작하지 않은 주문에는 없다."""
    report: ShiftItemReport | None = None
    """끝난 실행의 지표. 실행 보고서 없이 끝난 실행에는 없다."""


class ShiftRecord(Contract):
    """서버가 보존하는 교대의 주문 순서와 상태. 자동 재개에는 쓰지 않는다."""

    id: OpaqueId
    status: ShiftStatus
    created_at: datetime
    finished_at: datetime | None = None
    items: tuple[ShiftEntry, ...]


class ShiftItemView(Contract):
    """교대의 주문 1건과 그 실행의 상태."""

    name: str
    order_kind: OrderKind
    ordered_parts: Count
    fixture: FixtureOffset
    faults: PartFaults | None
    run_id: OpaqueId | None
    status: ShiftItemStatus
    """시작한 주문은 실행의 상태다. 기다리는 주문은 `queued`다.

    닫힌 교대에서 시작하지 않은 주문은 `skipped`다.
    """
    disposition: Disposition | None
    report: ShiftItemReport | None


class ShiftView(Contract):
    """교대 1개의 진행과 지표. 합계는 지표가 있는 주문에서 계산한다."""

    id: OpaqueId
    status: ShiftStatus
    created_at: datetime
    finished_at: datetime | None
    items: tuple[ShiftItemView, ...]
    totals: ShiftTotals


class SupplyFaults(Contract):
    """화면이 고르는 공급 결함 2종. 서버가 주문마다 부품과 색을 정한다."""

    missing: PartFaults
    """공급대에 오지 않는 부품 1개."""
    wrong_color: PartFaults | None
    """다른 색으로 공급되는 부품 1개. 계획한 색과 다른 색을 보장할 수 없으면 없다."""


class OrderPreview(Contract):
    """주문의 배치 검사, 지시문과 공급 결함. 위반이 있으면 지시문과 공급 결함은 없다."""

    part_count: Count
    issues: tuple[Issue, ...]
    instruction: str | None
    supply_faults: SupplyFaults | None


class IntakeRequest(Contract):
    """주문으로 읽을 자유 문장 1개."""

    text: IntakeText


class IntakeView(Contract):
    """접수 1회의 화면용 결과. 요청 근거와 모델의 자유 텍스트는 없다.

    `preview`는 모델이 제출한 주문의 배치 검사와 지시문이다. 주문이 없으면 없다.
    """

    intake_id: OpaqueId
    outcome: IntakeOutcome
    model: str
    order: Order | None
    preview: OrderPreview | None
    reject_reason: RejectReason | None


class PartShape(Contract):
    """부품 종류 1개의 한 회전 형태. 셀은 감싸는 사각형의 왼쪽 위 기준이다."""

    rotation_deg: Rotation
    cells: tuple[Cell, ...]


class PartOption(Contract):
    type: PartType
    name: str
    shapes: tuple[PartShape, ...]


class Range(Contract):
    min: float
    max: float


class FixtureRange(Contract):
    x_mm: Range
    y_mm: Range
    yaw_deg: Range


class ArrangementOption(Contract):
    value: Arrangement
    name: str


class KitPartOption(Contract):
    type: PartType
    color: Color
    max_count: Count


class KitOptions(Contract):
    """키트 주문의 한도와 선택지. 실제 배치 계획 모델 시험이 통과한 범위다."""

    max_parts: Count
    parts: tuple[KitPartOption, ...]
    arrangements: tuple[ArrangementOption, ...]


class OrderOptions(Contract):
    """주문 화면이 그리는 어휘와 한도. 화면은 이 값을 복제하지 않는다."""

    board: Board
    max_parts: Count
    colors: tuple[Color, ...]
    parts: tuple[PartOption, ...]
    kit: KitOptions
    fixture: FixtureRange
    seconds_base: Count
    seconds_per_part: Count


class ReferenceView(Contract):
    """새 실행과 구분하는 기존 보정 벤치마크."""

    variant: Literal["before", "after"]
    layout: Layout
    check: CheckReport
    description: str
