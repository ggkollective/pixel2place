"""조립 도구의 strict 인자와 모델에 전달하는 결과."""

from enum import StrEnum
from typing import Annotated, Literal

from pydantic import BaseModel, ConfigDict, Field

from pixel2place.cell.check import CheckFailure, FailureStage
from pixel2place.cell.image_check import PartStatus
from pixel2place.feasibility.checker import IssueCode
from pixel2place.layout.schema import Cell, Color, PartType, Rotation

OpaqueId = Annotated[str, Field(pattern=r"^[a-f0-9]{32}$")]
CallId = Annotated[str, Field(min_length=1, max_length=256, pattern=r"\S")]
Count = Annotated[int, Field(ge=0)]
Measure = Annotated[float, Field(ge=0)]
PartId = Annotated[str, Field(min_length=1, max_length=64)]
# 검사 요약의 목록 1개의 최대 길이. 주문의 최대 부품 수와 같다.
MAX_INSPECTED_ITEMS = 16


class Contract(BaseModel):
    """추가 필드, 형 변환과 무한 값을 거부한다."""

    model_config = ConfigDict(extra="forbid", strict=True, frozen=True, allow_inf_nan=False)


class ToolName(StrEnum):
    """모델이 호출할 수 있는 조립 도구."""

    PREPARE_LAYOUT = "prepare_layout"
    CALIBRATE_BOARD = "calibrate_board"
    EXECUTE_CELL = "execute_cell"
    REPLAN_LAYOUT = "replan_layout"
    """계획 검토가 켜진 실행에서만 쓴다."""
    REWORK_CELL = "rework_cell"
    """처분이 켜진 실행에서만 쓴다."""


class ToolState(StrEnum):
    """도구 실행의 현재 단계."""

    CREATED = "created"
    LAYOUT_RETRYABLE = "layout_retryable"
    LAYOUT_READY = "layout_ready"
    CALIBRATED = "calibrated"
    CHECKED = "checked"
    COMPLETED = "completed"
    ABORTED = "aborted"


class ErrorCode(StrEnum):
    """외부 오류 본문을 포함하지 않는 실패 코드."""

    LAYOUT_INVALID = "layout_invalid"
    ROUNDTRIP_MISMATCH = "roundtrip_mismatch"
    REASONER_TRUNCATED = "reasoner_truncated"
    REASONER_ERROR = "reasoner_error"
    CALIBRATION_FAILED = "calibration_failed"
    COMPILE_FAILED = "compile_failed"
    RUNNER_ERROR = "runner_error"
    RESULT_MISMATCH = "result_mismatch"
    CHECK_FAILED = "check_failed"
    MODEL_ERROR = "model_error"
    PROTOCOL_ERROR = "protocol_error"
    BUDGET_EXHAUSTED = "budget_exhausted"
    RECORDING_ERROR = "recording_error"
    AGENT_ABORTED = "agent_aborted"
    PLAN_REJECTED = "plan_rejected"
    """모델이 계획 검토에서 준비한 계획을 거부했다."""
    INSPECTION_REJECTED = "inspection_rejected"
    """참값 판정은 통과했다. 카메라 판정이 실패했거나 카메라 판정의 보고서가 없다."""


class ReviewReason(StrEnum):
    """모델이 재계획을 요청하는 사유. Cosmos에는 사유별 고정 문장이 간다."""

    PART_SET_MISMATCH = "part_set_mismatch"
    POSITION_MISMATCH = "position_mismatch"
    ARRANGEMENT_MISMATCH = "arrangement_mismatch"


class PrepareLayoutArguments(Contract):
    """첫 준비는 false다. 재계획은 true다."""

    retry: bool


class CalibrateBoardArguments(Contract):
    """현재 실행에서 준비한 레이아웃 ID."""

    layout_id: OpaqueId


class ExecuteCellArguments(Contract):
    """같은 실행과 레이아웃에 속한 준비·보정 ID."""

    layout_id: OpaqueId
    calibration_id: OpaqueId


class ReplanLayoutArguments(Contract):
    """현재 레이아웃 ID와 재계획 사유."""

    layout_id: OpaqueId
    reason: ReviewReason


class ReworkCellArguments(Contract):
    """다시 놓을 부품의 ID. 카메라 판정의 재작업 대상과 같아야 한다."""

    part_ids: Annotated[tuple[PartId, ...], Field(min_length=1, max_length=MAX_INSPECTED_ITEMS)]


class ArtifactRef(Contract):
    """모델에 전달하는 ID. 파일 경로와 해시는 제외한다."""

    id: OpaqueId
    kind: Literal["layout", "calibration", "check"]


class LayoutSummary(Contract):
    """수리한 계획과 파싱 결과의 검사 값."""

    part_count: Count
    planned_part_count: Count
    plan_feasible: bool
    parsed_feasible: bool
    repair_count: Count
    cell_accuracy: Annotated[float, Field(ge=0, le=1)]


class PlannedPart(Contract):
    """준비한 레이아웃의 부품 1개. 위치는 부품을 감싸는 최소 사각형의 왼쪽 위 셀이다."""

    part_id: str
    type: PartType
    color: Color
    rotation_deg: Rotation
    row: Count
    col: Count


class PartCount(Contract):
    """종류와 색이 같은 부품의 수."""

    type: PartType
    color: Color
    count: Count


class ReviewedLayoutSummary(LayoutSummary):
    """계획 검토가 켜진 실행의 준비 결과. 모델이 목표와 비교할 부품 목록이 있다."""

    parts: tuple[PlannedPart, ...]
    part_counts: tuple[PartCount, ...]


class CalibrationSummary(Contract):
    """이미지에서 구한 보정 품질과 이동량. 참값은 없다."""

    markers_found: Count
    markers_total: Count
    pins_used: Count
    pins_total: Count
    marker_max_residual_px: Measure
    pin_rms_mm: Measure
    pin_max_mm: Measure
    board_shift_mm: Measure
    board_dx_mm: float
    board_dy_mm: float
    board_yaw_deg: float


class FailedPart(Contract):
    """실패한 부품의 ID와 실패 단계."""

    part_id: str
    failures: tuple[CheckFailure, ...]
    failed_stage: FailureStage | None


class ExecutionSummary(Contract):
    """유효한 실행 결과의 판정. status와 배치 성공을 구분한다."""

    part_count: Count
    passed_count: Count
    passed: bool
    failed_parts: tuple[FailedPart, ...]


class InspectedPart(Contract):
    """카메라 판정이 일치로 읽지 않은 부품 1개."""

    part_id: str
    status: Literal[PartStatus.SHADOWED, PartStatus.MISMATCH]
    empty: bool
    """부품의 모든 칸에서 핀이 보인다. 그 칸에 부품이 없다."""


class InspectionSummary(Contract):
    """최종 이미지의 카메라 판정. 참값은 없다.

    두 목록의 길이는 `MAX_INSPECTED_ITEMS` 이하다. 전체 수는 수의 필드에 있다.
    """

    passed: bool
    """불일치 부품과 계획에 없는 점유 칸이 없다."""
    match_count: Count
    shadowed_count: Count
    mismatch_count: Count
    parts: Annotated[tuple[InspectedPart, ...], Field(max_length=MAX_INSPECTED_ITEMS)]
    """일치가 아닌 부품. 순서는 배치 순서다."""
    unexpected_cell_count: Count
    unexpected_cells: Annotated[tuple[Cell, ...], Field(max_length=MAX_INSPECTED_ITEMS)]
    """계획에 없는 칸 중에서 점유된 칸 (row, col)."""


class InspectedExecutionSummary(ExecutionSummary):
    """처분이 켜진 실행의 판정. 모델이 처분을 고를 카메라 판정이 있다."""

    inspection: InspectionSummary | None
    """카메라 판정의 보고서가 없는 실행은 None이다. 그 실행은 출하하지 않는다."""


class ErrorSummary(Contract):
    """실패한 도구와 준비 시도 번호. 원시 응답은 없다."""

    tool: ToolName
    layout_attempt: Count
    violations: tuple[IssueCode, ...] = ()
    cell_accuracy: Annotated[float | None, Field(ge=0, le=1)] = None


class _Result(Contract):
    schema_version: Literal[1] = 1
    run_id: OpaqueId
    tool_call_id: CallId
    artifacts: tuple[ArtifactRef, ...] = ()


class _Success(_Result):
    status: Literal["ok"] = "ok"
    error_code: None = None
    recoverable: Literal[False] = False


class LayoutResult(_Success):
    """레이아웃 준비의 성공 결과. 부품 목록이 있는 요약을 먼저 맞춘다."""

    summary: ReviewedLayoutSummary | LayoutSummary


class CalibrationResult(_Success):
    """보드 보정의 성공 결과."""

    summary: CalibrationSummary


class ExecutionResult(_Success):
    """셀 실행과 판정 파일을 만든 결과. 카메라 판정이 있는 요약을 먼저 맞춘다."""

    summary: InspectedExecutionSummary | ExecutionSummary


class ErrorResult(_Result):
    """도구 실패와 runtime이 정한 재계획 가능 여부."""

    status: Literal["error"] = "error"
    error_code: ErrorCode
    recoverable: bool
    summary: ErrorSummary


type ToolResult = LayoutResult | CalibrationResult | ExecutionResult | ErrorResult
