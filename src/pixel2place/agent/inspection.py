"""도메인 근거를 검사하고 모델에 전달할 요약을 만든다."""

import math
from collections import Counter

from pixel2place.agent.contracts import (
    MAX_INSPECTED_ITEMS,
    CalibrationSummary,
    Contract,
    ErrorCode,
    ExecutionSummary,
    FailedPart,
    InspectedExecutionSummary,
    InspectedPart,
    InspectionSummary,
    LayoutSummary,
    PartCount,
    PlannedPart,
    ReviewedLayoutSummary,
)
from pixel2place.cell.calibration import CalibrationReport
from pixel2place.cell.check import CheckReport
from pixel2place.cell.image_check import ImageCheckReport, PartStatus
from pixel2place.cell.rework import reworkable
from pixel2place.cell.task import MM_PER_M
from pixel2place.feasibility.checker import IssueCode, check_layout
from pixel2place.layout.schema import Board, Layout
from pixel2place.pipeline import compare_layouts


class LayoutInspection(Contract):
    """수리 후 계획과 파싱 결과의 검사. 빈 레이아웃은 실패다."""

    summary: LayoutSummary
    error_code: ErrorCode | None
    violations: tuple[IssueCode, ...]


def inspect_layout(
    planned: Layout, parsed: Layout, *, board: Board, repair_count: int
) -> LayoutInspection:
    """현재 두 레이아웃에서 검사와 비교 지표를 다시 계산한다."""
    planned_check = check_layout(planned)
    parsed_check = check_layout(parsed)
    comparison = compare_layouts(planned=planned, parsed=parsed)
    code = None
    if (
        planned.board != board
        or parsed.board != board
        or not planned_check.feasible
        or not parsed_check.feasible
    ):
        code = ErrorCode.LAYOUT_INVALID
    elif comparison.cell_accuracy != 1.0:
        code = ErrorCode.ROUNDTRIP_MISMATCH
    return LayoutInspection(
        summary=LayoutSummary(
            part_count=len(parsed.parts),
            planned_part_count=len(planned.parts),
            plan_feasible=planned_check.feasible,
            parsed_feasible=parsed_check.feasible,
            repair_count=repair_count,
            cell_accuracy=comparison.cell_accuracy,
        ),
        error_code=code,
        violations=tuple(
            sorted({issue.code for issue in (*planned_check.issues, *parsed_check.issues)})
        ),
    )


def review_summary(summary: LayoutSummary, *, layout: Layout) -> ReviewedLayoutSummary:
    """실행할 레이아웃의 부품 목록을 요약에 더한다. 모델이 목표와 비교한다.

    위치는 부품을 감싸는 최소 사각형의 왼쪽 위 셀이다. 지시문의 row, col과 같은 기준이다.
    """
    counts = Counter((part.type, part.color) for part in layout.parts)
    return ReviewedLayoutSummary(
        **summary.model_dump(),
        parts=tuple(
            PlannedPart(
                part_id=part.part_id,
                type=part.type,
                color=part.color,
                rotation_deg=part.rotation_deg,
                row=min(row for row, _ in part.cells),
                col=min(col for _, col in part.cells),
            )
            for part in layout.parts
        ),
        part_counts=tuple(
            PartCount(type=part_type, color=color, count=count)
            for (part_type, color), count in counts.items()
        ),
    )


def summarize_calibration(report: CalibrationReport) -> CalibrationSummary:
    """추정 보고에서 품질과 이동량을 고른다. 참값을 받지 않는다."""
    return CalibrationSummary(
        markers_found=sum(marker.found_px is not None for marker in report.markers),
        markers_total=len(report.markers),
        pins_used=report.pins_used,
        pins_total=len(report.pins),
        marker_max_residual_px=max(
            marker.residual_px for marker in report.markers if marker.residual_px is not None
        ),
        pin_rms_mm=report.pin_rms_mm,
        pin_max_mm=report.pin_max_mm,
        board_shift_mm=report.board_shift_mm,
        board_dx_mm=report.offset.x * MM_PER_M,
        board_dy_mm=report.offset.y * MM_PER_M,
        board_yaw_deg=math.degrees(report.offset.yaw),
    )


def summarize_execution(check: CheckReport) -> ExecutionSummary:
    """유효한 판정에서 통과 수와 실패 부품·단계를 고른다."""
    return ExecutionSummary(
        part_count=len(check.parts),
        passed_count=check.passed_count,
        passed=check.passed,
        failed_parts=tuple(
            FailedPart(
                part_id=part.part_id,
                failures=tuple(part.failures),
                failed_stage=part.failed_stage,
            )
            for part in check.parts
            if not part.passed
        ),
    )


def summarize_inspection(report: ImageCheckReport) -> InspectionSummary:
    """카메라 판정에서 일치가 아닌 부품과 계획에 없는 점유 칸을 고른다.

    두 목록은 `MAX_INSPECTED_ITEMS`에서 자른다. 전체 수는 수의 필드가 갖는다.
    """
    return InspectionSummary(
        passed=report.passed,
        match_count=report.match_count,
        shadowed_count=report.shadowed_count,
        mismatch_count=report.mismatch_count,
        parts=tuple(
            InspectedPart(part_id=part.part_id, status=part.status, empty=part.empty)
            for part in report.parts
            if part.status is not PartStatus.MATCH
        )[:MAX_INSPECTED_ITEMS],
        unexpected_cell_count=len(report.unexpected_cells),
        unexpected_cells=tuple(report.unexpected_cells[:MAX_INSPECTED_ITEMS]),
    )


def inspected_summary(
    summary: ExecutionSummary, *, report: ImageCheckReport | None
) -> InspectedExecutionSummary:
    """판정 요약에 카메라 판정을 더한다. 모델이 처분을 고른다. 보고서가 없으면 None을 둔다."""
    return InspectedExecutionSummary(
        **summary.model_dump(),
        inspection=None if report is None else summarize_inspection(report),
    )


def checked_error(*, check_passed: bool, report: ImageCheckReport | None) -> ErrorCode | None:
    """처분을 켠 실행이 `checked`에 두는 오류 코드. 출하할 수 있으면 None이다.

    참값 판정의 실패가 먼저다. 카메라 판정의 실패와 없는 보고서는 `inspection_rejected`다.
    """
    if not check_passed:
        return ErrorCode.CHECK_FAILED
    if report is None or not report.passed:
        return ErrorCode.INSPECTION_REJECTED
    return None


def rework_targets(report: ImageCheckReport) -> tuple[str, ...]:
    """`rework_cell`이 받는 부품 ID. 순서는 배치 순서다.

    대상은 `cell.rework.reworkable`이 정한다.
    일치가 아닌 부품이 검사 요약의 목록보다 많으면 대상은 없다. 모델은 목록 밖의 부품을 모른다.
    """
    unmatched = sum(part.status is not PartStatus.MATCH for part in report.parts)
    if unmatched > MAX_INSPECTED_ITEMS:
        return ()
    sequences = set(reworkable(report))
    return tuple(part.part_id for part in report.parts if part.sequence in sequences)
