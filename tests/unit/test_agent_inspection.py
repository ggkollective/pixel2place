"""계획·파싱 경계의 기하 검사. 도메인 함수와 입력을 그대로 쓴다."""

import pytest

from pixel2place.agent.contracts import (
    MAX_INSPECTED_ITEMS,
    ErrorCode,
    ExecutionSummary,
    InspectedPart,
    LayoutResult,
)
from pixel2place.agent.inspection import (
    inspect_layout,
    inspected_summary,
    review_summary,
    rework_targets,
    summarize_inspection,
)
from pixel2place.agent.runtime_contracts import AgentBudget
from pixel2place.cell.image_check import PartStatus
from pixel2place.feasibility.checker import IssueCode
from pixel2place.layout.schema import Board, Color, Layout, PartType, Rotation
from tests.factories import (
    make_image_check_report,
    make_layout,
    make_part,
    make_part_image_check,
)

pytestmark = pytest.mark.unit


def test_empty_layout_does_not_pass_with_perfect_cell_accuracy() -> None:
    empty = make_layout()
    result = inspect_layout(empty, empty, board=Board(), repair_count=0)
    assert result.summary.cell_accuracy == 1.0
    assert result.error_code == ErrorCode.LAYOUT_INVALID
    assert result.violations == (IssueCode.EMPTY_LAYOUT,)


def test_feasible_layouts_with_different_colors_are_not_ready(valid_layout: Layout) -> None:
    changed = valid_layout.model_copy(
        update={
            "parts": [
                part.model_copy(update={"color": Color.ORANGE}) for part in valid_layout.parts
            ]
        }
    )
    changed = changed.model_copy(update={"palette": [Color.ORANGE]})
    result = inspect_layout(valid_layout, changed, board=Board(), repair_count=0)
    assert result.summary.plan_feasible and result.summary.parsed_feasible
    assert result.error_code == ErrorCode.ROUNDTRIP_MISMATCH
    assert result.summary.cell_accuracy < 1.0


@pytest.mark.parametrize("invalid_stage", ["planned", "parsed"])
def test_both_layouts_must_be_feasible(
    valid_layout: Layout, overlap_layout: Layout, invalid_stage: str
) -> None:
    planned = overlap_layout if invalid_stage == "planned" else valid_layout
    parsed = overlap_layout if invalid_stage == "parsed" else valid_layout
    result = inspect_layout(planned, parsed, board=Board(), repair_count=1)
    assert result.error_code == ErrorCode.LAYOUT_INVALID
    assert IssueCode.OVERLAP in result.violations


def test_board_must_match_fixed_run_board(valid_layout: Layout) -> None:
    result = inspect_layout(
        valid_layout, valid_layout, board=Board(cell_size_mm=25), repair_count=0
    )
    assert result.error_code == ErrorCode.LAYOUT_INVALID


def test_matching_feasible_layouts_are_ready(valid_layout: Layout) -> None:
    result = inspect_layout(valid_layout, valid_layout, board=Board(), repair_count=2)
    assert result.error_code is None
    assert result.summary.part_count == 4
    assert result.summary.repair_count == 2


def test_review_summary_lists_each_part_at_its_top_left_cell(valid_layout: Layout) -> None:
    """부품 목록은 실행할 레이아웃의 순서를 따른다. 위치는 감싸는 사각형의 왼쪽 위 셀이다."""
    inspection = inspect_layout(valid_layout, valid_layout, board=Board(), repair_count=0)
    summary = review_summary(inspection.summary, layout=valid_layout)
    assert summary.part_count == inspection.summary.part_count
    assert [(part.part_id, part.row, part.col) for part in summary.parts] == [
        ("P001", 0, 0),
        ("P002", 0, 3),
        ("P003", 4, 4),
        ("P004", 7, 7),
    ]
    assert summary.parts[2].rotation_deg == Rotation.DEG_90
    assert sum(item.count for item in summary.part_counts) == len(valid_layout.parts)


def test_review_summary_of_a_full_board_fits_the_tool_result_limit() -> None:
    """부품 64개의 준비 결과가 모델용 도구 결과 한도 안에 있다."""
    full = make_layout(
        *(
            make_part(f"P{row * 8 + col + 1:03d}", PartType.SQUARE_TILE, origin=(row, col))
            for row in range(8)
            for col in range(8)
        )
    )
    inspection = inspect_layout(full, full, board=Board(), repair_count=0)
    result = LayoutResult(
        run_id="a" * 32,
        tool_call_id="call",
        summary=review_summary(inspection.summary, layout=full),
    )
    size = len(result.model_dump_json().encode("utf-8"))
    assert size <= AgentBudget().max_tool_bytes, f"The result has {size} bytes."


class TestInspectionSummary:
    """카메라 판정의 요약과 재작업 대상."""

    def test_summary_lists_only_parts_that_are_not_a_match(self) -> None:
        """일치한 부품은 수로만 준다. 다른 부품은 상태와 빈 칸 여부를 준다."""
        report = make_image_check_report(
            make_part_image_check(1, status=PartStatus.MATCH, empty=False),
            make_part_image_check(2, status=PartStatus.MISMATCH, empty=True),
            make_part_image_check(3, status=PartStatus.SHADOWED, empty=False),
            make_part_image_check(4, status=PartStatus.MISMATCH, empty=False),
            unexpected=((6, 6),),
        )
        summary = summarize_inspection(report)
        assert not summary.passed
        assert (summary.match_count, summary.shadowed_count, summary.mismatch_count) == (1, 1, 2)
        assert summary.parts == (
            InspectedPart(part_id="P002", status=PartStatus.MISMATCH, empty=True),
            InspectedPart(part_id="P003", status=PartStatus.SHADOWED, empty=False),
            InspectedPart(part_id="P004", status=PartStatus.MISMATCH, empty=False),
        )
        assert summary.unexpected_cell_count == 1 and summary.unexpected_cells == ((6, 6),)

    def test_shadowed_parts_do_not_fail_the_inspection(self) -> None:
        """그림자 부품만 있는 판정은 통과다. 재작업 대상은 없다."""
        report = make_image_check_report(
            make_part_image_check(1, status=PartStatus.MATCH, empty=False),
            make_part_image_check(2, status=PartStatus.SHADOWED, empty=False),
        )
        assert summarize_inspection(report).passed and rework_targets(report) == ()

    def test_targets_are_the_ids_of_the_empty_mismatch_parts(self) -> None:
        """대상은 불일치이고 빈 칸인 부품이다. 순서는 배치 순서다."""
        report = make_image_check_report(
            make_part_image_check(3, status=PartStatus.MISMATCH, empty=True),
            make_part_image_check(1, status=PartStatus.MISMATCH, empty=False),
            make_part_image_check(2, status=PartStatus.MISMATCH, empty=True),
        )
        assert rework_targets(report) == ("P003", "P002")

    def test_unexpected_cell_has_no_targets(self) -> None:
        """계획에 없는 점유 칸이 있으면 대상은 없다."""
        report = make_image_check_report(
            make_part_image_check(1, status=PartStatus.MISMATCH, empty=True), unexpected=((6, 6),)
        )
        assert rework_targets(report) == ()

    @pytest.mark.parametrize("extra", [0, 1])
    def test_lists_stop_at_the_limit_and_then_have_no_targets(self, extra: int) -> None:
        """목록은 상한에서 자른다. 모델이 모든 부품을 보지 못하면 재작업 대상은 없다."""
        count = MAX_INSPECTED_ITEMS + extra
        report = make_image_check_report(
            *(
                make_part_image_check(sequence, status=PartStatus.MISMATCH, empty=True)
                for sequence in range(1, count + 1)
            ),
            unexpected=tuple((7, col) for col in range(count)),
        )
        summary = summarize_inspection(report)
        assert summary.mismatch_count == summary.unexpected_cell_count == count
        assert len(summary.parts) == len(summary.unexpected_cells) == MAX_INSPECTED_ITEMS
        clear = report.model_copy(update={"unexpected_cells": []})
        assert len(rework_targets(clear)) == (0 if extra else MAX_INSPECTED_ITEMS)

    def test_summary_without_a_report_has_a_null_inspection(self) -> None:
        """보고서가 없으면 검사 요약은 None이다. 판정 요약의 값은 그대로다."""
        summary = ExecutionSummary(part_count=2, passed_count=2, passed=True, failed_parts=())
        inspected = inspected_summary(summary, report=None)
        assert inspected.inspection is None
        assert inspected.model_dump(exclude={"inspection"}) == summary.model_dump()
