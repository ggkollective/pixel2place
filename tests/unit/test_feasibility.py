import pytest

from pixel2place.feasibility.checker import IssueCode, apply_report, check_layout
from pixel2place.layout.schema import Color, Layout, Part, PartType
from tests.factories import make_layout, make_part


def _codes(layout: Layout) -> list[IssueCode]:
    return [issue.code for issue in check_layout(layout).issues]


@pytest.mark.unit
class TestCheckLayout:
    """배치 규칙 검사 테스트."""

    def test_success_with_valid_layout(self, valid_layout):
        """규칙을 모두 지킨 레이아웃은 위반이 0건이다."""
        report = check_layout(valid_layout)
        assert report.feasible, f"예상 밖 위반: {report.issues}"

    def test_fail_with_empty_layout(self):
        """부품이 없으면 EMPTY_LAYOUT을 낸다."""
        assert _codes(Layout(palette=[Color.BLUE], parts=[])) == [IssueCode.EMPTY_LAYOUT]

    def test_fail_with_duplicate_part_id(self):
        """같은 part_id가 두 번 있으면 DUPLICATE_PART_ID를 낸다."""
        layout = make_layout(
            make_part("P1", PartType.SQUARE_TILE, origin=(0, 0)),
            make_part("P1", PartType.SQUARE_TILE, origin=(5, 5)),
        )
        assert _codes(layout) == [IssueCode.DUPLICATE_PART_ID]

    def test_fail_with_color_not_in_palette(self):
        """팔레트에 없는 색을 쓰면 COLOR_NOT_IN_PALETTE를 낸다."""
        part = make_part("P1", PartType.SQUARE_TILE, origin=(0, 0), color=Color.ORANGE)
        layout = Layout(palette=[Color.BLUE], parts=[part])
        assert _codes(layout) == [IssueCode.COLOR_NOT_IN_PALETTE]

    def test_fail_with_shape_mismatch(self):
        """셀이 선언한 형태와 다르면 SHAPE_MISMATCH를 낸다."""
        part = Part(part_id="P1", type=PartType.T_TILE, color=Color.BLUE, cells=[(0, 0), (0, 1)])
        assert _codes(make_layout(part)) == [IssueCode.SHAPE_MISMATCH]

    def test_fail_with_out_of_bounds(self):
        """보드 밖의 셀마다 OUT_OF_BOUNDS를 낸다."""
        layout = make_layout(make_part("P1", PartType.RECT_TILE, origin=(7, 7)))
        assert _codes(layout) == [IssueCode.OUT_OF_BOUNDS]

    def test_fail_with_overlap(self):
        """두 부품이 같은 셀을 쓰면 OVERLAP을 낸다."""
        layout = make_layout(
            make_part("P1", PartType.RECT_TILE, origin=(0, 0)),
            make_part("P2", PartType.SQUARE_TILE, origin=(0, 1)),
        )
        report = check_layout(layout)
        assert [issue.code for issue in report.issues] == [IssueCode.OVERLAP]
        assert report.issues[0].part_id == "P2"


@pytest.mark.unit
class TestApplyReport:
    """검사 결과 기록 테스트."""

    def test_records_issues_without_mutating_input(self):
        """위반을 notes에 쓰고 feasible을 False로 둔다. 원본은 바뀌지 않는다."""
        layout = make_layout(make_part("P1", PartType.RECT_TILE, origin=(7, 7)))
        updated = apply_report(layout, check_layout(layout))
        assert updated.feasible is False
        assert updated.notes[0].startswith(IssueCode.OUT_OF_BOUNDS)
        assert layout.feasible is None and layout.notes == []
