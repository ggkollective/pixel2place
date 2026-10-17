import pytest

from pixel2place.feasibility.checker import check_layout
from pixel2place.feasibility.repair import RepairAction, repair_layout
from pixel2place.layout.schema import Board, Color, Layout, PartType, Rotation
from tests.factories import make_layout, make_part


@pytest.mark.unit
class TestRepairLayout:
    """배치 위반 자동 수리 테스트."""

    def test_feasible_layout_is_unchanged(self, valid_layout):
        """위반이 없으면 같은 레이아웃과 빈 수리 기록을 반환한다."""
        result = repair_layout(valid_layout)
        assert result.layout == valid_layout
        assert result.repairs == []

    def test_moves_overlap_to_nearest_free_origin(self, overlap_layout):
        """겹친 뒤 부품을 가장 가까운 빈 위치로 옮긴다. 앞 부품은 그대로 둔다."""
        result = repair_layout(overlap_layout)
        first, second = result.layout.parts
        assert first == overlap_layout.parts[0]
        # (0, 0)에서 거리 1인 위치는 (0, 1)과 (1, 0)이다. row가 작은 (0, 1)을 쓴다.
        assert second.cells == [(0, 1), (0, 2)]
        assert second.type == PartType.RECT_TILE
        assert second.rotation_deg == overlap_layout.parts[1].rotation_deg
        assert check_layout(result.layout).feasible
        (repair,) = result.repairs
        assert (repair.action, repair.part_id) == (RepairAction.MOVED, "P002")
        assert (repair.from_origin, repair.to_origin) == ((0, 0), (0, 1))
        assert repair.message == "Moved part P002 from [0, 0] to [0, 1]."

    def test_moves_only_violating_parts(self):
        """옮긴 부품이 뒤의 정상 부품 자리를 차지하지 않는다."""
        layout = make_layout(
            make_part("P001", PartType.SQUARE_TILE, origin=(0, 0)),
            make_part("P002", PartType.SQUARE_TILE, origin=(0, 0)),
            make_part("P003", PartType.SQUARE_TILE, origin=(0, 1), color=Color.GRAY),
        )
        result = repair_layout(layout)
        assert [repair.part_id for repair in result.repairs] == ["P002"]
        # (0, 1)은 P003이 쓴다. 그래서 다음으로 가까운 (1, 0)을 쓴다.
        assert result.repairs[0].to_origin == (1, 0)
        assert [part.part_id for part in result.layout.parts] == ["P001", "P002", "P003"]
        assert result.layout.parts[2] == layout.parts[2]

    def test_moves_out_of_bounds_part_inside(self):
        """보드 밖으로 나간 부품을 같은 회전으로 보드 안에 놓는다."""
        layout = make_layout(
            make_part("P001", PartType.L_TILE, origin=(6, 7), rotation=Rotation.DEG_90)
        )
        result = repair_layout(layout)
        (part,) = result.layout.parts
        assert part.rotation_deg == Rotation.DEG_90
        assert check_layout(result.layout).feasible
        assert result.repairs[0].to_origin == (6, 5)

    def test_removes_part_without_free_position(self):
        """놓을 자리가 없으면 부품을 지우고 팔레트에서 그 색을 뺀다."""
        board = Board(rows=1, cols=2)
        parts = [
            make_part("P001", PartType.RECT_TILE, origin=(0, 0)),
            make_part("P002", PartType.SQUARE_TILE, origin=(0, 1), color=Color.GRAY),
        ]
        layout = Layout(board=board, palette=[Color.BLUE, Color.GRAY], parts=parts)
        result = repair_layout(layout)
        assert [part.part_id for part in result.layout.parts] == ["P001"]
        assert result.layout.palette == [Color.BLUE]
        (repair,) = result.repairs
        assert (repair.action, repair.to_origin) == (RepairAction.REMOVED, None)
        assert repair.message == "Removed part P002 at [0, 1]. No free position."

    def test_keeps_palette_when_all_parts_removed(self):
        """부품이 모두 없어지면 원래 팔레트를 둔다."""
        board = Board(rows=1, cols=1)
        layout = Layout(
            board=board,
            palette=[Color.WHITE],
            parts=[make_part("P001", PartType.RECT_TILE, origin=(0, 0), color=Color.WHITE)],
        )
        result = repair_layout(layout)
        assert result.layout.parts == []
        assert result.layout.palette == [Color.WHITE]

    def test_does_not_change_input(self, overlap_layout):
        """입력 레이아웃은 바꾸지 않는다."""
        before = overlap_layout.model_copy(deep=True)
        repair_layout(overlap_layout)
        assert overlap_layout == before
