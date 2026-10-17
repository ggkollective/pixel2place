import pytest

from pixel2place.layout.parts import CANONICAL_SHAPES, matches_shape, normalize, place, rotate
from pixel2place.layout.schema import Color, Part, PartType, Rotation
from tests.factories import make_part


@pytest.mark.unit
class TestRotate:
    """셀 집합 회전 테스트."""

    def test_l_tile_90_degrees(self):
        """L 타일을 시계 방향 90도 돌리면 가로 3칸과 왼쪽 아래 1칸이 된다."""
        rotated = rotate(CANONICAL_SHAPES[PartType.L_TILE], Rotation.DEG_90)
        assert rotated == {(0, 0), (0, 1), (0, 2), (1, 0)}, f"실제: {sorted(rotated)}"

    @pytest.mark.parametrize("part_type", list(PartType))
    def test_four_quarter_turns_restore_shape(self, part_type):
        """90도 회전을 4번 하면 원래 형태로 돌아온다."""
        shape = CANONICAL_SHAPES[part_type]
        for _ in range(4):
            shape = rotate(shape, Rotation.DEG_90)
        assert shape == CANONICAL_SHAPES[part_type]

    def test_normalize_moves_to_origin(self):
        """정규화하면 최소 row와 최소 col이 0이 된다."""
        assert normalize([(3, 5), (4, 5)]) == {(0, 0), (1, 0)}


@pytest.mark.unit
class TestMatchesShape:
    """부품 형태 일치 검사 테스트."""

    @pytest.mark.parametrize("part_type", list(PartType))
    @pytest.mark.parametrize("rotation", list(Rotation))
    def test_success_with_placed_part(self, part_type, rotation):
        """`place`로 만든 셀은 선언한 종류·회전과 일치한다."""
        part = make_part("P1", part_type, origin=(2, 2), rotation=rotation)
        assert matches_shape(part), f"{part_type} {rotation} 불일치: {part.cells}"

    def test_fail_with_wrong_rotation(self):
        """셀은 L 타일 90도인데 0도로 선언하면 불일치다."""
        cells = place(PartType.L_TILE, Rotation.DEG_90, origin=(0, 0))
        part = Part(part_id="P1", type=PartType.L_TILE, color=Color.BLUE, cells=cells)
        assert not matches_shape(part)

    def test_fail_with_duplicate_cells(self):
        """같은 셀이 두 번 들어 있으면 불일치다."""
        part = Part(part_id="P1", type=PartType.RECT_TILE, color=Color.BLUE, cells=[(0, 0), (0, 0)])
        assert not matches_shape(part)
