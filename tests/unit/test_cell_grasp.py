import numpy as np
import pytest

from pixel2place.cell.grasp import canonical_grasp, centroid, placed_grasp
from pixel2place.layout.parts import shape_of
from pixel2place.layout.schema import PartType, Rotation

# 20 mm 셀에서 손가락 반폭 5 mm와 여유 1 mm
HALF_WIDTH = 0.3
# 셀 경계 위의 점을 피하는 작은 값
EDGE = 1e-6


def _cells_on_line(cells, *, point, axis):
    """파지점을 지나 닫는 방향으로 놓인 선이 지나는 부품 셀."""
    u, v = point
    # 회전한 축에는 부동소수점 잔차가 있다. 그래서 크기로 u 방향인지 판단한다.
    if abs(axis[0]) > abs(axis[1]):
        return [(row, col) for row, col in cells if row <= v < row + 1]
    return [(row, col) for row, col in cells if col <= u < col + 1]


@pytest.mark.unit
class TestGrasp:
    """파지점 테스트."""

    @pytest.mark.parametrize("part_type", list(PartType))
    @pytest.mark.parametrize("rotation", list(Rotation))
    def test_grasp_point_is_on_part(self, part_type, rotation):
        """회전한 부품의 파지점은 부품 셀 안에 있다. jaw 방향은 단위 벡터다."""
        grasp = placed_grasp(part_type, rotation, finger_half_width=HALF_WIDTH)
        cells = shape_of(part_type, rotation)
        u, v = centroid(cells) + grasp.offset
        assert any(col <= u <= col + 1 and row <= v <= row + 1 for row, col in cells)
        assert np.linalg.norm(grasp.axis) == pytest.approx(1.0)

    @pytest.mark.parametrize("part_type", list(PartType))
    @pytest.mark.parametrize("rotation", list(Rotation))
    def test_fingers_cross_one_cell(self, part_type, rotation):
        """손가락 띠의 모든 선은 부품 셀 1개만 지난다. jaw가 부품의 다른 셀 위에 내려앉지 않는다."""
        grasp = placed_grasp(part_type, rotation, finger_half_width=HALF_WIDTH)
        cells = shape_of(part_type, rotation)
        point = centroid(cells) + grasp.offset
        across = np.array([-grasp.axis[1], grasp.axis[0]])
        for shift in (-HALF_WIDTH + EDGE, 0.0, HALF_WIDTH - EDGE):
            line_point = point + across * shift
            assert len(_cells_on_line(cells, point=line_point, axis=grasp.axis)) == 1

    @pytest.mark.parametrize(
        "part_type,offset,axis",
        [
            (PartType.SQUARE_TILE, (0.0, 0.0), (1.0, 0.0)),
            (PartType.RECT_TILE, (0.0, 0.0), (0.0, 1.0)),
            (PartType.L_TILE, (-0.25, -0.05), (1.0, 0.0)),
            (PartType.T_TILE, (0.0, 0.55), (1.0, 0.0)),
        ],
    )
    def test_canonical_grasp(self, part_type, offset, axis):
        """손가락 띠가 자유 셀 안에 있는 파지 중에서 centroid에 가장 가까운 파지를 고른다.

        T는 막대 가운데를 잡으면 jaw가 기둥 셀 위에 내려앉는다. 그래서 기둥을 잡는다.
        """
        grasp = canonical_grasp(part_type, finger_half_width=HALF_WIDTH)
        np.testing.assert_allclose(grasp.offset, offset, atol=1e-12)
        np.testing.assert_allclose(grasp.axis, axis, atol=1e-12)

    def test_rotation_turns_axis(self):
        """90도 회전은 (u, v) 좌표에서 u축을 v축으로 돌린다."""
        grasp = placed_grasp(PartType.SQUARE_TILE, Rotation.DEG_90, finger_half_width=HALF_WIDTH)
        np.testing.assert_allclose(grasp.axis, (0.0, 1.0), atol=1e-12)

    def test_too_wide_fingers_raise(self):
        """손가락 띠가 셀보다 넓으면 파지가 없다."""
        with pytest.raises(ValueError, match="no grasp"):
            canonical_grasp(PartType.SQUARE_TILE, finger_half_width=0.6)
