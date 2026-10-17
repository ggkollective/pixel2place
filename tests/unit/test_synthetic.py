import pytest

from pixel2place.generation.synthetic import (
    BOARD_RGB,
    COLOR_RGB,
    PART_OUTLINE_RGB,
    SyntheticRenderer,
)
from pixel2place.layout.schema import Color, PartType
from tests.factories import make_layout, make_part

CELL_PX = 32


def _cell_center(row: int, col: int) -> tuple[int, int]:
    return (col * CELL_PX + CELL_PX // 2, row * CELL_PX + CELL_PX // 2)


@pytest.mark.unit
class TestSyntheticRenderer:
    """합성 렌더러 테스트."""

    def test_image_size_follows_board(self, valid_layout):
        """이미지 크기는 (cols x cell_px, rows x cell_px)이다."""
        image = SyntheticRenderer(cell_px=CELL_PX).render(valid_layout)
        assert image.size == (8 * CELL_PX, 8 * CELL_PX)

    def test_cells_use_palette_color(self):
        """부품 셀은 팔레트 색, 빈 셀은 보드 색이다."""
        layout = make_layout(
            make_part("P1", PartType.SQUARE_TILE, origin=(2, 3), color=Color.ORANGE)
        )
        image = SyntheticRenderer(cell_px=CELL_PX).render(layout)
        assert image.getpixel(_cell_center(2, 3)) == COLOR_RGB[Color.ORANGE]
        assert image.getpixel(_cell_center(0, 0)) == BOARD_RGB

    def test_no_edge_between_cells_of_one_part(self):
        """같은 부품의 두 셀 사이에는 외곽선이 없다. 다른 부품 사이에는 있다."""
        layout = make_layout(
            make_part("P1", PartType.RECT_TILE, origin=(0, 0)),
            make_part("P2", PartType.SQUARE_TILE, origin=(0, 2)),
        )
        image = SyntheticRenderer(cell_px=CELL_PX).render(layout)
        middle_y = CELL_PX // 2
        assert image.getpixel((CELL_PX, middle_y)) == COLOR_RGB[Color.BLUE]
        assert image.getpixel((2 * CELL_PX, middle_y)) == PART_OUTLINE_RGB

    def test_same_layout_same_image(self, valid_layout):
        """같은 레이아웃은 항상 같은 이미지가 된다."""
        renderer = SyntheticRenderer(cell_px=CELL_PX)
        assert renderer.render(valid_layout).tobytes() == renderer.render(valid_layout).tobytes()
