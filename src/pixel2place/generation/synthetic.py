"""합성 렌더러. 레이아웃을 평면 조명의 top-down 보드 이미지로 그린다."""

from PIL import Image, ImageDraw

from pixel2place.layout.schema import Color, Layout

RGB = tuple[int, int, int]

COLOR_RGB: dict[Color, RGB] = {
    Color.BLUE: (40, 90, 200),
    Color.GRAY: (130, 130, 130),
    Color.WHITE: (240, 240, 240),
    Color.ORANGE: (240, 140, 30),
}
BOARD_RGB: RGB = (30, 30, 34)
GRID_LINE_RGB: RGB = (70, 70, 76)
PART_OUTLINE_RGB: RGB = (10, 10, 10)

DEFAULT_CELL_PX = 64
GRID_LINE_WIDTH_PX = 1
PART_EDGE_WIDTH_PX = 4

# 셀의 네 변과 각 변 너머에 있는 이웃 셀의 오프셋.
_EDGE_NEIGHBORS = {"top": (-1, 0), "bottom": (1, 0), "left": (0, -1), "right": (0, 1)}


class SyntheticRenderer:
    """부품 셀을 팔레트 색으로 채운다.

    부품 외곽선은 다른 부품이나 빈 셀과 닿는 변에만 그린다.
    그래서 여러 셀로 된 부품이 한 덩어리로 보인다.
    같은 레이아웃은 항상 같은 이미지가 된다.
    """

    def __init__(self, *, cell_px: int = DEFAULT_CELL_PX) -> None:
        self._cell_px = cell_px

    def render(self, layout: Layout) -> Image.Image:
        cell_px = self._cell_px
        width = layout.board.cols * cell_px
        height = layout.board.rows * cell_px
        image = Image.new("RGB", (width, height), BOARD_RGB)
        draw = ImageDraw.Draw(image)

        for col in range(layout.board.cols + 1):
            x = col * cell_px
            draw.line([(x, 0), (x, height)], fill=GRID_LINE_RGB, width=GRID_LINE_WIDTH_PX)
        for row in range(layout.board.rows + 1):
            y = row * cell_px
            draw.line([(0, y), (width, y)], fill=GRID_LINE_RGB, width=GRID_LINE_WIDTH_PX)

        for part in layout.parts:
            fill = COLOR_RGB[part.color]
            for row, col in part.cells:
                left, top = col * cell_px, row * cell_px
                draw.rectangle([left, top, left + cell_px - 1, top + cell_px - 1], fill=fill)

        for part in layout.parts:
            own_cells = set(part.cells)
            for row, col in own_cells:
                for edge, (d_row, d_col) in _EDGE_NEIGHBORS.items():
                    if (row + d_row, col + d_col) not in own_cells:
                        draw.line(
                            _edge_segment(row=row, col=col, edge=edge, cell_px=cell_px),
                            fill=PART_OUTLINE_RGB,
                            width=PART_EDGE_WIDTH_PX,
                        )
        return image


def _edge_segment(*, row: int, col: int, edge: str, cell_px: int) -> list[tuple[int, int]]:
    """셀 한 변의 양 끝점 픽셀 좌표를 반환한다."""
    left, top = col * cell_px, row * cell_px
    right, bottom = left + cell_px - 1, top + cell_px - 1
    match edge:
        case "top":
            return [(left, top), (right, top)]
        case "bottom":
            return [(left, bottom), (right, bottom)]
        case "left":
            return [(left, top), (left, bottom)]
        case _:
            return [(right, top), (right, bottom)]
