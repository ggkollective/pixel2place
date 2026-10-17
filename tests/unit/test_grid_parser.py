import logging

import pytest
from PIL import ImageDraw

from pixel2place.benchmark import generate_layouts
from pixel2place.generation.synthetic import COLOR_RGB, SyntheticRenderer
from pixel2place.layout.schema import Board, Color, Layout, PartType
from pixel2place.parsing.grid import GridImageParser, GridParseError
from tests.factories import make_layout, make_part


def _signature(layout: Layout) -> list[tuple[str, str, tuple[tuple[int, int], ...]]]:
    """회전 표기와 부품 ID를 뺀 비교 값. 대칭 형태는 여러 회전이 같은 셀을 만든다."""
    return sorted((p.type.value, p.color.value, tuple(sorted(p.cells))) for p in layout.parts)


@pytest.mark.unit
class TestGridImageParser:
    """픽셀 기반 격자 판독기 테스트."""

    @pytest.mark.parametrize("cell_px", [16, 64])
    def test_reads_generated_layouts_exactly(self, cell_px):
        """생성한 레이아웃을 그린 이미지에서 모든 부품의 종류, 색, 셀을 읽는다."""
        renderer = SyntheticRenderer(cell_px=cell_px)
        for layout in generate_layouts(count=48, seed=7):
            parsed = GridImageParser().parse(renderer.render(layout))
            assert _signature(parsed) == _signature(layout)

    def test_splits_touching_parts_with_same_color(self):
        """같은 색 부품이 닿아 있어도 외곽선으로 나눈다."""
        layout = make_layout(
            make_part("P1", PartType.RECT_TILE, origin=(0, 0)),
            make_part("P2", PartType.RECT_TILE, origin=(0, 2)),
        )
        parsed = GridImageParser().parse(SyntheticRenderer().render(layout))
        assert _signature(parsed) == _signature(layout)

    def test_assigns_ids_in_reading_order(self):
        """부품 ID는 첫 셀의 위→아래, 왼→오른쪽 순서로 붙인다."""
        layout = make_layout(
            make_part("A", PartType.SQUARE_TILE, origin=(5, 0)),
            make_part("B", PartType.SQUARE_TILE, origin=(1, 6), color=Color.GRAY),
        )
        parsed = GridImageParser().parse(SyntheticRenderer().render(layout))
        assert [(p.part_id, p.cells) for p in parsed.parts] == [
            ("P001", [(1, 6)]),
            ("P002", [(5, 0)]),
        ]

    def test_empty_board_raises(self):
        """부품이 없는 이미지는 GridParseError를 낸다."""
        empty = SyntheticRenderer().render(Layout(palette=[Color.BLUE], parts=[]))
        with pytest.raises(GridParseError):
            GridImageParser().parse(empty)

    def test_unknown_shape_goes_to_notes(self, caplog):
        """어휘에 없는 형태는 부품으로 만들지 않고 notes와 로그에 남긴다."""
        layout = make_layout(make_part("P1", PartType.SQUARE_TILE, origin=(6, 6)))
        image = SyntheticRenderer().render(layout)
        # 외곽선 없는 2x2 파란 덩어리를 직접 그린다.
        ImageDraw.Draw(image).rectangle([0, 0, 127, 127], fill=COLOR_RGB[Color.BLUE])
        with caplog.at_level(logging.WARNING):
            parsed = GridImageParser().parse(image)
        assert _signature(parsed) == _signature(layout)
        assert parsed.notes == ["Unknown shape at cells [[0, 0], [0, 1], [1, 0], [1, 1]]."]
        assert "unknown shape" in caplog.text

    def test_board_size_follows_parser_board(self):
        """보드 크기가 다르면 셀 크기를 그 보드에 맞춰 계산한다."""
        board = Board(rows=4, cols=6)
        layout = Layout(
            board=board,
            palette=[Color.WHITE],
            parts=[make_part("P1", PartType.T_TILE, origin=(1, 2), color=Color.WHITE)],
        )
        image = SyntheticRenderer(cell_px=20).render(layout)
        parsed = GridImageParser(board=board).parse(image)
        assert parsed.board == board
        assert _signature(parsed) == _signature(layout)
