import pytest

from pixel2place.layout.schema import Board, Color, PartType, Rotation
from pixel2place.reasoner.response import (
    ParserLayout,
    ParserPart,
    PlannerLayout,
    PlannerPart,
    parser_json_schema,
    parser_to_layout,
    planner_json_schema,
    planner_to_layout,
)


def _part(part_type: PartType, *, row: int, col: int, rotation: Rotation, color: Color):
    return PlannerPart(type=part_type, color=color, row=row, col=col, rotation_deg=rotation)


@pytest.mark.unit
class TestToLayout:
    """Reasoner 응답 → Layout 변환 테스트."""

    def test_computes_cells_ids_and_palette(self):
        """형태 표로 셀을 계산하고 ID를 순서대로 붙이고 팔레트를 부품 색으로 채운다."""
        response = PlannerLayout(
            parts=[
                _part(PartType.L_TILE, row=1, col=2, rotation=Rotation.DEG_0, color=Color.GRAY),
                _part(
                    PartType.SQUARE_TILE, row=0, col=0, rotation=Rotation.DEG_0, color=Color.BLUE
                ),
            ]
        )
        board = Board(rows=6, cols=6)
        layout = planner_to_layout(response, board=board)
        assert layout.board == board
        assert [part.part_id for part in layout.parts] == ["P001", "P002"]
        assert layout.parts[0].cells == [(1, 2), (2, 2), (3, 2), (3, 3)]
        assert layout.palette == [Color.BLUE, Color.GRAY]

    def test_origin_is_bounding_box_corner(self):
        """기준 셀은 외접 사각형의 왼쪽 위다. T_tile 180도는 기준 셀을 비워 둔다."""
        response = PlannerLayout(
            parts=[
                _part(PartType.T_TILE, row=2, col=3, rotation=Rotation.DEG_180, color=Color.BLUE)
            ]
        )
        cells = planner_to_layout(response, board=Board()).parts[0].cells
        assert cells == [(2, 4), (3, 3), (3, 4), (3, 5)]


@pytest.mark.unit
class TestResponseJsonSchema:
    """structured output용 JSON Schema 테스트."""

    def test_bounds_follow_board(self):
        """row, col, 부품 수의 상한이 보드 크기를 따른다."""
        schema = planner_json_schema(Board(rows=6, cols=7))
        properties = schema["$defs"]["PlannerPart"]["properties"]
        assert (properties["row"]["maximum"], properties["col"]["maximum"]) == (5, 6)
        assert schema["properties"]["parts"]["maxItems"] == 42

    def test_has_no_descriptions(self):
        """한국어 docstring을 서버에 보내지 않는다."""
        schema = planner_json_schema(Board())
        definitions = [schema, *schema["$defs"].values()]
        assert all("description" not in definition for definition in definitions)


def _parser_part(part_type: PartType, bbox: list[int], *, rotation: Rotation) -> ParserPart:
    return ParserPart(bbox_2d=bbox, type=part_type, color=Color.BLUE, rotation_deg=rotation)


@pytest.mark.unit
class TestParserToLayout:
    """판독기 응답 bbox → Layout 변환 테스트."""

    def test_exact_bbox(self):
        """셀 경계와 같은 bbox는 그 셀에 부품을 놓는다."""
        response = ParserLayout(
            parts=[_parser_part(PartType.L_TILE, [250, 125, 500, 500], rotation=Rotation.DEG_0)]
        )
        layout = parser_to_layout(response, board=Board())
        assert layout.parts[0].cells == [(1, 2), (2, 2), (3, 2), (3, 3)]
        assert layout.parts[0].part_id == "P001"

    def test_snaps_to_nearest_grid_line(self):
        """bbox 경계를 가장 가까운 격자선으로 옮긴다."""
        response = ParserLayout(
            parts=[_parser_part(PartType.RECT_TILE, [130, 240, 370, 380], rotation=Rotation.DEG_0)]
        )
        # x 130 -> 1, 370 -> 3, y 240 -> 2, 380 -> 3
        assert parser_to_layout(response, board=Board()).parts[0].cells == [(2, 1), (2, 2)]

    def test_corrects_rotation_to_bbox_size(self):
        """모델의 회전이 bbox 크기와 맞지 않으면 크기가 같은 회전을 쓴다."""
        # 2행 3열 bbox. L_tile 0도는 3행 2열이다.
        response = ParserLayout(
            parts=[_parser_part(PartType.L_TILE, [0, 0, 375, 250], rotation=Rotation.DEG_0)]
        )
        part = parser_to_layout(response, board=Board()).parts[0]
        assert part.rotation_deg == Rotation.DEG_90
        assert max(row for row, _ in part.cells) == 1

    def test_keeps_rotation_when_no_size_matches(self):
        """크기가 같은 회전이 없으면 모델의 회전을 그대로 쓴다."""
        response = ParserLayout(
            parts=[_parser_part(PartType.T_TILE, [0, 0, 500, 500], rotation=Rotation.DEG_180)]
        )
        part = parser_to_layout(response, board=Board()).parts[0]
        assert part.rotation_deg == Rotation.DEG_180
        assert part.cells == [(0, 1), (1, 0), (1, 1), (1, 2)]


@pytest.mark.unit
class TestParserJsonSchema:
    """판독기 structured output JSON Schema 테스트."""

    def test_bbox_bounds_and_part_limit(self):
        """bbox는 0–1000 정수 4개이고 부품 수 상한은 보드 셀 수다."""
        schema = parser_json_schema(Board(rows=6, cols=7))
        bbox = schema["$defs"]["ParserPart"]["properties"]["bbox_2d"]
        assert (bbox["minItems"], bbox["maxItems"]) == (4, 4)
        assert (bbox["items"]["minimum"], bbox["items"]["maximum"]) == (0, 1000)
        assert schema["properties"]["parts"]["maxItems"] == 42
        assert all("description" not in d for d in [schema, *schema["$defs"].values()])
