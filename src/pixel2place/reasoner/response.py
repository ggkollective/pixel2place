"""Reasoner 응답 스키마.

모델은 부품 셀 목록을 쓰지 않는다. 부품 셀은 코드가 형태 표로 계산한다.
그래서 모델 응답에는 형태 불일치가 생기지 않는다.

- 배치 계획 모델: 부품마다 종류, 색, 회전, 기준 셀을 쓴다.
- 판독기: 부품마다 종류, 색, 회전, 이미지 상대 좌표의 bbox를 쓴다. 코드가 bbox를 격자에 맞춘다.
"""

from collections.abc import Callable
from typing import Annotated, Any

from pydantic import BaseModel, ConfigDict, Field

from pixel2place.layout.parts import place, shape_of
from pixel2place.layout.schema import Board, Color, Layout, Part, PartType, Rotation

PART_ID_FORMAT = "P{index:03d}"
RESPONSE_SCHEMA_NAME = "layout"
# bbox 좌표의 최댓값. Qwen3-VL grounding 형식처럼 이미지 크기를 0–1000으로 나타낸다.
BBOX_SCALE = 1000
BBOX_LENGTH = 4

BboxCoordinate = Annotated[int, Field(ge=0, le=BBOX_SCALE)]
LayoutConverter = Callable[[dict[str, Any], Board], Layout]
"""응답 JSON을 검증하고 `Layout`으로 바꾸는 함수."""


class _ResponseModel(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)


class PlannerPart(_ResponseModel):
    """planner가 쓰는 부품 1개.

    `row`, `col`은 회전한 형태를 둘러싸는 가장 작은 사각형의 왼쪽 위 셀이다.
    회전에 따라 이 셀이 비어 있을 수 있다. 예: T_tile 180도.
    """

    type: PartType
    color: Color
    row: int = Field(ge=0)
    col: int = Field(ge=0)
    rotation_deg: Rotation


class PlannerLayout(_ResponseModel):
    """배치 계획 모델 응답 1개."""

    parts: list[PlannerPart] = Field(min_length=1)


class ParserPart(_ResponseModel):
    """판독기가 쓰는 부품 1개. `bbox_2d`는 `[x1, y1, x2, y2]`, 이미지 상대 좌표 0–1000이다."""

    bbox_2d: list[BboxCoordinate] = Field(min_length=BBOX_LENGTH, max_length=BBOX_LENGTH)
    type: PartType
    color: Color
    rotation_deg: Rotation


class ParserLayout(_ResponseModel):
    """판독기 응답 1개."""

    parts: list[ParserPart] = Field(min_length=1)


def planner_json_schema(board: Board) -> dict[str, Any]:
    """배치 계획 모델 요청의 vLLM structured output JSON Schema를 만든다.

    정수에 상한이 없으면 모델이 자릿수를 끝없이 늘릴 수 있다.
    그래서 `row`, `col`, 부품 수에 보드 크기로 상한을 둔다.
    """
    schema = _schema_without_descriptions(PlannerLayout, board=board)
    part_properties = schema["$defs"][PlannerPart.__name__]["properties"]
    part_properties["row"]["maximum"] = board.rows - 1
    part_properties["col"]["maximum"] = board.cols - 1
    return schema


def parser_json_schema(board: Board) -> dict[str, Any]:
    """판독기 요청의 vLLM structured output JSON Schema를 만든다. 좌표 상한은 `BBOX_SCALE`이다."""
    return _schema_without_descriptions(ParserLayout, board=board)


def _schema_without_descriptions(model: type[BaseModel], *, board: Board) -> dict[str, Any]:
    schema = model.model_json_schema()
    # description에는 한국어 docstring이 들어간다. 서버에 보내는 값은 영어만 쓴다.
    for definition in [schema, *schema["$defs"].values()]:
        definition.pop("description", None)
    schema["properties"]["parts"]["maxItems"] = board.rows * board.cols
    return schema


def planner_to_layout(response: PlannerLayout, *, board: Board) -> Layout:
    """배치 계획 모델 응답을 `Layout`으로 바꾼다. 부품 ID는 응답 순서대로 붙인다."""
    parts = [
        _part(index, part.type, part.color, part.rotation_deg, origin=(part.row, part.col))
        for index, part in enumerate(response.parts, start=1)
    ]
    return _layout(parts, board=board)


def parser_to_layout(response: ParserLayout, *, board: Board) -> Layout:
    """판독기 응답의 bbox를 격자에 맞추고 `Layout`으로 바꾼다.

    단계:
    1. bbox 경계를 가장 가까운 격자선으로 옮긴다. 높이와 폭은 최소 1셀이다.
    2. 모델의 회전으로 만든 형태의 크기가 bbox 크기와 같으면 그 회전을 쓴다.
    3. 다르면 크기가 같은 첫 회전을 쓴다. 그런 회전이 없으면 모델의 회전을 쓴다.
    4. bbox의 왼쪽 위 셀에 형태를 놓는다.
    """
    parts = []
    for index, part in enumerate(response.parts, start=1):
        x1, y1, x2, y2 = part.bbox_2d
        top, bottom = _to_grid(y1, board.rows), _to_grid(y2, board.rows)
        left, right = _to_grid(x1, board.cols), _to_grid(x2, board.cols)
        size = (max(1, bottom - top), max(1, right - left))
        rotation = _rotation_for_size(part.type, part.rotation_deg, size=size)
        parts.append(_part(index, part.type, part.color, rotation, origin=(top, left)))
    return _layout(parts, board=board)


def parse_planner_response(data: dict[str, Any], board: Board) -> Layout:
    """배치 계획 모델 응답 JSON을 검증하고 `Layout`으로 바꾼다."""
    return planner_to_layout(PlannerLayout.model_validate(data), board=board)


def parse_parser_response(data: dict[str, Any], board: Board) -> Layout:
    """판독기 응답 JSON을 검증하고 `Layout`으로 바꾼다."""
    return parser_to_layout(ParserLayout.model_validate(data), board=board)


def _to_grid(coordinate: int, cells: int) -> int:
    return round(coordinate / BBOX_SCALE * cells)


def _shape_size(part_type: PartType, rotation: Rotation) -> tuple[int, int]:
    shape = shape_of(part_type, rotation)
    return (max(row for row, _ in shape) + 1, max(col for _, col in shape) + 1)


def _rotation_for_size(
    part_type: PartType, rotation: Rotation, *, size: tuple[int, int]
) -> Rotation:
    if _shape_size(part_type, rotation) == size:
        return rotation
    for candidate in Rotation:
        if _shape_size(part_type, candidate) == size:
            return candidate
    return rotation


def _part(
    index: int, part_type: PartType, color: Color, rotation: Rotation, *, origin: tuple[int, int]
) -> Part:
    return Part(
        part_id=PART_ID_FORMAT.format(index=index),
        type=part_type,
        color=color,
        cells=place(part_type, rotation, origin=origin),
        rotation_deg=rotation,
    )


def _layout(parts: list[Part], *, board: Board) -> Layout:
    palette = sorted({part.color for part in parts})
    return Layout(board=board, palette=palette, parts=parts)
