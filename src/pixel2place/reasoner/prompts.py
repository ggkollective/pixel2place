"""패키지에 포함된 reasoner 시스템 지시문을 읽고 자리 표시자를 채운다."""

import json
from importlib.resources import files
from typing import Literal

import yaml

from pixel2place.layout.parts import CANONICAL_SHAPES, shape_of
from pixel2place.layout.schema import Board, Color, Rotation

PromptName = Literal["planner", "parser", "expand_image", "expand_video"]

PROMPT_PACKAGE = "pixel2place.prompts"
SHAPE_TABLE_INDENT = "  "

# 배치 계획 모델 응답 예시. 형식은 `pixel2place.reasoner.response.PlannerLayout`이다.
PLANNER_EXAMPLE = {
    "parts": [{"type": "L_tile", "color": "blue", "row": 1, "col": 2, "rotation_deg": 0}]
}
# 판독기 응답 예시. 형식은 `pixel2place.reasoner.response.ParserLayout`이다.
# 8x8 보드에서 PLANNER_EXAMPLE과 같은 부품이다. 행 1–3, 열 2–3을 채운다.
PARSER_EXAMPLE = {
    "parts": [
        {"bbox_2d": [250, 125, 500, 500], "type": "L_tile", "color": "blue", "rotation_deg": 0}
    ]
}


def load_system_prompt(name: PromptName, *, board: Board) -> str:
    """`<name>.yaml`의 `system` 값을 읽고 보드·부품·팔레트 정보를 채운다.

    지시문 안의 JSON 예시에도 중괄호가 있다. 그래서 `str.format` 대신 이름으로 치환한다.
    """
    raw = files(PROMPT_PACKAGE).joinpath(f"{name}.yaml").read_text(encoding="utf-8")
    template: str = yaml.safe_load(raw)["system"]
    replacements = {
        "{rows}": str(board.rows),
        "{cols}": str(board.cols),
        "{colors}": ", ".join(color.value for color in Color),
        "{shape_table}": _shape_table(),
        "{schema_example}": json.dumps(PLANNER_EXAMPLE),
        "{parser_example}": json.dumps(PARSER_EXAMPLE),
    }
    for placeholder, value in replacements.items():
        template = template.replace(placeholder, value)
    return template


def _shape_table() -> str:
    """부품 종류마다 회전별 셀 목록을 한 줄로 쓴다."""
    lines = []
    for part_type in CANONICAL_SHAPES:
        by_rotation = "; ".join(
            f"{int(rotation)}: {sorted([list(cell) for cell in shape_of(part_type, rotation)])}"
            for rotation in Rotation
        )
        lines.append(f"{SHAPE_TABLE_INDENT}- {part_type.value} -> {by_rotation}")
    return "\n".join(lines).lstrip()
