"""작업 주문을 레이아웃과 영어 지시문으로 바꾼다.

주문은 폼 입력의 계약이다. 부품 형태는 `layout.parts`가, 배치 규칙은 `feasibility`가 소유한다.
지시문은 모델에 그대로 전달하는 원문이다. 같은 입력은 항상 같은 문장을 만든다.
보드 주문은 사용자가 위치를 정한다. 키트 주문은 planner가 위치를 정한다.
"""

from collections.abc import Sequence
from enum import StrEnum
from typing import Annotated, Literal, Self

from pydantic import AfterValidator, Field, TypeAdapter, model_validator

from pixel2place.agent.contracts import Contract
from pixel2place.feasibility.checker import Issue, check_layout
from pixel2place.layout.parts import canonical_cols, place
from pixel2place.layout.schema import Board, Color, Layout, Part, PartType, Rotation

# 기준 실행의 부품 수다. Isaac 배치 timeout 안에 끝난다.
MAX_ORDER_PARTS = 16
# 실제 배치 계획 모델 시험이 통과한 키트의 범위다.
# 이 범위 밖에서는 자동 수리나 수량 불일치가 나왔다.
MAX_KIT_PARTS = 6
# 키트가 받는 종류와 그 색이다. 지시문은 이 순서로 종류를 쓴다.
KIT_COLORS: dict[PartType, Color] = {
    PartType.SQUARE_TILE: Color.BLUE,
    PartType.RECT_TILE: Color.GRAY,
}
# 종류별 수량의 한도다. 키트는 한 종류를 한 row에 놓는다. 한 row에 놓을 수 없는 수량은 받지 않는다.
KIT_MAX_COUNTS: dict[PartType, int] = {
    part_type: min(MAX_KIT_PARTS, Board().cols // canonical_cols(part_type))
    for part_type in KIT_COLORS
}

_COUNT_WORDS = (
    "zero",
    "one",
    "two",
    "three",
    "four",
    "five",
    "six",
    "seven",
    "eight",
    "nine",
    "ten",
    "eleven",
    "twelve",
    "thirteen",
    "fourteen",
    "fifteen",
    "sixteen",
)
PART_NOUNS: dict[PartType, str] = {
    PartType.SQUARE_TILE: "square tile",
    PartType.RECT_TILE: "rectangular tile",
    PartType.L_TILE: "L tile",
    PartType.T_TILE: "T tile",
}
_VOWELS = "aeiou"
_VOWEL_NUMBERS = ("11", "18")


class Arrangement(StrEnum):
    """키트 주문의 배치 방식. 실제 배치 계획 모델 시험이 통과한 방식만 있다."""

    TYPE_ROWS = "type_rows"


ARRANGEMENT_SENTENCES: dict[Arrangement, str] = {
    Arrangement.TYPE_ROWS: "Put parts of the same type in the same row.",
}


class OrderPart(Contract):
    """보드에 놓을 부품 1개. 위치는 부품을 감싸는 최소 사각형의 왼쪽 위 셀이다."""

    type: PartType
    color: Color
    rotation_deg: Rotation = Rotation.DEG_0
    row: Annotated[int, Field(ge=0)]
    col: Annotated[int, Field(ge=0)]


class BoardOrder(Contract):
    """부품마다 위치를 정한 주문. 보드 경계와 겹침은 `check_layout`이 검사한다."""

    kind: Literal["board"] = "board"
    parts: Annotated[tuple[OrderPart, ...], Field(max_length=MAX_ORDER_PARTS)]

    @property
    def part_count(self) -> int:
        return len(self.parts)


def _kit_part_type(part_type: PartType) -> PartType:
    if part_type not in KIT_COLORS:
        raise ValueError("A kit order cannot have this part type.")
    return part_type


class KitItem(Contract):
    """키트의 부품 종류 1개와 그 수량. 색은 종류가 정한다."""

    type: Annotated[PartType, AfterValidator(_kit_part_type)]
    count: Annotated[int, Field(ge=1, le=MAX_KIT_PARTS)]

    @model_validator(mode="after")
    def _check_count(self) -> Self:
        limit = KIT_MAX_COUNTS[self.type]
        if self.count > limit:
            raise ValueError(f"A kit order can have a maximum of {limit} parts of this type.")
        return self

    @property
    def color(self) -> Color:
        return KIT_COLORS[self.type]


class KitOrder(Contract):
    """수량과 배치 방식만 정한 주문. 위치는 planner가 정한다.

    주문은 배치 방식의 충족을 보장하지 않는다. 비교는 종류와 색별 수량만 본다.
    """

    kind: Literal["kit"]
    items: Annotated[tuple[KitItem, ...], Field(min_length=1)]
    arrangement: Arrangement

    @model_validator(mode="after")
    def _check_items(self) -> Self:
        types = [item.type for item in self.items]
        if len(set(types)) != len(types):
            raise ValueError("A kit order can have each part type only one time.")
        if self.part_count > MAX_KIT_PARTS:
            raise ValueError(f"A kit order can have a maximum of {MAX_KIT_PARTS} parts.")
        return self

    @property
    def part_count(self) -> int:
        return sum(item.count for item in self.items)


Order = BoardOrder | KitOrder
_ORDER_ADAPTER: TypeAdapter[Order] = TypeAdapter(Order)


def parse_order(text: str | bytes) -> Order:
    """주문 JSON을 보드 주문 또는 키트 주문으로 읽는다. `kind`가 없으면 보드 주문이다.

    Raises:
        pydantic.ValidationError: 본문이 두 계약 모두와 다를 때.
    """
    return _ORDER_ADAPTER.validate_json(text)


def order_layout(order: BoardOrder) -> Layout:
    """주문을 기본 보드의 레이아웃으로 바꾼다. 배치 규칙은 검사하지 않는다."""
    parts = [
        Part(
            part_id=f"P{index:03d}",
            type=part.type,
            color=part.color,
            cells=place(part.type, part.rotation_deg, origin=(part.row, part.col)),
            rotation_deg=part.rotation_deg,
        )
        for index, part in enumerate(order.parts, start=1)
    ]
    return Layout(palette=list(Color), parts=parts)


class OrderNotFeasibleError(ValueError):
    """보드 주문이 배치 규칙을 어긴다. `issues`에 위반 목록이 있다."""

    def __init__(self, issues: Sequence[Issue]) -> None:
        super().__init__("The work order is not feasible.")
        self.issues = tuple(issues)


def order_instruction(order: Order) -> str:
    """주문의 지시문을 쓴다. 보드 주문은 배치 규칙을 먼저 검사한다.

    키트 주문에는 위치가 없다. 배치 규칙은 planner의 레이아웃에서 검사한다.

    Raises:
        OrderNotFeasibleError: 보드 주문이 배치 규칙을 어길 때.
    """
    if isinstance(order, KitOrder):
        return compose_kit_instruction(order)
    layout = order_layout(order)
    issues = check_layout(layout).issues
    if issues:
        raise OrderNotFeasibleError(issues)
    return compose_instruction(layout)


def compose_instruction(layout: Layout) -> str:
    """레이아웃의 모든 부품과 위치를 영어 지시문으로 쓴다.

    종류, 색과 회전이 같은 부품을 한 문장으로 묶는다. 문장 순서는 부품 순서를 따른다.
    회전이 모두 0도이면 회전을 끝에 한 번만 쓴다.
    위치는 planner의 출력 필드와 같은 row, col이다. 셀 목록은 쓰지 않는다.
    planner는 셀 목록의 첫 셀을 위치로 읽어서 회전한 부품을 옮긴다.

    Raises:
        ValueError: 레이아웃에 부품이 없을 때.
    """
    if not layout.parts:
        raise ValueError("Layout has no parts.")
    groups: dict[tuple[PartType, Color, Rotation], list[Part]] = {}
    for part in layout.parts:
        groups.setdefault((part.type, part.color, part.rotation_deg), []).append(part)
    is_unrotated = all(rotation == Rotation.DEG_0 for _, _, rotation in groups)

    targets = []
    for (part_type, color, rotation), parts in groups.items():
        rotation_text = "" if is_unrotated else f" with a rotation of {int(rotation)} degrees"
        targets.append(
            f"{_quantity(len(parts), color=color)} {color} {PART_NOUNS[part_type]}"
            f"{'s' if len(parts) > 1 else ''}{rotation_text} at {_positions(parts)}."
        )
    board = layout.board
    sentences = [
        f"On {_article(str(board.rows))} {board.rows} by {board.cols} board, place {targets[0]}"
    ]
    sentences += [f"Place {target}" for target in targets[1:]]
    if is_unrotated:
        sentences.append("All rotations are 0 degrees.")
    sentences.append("Each row and col is the position of the part. Rows and cols start at 0.")
    return " ".join(sentences)


def compose_kit_instruction(order: KitOrder) -> str:
    """키트 주문의 수량과 배치 방식을 영어 지시문으로 쓴다. 위치는 쓰지 않는다.

    문장은 실제 배치 계획 모델 시험에서 쓴 문구다. 회전은 모두 0도다.
    종류는 주문의 순서와 관계없이 `KIT_COLORS`의 순서로 쓴다. 시험한 문장만 만든다.
    """
    board = Board()
    counts = {item.type: item.count for item in order.items}
    items = [
        f"{_COUNT_WORDS[counts[part_type]]} {color} {PART_NOUNS[part_type]}"
        f"{'s' if counts[part_type] > 1 else ''}"
        for part_type, color in KIT_COLORS.items()
        if part_type in counts
    ]
    return (
        f"On {_article(str(board.rows))} {board.rows} by {board.cols} board, "
        f"place {_series(items)}. All rotations are 0 degrees. "
        f"{ARRANGEMENT_SENTENCES[order.arrangement]}"
    )


def arrangement_satisfied(order: KitOrder, *, layout: Layout) -> bool:
    """레이아웃이 키트 주문의 배치 방식을 지키는지 검사한다. 수량은 비교하지 않는다.

    `type_rows`는 같은 종류의 부품이 쓰는 모든 셀이 한 row에 있을 때 지킨 것이다.
    두 row에 걸친 부품이 있는 종류는 지키지 않은 것이다.
    """
    assert order.arrangement == Arrangement.TYPE_ROWS
    rows_by_type: dict[PartType, set[int]] = {}
    for part in layout.parts:
        rows_by_type.setdefault(part.type, set()).update(row for row, _ in part.cells)
    return all(len(rows) == 1 for rows in rows_by_type.values())


def _article(spoken: str) -> str:
    """다음 낱말의 첫소리에 맞는 부정관사. 숫자는 8, 11, 18만 모음으로 읽는다."""
    return "an" if spoken[0] in f"{_VOWELS}8" or spoken in _VOWEL_NUMBERS else "a"


def _quantity(count: int, *, color: Color) -> str:
    if count == 1:
        return _article(color)
    return _COUNT_WORDS[count] if count < len(_COUNT_WORDS) else str(count)


def _positions(parts: list[Part]) -> str:
    """부품마다 감싸는 최소 사각형의 왼쪽 위 셀을 쓴다."""
    positions = [
        f"row {min(row for row, _ in part.cells)} col {min(col for _, col in part.cells)}"
        for part in parts
    ]
    return _series(positions)


def _series(phrases: list[str]) -> str:
    """구를 쉼표로 나눈다. 마지막 구 앞에 and를 쓴다."""
    if len(phrases) == 1:
        return phrases[0]
    return f"{', '.join(phrases[:-1])} and {phrases[-1]}"
