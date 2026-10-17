"""작업 주문의 계약, 레이아웃 변환과 지시문 문장을 검사한다."""

from pathlib import Path

import pytest
from pydantic import ValidationError

from pixel2place.feasibility.checker import IssueCode, check_layout
from pixel2place.layout.schema import Color, PartType, Rotation
from pixel2place.order import (
    KIT_COLORS,
    KIT_MAX_COUNTS,
    MAX_KIT_PARTS,
    MAX_ORDER_PARTS,
    Arrangement,
    BoardOrder,
    KitItem,
    KitOrder,
    OrderPart,
    arrangement_satisfied,
    compose_instruction,
    compose_kit_instruction,
    order_layout,
    parse_order,
)
from tests.factories import make_layout, make_part

ORDERS = Path(__file__).parents[2] / "examples" / "orders"
# 실제 Cosmos 배치 계획 모델 왕복 시험에서 쓴 문구다. 문장이 바뀌면 그 근거와 달라진다.
CHANGEOVER_INSTRUCTION = (
    "On an 8 by 8 board, place two blue square tiles at row 1 col 1 and row 1 col 6. "
    "Place two gray rectangular tiles at row 3 col 1 and row 3 col 5. "
    "Place an orange L tile at row 5 col 1. "
    "Place a white T tile at row 5 col 4. "
    "All rotations are 0 degrees. "
    "Each row and col is the position of the part. Rows and cols start at 0."
)
# 실제 Cosmos 배치 계획 모델 배치 방식 시험에서 쓴 문구다. 문장이 바뀌면 그 근거와 달라진다.
KIT_INSTRUCTION = (
    "On an 8 by 8 board, place four blue square tiles and two gray rectangular tiles. "
    "All rotations are 0 degrees. "
    "Put parts of the same type in the same row."
)
KIT_ITEM = '{"type": "square_tile", "count": 2}'


def _kit_body(items: str, *, rest: str = '"arrangement": "type_rows"') -> str:
    return f'{{"kind": "kit", "items": [{items}], {rest}}}'


def _kit(*items: KitItem) -> KitOrder:
    return KitOrder(kind="kit", items=items, arrangement=Arrangement.TYPE_ROWS)


def _square(row: int, col: int) -> OrderPart:
    return OrderPart(type=PartType.SQUARE_TILE, color=Color.BLUE, row=row, col=col)


@pytest.mark.unit
class TestBoardOrder:
    """보드 주문 계약 테스트."""

    def test_example_orders_are_feasible(self):
        """저장소의 예시 보드 주문은 모두 배치할 수 있다."""
        orders = {path.name: parse_order(path.read_text()) for path in ORDERS.glob("*.json")}
        boards = {name: order for name, order in orders.items() if isinstance(order, BoardOrder)}
        assert boards, f"예시 보드 주문이 없다: {ORDERS}"
        for name, order in boards.items():
            report = check_layout(order_layout(order))
            assert report.feasible, f"{name}: {report.issues}"

    @pytest.mark.parametrize(
        "body",
        [
            '{"kind": "kit", "parts": []}',
            '{"parts": [{"type": "hex_tile", "color": "blue", "row": 0, "col": 0}]}',
            '{"parts": [{"type": "square_tile", "color": "red", "row": 0, "col": 0}]}',
            '{"parts": [{"type": "square_tile", "color": "blue", "row": -1, "col": 0}]}',
            '{"parts": [{"type": "square_tile", "color": "blue", "row": "1", "col": 0}]}',
            '{"parts": [{"type": "L_tile", "color": "blue", "rotation_deg": 45, '
            '"row": 0, "col": 0}]}',
            '{"parts": [], "command": "private"}',
        ],
    )
    def test_fail_with_input_outside_contract(self, body: str):
        """어휘, 범위나 필드가 계약을 벗어나면 검증 오류가 난다."""
        with pytest.raises(ValidationError):
            BoardOrder.model_validate_json(body)

    def test_fail_with_too_many_parts(self):
        """부품이 상한보다 많으면 검증 오류가 난다."""
        parts = tuple(_square(index // 8, index % 8) for index in range(MAX_ORDER_PARTS + 1))
        with pytest.raises(ValidationError):
            BoardOrder(parts=parts)


@pytest.mark.unit
class TestKitOrder:
    """키트 주문 계약 테스트."""

    def test_part_count_is_sum_of_item_counts(self):
        """부품 수는 종류별 수량의 합이다."""
        order = _kit(
            KitItem(type=PartType.SQUARE_TILE, count=4),
            KitItem(type=PartType.RECT_TILE, count=2),
        )
        assert order.part_count == MAX_KIT_PARTS

    def test_part_type_limit_is_the_count_that_one_row_holds(self):
        """종류별 한도는 한 row에 들어가는 수량이다. 합계 한도를 넘지 않는다."""
        assert KIT_MAX_COUNTS == {PartType.SQUARE_TILE: 6, PartType.RECT_TILE: 4}
        for part_type, limit in KIT_MAX_COUNTS.items():
            assert _kit(KitItem(type=part_type, count=limit)).part_count == limit

    def test_color_comes_from_part_type(self):
        """색은 종류가 정한다. 주문은 색을 받지 않는다."""
        colors = {part_type: KitItem(type=part_type, count=1).color for part_type in KIT_COLORS}
        assert colors == {PartType.SQUARE_TILE: Color.BLUE, PartType.RECT_TILE: Color.GRAY}

    @pytest.mark.parametrize(
        "body",
        [
            _kit_body(""),
            _kit_body(KIT_ITEM, rest='"arrangement": "symmetric"'),
            _kit_body(KIT_ITEM, rest='"arrangement": "type_rows", "spacing": true'),
            _kit_body(f"{KIT_ITEM}, {KIT_ITEM}"),
            _kit_body('{"type": "L_tile", "count": 1}'),
            _kit_body('{"type": "square_tile", "color": "orange", "count": 1}'),
            _kit_body('{"type": "square_tile", "count": 0}'),
            _kit_body('{"type": "square_tile", "count": 7}'),
            _kit_body('{"type": "rect_tile", "count": 5}'),
            _kit_body('{"type": "square_tile", "count": 4}, {"type": "rect_tile", "count": 3}'),
            f'{{"kind": "kit", "items": [{KIT_ITEM}]}}',
            f'{{"items": [{KIT_ITEM}], "arrangement": "type_rows"}}',
        ],
    )
    def test_fail_with_input_outside_tested_range(self, body: str):
        """종류, 색, 수량, 배치 방식이나 필드가 시험한 범위를 벗어나면 검증 오류가 난다."""
        with pytest.raises(ValidationError):
            parse_order(body)


@pytest.mark.unit
class TestParseOrder:
    """주문 JSON을 두 주문 계약으로 읽는 테스트."""

    def test_body_without_kind_is_board_order(self):
        """`kind`가 없는 본문은 보드 주문으로 읽는다."""
        order = parse_order('{"parts": []}')
        assert isinstance(order, BoardOrder) and order.part_count == 0

    def test_kit_body_is_kit_order(self):
        """`kind`가 kit인 본문은 키트 주문으로 읽는다."""
        order = parse_order((ORDERS / "kit.json").read_text())
        assert isinstance(order, KitOrder), f"실제: {type(order).__name__}"
        assert order.arrangement == Arrangement.TYPE_ROWS and order.part_count == 6


@pytest.mark.unit
class TestOrderLayout:
    """주문에서 레이아웃을 만드는 테스트."""

    def test_position_is_top_left_of_bounding_box(self):
        """회전한 부품은 감싸는 사각형의 왼쪽 위 셀을 기준으로 놓인다."""
        order = BoardOrder(
            parts=(
                OrderPart(
                    type=PartType.T_TILE,
                    color=Color.WHITE,
                    rotation_deg=Rotation.DEG_180,
                    row=1,
                    col=5,
                ),
            )
        )
        part = order_layout(order).parts[0]
        assert part.part_id == "P001"
        assert part.cells == [(1, 6), (2, 5), (2, 6), (2, 7)], f"실제: {part.cells}"

    def test_overlap_and_bounds_are_reported_by_checker(self):
        """겹치거나 보드를 벗어난 주문은 기존 배치 검사가 위반으로 보고한다."""
        order = BoardOrder(
            parts=(
                _square(0, 0),
                OrderPart(type=PartType.RECT_TILE, color=Color.GRAY, row=0, col=0),
                OrderPart(type=PartType.L_TILE, color=Color.ORANGE, row=6, col=7),
            )
        )
        codes = {issue.code for issue in check_layout(order_layout(order)).issues}
        assert codes == {IssueCode.OVERLAP, IssueCode.OUT_OF_BOUNDS}, f"실제: {codes}"

    def test_empty_order_is_reported_as_empty_layout(self):
        """부품이 없는 주문은 빈 레이아웃 위반이 된다."""
        issues = check_layout(order_layout(BoardOrder(parts=()))).issues
        assert [issue.code for issue in issues] == [IssueCode.EMPTY_LAYOUT]


@pytest.mark.unit
class TestComposeInstruction:
    """지시문 문장 테스트."""

    def test_changeover_matches_verified_instruction(self):
        """기본 6부품 주문은 배치 계획 모델 왕복 시험에서 쓴 문구와 같은 문장을 만든다."""
        order = BoardOrder.model_validate_json((ORDERS / "changeover.json").read_text())
        assert compose_instruction(order_layout(order)) == CHANGEOVER_INSTRUCTION

    def test_parts_in_one_group_are_separated(self):
        """같은 묶음의 부품이 3개 이상이면 쉼표와 and로 부품을 나눈다."""
        layout = make_layout(
            make_part("P1", PartType.RECT_TILE, origin=(0, 0), color=Color.GRAY),
            make_part("P2", PartType.RECT_TILE, origin=(2, 0), color=Color.GRAY),
            make_part("P3", PartType.RECT_TILE, origin=(4, 0), color=Color.GRAY),
        )
        assert compose_instruction(layout) == (
            "On an 8 by 8 board, place three gray rectangular tiles at "
            "row 0 col 0, row 2 col 0 and row 4 col 0. "
            "All rotations are 0 degrees. "
            "Each row and col is the position of the part. Rows and cols start at 0."
        )

    def test_rotated_parts_state_each_rotation(self):
        """회전한 부품이 있으면 묶음마다 회전을 쓰고 공통 회전 문장을 뺀다."""
        layout = make_layout(
            make_part(
                "P1", PartType.T_TILE, origin=(1, 5), color=Color.WHITE, rotation=Rotation.DEG_180
            ),
            make_part("P2", PartType.SQUARE_TILE, origin=(6, 6)),
        )
        # 180도 T 타일은 감싸는 사각형의 왼쪽 위 셀 (1, 5)가 비어 있다. 위치는 그 셀이다.
        assert (1, 5) not in layout.parts[0].cells
        assert compose_instruction(layout) == (
            "On an 8 by 8 board, place a white T tile with a rotation of 180 degrees at "
            "row 1 col 5. "
            "Place a blue square tile with a rotation of 0 degrees at row 6 col 6. "
            "Each row and col is the position of the part. Rows and cols start at 0."
        )

    def test_same_type_with_other_rotation_is_other_group(self):
        """종류와 색이 같아도 회전이 다르면 다른 문장으로 쓴다."""
        layout = make_layout(
            make_part("P1", PartType.RECT_TILE, origin=(0, 0)),
            make_part("P2", PartType.RECT_TILE, origin=(2, 0), rotation=Rotation.DEG_90),
        )
        text = compose_instruction(layout)
        assert text.count("blue rectangular tile ") == 2, text
        assert "rotation of 0 degrees at row 0 col 0." in text, text
        assert "rotation of 90 degrees at row 2 col 0." in text, text

    def test_full_order_fits_prompt_limit(self):
        """부품 상한의 주문도 실행 입력의 4,000자 한도 안에 든다."""
        parts = tuple(
            OrderPart(
                type=PartType.L_TILE if index % 2 else PartType.T_TILE,
                color=list(Color)[index % 4],
                rotation_deg=list(Rotation)[index % 4],
                row=0,
                col=0,
            )
            for index in range(MAX_ORDER_PARTS)
        )
        text = compose_instruction(order_layout(BoardOrder(parts=parts)))
        assert len(text) < 4000, f"길이: {len(text)}"

    def test_fail_with_empty_layout(self):
        """부품이 없으면 지시문을 만들지 않는다."""
        with pytest.raises(ValueError, match="no parts"):
            compose_instruction(make_layout())


@pytest.mark.unit
class TestComposeKitInstruction:
    """키트 지시문 문장 테스트."""

    def test_example_kit_matches_verified_instruction(self):
        """예시 키트 주문은 배치 계획 모델 배치 방식 시험에서 쓴 문구와 같은 문장을 만든다."""
        order = parse_order((ORDERS / "kit.json").read_text())
        assert isinstance(order, KitOrder)
        assert compose_kit_instruction(order) == KIT_INSTRUCTION

    def test_single_part_uses_count_word_and_singular_noun(self):
        """수량이 1이면 one과 단수 명사를 쓴다. 위치는 쓰지 않는다."""
        order = _kit(
            KitItem(type=PartType.SQUARE_TILE, count=1),
            KitItem(type=PartType.RECT_TILE, count=1),
        )
        text = compose_kit_instruction(order)
        assert text.startswith(
            "On an 8 by 8 board, place one blue square tile and one gray rectangular tile. "
        ), text
        assert "row " not in text and "col" not in text, text

    def test_one_item_has_no_conjunction(self):
        """종류가 1개면 and 없이 쓴다."""
        text = compose_kit_instruction(_kit(KitItem(type=PartType.RECT_TILE, count=3)))
        assert text.startswith("On an 8 by 8 board, place three gray rectangular tiles. "), text

    def test_item_order_does_not_change_instruction(self):
        """주문의 종류 순서가 달라도 시험한 문장과 같은 순서로 쓴다."""
        square = KitItem(type=PartType.SQUARE_TILE, count=4)
        rect = KitItem(type=PartType.RECT_TILE, count=2)
        assert compose_kit_instruction(_kit(rect, square)) == KIT_INSTRUCTION
        assert compose_kit_instruction(_kit(square, rect)) == KIT_INSTRUCTION


@pytest.mark.unit
class TestArrangementSatisfied:
    """키트 배치 방식 검사 테스트."""

    ORDER = _kit(
        KitItem(type=PartType.SQUARE_TILE, count=2), KitItem(type=PartType.RECT_TILE, count=2)
    )

    def test_each_type_in_one_row_is_satisfied(self):
        """종류마다 모든 부품이 한 row에 있으면 배치 방식을 지킨 것이다."""
        layout = make_layout(
            make_part("P001", PartType.SQUARE_TILE, origin=(2, 0)),
            make_part("P002", PartType.SQUARE_TILE, origin=(2, 5)),
            make_part("P003", PartType.RECT_TILE, origin=(6, 0), color=Color.GRAY),
            make_part("P004", PartType.RECT_TILE, origin=(6, 4), color=Color.GRAY),
        )
        assert arrangement_satisfied(self.ORDER, layout=layout)

    def test_two_types_can_share_a_row(self):
        """두 종류가 같은 row를 써도 종류마다 row가 1개면 지킨 것이다."""
        layout = make_layout(
            make_part("P001", PartType.SQUARE_TILE, origin=(3, 0)),
            make_part("P002", PartType.RECT_TILE, origin=(3, 2), color=Color.GRAY),
        )
        assert arrangement_satisfied(self.ORDER, layout=layout)

    def test_type_in_two_rows_is_not_satisfied(self):
        """한 종류의 부품이 두 row에 있으면 지키지 않은 것이다."""
        layout = make_layout(
            make_part("P001", PartType.SQUARE_TILE, origin=(2, 0)),
            make_part("P002", PartType.SQUARE_TILE, origin=(2, 5)),
            make_part("P003", PartType.RECT_TILE, origin=(4, 0), color=Color.GRAY),
            make_part("P004", PartType.RECT_TILE, origin=(5, 0), color=Color.GRAY),
        )
        assert not arrangement_satisfied(self.ORDER, layout=layout)

    def test_part_that_spans_two_rows_is_not_satisfied(self):
        """90도 돌린 직사각은 두 row를 쓴다. 1개여도 지키지 않은 것이다."""
        layout = make_layout(
            make_part(
                "P001",
                PartType.RECT_TILE,
                origin=(1, 1),
                color=Color.GRAY,
                rotation=Rotation.DEG_90,
            )
        )
        assert not arrangement_satisfied(self.ORDER, layout=layout)

    def test_quantities_are_not_compared(self):
        """수량이 주문과 달라도 배치 방식만 본다."""
        layout = make_layout(make_part("P001", PartType.SQUARE_TILE, origin=(0, 0)))
        assert arrangement_satisfied(self.ORDER, layout=layout)
