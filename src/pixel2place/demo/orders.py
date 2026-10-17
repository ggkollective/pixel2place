"""주문 화면의 선택지, 미리보기, 공급 결함과 계획 비교.

주문 계약과 지시문 문장은 `pixel2place.order`가 소유한다.
결함의 계약은 `pixel2place.agent.fixture`가 소유한다.
이 모듈은 화면이 고르는 결함의 부품과 색만 정한다.
이 모듈은 실행을 시작하지 않는다. 모델 결과를 고치지 않는다.
"""

from collections import Counter

from pixel2place.agent.fixture import (
    FIXTURE_X_LIMIT_MM,
    FIXTURE_Y_LIMIT_MM,
    FIXTURE_YAW_LIMIT_DEG,
    PartFaults,
    WrongColorFault,
)
from pixel2place.cell.config import CellConfig
from pixel2place.cell.task import MM_PER_M, BoardFrame, pan_distance
from pixel2place.demo.contracts import (
    ArrangementOption,
    FixtureRange,
    KitOptions,
    KitPartOption,
    OrderMatch,
    OrderOptions,
    OrderPreview,
    PartOption,
    PartShape,
    Range,
    SupplyFaults,
)
from pixel2place.layout.parts import shape_of
from pixel2place.layout.schema import Board, Cell, Color, Layout, PartType, Rotation
from pixel2place.order import (
    KIT_COLORS,
    KIT_MAX_COUNTS,
    MAX_KIT_PARTS,
    MAX_ORDER_PARTS,
    PART_NOUNS,
    Arrangement,
    KitOrder,
    Order,
    OrderNotFeasibleError,
    order_instruction,
    order_layout,
)

# 두 계수는 부품 6개의 실행에서 쟀다.
# 발표 영상을 쓰는 Isaac 배치(`sim-place --present`)는 220초다.
# 나머지 단계는 실제 웹 실행 1건에서 52초다.
ESTIMATE_BASE_S = 50
ESTIMATE_PER_PART_S = 37

ARRANGEMENT_NAMES: dict[Arrangement, str] = {
    Arrangement.TYPE_ROWS: "One row for each part type",
}

# 화면이 고르는 결함은 배치 순서의 첫 부품에 준다. 모든 계획에 첫 부품이 있다.
SUPPLY_FAULT_SEQUENCE = 1
# 다른 색 결함의 색 후보다. 회색은 없다. 카메라 판정은 흰색 자리의 회색 부품을 불일치로 읽지 못한다.
WRONG_COLOR_CANDIDATES = (Color.ORANGE, Color.BLUE, Color.WHITE)
# 첫 부품이 될 수 있는 부품의 거리 범위(m). 작업은 pan 축에서 먼 부품부터 놓는다.
# 지그 오차와 보정 오차는 두 부품의 거리 차이를 약 18 mm까지 바꾼다. 이 값은 그보다 크다.
FIRST_PART_MARGIN_M = 0.020

PartKey = tuple[PartType, Color, frozenset[Cell]]


def order_options() -> OrderOptions:
    """부품 형태는 `CANONICAL_SHAPES`에서 계산한다. 화면에 형태 표를 따로 두지 않는다."""
    return OrderOptions(
        board=Board(),
        max_parts=MAX_ORDER_PARTS,
        colors=tuple(Color),
        parts=tuple(
            PartOption(
                type=part_type,
                name=PART_NOUNS[part_type],
                shapes=tuple(
                    PartShape(
                        rotation_deg=rotation, cells=tuple(sorted(shape_of(part_type, rotation)))
                    )
                    for rotation in Rotation
                ),
            )
            for part_type in PartType
        ),
        kit=KitOptions(
            max_parts=MAX_KIT_PARTS,
            parts=tuple(
                KitPartOption(type=part_type, color=color, max_count=KIT_MAX_COUNTS[part_type])
                for part_type, color in KIT_COLORS.items()
            ),
            arrangements=tuple(
                ArrangementOption(value=arrangement, name=ARRANGEMENT_NAMES[arrangement])
                for arrangement in Arrangement
            ),
        ),
        fixture=FixtureRange(
            x_mm=Range(min=-FIXTURE_X_LIMIT_MM, max=FIXTURE_X_LIMIT_MM),
            y_mm=Range(min=-FIXTURE_Y_LIMIT_MM, max=FIXTURE_Y_LIMIT_MM),
            yaw_deg=Range(min=-FIXTURE_YAW_LIMIT_DEG, max=FIXTURE_YAW_LIMIT_DEG),
        ),
        seconds_base=ESTIMATE_BASE_S,
        seconds_per_part=ESTIMATE_PER_PART_S,
    )


def preview_order(order: Order) -> OrderPreview:
    """배치할 수 있는 주문에만 지시문을 만든다.

    키트 주문에는 위치가 없다. 배치 위반은 없다.
    """
    try:
        instruction = order_instruction(order)
    except OrderNotFeasibleError as exc:
        return OrderPreview(
            part_count=order.part_count, issues=exc.issues, instruction=None, supply_faults=None
        )
    return OrderPreview(
        part_count=order.part_count,
        issues=(),
        instruction=instruction,
        supply_faults=supply_faults(order),
    )


def supply_faults(order: Order) -> SupplyFaults:
    """화면이 고르는 공급 결함 2종. 둘 다 배치 순서의 첫 부품에 준다.

    다른 색 결함의 색은 첫 부품이 될 수 있는 모든 부품의 색과 다르다.
    그런 색이 없으면 다른 색 결함은 없다. 주문은 배치할 수 있는 주문이어야 한다.
    """
    planned = _first_part_colors(order)
    color = next((item for item in WRONG_COLOR_CANDIDATES if item not in planned), None)
    return SupplyFaults(
        missing=PartFaults(missing=(SUPPLY_FAULT_SEQUENCE,)),
        wrong_color=None
        if color is None
        else PartFaults(
            wrong_color=(WrongColorFault(sequence=SUPPLY_FAULT_SEQUENCE, color=color),)
        ),
    )


def _first_part_colors(order: Order) -> frozenset[Color]:
    """배치 순서의 첫 부품이 될 수 있는 부품의 색.

    보드 주문은 공칭 셀에서 가장 먼 부품과 그 거리의 `FIRST_PART_MARGIN_M` 안에 있는 부품이다.
    실제 순서는 보정한 셀에서 정한다. 그 순서의 첫 부품은 이 부품 중 하나다.
    키트 주문에는 위치가 없다. 모든 부품이 후보다.
    """
    if isinstance(order, KitOrder):
        return frozenset(item.color for item in order.items)
    layout = order_layout(order)
    frame = BoardFrame(cell=CellConfig(), cell_size=layout.board.cell_size_mm / MM_PER_M)
    distances = [(pan_distance(part, frame), part.color) for part in layout.parts]
    nearest_candidate = max(distance for distance, _ in distances) - FIRST_PART_MARGIN_M
    return frozenset(color for distance, color in distances if distance >= nearest_candidate)


def match_order(order: Order, *, planned: Layout) -> OrderMatch:
    """주문의 부품과 준비한 레이아웃의 부품을 짝짓는다.

    보드 주문은 종류, 색과 셀이 같은 부품끼리 짝짓는다. 회전은 비교하지 않는다.
    대칭 형태는 다른 회전 값으로 같은 셀을 쓴다.
    키트 주문은 종류와 색이 같은 부품끼리 짝짓는다. 위치와 배치 방식은 비교하지 않는다.
    """
    if isinstance(order, KitOrder):
        return _match(
            Counter({(item.type, item.color): item.count for item in order.items}),
            Counter((part.type, part.color) for part in planned.parts),
        )
    return _match(_part_keys(order_layout(order)), _part_keys(planned))


def _match[Key](ordered: Counter[Key], prepared: Counter[Key]) -> OrderMatch:
    missing = sum((ordered - prepared).values())
    extra = sum((prepared - ordered).values())
    return OrderMatch(matches=missing == 0 and extra == 0, missing=missing, extra=extra)


def _part_keys(layout: Layout) -> Counter[PartKey]:
    return Counter((part.type, part.color, frozenset(part.cells)) for part in layout.parts)
