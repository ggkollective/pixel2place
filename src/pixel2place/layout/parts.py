"""부품 형태 라이브러리.

각 부품 형태는 회전 0도일 때의 셀 오프셋으로 정의한다.
회전한 형태는 이 표에서 계산한다.
"""

from collections.abc import Iterable

from pixel2place.layout.schema import Cell, Part, PartType, Rotation

CANONICAL_SHAPES: dict[PartType, frozenset[Cell]] = {
    PartType.SQUARE_TILE: frozenset({(0, 0)}),
    PartType.RECT_TILE: frozenset({(0, 0), (0, 1)}),
    # 세로 3칸과 오른쪽 아래 1칸.
    PartType.L_TILE: frozenset({(0, 0), (1, 0), (2, 0), (2, 1)}),
    # 가로 3칸과 가운데 아래 1칸.
    PartType.T_TILE: frozenset({(0, 0), (0, 1), (0, 2), (1, 1)}),
}

QUARTER_TURN_DEG = 90


def normalize(cells: Iterable[Cell]) -> frozenset[Cell]:
    """최소 row와 최소 col이 0이 되도록 셀 집합을 옮긴다."""
    cell_set = frozenset(cells)
    min_row = min(row for row, _ in cell_set)
    min_col = min(col for _, col in cell_set)
    return frozenset((row - min_row, col - min_col) for row, col in cell_set)


def rotate(cells: Iterable[Cell], rotation: Rotation) -> frozenset[Cell]:
    """셀 집합을 시계 방향으로 돌리고 정규화한다."""
    rotated = frozenset(cells)
    for _ in range(rotation // QUARTER_TURN_DEG):
        # 시계 방향 90도: (row, col) -> (col, -row)
        rotated = frozenset((col, -row) for row, col in rotated)
    return normalize(rotated)


def shape_of(part_type: PartType, rotation: Rotation) -> frozenset[Cell]:
    """부품 종류와 회전에 맞는 정규화된 형태를 반환한다."""
    return rotate(CANONICAL_SHAPES[part_type], rotation)


def canonical_cols(part_type: PartType) -> int:
    """회전 0도의 형태가 쓰는 col의 수."""
    return 1 + max(col for _, col in CANONICAL_SHAPES[part_type])


def matches_shape(part: Part) -> bool:
    """부품 셀이 선언한 종류·회전의 형태와 같으면 True를 반환한다."""
    if len(set(part.cells)) != len(part.cells):
        return False
    return normalize(part.cells) == shape_of(part.type, part.rotation_deg)


def place(part_type: PartType, rotation: Rotation, *, origin: Cell) -> list[Cell]:
    """형태를 `origin`(왼쪽 위 셀)에 놓았을 때의 보드 셀 목록을 반환한다."""
    origin_row, origin_col = origin
    return sorted(
        (origin_row + row, origin_col + col) for row, col in shape_of(part_type, rotation)
    )
