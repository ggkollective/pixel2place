"""부품을 놓는 시점의 이웃 틈을 계산한다.

틈은 먼저 놓인 부품과 내려가는 부품의 목표 자리 사이의 최소 수평 거리다.
- 먼저 놓인 부품은 놓은 직후 자세(`placed`)로 옮긴다. 놓을 때 끌려서 틈이 준 것이 틈에 들어간다.
- 내려가는 부품은 목표 자리에 있다고 본다. 내려가는 부품 자신의 오차는 넣지 않는다.
  목표 자리는 task의 셀 설정이 믿는 보드의 자리다. 로봇이 부품을 내리는 곳이다.
- 블록은 셀마다 정사각형으로 근사한다. 한 변은 셀 크기에서 `part_inset`을 양쪽으로 뺀 값이다.
- 높이와 기울기는 보지 않는다.

틈이 작으면 내려가는 부품이 이웃 윗모서리에 걸리기 쉽다.
"""

import numpy as np

from pixel2place.cell.grasp import Vector, cell_centers
from pixel2place.cell.result import SimResult
from pixel2place.cell.task import MM_PER_M, BoardFrame, Task
from pixel2place.layout.schema import Cell, Layout

# 이 거리(셀 단위)보다 멀리 있는 부품은 이웃으로 보지 않는다.
NEIGHBOR_RANGE_CELLS = 0.5
# 틈(mm)을 이 자릿수로 반올림한다.
CLEARANCE_DECIMALS = 2
# 정사각형의 꼭짓점. 중심 기준이고 반폭 1이다. 순서는 둘레를 따라간다.
_UNIT_SQUARE = np.array([[-1.0, -1.0], [1.0, -1.0], [1.0, 1.0], [-1.0, 1.0]])


def placement_clearances(layout: Layout, task: Task, result: SimResult) -> dict[str, float | None]:
    """부품마다 놓는 시점의 최소 이웃 틈(mm)을 계산한다. 키는 `part_id`다.

    반 셀 안에 먼저 놓인 부품이 없으면 None이다. 블록이 겹치면 0이다.
    """
    cell_size = task.board.cell_size_mm / MM_PER_M
    frame = BoardFrame(cell=task.cell, cell_size=cell_size)
    half = cell_size / 2 - task.cell.part_inset
    cells_by_id = {part.part_id: part.cells for part in layout.parts}
    placed_by_id = {pose.part_id: pose.placed for pose in result.parts}
    neighbor_range = NEIGHBOR_RANGE_CELLS * cell_size
    # 놓인 부품의 yaw는 실제 보드의 u축에서 잰 값이다.
    # 그래서 world에서 부품이 돈 각도는 두 보드의 방위각 차이를 더한 값이다.
    board_turn = 0.0 if result.board is None else result.board.board_yaw - task.board_yaw
    placed: list[np.ndarray] = []
    clearances: dict[str, float | None] = {}
    for part in task.parts:
        centers = _cell_points(cells_by_id[part.part_id], frame)
        target = _squares(centers, half=half, yaw=task.board_yaw)
        gaps = [gap for squares in placed if (gap := _distance(target, squares)) < neighbor_range]
        clearances[part.part_id] = round(min(gaps) * MM_PER_M, CLEARANCE_DECIMALS) if gaps else None
        pose = placed_by_id[part.part_id]
        delta = pose.yaw - part.target_yaw + board_turn
        moved = _rotate(centers - np.array(part.target_center), delta) + np.array(pose.position[:2])
        placed.append(_squares(moved, half=half, yaw=task.board_yaw + delta))
    return clearances


def _cell_points(cells: list[Cell], frame: BoardFrame) -> Vector:
    """셀 중심의 base_link xy(m) 배열(N×2)."""
    return np.array([frame.point(uv) for uv in cell_centers(cells)])


def _squares(centers: Vector, *, half: float, yaw: float) -> np.ndarray:
    """셀 중심마다 반폭 `half`, 방위각 `yaw`인 정사각형의 꼭짓점 배열(N×4×2)."""
    corners = _rotate(_UNIT_SQUARE * half, yaw)
    return centers[:, np.newaxis, :] + corners[np.newaxis, :, :]


def _rotate(points: Vector, angle: float) -> Vector:
    cosine, sine = np.cos(angle), np.sin(angle)
    return points @ np.array([[cosine, sine], [-sine, cosine]])


def _distance(first: np.ndarray, second: np.ndarray) -> float:
    """두 정사각형 묶음(N×4×2, M×4×2) 사이의 최소 거리(m)."""
    return min(_polygon_distance(a, b) for a in first for b in second)


def _polygon_distance(first: np.ndarray, second: np.ndarray) -> float:
    """볼록 다각형 두 개 사이의 최소 거리(m). 겹치면 0이다."""
    if not _separated(first, second):
        return 0.0
    return min(_vertex_edge_distance(first, second), _vertex_edge_distance(second, first))


def _separated(first: np.ndarray, second: np.ndarray) -> bool:
    """분리축 검사. 변의 법선 중 하나에 투영한 두 구간이 떨어져 있으면 겹치지 않는다."""
    for polygon in (first, second):
        edges = np.roll(polygon, -1, axis=0) - polygon
        for normal in np.column_stack([-edges[:, 1], edges[:, 0]]):
            a, b = first @ normal, second @ normal
            if a.max() < b.min() or b.max() < a.min():
                return True
    return False


def _vertex_edge_distance(vertices: np.ndarray, polygon: np.ndarray) -> float:
    """`vertices`의 꼭짓점에서 `polygon`의 변까지 최소 거리(m)."""
    starts = polygon
    edges = np.roll(polygon, -1, axis=0) - polygon
    offsets = vertices[:, np.newaxis, :] - starts[np.newaxis, :, :]
    ratios = np.clip(np.sum(offsets * edges, axis=2) / np.sum(edges * edges, axis=1), 0.0, 1.0)
    nearest = starts[np.newaxis, :, :] + ratios[:, :, np.newaxis] * edges[np.newaxis, :, :]
    return float(np.min(np.linalg.norm(vertices[:, np.newaxis, :] - nearest, axis=2)))
