"""부품의 파지점과 jaw가 닫히는 방향.

좌표는 보드 격자 단위의 (u, v)다. u는 col 방향, v는 row 방향이다. 셀 (row, col)의 중심은
(col + 0.5, row + 0.5)다. 회전 θ는 이미지에서 시계 방향이다. v축이 아래를 향하므로 (u, v) 좌표에서는
표준 회전 행렬 [[cos θ, -sin θ], [sin θ, cos θ]]과 같다. `layout.parts.rotate`와 같은 규약이다.
"""

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

from pixel2place.layout.parts import shape_of
from pixel2place.layout.schema import Cell, PartType, Rotation

Vector = NDArray[np.float64]


@dataclass(frozen=True)
class Grasp:
    """회전 0°인 부품의 파지. 모든 값은 셀 중심 평균(centroid) 기준이다.

    `offset`은 centroid에서 파지점까지의 (u, v) 벡터다. `axis`는 jaw가 닫히는 방향의 단위 벡터다.
    """

    offset: Vector
    axis: Vector


def cell_centers(cells: list[Cell] | frozenset[Cell]) -> Vector:
    """셀 목록의 (u, v) 중심 좌표 배열(N×2)."""
    return np.array([(col + 0.5, row + 0.5) for row, col in sorted(cells)], dtype=np.float64)


def centroid(cells: list[Cell] | frozenset[Cell]) -> Vector:
    return cell_centers(cells).mean(axis=0)


def rotation_matrix(rotation: Rotation) -> Vector:
    theta = np.radians(int(rotation))
    return np.array([[np.cos(theta), -np.sin(theta)], [np.sin(theta), np.cos(theta)]])


def canonical_grasp(part_type: PartType, *, finger_half_width: float) -> Grasp:
    """회전 0°인 부품의 파지를 정한다.

    jaw 손가락은 닫는 방향에 수직으로 폭이 있다. `finger_half_width`는 그 반폭에 여유를 더한 값이다.
    단위는 셀이다. 손가락이 지나는 띠 안에서는 닫는 방향의 선이 부품 셀 1개만 지나야 한다.
    그렇지 않으면 jaw가 부품의 다른 셀 위에 내려앉는다.

    단계:
    1. 닫는 방향(u, v)마다 자유 셀을 찾는다.
       자유 셀은 닫는 방향의 줄에 부품 셀이 그 셀 하나뿐인 셀이다.
    2. 닫는 방향 좌표가 같고 이어진 자유 셀들이 한 구간을 이룬다.
       손가락 띠가 구간 안에 들어가는 파지점 중에서 centroid에 가장 가까운 점을 고른다.
    3. 모든 후보 중에서 centroid에 가장 가까운 파지를 고른다. 거리가 같으면 u 방향으로 닫는다.
       무게중심에 가깝게 잡아서 들 때 부품이 덜 기운다.
    """
    cells = shape_of(part_type, Rotation.DEG_0)
    center = centroid(cells)
    candidates: list[tuple[float, int, Vector]] = []
    # 닫는 축 index. 0은 u(같은 row의 셀을 가로지름), 1은 v(같은 col의 셀을 가로지름)다.
    for closing in (0, 1):
        along = 1 - closing
        for closing_index, spans in _free_runs(cells, closing=closing):
            for first, last in spans:
                low, high = first + finger_half_width, last + 1 - finger_half_width
                if low > high:
                    continue
                point = np.empty(2)
                point[closing] = closing_index + 0.5
                point[along] = float(np.clip(center[along], low, high))
                candidates.append((round(float(np.linalg.norm(point - center)), 9), closing, point))
    if not candidates:
        raise ValueError(f"Part type {part_type} has no grasp for the jaw fingers.")
    _, closing, point = min(candidates, key=lambda candidate: candidate[:2])
    axis = np.zeros(2)
    axis[closing] = 1.0
    return Grasp(offset=point - center, axis=axis)


def _free_runs(cells: frozenset[Cell], *, closing: int) -> list[tuple[int, list[tuple[int, int]]]]:
    """닫는 축 좌표(셀 index)별로 이어진 자유 셀 구간 (처음, 마지막) 목록을 반환한다.

    `closing`이 0이면 닫는 축은 u(col)이고 구간은 row 방향이다. 1이면 반대다.
    """

    # (row, col)에서 닫는 축 좌표와 줄 좌표를 꺼낸다. u는 col, v는 row다.
    # 줄은 닫는 방향으로 놓인 셀들이다. 줄 좌표는 구간 방향 좌표와 같다.
    def split(cell: Cell) -> tuple[int, int]:
        row, col = cell
        return (col, row) if closing == 0 else (row, col)

    # 같은 줄에 셀이 1개뿐이면 자유 셀이다.
    per_line: dict[int, list[int]] = {}
    for cell in cells:
        closing_index, line_index = split(cell)
        per_line.setdefault(line_index, []).append(closing_index)
    free: dict[int, list[int]] = {}
    for line_index, closing_indices in per_line.items():
        if len(closing_indices) == 1:
            free.setdefault(closing_indices[0], []).append(line_index)
    runs = []
    for closing_index, line_indices in sorted(free.items()):
        ordered = sorted(line_indices)
        spans: list[tuple[int, int]] = []
        for index in ordered:
            if spans and index == spans[-1][1] + 1:
                spans[-1] = (spans[-1][0], index)
            else:
                spans.append((index, index))
        runs.append((closing_index, spans))
    return runs


def placed_grasp(part_type: PartType, rotation: Rotation, *, finger_half_width: float) -> Grasp:
    """회전한 부품의 파지. 회전 0°의 파지를 부품과 함께 돌린다."""
    grasp = canonical_grasp(part_type, finger_half_width=finger_half_width)
    matrix = rotation_matrix(rotation)
    return Grasp(offset=matrix @ grasp.offset, axis=matrix @ grasp.axis)
