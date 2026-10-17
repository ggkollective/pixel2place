"""레이아웃을 SO-101 pick-and-place 작업(`task.json`)으로 컴파일한다.

Isaac 실행기와 실물 실행기가 같은 `task.json`을 읽는다.
그래서 관절 값과 그리퍼 각도까지 여기서 계산한다.
waypoint 위치는 TCP(jaw 끝 높이의 jaw 가운데) 위치다. `pre_pick`, `pick_approach`는 pick TCP 위치다.
"""

from enum import StrEnum
from itertools import pairwise
from typing import Literal

import numpy as np
from pydantic import BaseModel, ConfigDict, Field

from pixel2place.cell.config import CellConfig, Vector2
from pixel2place.cell.grasp import (
    Vector,
    canonical_grasp,
    cell_centers,
    centroid,
    placed_grasp,
    rotation_matrix,
)
from pixel2place.feasibility.checker import check_layout
from pixel2place.generation.synthetic import COLOR_RGB, RGB
from pixel2place.layout.parts import shape_of
from pixel2place.layout.schema import Board, Cell, Color, Layout, Part, PartType, Rotation
from pixel2place.robot.so101 import (
    ARM_JOINTS,
    JAW_TIP_HALF_WIDTH,
    Vector3,
    grasp_tcp,
    gripper_angle_for_gap,
)
from pixel2place.robot.so101_ik import (
    PAN_AXIS_X,
    Joints,
    TopDownTarget,
    UnreachableError,
    solve_top_down,
)

TASK_VERSION = "0.8"
MM_PER_M = 1000.0
# 파지 폭에 더하는 jaw 간격(m). 음수는 부품을 쥐는 힘을 만든다. drive 힘 제한이 쥐는 힘을 정한다.
OPEN_GAP_MARGIN = 0.015
CLOSE_GAP_MARGIN = -0.004
# 놓을 때는 조금만 연다. 움직이는 jaw가 이웃 부품 위로 크게 나가지 않게 한다.
RELEASE_GAP_MARGIN = 0.006
# jaw 손가락 옆과 부품의 다른 셀 사이에 두는 여유(m)
FINGER_CLEARANCE = 0.001
# 부품 바닥에서 구멍 입구와 바깥 모따기 사이에 남는 평면의 최소 폭(m)
MIN_FLOOR_WIDTH = 0.001
U_AXIS = np.array([1.0, 0.0])
# 순서 비교에서 거리의 부동소수점 차이를 없앤다.
DISTANCE_DECIMALS = 6

WaypointName = Literal["pre_pick", "pick_approach", "pick", "lift", "pre_place", "place", "retreat"]


class GripperState(StrEnum):
    """waypoint에 도착한 뒤의 그리퍼 상태."""

    OPEN = "open"
    CLOSED = "closed"
    RELEASED = "released"


class TaskCompileError(ValueError):
    """레이아웃을 작업으로 바꿀 수 없다."""


class _TaskModel(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)


class Waypoint(_TaskModel):
    """툴 끝점의 목표 1개와 그 관절 값."""

    name: WaypointName
    position: tuple[float, float, float]
    yaw: float
    """jaw가 닫히는 방향의 수평 방위각(rad)."""
    joints: list[float] = Field(min_length=len(ARM_JOINTS), max_length=len(ARM_JOINTS))
    """팔 관절 값(rad). 순서는 `pixel2place.robot.so101.JOINT_NAMES`다."""
    gripper: GripperState


class PartTask(_TaskModel):
    """부품 1개의 pick-and-place."""

    sequence: int
    part_id: str
    type: PartType
    color: Color
    rotation_deg: Rotation
    target_center: Vector2
    """보드에 놓인 부품의 셀 중심 평균 위치(m)."""
    target_yaw: float
    """부품의 회전을 base_link의 방위각(rad)으로 바꾼 값."""
    color_rgb: RGB
    feeder_cells: list[Vector2]
    """공급대에 놓인 부품의 셀 중심 위치(m). 부품은 공급대 방향에서 회전 0°다."""
    block_cells: list[tuple[int, int]]
    """부품 블록 좌표의 셀 번호. 순서는 `feeder_cells`와 같다.

    블록 x축은 회전 0°인 부품의 u축(col 방향)이다. 공급대에서는 공급대 u축이다.
    y축은 x축을 위에서 보아 반시계로 90° 돌린 방향이다.
    실행기는 이 번호로 같은 부품의 이웃 셀을 찾는다.
    """
    waypoints: list[Waypoint]


class Task(_TaskModel):
    """`task.json` 전체. 부품은 놓는 순서대로 있다."""

    task_version: str = TASK_VERSION
    board: Board
    cell: CellConfig
    board_yaw: float
    """보드 col 방향(u축)의 방위각(rad). 보드에 놓인 부품 블록의 셀 축이 이 방향이다."""
    feeder_yaw: float
    """공급대 col 방향(u축)의 방위각(rad). 공급대의 부품 블록 x축이 이 방향이다."""
    grip_width: float
    """jaw 사이의 부품 폭(m). 셀 크기에서 `part_inset`을 양쪽으로 뺀 값이다."""
    tcp: Vector3
    """툴 끝점(`gripper_frame_link`) 좌표에서 본 TCP 위치(m)."""
    pick_tcp: Vector3
    """`pre_pick`, `pick_approach`의 TCP 위치(m).

    `tcp`에서 툴 −x 방향으로 `pick_clearance`만큼 간 점이다.
    """
    gripper_angles: dict[GripperState, float]
    """그리퍼 상태별 그리퍼 관절 목표(rad)."""
    parts: list[PartTask]


def compile_task(layout: Layout, cell: CellConfig) -> Task:
    """레이아웃을 작업으로 바꾼다.

    단계:
    1. 배치 규칙을 검사한다. 위반이 있으면 `TaskCompileError`를 낸다.
    2. 부품 바닥의 평면 폭을 확인한다. 셀이 작아서 구멍과 모따기가 겹치면 `TaskCompileError`를 낸다.
    3. 부품마다 목표 위치, 방위각, 파지를 계산한다.
    4. pan 축에서 먼 부품부터 놓는다. 거리가 같으면 기준 셀의 (row, col)이 작은 부품이 먼저다.
    5. 부품마다 waypoint 7개의 관절 값을 푼다. 앞 waypoint의 해에서 시작한다.
    """
    issues = check_layout(layout).issues
    if issues:
        details = "; ".join(f"{issue.code}: {issue.message}" for issue in issues)
        raise TaskCompileError(f"Layout breaks the placement rules. {details}")

    cell_size = layout.board.cell_size_mm / MM_PER_M
    _require_part_floor(cell, cell_size=cell_size)
    frame = BoardFrame(cell=cell, cell_size=cell_size)
    grip_width = cell_size - 2 * cell.part_inset
    tcp = grasp_tcp(grip_width)
    # 툴 x축은 움직이는 jaw에서 고정 jaw 쪽을 향한다.
    # 그래서 TCP를 −x로 옮기면 고정 jaw가 부품 면에서 떨어진다.
    pick_tcp = (tcp[0] - cell.pick_clearance, tcp[1], tcp[2])
    ordered = sorted(layout.parts, key=lambda part: _order_key(part, frame))
    parts: list[PartTask] = []
    seed: Joints | None = None
    for sequence, part in enumerate(ordered, start=1):
        part_task, seed = _compile_part(
            part, sequence=sequence, frame=frame, cell=cell, tcp=tcp, pick_tcp=pick_tcp, seed=seed
        )
        parts.append(part_task)
    margins = {
        GripperState.OPEN: OPEN_GAP_MARGIN,
        GripperState.CLOSED: CLOSE_GAP_MARGIN,
        GripperState.RELEASED: RELEASE_GAP_MARGIN,
    }
    return Task(
        board=layout.board,
        cell=cell,
        board_yaw=frame.yaw(U_AXIS),
        feeder_yaw=frame.feeder_yaw(U_AXIS),
        grip_width=grip_width,
        tcp=tcp,
        pick_tcp=pick_tcp,
        gripper_angles={
            state: gripper_angle_for_gap(grip_width + margin) for state, margin in margins.items()
        },
        parts=parts,
    )


class BoardFrame:
    """보드와 공급대의 격자 (u, v) 좌표를 base_link xy(m)로 바꾼다."""

    def __init__(self, *, cell: CellConfig, cell_size: float) -> None:
        self._cell_size = cell_size
        self._origin = np.array(cell.board_origin)
        self._feeder = np.array(cell.feeder_center)
        # 열 벡터는 u(col)와 v(row) 방향의 1셀 길이 벡터다.
        self._basis = _basis(cell.row_direction, cell.col_direction, cell_size=cell_size)
        feeder_row, feeder_col = cell.feeder_directions
        self._feeder_basis = _basis(feeder_row, feeder_col, cell_size=cell_size)
        self.finger_half_width = (JAW_TIP_HALF_WIDTH + FINGER_CLEARANCE) / cell_size

    def point(self, uv: Vector) -> Vector:
        return self._origin + self._basis @ uv

    def grid_uv(self, points: Vector) -> Vector:
        """base_link xy(m)의 보드 격자 (u, v). `point`의 역변환이다. 입력은 (2,) 또는 (N, 2)다."""
        return np.linalg.solve(self._basis, (np.asarray(points) - self._origin).T).T

    def feeder_point(self, offset_uv: Vector) -> Vector:
        return self._feeder + self._feeder_basis @ offset_uv

    def yaw(self, direction_uv: Vector) -> float:
        """보드 격자 방향의 방위각(rad)."""
        return _azimuth(self._basis @ direction_uv)

    def feeder_yaw(self, direction_uv: Vector) -> float:
        """공급대 격자 방향의 방위각(rad)."""
        return _azimuth(self._feeder_basis @ direction_uv)

    def block_cell(self, cell: Cell) -> tuple[int, int]:
        """회전 0°인 부품의 셀 (row, col)을 블록 좌표의 셀 번호로 바꾼다.

        블록 x축은 공급대 u축이다. 실행기가 `feeder_cells`로 블록을 만드는 좌표와 같다.
        """
        row, col = cell
        angle = -self.feeder_yaw(U_AXIS)
        rotation = np.array([[np.cos(angle), -np.sin(angle)], [np.sin(angle), np.cos(angle)]])
        # 정수 셀의 변위라서 반올림은 부동소수점 오차만 없앤다.
        block = rotation @ self._feeder_basis @ np.array([col, row]) / self._cell_size
        return int(np.rint(block[0])), int(np.rint(block[1]))


def _basis(row_direction: Vector2, col_direction: Vector2, *, cell_size: float) -> Vector:
    """열이 u(col)와 v(row) 방향의 1셀 길이 벡터인 2×2 행렬."""
    return np.column_stack([col_direction, row_direction]) * cell_size


def _azimuth(direction: Vector) -> float:
    return float(np.arctan2(direction[1], direction[0]))


def _require_part_floor(cell: CellConfig, *, cell_size: float) -> None:
    """부품 바닥에 구멍 입구와 바깥 모따기 사이의 평면이 `MIN_FLOOR_WIDTH` 이상 남는지 확인한다.

    바깥 면 쪽이 가장 좁다. 블록 반폭에서 입구 반폭과 모따기를 뺀 값이다.
    """
    mouth_half_width = cell.locator.mouth_half_width if cell.locator is not None else 0.0
    floor_width = cell_size / 2 - cell.part_inset - cell.part_chamfer - mouth_half_width
    if floor_width < MIN_FLOOR_WIDTH:
        raise TaskCompileError(
            f"Part floor width {floor_width * MM_PER_M:.2f} mm is less than "
            f"{MIN_FLOOR_WIDTH * MM_PER_M:.2f} mm. Use a larger cell, "
            "a smaller part_chamfer, or a smaller locator hole."
        )


def pan_distance(part: Part, frame: BoardFrame) -> float:
    """pan 축에서 부품 중심까지의 수평 거리(m). 배치 순서의 첫 기준이다."""
    center = frame.point(centroid(part.cells))
    return float(np.hypot(center[0] - PAN_AXIS_X, center[1]))


def _order_key(part: Part, frame: BoardFrame) -> tuple[float, int, int]:
    origin_row = min(row for row, _ in part.cells)
    origin_col = min(col for _, col in part.cells)
    return (-round(pan_distance(part, frame), DISTANCE_DECIMALS), origin_row, origin_col)


def _compile_part(
    part: Part,
    *,
    sequence: int,
    frame: BoardFrame,
    cell: CellConfig,
    tcp: Vector3,
    pick_tcp: Vector3,
    seed: Joints | None,
) -> tuple[PartTask, Joints]:
    """부품 1개의 작업과 마지막 waypoint의 관절 값을 반환한다."""
    part_center = centroid(part.cells)
    pick = canonical_grasp(part.type, finger_half_width=frame.finger_half_width)
    place = placed_grasp(part.type, part.rotation_deg, finger_half_width=frame.finger_half_width)
    pick_xy = frame.feeder_point(pick.offset)
    place_xy = frame.point(part_center + place.offset)
    pick_yaw, place_yaw = frame.feeder_yaw(pick.axis), frame.yaw(place.axis)
    # 옆 접근 파지. 고정 jaw는 부품 면에서 pick_clearance만큼 떨어진 곳으로 내려간다.
    # 그 다음 파지 높이에서 옆으로 움직여 고정 jaw 면을 부품 면에 댄다.
    # 닫는 jaw가 부품을 밀지 않는다. 그래서 centroid 밖을 잡은 부품도 돌지 않는다.
    plan: _Plan = (
        ("pre_pick", pick_xy, cell.approach_z, pick_yaw, pick_tcp, GripperState.OPEN),
        ("pick_approach", pick_xy, cell.grasp_z, pick_yaw, pick_tcp, GripperState.OPEN),
        ("pick", pick_xy, cell.grasp_z, pick_yaw, tcp, GripperState.CLOSED),
        ("lift", pick_xy, cell.approach_z, pick_yaw, tcp, GripperState.CLOSED),
        ("pre_place", place_xy, cell.approach_z, place_yaw, tcp, GripperState.CLOSED),
        ("place", place_xy, cell.place_z, place_yaw, tcp, GripperState.RELEASED),
        ("retreat", place_xy, cell.approach_z, place_yaw, tcp, GripperState.RELEASED),
    )
    # 180° 돌려도 같은 부품은 jaw 방향만 맞으면 된다.
    # 다른 부품은 손목이 부품 회전만큼 정확히 돌아야 한다.
    # 그래서 공급대에서 잡는 두 방향(yaw, yaw+180°)을 모두 풀고 관절 이동이 작은 쪽을 고른다.
    symmetric = shape_of(part.type, Rotation.DEG_0) == shape_of(part.type, Rotation.DEG_180)
    flips = (None,) if symmetric else (0.0, np.pi)
    solutions: list[list[Waypoint]] = []
    error: TaskCompileError | None = None
    for flip in flips:
        try:
            solutions.append(_solve_plan(plan, part_id=part.part_id, seed=seed, flip=flip))
        except TaskCompileError as exc:
            error = exc
    if not solutions:
        assert error is not None
        raise error
    waypoints = min(solutions, key=lambda solution: _joint_motion(solution, seed=seed))
    center_xy = frame.point(part_center)
    # 부품의 u축을 부품과 함께 돌린 방향의 방위각에서 돌리기 전 방위각을 뺀다.
    target_yaw = frame.yaw(rotation_matrix(part.rotation_deg) @ U_AXIS) - frame.yaw(U_AXIS)
    canonical_shape = sorted(shape_of(part.type, Rotation.DEG_0))
    canonical_cells = cell_centers(canonical_shape)
    feeder_cells = [
        frame.feeder_point(center - canonical_cells.mean(axis=0)) for center in canonical_cells
    ]
    part_task = PartTask(
        sequence=sequence,
        part_id=part.part_id,
        type=part.type,
        color=part.color,
        rotation_deg=part.rotation_deg,
        target_center=(float(center_xy[0]), float(center_xy[1])),
        target_yaw=_wrap(target_yaw),
        color_rgb=COLOR_RGB[part.color],
        feeder_cells=[(float(point[0]), float(point[1])) for point in feeder_cells],
        block_cells=[frame.block_cell(cell) for cell in canonical_shape],
        waypoints=waypoints,
    )
    return part_task, np.array(waypoints[-1].joints)


# waypoint마다 (이름, xy, 높이, yaw, 툴 좌표의 TCP, 그리퍼 상태)
_Plan = tuple[tuple[WaypointName, Vector, float, float, Vector3, GripperState], ...]


def _solve_plan(
    plan: _Plan, *, part_id: str, seed: Joints | None, flip: float | None
) -> list[Waypoint]:
    """waypoint의 관절 값을 차례로 푼다.

    `flip`이 None이면 jaw 방향의 180° 대칭을 쓴다. 아니면 모든 yaw에 `flip`을 더하고 그 yaw만 푼다.
    """
    waypoints: list[Waypoint] = []
    for name, xy, z, yaw, tcp, gripper in plan:
        target = TopDownTarget(
            x=float(xy[0]),
            y=float(xy[1]),
            z=z,
            yaw=yaw if flip is None else _wrap(yaw + flip),
            tcp=tcp,
            half_turn_symmetric=flip is None,
        )
        try:
            joints = solve_top_down(target, seed=seed)
        except UnreachableError as exc:
            raise TaskCompileError(f"Part {part_id} waypoint {name}: {exc}") from exc
        waypoints.append(
            Waypoint(
                name=name,
                position=(target.x, target.y, target.z),
                yaw=target.yaw,
                joints=[float(value) for value in joints],
                gripper=gripper,
            )
        )
        seed = joints
    return waypoints


def _joint_motion(waypoints: list[Waypoint], *, seed: Joints | None) -> float:
    """seed에서 시작해 waypoint를 따라갈 때 관절 변화량의 합(rad)."""
    path = [np.array(waypoint.joints) for waypoint in waypoints]
    if seed is not None:
        path.insert(0, np.asarray(seed))
    return float(sum(np.abs(after - before).sum() for before, after in pairwise(path)))


def _wrap(angle: float) -> float:
    return float(np.arctan2(np.sin(angle), np.cos(angle)))
