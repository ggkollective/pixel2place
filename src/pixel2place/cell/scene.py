"""Isaac 셀의 실제 장면. 컴파일러가 믿는 셀 설정과 다를 수 있다.

`task.json`은 계획이다. 컴파일러는 `CellConfig`의 보드 자세를 믿고 waypoint를 만든다.
실물 로봇도 같은 task를 쓴다. 실물에서는 보드가 믿는 자세와 다른 곳에 놓일 수 있다.
`scene.json`은 시뮬레이터에서 그 차이를 만든다.

- 실행기는 보드 판, 격자, 핀을 `SimScene.board`에 만든다. 공급대는 task의 셀 설정에 만든다.
- `board`가 None이면 실행기는 보드를 task의 셀 설정에 만든다.
- 판정은 목표 셀 중심을 믿는 보드에서 실제 보드로 옮긴 뒤 비교한다.
- `camera`가 있으면 실행기는 top-down 카메라를 공칭 자세에서 그만큼 옮기고 돌린다.
- `joint_error`가 있으면 실행기는 팔 관절의 측정값과 실제 각도를 그만큼 다르게 한다.
- `faults`가 있으면 실행기는 그 부품을 공급하지 않거나 다른 색으로 만든다.
- `preplaced`가 있으면 실행기는 그 부품을 주어진 자세에 두고 놓지 않는다. 재작업 실행이 쓴다.

실행기(`sim/place_task.py`)는 pixel2place를 import하지 않는다. 그래서 이 모델이 형식을 정한다.
"""

import math
from typing import Annotated

import numpy as np
from pydantic import BaseModel, ConfigDict, Field, model_validator

from pixel2place.cell.config import CellConfig, Vector2, check_axes
from pixel2place.cell.grasp import Vector
from pixel2place.cell.task import MM_PER_M, U_AXIS, BoardFrame, Task
from pixel2place.generation.synthetic import COLOR_RGB, RGB
from pixel2place.layout.schema import Board, Color

# 팔 관절 수. 관절 센서 오차의 열 수다. 순서는 실행기의 `ARM_DOFS`와 같다.
ARM_JOINT_COUNT = 5
# 실행기의 프레임 속도(Hz). 실행기 `--fps`의 기본값과 같다.
SIM_FPS = 30
# 바뀌는 관절 센서 오차의 행 수. 실행 1회의 프레임 수보다 크다.
# 스크립트 실행은 구간마다 대기 한도까지 가도 약 450 프레임이다.
JOINT_ERROR_ROWS = 900

JointErrorRow = tuple[float, float, float, float, float]
Quaternion = tuple[float, float, float, float]
Sequence = Annotated[int, Field(ge=1)]
# 단위 quaternion으로 보는 크기의 허용 차이. 실행기의 값은 float32다.
QUATERNION_NORM_TOLERANCE = 1e-3


class _SceneModel(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)


class BoardOffset(_SceneModel):
    """보드 놓기 오차. 보드 중심을 축으로 `yaw`만큼 돌리고 (`x`, `y`)만큼 옮긴다."""

    x: float
    """base_link x 방향 이동(m)."""
    y: float
    """base_link y 방향 이동(m)."""
    yaw: float
    """위에서 보아 반시계 회전(rad)."""


class BoardPlacement(_SceneModel):
    """보드의 자세. 필드의 뜻은 `CellConfig`의 같은 이름 필드와 같다."""

    board_origin: Vector2
    row_direction: Vector2
    col_direction: Vector2
    board_yaw: float
    """col 방향(u축)의 방위각(rad). `Task.board_yaw`와 같은 식으로 계산한 값이다."""

    @model_validator(mode="after")
    def _axes_are_orthonormal(self) -> "BoardPlacement":
        check_axes(self.row_direction, self.col_direction, name="Board")
        return self


class CameraOffset(_SceneModel):
    """top-down 카메라의 자세 오차. 카메라 중심을 옮기고 카메라 중심을 축으로 돌린다."""

    position: tuple[float, float, float]
    """base_link 방향의 카메라 중심 이동(m)."""
    rotation: tuple[float, float, float]
    """base_link 축의 회전 벡터(rad). 방향이 회전축이고 길이가 회전 각도다."""


class JointErrorLevel(_SceneModel):
    """관절 센서 오차의 크기와 지속 시간. 실행마다 이 값으로 오차를 뽑는다."""

    sigma: float = Field(ge=0)
    """관절마다의 표준편차(rad)."""
    tau: float | None = Field(default=None, gt=0)
    """오차가 유지되는 시간(s). None이면 실행 동안 오차가 같다."""


class WrongColor(_SceneModel):
    """다른 색으로 공급되는 부품 1개."""

    sequence: Sequence
    color: Color
    """실제로 공급되는 부품의 색."""
    color_rgb: RGB
    """실행기가 부품에 주는 색. 실행기는 팔레트를 import하지 않는다. 팔레트의 `color` 값과 같다."""

    @model_validator(mode="after")
    def _rgb_is_palette_color(self) -> "WrongColor":
        if self.color_rgb != COLOR_RGB[self.color]:
            raise ValueError("color_rgb is different from the palette value of color.")
        return self


class SceneFaults(_SceneModel):
    """부품의 결함. 부품은 task의 배치 순서(`PartTask.sequence`)로 지정한다.

    부품 ID는 계획을 만든 뒤에 생긴다. 장면은 그 전에 정한다. 그래서 순서로 지정한다.
    """

    missing: tuple[Sequence, ...] = ()
    """공급대에 오지 않는 부품. 팔은 빈 공급대에서 동작한다. 보드의 칸은 빈다."""
    wrong_color: tuple[WrongColor, ...] = ()
    """다른 색으로 공급되는 부품. 부품은 계획한 칸에 놓인다."""

    @model_validator(mode="after")
    def _each_part_has_one_fault(self) -> "SceneFaults":
        sequences = [*self.missing, *(item.sequence for item in self.wrong_color)]
        if len(sequences) != len(set(sequences)):
            raise ValueError("A part has more than one fault.")
        return self


class PreplacedPart(_SceneModel):
    """실행 전에 이미 놓여 있는 부품 1개의 자세."""

    sequence: Sequence
    position: tuple[float, float, float]
    """블록 중심의 base_link 좌표(m)."""
    orientation: Quaternion
    """base_link 좌표의 단위 quaternion (w, x, y, z)."""

    @model_validator(mode="after")
    def _pose_is_usable(self) -> "PreplacedPart":
        if not all(math.isfinite(value) for value in (*self.position, *self.orientation)):
            raise ValueError("Preplaced pose has a value that is not finite.")
        norm = math.sqrt(sum(value * value for value in self.orientation))
        if abs(norm - 1.0) > QUATERNION_NORM_TOLERANCE:
            raise ValueError("Preplaced orientation is not a unit quaternion.")
        return self


class SimScene(_SceneModel):
    """`scene.json` 전체."""

    board: BoardPlacement | None = None
    """보드의 실제 자세. None이면 task의 셀 설정과 같다."""
    camera: CameraOffset | None = None
    """top-down 카메라의 자세 오차. None이면 카메라가 공칭 자세에 있다."""
    joint_error: tuple[JointErrorRow, ...] | None = None
    """관절 센서 오차 `b`(rad). 행은 프레임, 열은 팔 관절이다. None이면 오차가 없다.

    측정 관절 값은 실제 각도에서 `b`를 뺀 값이다. 그래서 실제 각도는 관절 목표에 `b`를 더한 값이다.
    1행이면 모든 프레임에 그 행을 쓴다. 여러 행이면 프레임 번호가 행 번호다.
    """
    faults: SceneFaults | None = None
    """부품의 결함. None이면 결함이 없다."""
    preplaced: tuple[PreplacedPart, ...] | None = None
    """이미 놓인 부품. 실행기는 이 부품을 놓지 않는다. None이면 모든 부품을 놓는다."""

    @model_validator(mode="after")
    def _joint_error_is_usable(self) -> "SimScene":
        if self.joint_error is None:
            return self
        if not self.joint_error:
            raise ValueError("Joint error has no rows.")
        if not all(math.isfinite(value) for row in self.joint_error for value in row):
            raise ValueError("Joint error has a value that is not finite.")
        return self

    @model_validator(mode="after")
    def _preplaced_parts_are_usable(self) -> "SimScene":
        placed = [part.sequence for part in self.preplaced or ()]
        if len(placed) != len(set(placed)):
            raise ValueError("A preplaced part is listed more than one time.")
        if self.faults is not None and set(placed) & set(self.faults.missing):
            raise ValueError("A preplaced part cannot be a missing part.")
        return self


def check_scene_parts(scene: SimScene, task: Task, *, policy: bool = False) -> None:
    """장면의 결함과 이미 놓인 부품이 작업과 맞는지 본다. 실행기를 띄우기 전에 부른다.

    실행기도 같은 조건을 검사한다. 이 함수는 Isaac을 띄우기 전에 입력 오류를 찾는다.

    Raises:
        ValueError: 작업에 없는 부품이 있을 때. 모든 부품이 이미 놓여 있을 때.
            제어기 실행에 결함이나 이미 놓인 부품이 있을 때.
    """
    faults = scene.faults or SceneFaults()
    changed = {*faults.missing, *(item.sequence for item in faults.wrong_color)}
    placed = {part.sequence for part in scene.preplaced or ()}
    known = {part.sequence for part in task.parts}
    unknown = (changed | placed) - known
    if unknown:
        raise ValueError(f"Scene names parts that the task does not have: {sorted(unknown)}.")
    if known and placed == known:
        raise ValueError("All parts are preplaced. The task has no part to place.")
    if policy and (changed or placed):
        raise ValueError("A controller run does not accept faults or preplaced parts.")


def placement_of(cell: CellConfig, board: Board) -> BoardPlacement:
    """셀 설정이 믿는 보드 자세."""
    frame = BoardFrame(cell=cell, cell_size=board.cell_size_mm / MM_PER_M)
    return BoardPlacement(
        board_origin=cell.board_origin,
        row_direction=cell.row_direction,
        col_direction=cell.col_direction,
        board_yaw=frame.yaw(U_AXIS),
    )


def displace_board(cell: CellConfig, board: Board, *, offset: BoardOffset) -> BoardPlacement:
    """셀 설정이 믿는 보드에 놓기 오차를 더한 자세."""
    cell_size = board.cell_size_mm / MM_PER_M
    frame = BoardFrame(cell=cell, cell_size=cell_size)
    center = frame.point(np.array([board.cols / 2, board.rows / 2]))
    rotation = _rotation(offset.yaw)
    shift = np.array([offset.x, offset.y])
    origin = center + shift + rotation @ (np.array(cell.board_origin) - center)
    row_direction = rotation @ np.array(cell.row_direction)
    col_direction = rotation @ np.array(cell.col_direction)
    moved = cell.model_copy(
        update={
            "board_origin": _pair(origin),
            "row_direction": _pair(row_direction),
            "col_direction": _pair(col_direction),
        }
    )
    return placement_of(moved, board)


def sample_joint_error(
    rng: np.random.Generator, *, level: JointErrorLevel, rows: int = JOINT_ERROR_ROWS
) -> tuple[JointErrorRow, ...]:
    """관절 센서 오차를 Ornstein-Uhlenbeck 과정으로 뽑는다.

    모든 행의 표준편차는 `level.sigma`다. 이웃한 두 행의 상관은 exp(-1 / (`SIM_FPS` * tau))다.
    `level.tau`가 None이면 1행을 준다.
    """
    first = rng.normal(0.0, level.sigma, size=ARM_JOINT_COUNT)
    if level.tau is None:
        return (_joint_row(first),)
    keep = math.exp(-1.0 / (SIM_FPS * level.tau))
    steps = rng.normal(
        0.0, level.sigma * math.sqrt(1.0 - keep**2), size=(rows - 1, ARM_JOINT_COUNT)
    )
    values = [first]
    for step in steps:
        values.append(keep * values[-1] + step)
    return tuple(_joint_row(value) for value in values)


def joint_error_frames(joint_error: tuple[JointErrorRow, ...], *, frames: int) -> np.ndarray:
    """처음 `frames`개 프레임의 관절 센서 오차. 모양은 (`frames`, 팔 관절 수)다.

    Raises:
        ValueError: 여러 행의 오차가 `frames`보다 짧을 때.
    """
    values = np.asarray(joint_error, dtype=np.float64)
    if len(values) == 1:
        return np.repeat(values, frames, axis=0)
    if len(values) < frames:
        raise ValueError(f"Joint error has {len(values)} rows. The run has {frames} frames.")
    return values[:frames]


def placed_cell(cell: CellConfig, *, placement: BoardPlacement) -> CellConfig:
    """보드 자세를 `placement`로 바꾼 셀 설정. 공급대 방향은 `cell`의 방향 그대로다.

    `cell`의 공급대 방향이 비어 있으면 보드 방향을 따른다. 그래서 방향을 값으로 채운다.
    """
    feeder_row, feeder_col = cell.feeder_directions
    return CellConfig.model_validate(
        {
            **cell.model_dump(),
            "board_origin": placement.board_origin,
            "row_direction": placement.row_direction,
            "col_direction": placement.col_direction,
            "feeder_row_direction": feeder_row,
            "feeder_col_direction": feeder_col,
        }
    )


def to_placed_board(point: Vector2, *, cell: CellConfig, placement: BoardPlacement) -> Vector:
    """믿는 보드 위의 점을 실제 보드의 같은 격자 위치로 옮긴다(m).

    두 보드의 격자 좌표 (u, v)가 같은 점이다. 셀 크기는 두 보드가 같아서 식에서 빠진다.
    """
    planned = np.column_stack([cell.col_direction, cell.row_direction])
    placed = np.column_stack([placement.col_direction, placement.row_direction])
    uv = np.linalg.solve(planned, np.asarray(point) - np.array(cell.board_origin))
    return np.array(placement.board_origin) + placed @ uv


def board_shift_mm(cell: CellConfig, board: Board, *, placement: BoardPlacement) -> float:
    """보드 셀 중심이 믿는 자리에서 실제 자리까지 움직인 거리의 최댓값(mm)."""
    frame = BoardFrame(cell=cell, cell_size=board.cell_size_mm / MM_PER_M)
    shifts = []
    for row in range(board.rows):
        for col in range(board.cols):
            point = frame.point(np.array([col + 0.5, row + 0.5]))
            moved = to_placed_board(_pair(point), cell=cell, placement=placement)
            shifts.append(float(np.hypot(*(moved - point))))
    return max(shifts) * MM_PER_M


def _rotation(angle: float) -> np.ndarray:
    return np.array([[np.cos(angle), -np.sin(angle)], [np.sin(angle), np.cos(angle)]])


def _pair(vector: Vector) -> Vector2:
    return float(vector[0]), float(vector[1])


def _joint_row(values: np.ndarray) -> JointErrorRow:
    first, second, third, fourth, fifth = (float(value) for value in values)
    return first, second, third, fourth, fifth
