"""Isaac 실행 결과(`result.json`)를 작업 목표(`task.json`)와 비교해서 판정한다.

부품마다 위치 오차, yaw 오차, 기울기, 높이 오차를 계산한다.
실행기가 보드를 task와 다른 자세에 만들었으면 목표를 실제 보드의 같은 셀로 옮긴다.
모든 부품이 허용 오차 안에 있으면 작업이 성공이다.
실패한 부품은 놓은 직후 자세로 실패 단계를 나눈다.
"""

from collections import Counter
from enum import StrEnum
from typing import Annotated, NamedTuple

import numpy as np
from pydantic import BaseModel, ConfigDict, Field, computed_field

from pixel2place.cell.config import Vector2
from pixel2place.cell.result import PartPose, Pose, SimResult
from pixel2place.cell.scene import BoardPlacement, to_placed_board
from pixel2place.cell.task import MM_PER_M, PartTask, Task
from pixel2place.layout.parts import shape_of
from pixel2place.layout.schema import PartType, Rotation

FULL_TURN_DEG = 360.0
Tolerance = Annotated[float, Field(ge=0)]


class CheckFailure(StrEnum):
    """부품이 허용 오차를 넘은 항목."""

    POSITION = "position"
    YAW = "yaw"
    TILT = "tilt"
    HEIGHT = "height"
    FELL = "fell"


class FailureStage(StrEnum):
    """실패한 부품이 허용 오차를 벗어난 단계. 놓은 직후 자세로 나눈다."""

    TRANSPORT = "transport"
    """놓은 직후 부품 중심이 목표에서 셀 크기 이상 떨어져 있다. 파지나 이동 중에 빠졌다."""
    PLACE = "place"
    """놓은 직후 허용 오차 밖이다. 이웃 모서리에 걸렸거나 놓을 때 밀렸다."""
    LATER = "later"
    """놓은 직후 허용 오차 안이다. 뒤 부품을 놓을 때 밀렸다."""


class ResultMismatchError(ValueError):
    """`result.json`의 부품이나 작업 버전이 `task.json`과 다르다."""


class _CheckModel(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)


class CheckTolerance(_CheckModel):
    """부품 1개의 허용 오차."""

    position_ratio: Tolerance = 0.2
    """위치 오차 한계를 셀 크기에 대한 비율로 준다. 20 mm 셀에서 0.2는 4 mm다."""
    yaw_deg: Tolerance = 10.0
    """회전 대칭을 뺀 yaw 오차 한계(도)."""
    tilt_deg: Tolerance = 5.0
    """부품 z축과 수직 사이 각도의 한계(도)."""
    height_mm: Tolerance = 0.5
    """부품 중심 높이와 보드에 앉은 높이의 차이 한계(mm).

    덜 앉은 부품과 핀 끝에 수평으로 얹힌 부품은 기울기가 작다. 이 한계가 그 부품을 찾는다.
    """


DEFAULT_TOLERANCE = CheckTolerance()


class PartCheck(_CheckModel):
    """부품 1개의 판정."""

    sequence: int
    part_id: str
    type: PartType
    position_error_mm: float
    """부품 중심과 목표 위치의 수평 거리(mm). 목표 위치는 실제 보드의 목표 셀 중심 평균이다."""
    yaw_error_deg: float
    """회전 대칭을 뺀 yaw 오차(도). 위에서 보아 반시계 방향이 양수다."""
    tilt_deg: float
    height_error_mm: float
    """부품 중심 높이에서 보드에 앉은 높이를 뺀 값(mm). 위가 양수다."""
    fell: bool
    failures: list[CheckFailure]
    failed_stage: FailureStage | None
    """실패한 단계. 통과한 부품은 None이다."""

    @computed_field
    @property
    def passed(self) -> bool:
        return not self.failures


class CheckReport(_CheckModel):
    """작업 전체의 판정. 부품은 task의 순서와 같다."""

    tolerance: CheckTolerance
    position_limit_mm: float
    """이 보드의 셀 크기로 계산한 위치 오차 한계(mm)."""
    parts: list[PartCheck]

    @computed_field
    @property
    def passed_count(self) -> int:
        return sum(part.passed for part in self.parts)

    @computed_field
    @property
    def passed(self) -> bool:
        return all(part.passed for part in self.parts)


def check_result(
    task: Task, result: SimResult, *, tolerance: CheckTolerance = DEFAULT_TOLERANCE
) -> CheckReport:
    """부품마다 최종 자세를 목표 자세와 비교한다.

    `result`의 작업 버전이나 부품 목록이 `task`와 다르면 `ResultMismatchError`를 낸다.
    """
    _require_matching_parts(task, result)
    poses = {pose.part_id: pose for pose in result.parts}
    position_limit_mm = tolerance.position_ratio * task.board.cell_size_mm
    parts = [
        _check_part(
            part,
            poses[part.part_id],
            target=_target_center(part, task=task, board=result.board),
            tolerance=tolerance,
            position_limit_mm=position_limit_mm,
            transport_limit_mm=task.board.cell_size_mm,
            seated_z=task.cell.table_z + task.cell.part_height / 2,
        )
        for part in task.parts
    ]
    return CheckReport(tolerance=tolerance, position_limit_mm=position_limit_mm, parts=parts)


def yaw_period_deg(part_type: PartType) -> float:
    """부품을 돌려도 형태가 같은 가장 작은 회전(도). 회전 대칭이 없으면 360°다.

    정사각은 90°, 직사각은 180°다. yaw 오차는 이 주기로 접는다.
    """
    canonical = shape_of(part_type, Rotation.DEG_0)
    for rotation in (Rotation.DEG_90, Rotation.DEG_180):
        if shape_of(part_type, rotation) == canonical:
            return float(rotation)
    return FULL_TURN_DEG


def _require_matching_parts(task: Task, result: SimResult) -> None:
    if result.task_version != task.task_version:
        raise ResultMismatchError(
            f"Result task_version {result.task_version} is not "
            f"task task_version {task.task_version}."
        )
    expected = Counter(part.part_id for part in task.parts)
    actual = Counter(pose.part_id for pose in result.parts)
    if actual != expected:
        missing = sorted((expected - actual).elements())
        extra = sorted((actual - expected).elements())
        raise ResultMismatchError(
            f"Result parts are not the task parts. Missing: {missing}. Extra: {extra}."
        )


def _target_center(part: PartTask, *, task: Task, board: BoardPlacement | None) -> Vector2:
    """부품의 목표 위치(m). 실제 보드가 있으면 그 보드의 같은 격자 위치다."""
    if board is None:
        return part.target_center
    center = to_placed_board(part.target_center, cell=task.cell, placement=board)
    return float(center[0]), float(center[1])


def _check_part(
    part: PartTask,
    pose: PartPose,
    *,
    target: Vector2,
    tolerance: CheckTolerance,
    position_limit_mm: float,
    transport_limit_mm: float,
    seated_z: float,
) -> PartCheck:
    """`seated_z`는 보드에 앉은 부품의 중심 높이(m)다."""
    final, placed = (
        _pose_errors(
            part,
            checked,
            target=target,
            tolerance=tolerance,
            position_limit_mm=position_limit_mm,
            seated_z=seated_z,
        )
        for checked in (pose, pose.placed)
    )
    return PartCheck(
        sequence=part.sequence,
        part_id=part.part_id,
        type=part.type,
        position_error_mm=final.position_error_mm,
        yaw_error_deg=final.yaw_error_deg,
        tilt_deg=final.tilt_deg,
        height_error_mm=final.height_error_mm,
        fell=pose.fell,
        failures=final.failures,
        failed_stage=_failed_stage(final, placed, transport_limit_mm=transport_limit_mm),
    )


class _PoseErrors(NamedTuple):
    position_error_mm: float
    yaw_error_deg: float
    tilt_deg: float
    height_error_mm: float
    failures: list[CheckFailure]


def _pose_errors(
    part: PartTask,
    pose: Pose,
    *,
    target: Vector2,
    tolerance: CheckTolerance,
    position_limit_mm: float,
    seated_z: float,
) -> _PoseErrors:
    offset = np.array(pose.position[:2]) - np.array(target)
    position_error_mm = float(np.hypot(*offset) * MM_PER_M)
    yaw_error_deg = _yaw_error_deg(pose.yaw, part.target_yaw, period_deg=yaw_period_deg(part.type))
    tilt_deg = float(np.degrees(pose.tilt))
    height_error_mm = (pose.position[2] - seated_z) * MM_PER_M
    failures = [
        failure
        for failure, failed in (
            (CheckFailure.POSITION, position_error_mm > position_limit_mm),
            (CheckFailure.YAW, abs(yaw_error_deg) > tolerance.yaw_deg),
            (CheckFailure.TILT, tilt_deg > tolerance.tilt_deg),
            (CheckFailure.HEIGHT, abs(height_error_mm) > tolerance.height_mm),
            (CheckFailure.FELL, pose.fell),
        )
        if failed
    ]
    return _PoseErrors(position_error_mm, yaw_error_deg, tilt_deg, height_error_mm, failures)


def _failed_stage(
    final: _PoseErrors, placed: _PoseErrors, *, transport_limit_mm: float
) -> FailureStage | None:
    if not final.failures:
        return None
    if placed.position_error_mm >= transport_limit_mm or CheckFailure.FELL in placed.failures:
        return FailureStage.TRANSPORT
    if placed.failures:
        return FailureStage.PLACE
    return FailureStage.LATER


def _yaw_error_deg(measured: float, target: float, *, period_deg: float) -> float:
    """두 yaw(rad)의 차이를 [−주기/2, 주기/2) 범위로 접은 값(도)."""
    difference = np.degrees(np.arctan2(np.sin(measured - target), np.cos(measured - target)))
    return float((difference + period_deg / 2) % period_deg - period_deg / 2)
