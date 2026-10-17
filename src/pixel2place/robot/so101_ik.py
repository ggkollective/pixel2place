"""SO-101 top-down 파지 자세의 역기구학(IK).

툴 z축이 아래를 향하고, jaw가 닫히는 방향(툴 x축)이 주어진 yaw를 향하는 관절 값을 찾는다.
위치는 TCP(툴 좌표에서 옮긴 점)를 목표에 맞춘다.
툴 끝점이 `wrist_roll` 축에서 벗어나 있어서 닫힌 식이 복잡하다.
그래서 FK로 damped least squares 반복을 푼다.

- 평행 그리퍼는 180° 대칭이다. 기본으로 yaw와 yaw+180°를 모두 시도한다.
  TCP는 툴 좌표의 점이라서 yaw+180°의 해에서도 같은 위치에 온다.
- 잡은 부품의 방향이 중요하면 `half_turn_symmetric=False`로 yaw만 푼다.
- 관절 한계는 반복마다 적용한다.
- 위치 오차, 기울기, yaw 오차가 허용값 안이어야 성공이다.
"""

from dataclasses import dataclass
from typing import Final

import numpy as np
from numpy.typing import NDArray

from pixel2place.robot.so101 import ARM_JOINTS, JOINT_LOWER, JOINT_UPPER, Vector3, forward

Joints = NDArray[np.float64]

POSITION_TOLERANCE_M: Final = 1e-3
TILT_TOLERANCE_RAD: Final = np.radians(2.0)
YAW_TOLERANCE_RAD: Final = np.radians(2.0)
# 자세 오차(rad)를 위치 오차(m)와 같은 식에 넣는 가중치. 1°가 약 0.9 mm에 해당한다.
ORIENTATION_WEIGHT: Final = 0.05
DAMPING: Final = 1e-4
MAX_ITERATIONS: Final = 100
MIN_STEP: Final = 1e-9
JACOBIAN_STEP: Final = 1e-6
# 반복의 시작 관절 값. shoulder_pan은 목표 방위각으로 바꾼다. 팔을 굽혀 아래를 향하는 자세들이다.
SEED_POSTURES: Final = (
    (0.0, -0.5, 0.5, 1.2, 0.0),
    (0.0, 0.5, -0.5, 1.5, 0.0),
    (0.0, -1.0, 1.2, 1.0, 0.0),
    (0.0, 0.0, 0.0, 0.0, 0.0),
)
# 방위각은 shoulder_pan 축에서 잰다. 축의 x 위치는 URDF의 shoulder_pan 원점이다.
PAN_AXIS_X: Final = ARM_JOINTS[0].xyz[0]


class UnreachableError(ValueError):
    """top-down 자세로 목표에 도달할 수 없다."""


@dataclass(frozen=True)
class TopDownTarget:
    """TCP의 목표. 위치는 base_link 기준 m, yaw는 jaw가 닫히는 방향의 rad다.

    `tcp`는 툴 좌표에서 본 TCP 위치다. 기본값은 툴 끝점이다.
    `half_turn_symmetric`이 True이면 yaw+180°도 같은 목표로 본다.
    """

    x: float
    y: float
    z: float
    yaw: float
    tcp: Vector3 = (0.0, 0.0, 0.0)
    half_turn_symmetric: bool = True


def solve_top_down(target: TopDownTarget, *, seed: Joints | None = None) -> Joints:
    """목표에 맞는 팔 관절 값 5개를 반환한다.

    `seed`가 있으면 먼저 그 값에서 시작한다. 연속한 waypoint를 풀 때 이전 결과를 넘기면
    빠르게 수렴하고 관절이 크게 바뀌지 않는다.
    `half_turn_symmetric`이면 yaw와 yaw+180°를 모두 푼다.
    두 해가 모두 있으면 `seed`(없으면 0)에 더 가까운 해를 반환한다.
    해가 없으면 `UnreachableError`를 낸다.
    """
    reference = np.zeros(len(ARM_JOINTS)) if seed is None else np.asarray(seed, dtype=np.float64)
    solutions = [
        joints
        for yaw in _candidate_yaws(target)
        if (joints := _solve(target, yaw=yaw, seed=seed)) is not None
    ]
    if not solutions:
        raise UnreachableError(
            f"No top-down pose reaches x={target.x:.3f}, y={target.y:.3f}, z={target.z:.3f} m "
            f"at yaw {np.degrees(target.yaw):.1f} degrees."
        )
    return min(solutions, key=lambda joints: float(np.linalg.norm(joints - reference)))


def top_down_error(joints: Joints, target: TopDownTarget) -> tuple[float, float, float]:
    """관절 값의 (위치 오차 m, 기울기 rad, yaw 오차 rad)를 반환한다.

    `half_turn_symmetric`이면 yaw는 180° 대칭으로 잰다.
    """
    pose = forward(joints, tcp=target.tcp)
    position_error = float(np.linalg.norm(pose[:3, 3] - [target.x, target.y, target.z]))
    tilt = float(np.arccos(np.clip(-pose[2, 2], -1.0, 1.0)))
    wrap = _wrap_half_turn if target.half_turn_symmetric else _wrap_full_turn
    yaw_error = abs(wrap(_tool_yaw(pose) - target.yaw))
    return position_error, tilt, yaw_error


def _candidate_yaws(target: TopDownTarget) -> tuple[float, ...]:
    if target.half_turn_symmetric:
        return (target.yaw, target.yaw + np.pi)
    return (target.yaw,)


def _solve(target: TopDownTarget, *, yaw: float, seed: Joints | None) -> Joints | None:
    pan = float(np.arctan2(target.y, target.x - PAN_AXIS_X))
    starts = [] if seed is None else [np.asarray(seed, dtype=np.float64)]
    starts += [np.array([pan, *posture[1:]]) for posture in SEED_POSTURES]
    for start in starts:
        joints = _iterate(start, target, yaw=yaw)
        if _within_tolerance(joints, target):
            return joints
    return None


def _iterate(start: Joints, target: TopDownTarget, *, yaw: float) -> Joints:
    joints = np.clip(start, JOINT_LOWER, JOINT_UPPER)
    for _ in range(MAX_ITERATIONS):
        residual = _residual(joints, target, yaw=yaw)
        jacobian = np.empty((residual.size, joints.size))
        for index in range(joints.size):
            shifted = joints.copy()
            shifted[index] += JACOBIAN_STEP
            jacobian[:, index] = (_residual(shifted, target, yaw=yaw) - residual) / JACOBIAN_STEP
        normal = jacobian.T @ jacobian + DAMPING * np.eye(joints.size)
        step = np.linalg.solve(normal, -jacobian.T @ residual)
        joints = np.clip(joints + step, JOINT_LOWER, JOINT_UPPER)
        if np.linalg.norm(step) < MIN_STEP:
            break
    return joints


def _residual(joints: Joints, target: TopDownTarget, *, yaw: float) -> NDArray[np.float64]:
    """위치 오차 3개, 툴 z축의 수평 성분 2개, yaw 오차 1개."""
    pose = forward(joints, tcp=target.tcp)
    position = pose[:3, 3] - [target.x, target.y, target.z]
    tilt = pose[:2, 2]
    yaw_error = _wrap_full_turn(_tool_yaw(pose) - yaw)
    return np.concatenate([position, ORIENTATION_WEIGHT * tilt, [ORIENTATION_WEIGHT * yaw_error]])


def _within_tolerance(joints: Joints, target: TopDownTarget) -> bool:
    position_error, tilt, yaw_error = top_down_error(joints, target)
    return (
        position_error <= POSITION_TOLERANCE_M
        and tilt <= TILT_TOLERANCE_RAD
        and yaw_error <= YAW_TOLERANCE_RAD
    )


def _tool_yaw(pose: NDArray[np.float64]) -> float:
    """툴 x축(jaw가 닫히는 방향)의 수평 방위각."""
    return float(np.arctan2(pose[1, 0], pose[0, 0]))


def _wrap_full_turn(angle: float) -> float:
    return float(np.arctan2(np.sin(angle), np.cos(angle)))


def _wrap_half_turn(angle: float) -> float:
    """180° 대칭 각도 차이를 [-90°, 90°]로 줄인다."""
    return float(np.arctan2(np.sin(2 * angle), np.cos(2 * angle)) / 2)
