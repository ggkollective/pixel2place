"""SO-101 팔의 기구 모델과 정기구학(FK).

상수는 SO-ARM100 저장소의 `Simulation/SO101/so101_new_calib.urdf`에서 옮겼다.
출처 커밋은 `5f6d2b876a53a4872e405b991dd925556c9e38a4`이고 라이선스는 Apache-2.0이다.
URDF를 받는 방법은 `sim/scripts/fetch_so101.sh`에 있다.

좌표 규약:
- 기준 좌표계는 `base_link`다. z축이 위를 향한다. 단위는 m와 rad다.
- 툴 끝점은 `gripper_frame_link`다. 툴 z축은 jaw가 향하는 방향이다.
- 모든 팔 관절은 자기 좌표계의 z축으로 돈다.

jaw 형상:
- 툴 끝점은 고정 jaw의 안쪽 면 위에 있다. 움직이는 jaw는 툴 -x 쪽에서 닫힌다.
- jaw 끝은 툴 끝점보다 툴 z축 방향으로 더 나온다.
- 그래서 부품을 잡는 점(TCP)은 툴 끝점에서 옮긴 점이다. `grasp_tcp`가 계산한다.
"""

from dataclasses import dataclass
from typing import Final

import numpy as np
from numpy.typing import NDArray

Matrix = NDArray[np.float64]
Vector3 = tuple[float, float, float]


@dataclass(frozen=True)
class JointSpec:
    """관절 1개. `xyz`, `rpy`는 부모 link 좌표계에서 본 관절 원점이다."""

    name: str
    xyz: tuple[float, float, float]
    rpy: tuple[float, float, float]
    lower: float
    upper: float


ARM_JOINTS: Final = (
    JointSpec(
        name="shoulder_pan",
        xyz=(0.0388353, -8.97657e-09, 0.0624),
        rpy=(3.14159, 4.18253e-17, -3.14159),
        lower=-1.91986,
        upper=1.91986,
    ),
    JointSpec(
        name="shoulder_lift",
        xyz=(-0.0303992, -0.0182778, -0.0542),
        rpy=(-1.5708, -1.5708, 0.0),
        lower=-1.74533,
        upper=1.74533,
    ),
    JointSpec(
        name="elbow_flex",
        xyz=(-0.11257, -0.028, 1.73763e-16),
        rpy=(-3.63608e-16, 8.74301e-16, 1.5708),
        lower=-1.69,
        upper=1.69,
    ),
    JointSpec(
        name="wrist_flex",
        xyz=(-0.1349, 0.0052, 3.62355e-17),
        rpy=(4.02456e-15, 8.67362e-16, -1.5708),
        lower=-1.65806,
        upper=1.65806,
    ),
    JointSpec(
        name="wrist_roll",
        xyz=(5.55112e-17, -0.0611, 0.0181),
        rpy=(1.5708, 0.0486795, 3.14159),
        lower=-2.74385,
        upper=2.84121,
    ),
)
"""팔 관절 5개. base에서 툴 쪽 순서다."""

TOOL_XYZ: Final = (-0.0079, -0.000218121, -0.0981274)
TOOL_RPY: Final = (0.0, 3.14159, 0.0)
"""`gripper_link`에서 본 툴 끝점(`gripper_frame_link`)의 고정 변환."""

GRIPPER_LOWER: Final = -0.174533
GRIPPER_UPPER: Final = 1.74533
"""그리퍼 관절 한계(rad). 작은 값이 닫힘이다."""

JAW_TIP_DEPTH: Final = 0.008
"""툴 끝점에서 가장 깊은 jaw 끝까지의 툴 z축 거리(m). 움직이는 jaw가 조금 열렸을 때의 값이다.

URDF 메시에서 쟀다. 고정 jaw 끝은 약 6.3 mm, 움직이는 jaw 끝은 그리퍼 0–12°에서 7.3–8.0 mm다.
"""

JAW_TIP_HALF_WIDTH: Final = 0.005
"""jaw 끝 손가락 폭의 반(m). 닫는 방향에 수직인 폭이다. URDF 메시에서 쟀다."""

GRIPPER_GAP_TABLE: Final = (
    np.radians([-10.0, 0.0, 10.0, 20.0, 30.0, 50.0, 100.0]),
    np.array([0.0029, 0.0168, 0.0312, 0.0454, 0.0593, 0.0842, 0.1231]),
)
"""그리퍼 각도(rad)와 jaw 간격(m). Isaac Sim 측정값이다([so101-model] 문서).
각도의 양 끝은 그리퍼 관절 한계다."""

JOINT_NAMES: Final = tuple(joint.name for joint in ARM_JOINTS)
JOINT_LOWER: Final = np.array([joint.lower for joint in ARM_JOINTS])
JOINT_UPPER: Final = np.array([joint.upper for joint in ARM_JOINTS])


def origin_matrix(xyz: tuple[float, float, float], rpy: tuple[float, float, float]) -> Matrix:
    """URDF origin(xyz, rpy)을 4x4 변환으로 바꾼다. 회전 순서는 URDF 규약 Rz·Ry·Rx다."""
    roll, pitch, yaw = rpy
    cr, sr = np.cos(roll), np.sin(roll)
    cp, sp = np.cos(pitch), np.sin(pitch)
    cy, sy = np.cos(yaw), np.sin(yaw)
    matrix = np.eye(4)
    matrix[:3, :3] = [
        [cy * cp, cy * sp * sr - sy * cr, cy * sp * cr + sy * sr],
        [sy * cp, sy * sp * sr + cy * cr, sy * sp * cr - cy * sr],
        [-sp, cp * sr, cp * cr],
    ]
    matrix[:3, 3] = xyz
    return matrix


_JOINT_ORIGINS: Final = tuple(origin_matrix(joint.xyz, joint.rpy) for joint in ARM_JOINTS)
_TOOL_ORIGIN: Final = origin_matrix(TOOL_XYZ, TOOL_RPY)


def grasp_tcp(grip_width: float) -> Vector3:
    """폭 `grip_width`(m)인 부품을 잡는 점의 툴 좌표.

    x는 두 jaw 사이의 가운데다. z는 가장 깊은 jaw 끝의 높이다.
    """
    return (-grip_width / 2, 0.0, JAW_TIP_DEPTH)


def gripper_angle_for_gap(gap: float) -> float:
    """jaw 간격(m)에 맞는 그리퍼 각도(rad). 측정 표를 선형 보간한다. 표 밖이면 관절 한계다."""
    angles, gaps = GRIPPER_GAP_TABLE
    return float(np.interp(gap, gaps, angles))


def forward(joints: NDArray[np.float64] | list[float], *, tcp: Vector3 = (0.0, 0.0, 0.0)) -> Matrix:
    """팔 관절 값 5개(rad)에서 TCP의 자세(base_link 기준 4x4)를 계산한다.

    `tcp`는 툴 좌표에서 본 TCP 위치다. 기본값은 툴 끝점이다. 자세의 축은 툴 축과 같다.
    """
    values = np.asarray(joints, dtype=np.float64)
    if values.shape != (len(ARM_JOINTS),):
        raise ValueError(f"Expected {len(ARM_JOINTS)} joint values. Got shape {values.shape}.")
    pose = np.eye(4)
    for origin, angle in zip(_JOINT_ORIGINS, values, strict=True):
        pose = pose @ origin @ _rotation_z(float(angle))
    offset = np.eye(4)
    offset[:3, 3] = tcp
    return pose @ _TOOL_ORIGIN @ offset


def _rotation_z(angle: float) -> Matrix:
    matrix = np.eye(4)
    cosine, sine = np.cos(angle), np.sin(angle)
    matrix[:2, :2] = [[cosine, -sine], [sine, cosine]]
    return matrix
