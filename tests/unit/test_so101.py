import json
from pathlib import Path

import numpy as np
import pytest

from pixel2place.robot.so101 import (
    GRIPPER_LOWER,
    GRIPPER_UPPER,
    JOINT_NAMES,
    forward,
    gripper_angle_for_gap,
    origin_matrix,
)

ISAAC_FK_FILE = Path(__file__).parents[1] / "data" / "so101_isaac_fk.json"
# Isaac 측정값은 float32로 저장된다. 그래서 허용 오차를 0.1 mm로 둔다.
POSITION_TOLERANCE_M = 1e-4
ROTATION_TOLERANCE = 1e-3


def _isaac_samples() -> list[dict]:
    return json.loads(ISAAC_FK_FILE.read_text())["samples"]


@pytest.mark.unit
class TestForward:
    """SO-101 FK 테스트."""

    def test_joint_order(self):
        """관절 순서는 base에서 툴 쪽이다."""
        assert JOINT_NAMES == (
            "shoulder_pan",
            "shoulder_lift",
            "elbow_flex",
            "wrist_flex",
            "wrist_roll",
        )

    @pytest.mark.parametrize("sample", _isaac_samples(), ids=lambda sample: sample["name"])
    def test_matches_isaac(self, sample):
        """같은 관절 값에서 Isaac Sim이 잰 툴 자세와 같다."""
        pose = forward(sample["joints"])
        np.testing.assert_allclose(pose[:3, 3], sample["tool_position"], atol=POSITION_TOLERANCE_M)
        np.testing.assert_allclose(pose[:3, :3], sample["tool_rotation"], atol=ROTATION_TOLERANCE)

    def test_tcp_moves_along_tool_axes(self):
        """TCP 오프셋은 툴 좌표의 축을 따라 옮긴다."""
        joints = [0.3, -0.2, 0.4, 0.8, -0.5]
        tool = forward(joints)
        tcp = forward(joints, tcp=(0.01, 0.0, 0.02))
        np.testing.assert_allclose(tcp[:3, 3], tool[:3, 3] + tool[:3, :3] @ [0.01, 0.0, 0.02])
        np.testing.assert_allclose(tcp[:3, :3], tool[:3, :3])

    def test_rejects_wrong_joint_count(self):
        """관절 값이 5개가 아니면 ValueError다."""
        with pytest.raises(ValueError, match="Expected 5 joint values"):
            forward([0.0, 0.0, 0.0])


@pytest.mark.unit
class TestOriginMatrix:
    """URDF origin 변환 테스트."""

    def test_yaw_rotates_x_to_y(self):
        """yaw 90도는 x축을 y축으로 돌린다."""
        matrix = origin_matrix((1.0, 2.0, 3.0), (0.0, 0.0, np.pi / 2))
        np.testing.assert_allclose(matrix[:3, :3] @ [1.0, 0.0, 0.0], [0.0, 1.0, 0.0], atol=1e-12)
        np.testing.assert_allclose(matrix[:3, 3], [1.0, 2.0, 3.0])

    def test_matches_rz_ry_rx(self):
        """회전은 Rz·Ry·Rx 순서다."""
        roll, pitch, yaw = 0.3, -0.4, 0.5
        rx = origin_matrix((0, 0, 0), (roll, 0, 0))[:3, :3]
        ry = origin_matrix((0, 0, 0), (0, pitch, 0))[:3, :3]
        rz = origin_matrix((0, 0, 0), (0, 0, yaw))[:3, :3]
        expected = rz @ ry @ rx
        np.testing.assert_allclose(origin_matrix((0, 0, 0), (roll, pitch, yaw))[:3, :3], expected)


@pytest.mark.unit
class TestGripperAngle:
    """jaw 간격-그리퍼 각도 변환 테스트."""

    def test_interpolates_measured_table(self):
        """측정 표 사이는 선형 보간한다. 0°는 16.8 mm, 10°는 31.2 mm다."""
        assert gripper_angle_for_gap(0.024) == pytest.approx(np.radians(5.0))

    @pytest.mark.parametrize(("gap", "expected"), [(0.0, GRIPPER_LOWER), (0.2, GRIPPER_UPPER)])
    def test_clamps_to_joint_limits(self, gap, expected):
        """표 밖의 간격은 그리퍼 관절 한계다."""
        assert gripper_angle_for_gap(gap) == pytest.approx(expected, abs=1e-5)
