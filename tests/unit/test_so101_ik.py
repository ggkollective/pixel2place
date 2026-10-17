import numpy as np
import pytest

from pixel2place.robot.so101 import JOINT_LOWER, JOINT_UPPER, forward, grasp_tcp
from pixel2place.robot.so101_ik import (
    POSITION_TOLERANCE_M,
    TILT_TOLERANCE_RAD,
    YAW_TOLERANCE_RAD,
    TopDownTarget,
    UnreachableError,
    solve_top_down,
    top_down_error,
)

# 보드 안쪽과 공급대 근처의 점. 파지 높이와 접근 높이를 모두 검사한다.
REACHABLE_POINTS = [(0.125, 0.07), (0.265, -0.07), (0.20, 0.0), (0.14, 0.16)]
HEIGHTS = [0.005, 0.045]
YAWS = [0.0, np.pi / 4, np.pi / 2, -np.pi / 3]


@pytest.mark.unit
class TestSolveTopDown:
    """top-down IK 테스트."""

    @pytest.mark.parametrize("x,y", REACHABLE_POINTS)
    @pytest.mark.parametrize("z", HEIGHTS)
    def test_reaches_target_within_tolerance(self, x, y, z):
        """FK(IK(목표))가 허용 오차 안에서 목표와 같다."""
        for yaw in YAWS:
            target = TopDownTarget(x=x, y=y, z=z, yaw=yaw)
            joints = solve_top_down(target)
            position_error, tilt, yaw_error = top_down_error(joints, target)
            assert position_error <= POSITION_TOLERANCE_M
            assert tilt <= TILT_TOLERANCE_RAD
            assert yaw_error <= YAW_TOLERANCE_RAD
            assert np.all(joints >= JOINT_LOWER) and np.all(joints <= JOINT_UPPER)

    def test_tool_points_down(self):
        """해의 툴 z축은 아래를 향한다."""
        joints = solve_top_down(TopDownTarget(x=0.2, y=0.0, z=0.02, yaw=0.3))
        assert forward(joints)[2, 2] < -0.99

    def test_uses_half_turn_symmetry(self):
        """wrist_roll 한계 때문에 yaw 0은 yaw 180°의 해로 푼다."""
        target = TopDownTarget(x=0.2, y=0.0, z=0.02, yaw=0.0)
        joints = solve_top_down(target)
        assert top_down_error(joints, target)[2] <= YAW_TOLERANCE_RAD

    def test_exact_yaw_when_not_symmetric(self):
        """half_turn_symmetric=False이면 yaw+180°의 해를 쓰지 않는다."""
        for yaw in (np.pi / 2, -np.pi / 2):
            target = TopDownTarget(x=0.2, y=0.0, z=0.02, yaw=yaw, half_turn_symmetric=False)
            pose = forward(solve_top_down(target))
            assert np.arctan2(pose[1, 0], pose[0, 0]) == pytest.approx(yaw, abs=YAW_TOLERANCE_RAD)

    def test_prefers_solution_near_seed(self):
        """두 해가 있으면 seed에 가까운 해를 고른다."""
        target = TopDownTarget(x=0.2, y=0.05, z=0.02, yaw=np.pi / 2)
        first = solve_top_down(target)
        again = solve_top_down(target, seed=first)
        np.testing.assert_allclose(again, first, atol=1e-4)

    def test_places_tcp_on_target_for_both_symmetric_solutions(self):
        """yaw와 yaw+180° 해 모두 TCP가 목표에 온다. 툴 끝점은 서로 반대쪽에 있다."""
        tcp = grasp_tcp(0.019)
        tool_points = []
        for yaw in (np.pi / 2, -np.pi / 2):
            target = TopDownTarget(x=0.2, y=0.0, z=0.02, yaw=yaw, tcp=tcp)
            joints = solve_top_down(target)
            assert top_down_error(joints, target)[0] <= POSITION_TOLERANCE_M
            np.testing.assert_allclose(
                forward(joints, tcp=tcp)[:3, 3], (0.2, 0.0, 0.02), atol=POSITION_TOLERANCE_M
            )
            tool_points.append(forward(joints)[:3, 3])
        # 툴 끝점은 TCP에서 파지 폭의 반만큼 옆, jaw 끝 깊이만큼 위에 있다.
        for point in tool_points:
            assert np.hypot(point[0] - 0.2, point[1]) == pytest.approx(0.0095, abs=1e-3)

    @pytest.mark.parametrize(
        "target",
        [
            TopDownTarget(x=0.6, y=0.0, z=0.02, yaw=0.0),
            TopDownTarget(x=0.0, y=0.0, z=0.02, yaw=0.0),
            TopDownTarget(x=0.2, y=0.0, z=0.5, yaw=0.0),
        ],
        ids=["too-far", "inside-base", "too-high"],
    )
    def test_unreachable_raises(self, target):
        """도달할 수 없으면 UnreachableError다."""
        with pytest.raises(UnreachableError, match="No top-down pose"):
            solve_top_down(target)
