import numpy as np
import pytest

from pixel2place.cell.capture import CaptureTask, compile_capture
from pixel2place.cell.config import CellConfig
from pixel2place.cell.task import GripperState, TaskCompileError, compile_task
from pixel2place.layout.schema import Board, PartType
from pixel2place.robot.so101 import forward, grasp_tcp
from pixel2place.robot.so101_ik import POSITION_TOLERANCE_M
from tests.factories import make_layout, make_part

CELL = CellConfig()
BOARD = Board()


@pytest.mark.unit
class TestCompileCapture:
    """캡처 작업 컴파일 테스트."""

    def test_tcp_above_feeder_center(self):
        """TCP는 공급대 중심 위의 접근 높이에 있다."""
        capture = compile_capture(CELL, BOARD)
        grip_width = BOARD.cell_size_mm / 1000 - 2 * CELL.part_inset
        position = forward(capture.joints, tcp=grasp_tcp(grip_width))[:3, 3]
        np.testing.assert_allclose(
            position, (*CELL.feeder_center, CELL.approach_z), atol=POSITION_TOLERANCE_M
        )

    def test_matches_task_geometry(self):
        """보드와 공급대 방위각, 열린 그리퍼 각도는 부품 작업과 같다."""
        capture = compile_capture(CELL, BOARD)
        task = compile_task(
            make_layout(make_part("P001", PartType.SQUARE_TILE, origin=(0, 0))), CELL
        )
        assert capture.board_yaw == task.board_yaw
        assert capture.feeder_yaw == task.feeder_yaw
        assert capture.gripper == task.gripper_angles[GripperState.OPEN]

    def test_json_round_trip(self):
        capture = compile_capture(CELL, BOARD)
        assert CaptureTask.model_validate_json(capture.model_dump_json()) == capture

    def test_unreachable_feeder(self):
        """공급대 위에 닿지 않으면 TaskCompileError다."""
        with pytest.raises(TaskCompileError, match="Capture pose"):
            compile_capture(CellConfig(feeder_center=(0.6, 0.6)), BOARD)
