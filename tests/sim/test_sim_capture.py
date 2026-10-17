"""Isaac Sim 캡처와 보드 캘리브레이션 smoke 테스트.

`pytest -m sim`으로만 실행한다. 캡처 1회에 약 10초가 걸린다.
Isaac 설정(`ISAAC_SIM_PATH`)이나 SO-101 USD가 없으면 skip한다.
"""

import numpy as np
import pytest
from PIL import Image

from pixel2place.cell.calibration import calibrate_board
from pixel2place.cell.capture import compile_capture
from pixel2place.cell.config import CellConfig
from pixel2place.cell.scene import (
    BoardOffset,
    CameraOffset,
    SimScene,
    board_shift_mm,
    displace_board,
)
from pixel2place.cell.sim import run_capture
from pixel2place.layout.schema import Board

# 계획의 목표. 64셀 중심의 최대 오차(mm)다.
TARGET_ERROR_MM = 0.5


@pytest.mark.sim
def test_calibrates_moved_board(sim_settings, tmp_path):
    """보드 오차와 카메라 흔들림이 있는 캡처에서 셀 중심을 0.5 mm 안으로 맞춘다."""
    cell, board = CellConfig(), Board()
    scene = SimScene(
        board=displace_board(cell, board, offset=BoardOffset(x=0.003, y=-0.004, yaw=0.02)),
        camera=CameraOffset(position=(0.008, -0.006, 0.005), rotation=(0.012, -0.01, 0.015)),
    )
    capture_path, scene_path = tmp_path / "capture.json", tmp_path / "scene.json"
    capture_path.write_text(compile_capture(cell, board).model_dump_json())
    scene_path.write_text(scene.model_dump_json())

    captured = run_capture(
        capture_path, out_dir=tmp_path / "out", settings=sim_settings, scene_path=scene_path
    )

    assert np.linalg.norm(captured.camera_truth.center - captured.camera.center) > 0.005
    with Image.open(captured.image_path) as image:
        calibration = calibrate_board(image.convert("RGB"), captured.camera, cell=cell, board=board)
    assert scene.board is not None
    assert board_shift_mm(calibration.cell, board, placement=scene.board) < TARGET_ERROR_MM
