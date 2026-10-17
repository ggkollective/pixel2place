"""Isaac Sim에서 보정 폐루프 실행 1회의 smoke 테스트.

`pytest -m sim`으로만 실행한다. 캡처(약 8초)와 부품 실행(약 30초)이 있다.
Isaac 설정(`ISAAC_SIM_PATH`)이나 SO-101 USD가 없으면 skip한다.
"""

import math

import pytest

from pixel2place.cell import bench
from pixel2place.cell.bench import BenchLayout, run_cell_benchmark
from pixel2place.cell.config import CellConfig
from pixel2place.cell.scene import BoardOffset, CameraOffset
from pixel2place.layout.schema import PartType
from tests.factories import make_layout, make_part

# 보드 오차의 한계. 정사각 자리의 이동은 약 5.8 mm다. 핀이 받아들이는 범위(약 3 mm)보다 크다.
BOARD_OFFSET = BoardOffset(x=0.003, y=0.005, yaw=math.radians(1.5))
ONE_DEGREE = math.radians(1.0)
CAMERA_ERROR = CameraOffset(
    position=(0.01, 0.01, 0.01), rotation=(ONE_DEGREE, ONE_DEGREE, ONE_DEGREE)
)
MAX_CALIBRATION_ERROR_MM = 0.5


@pytest.mark.sim
def test_calibrated_run_places_part_on_moved_board(sim_settings, tmp_path, monkeypatch):
    """보드와 카메라가 벗어나도 캡처, 추정, 보정 컴파일로 정사각을 실제 보드의 목표 셀에 놓는다."""
    monkeypatch.setattr(bench, "sample_board_offset", lambda rng, *, limit: BOARD_OFFSET)
    layout = make_layout(make_part("P001", PartType.SQUARE_TILE, origin=(3, 3)))

    summary = run_cell_benchmark(
        [BenchLayout("one", layout)],
        out_dir=tmp_path / "bench",
        cell=CellConfig(),
        sim_settings=sim_settings,
        board_error=BOARD_OFFSET,
        camera_error=CAMERA_ERROR,
        calibrate=True,
    )

    (run,) = summary.runs
    assert run.error is None, run.error
    assert run.calibration_error_mm is not None
    assert run.calibration_error_mm < MAX_CALIBRATION_ERROR_MM
    (part,) = run.parts
    assert part.passed, part
