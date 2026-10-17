"""Isaac Sim에서 부품 1개를 실제로 놓는 smoke 테스트.

`pytest -m sim`으로만 실행한다. 기동과 실행에 약 30초가 걸린다.
첫 실행은 셰이더 캐시 때문에 더 걸린다.
Isaac 설정(`ISAAC_SIM_PATH`)이나 SO-101 USD가 없으면 skip한다.
"""

import subprocess
from pathlib import Path

import numpy as np
import pytest

from pixel2place.cell.check import check_result
from pixel2place.cell.config import CellConfig
from pixel2place.cell.scene import BoardOffset, SimScene, displace_board
from pixel2place.cell.sim import FRONT_VIDEO_FILE, PRESENT_VIDEO_FILE, run_place_task
from pixel2place.cell.task import compile_task
from pixel2place.layout.schema import PartType
from tests.factories import make_layout, make_part

PRESENT_VIDEO_SIZE = (1280, 720)


def _video_stream(path: Path) -> tuple[int, int, int]:
    """영상의 (가로, 세로, 프레임 수)."""
    output = subprocess.run(
        [
            "ffprobe",
            "-v",
            "error",
            "-select_streams",
            "v:0",
            "-count_frames",
            "-show_entries",
            "stream=width,height,nb_read_frames",
            "-of",
            "csv=p=0",
            str(path),
        ],
        check=True,
        capture_output=True,
        text=True,
    ).stdout
    width, height, frames = (int(value) for value in output.strip().split(","))
    return width, height, frames


@pytest.mark.sim
def test_places_one_square(sim_settings, tmp_path):
    """정사각 부품 1개를 공급대에서 집어 허용 오차 안에 놓는다. 영상과 top-down 이미지를 쓴다."""
    layout = make_layout(make_part("P001", PartType.SQUARE_TILE, origin=(3, 3)))
    task = compile_task(layout, CellConfig())
    task_path = tmp_path / "task.json"
    task_path.write_text(task.model_dump_json())
    out_dir = tmp_path / "out"

    result = run_place_task(task_path, out_dir=out_dir, settings=sim_settings)

    report = check_result(task, result)
    assert report.passed, report.parts
    assert (out_dir / "cell.mp4").stat().st_size > 0
    assert (out_dir / "topdown_final.png").is_file()


@pytest.mark.sim
def test_present_writes_presentation_video(sim_settings, tmp_path):
    """발표 영상을 켜면 1280×720 영상을 쓴다. 프레임 수는 사선 카메라 영상과 같다."""
    layout = make_layout(make_part("P001", PartType.SQUARE_TILE, origin=(3, 3)))
    task = compile_task(layout, CellConfig())
    task_path = tmp_path / "task.json"
    task_path.write_text(task.model_dump_json())
    out_dir = tmp_path / "out"

    result = run_place_task(task_path, out_dir=out_dir, settings=sim_settings, present=True)

    assert check_result(task, result).passed
    *size, frames = _video_stream(out_dir / PRESENT_VIDEO_FILE)
    assert tuple(size) == PRESENT_VIDEO_SIZE
    assert frames == _video_stream(out_dir / FRONT_VIDEO_FILE)[2] > 0


@pytest.mark.sim
def test_pins_hold_part_on_moved_board(sim_settings, tmp_path):
    """실제 보드가 x로 3 mm, yaw로 1° 벗어나도 핀이 L 부품을 실제 보드의 목표 셀로 끈다."""
    layout = make_layout(make_part("P001", PartType.L_TILE, origin=(3, 3)))
    task = compile_task(layout, CellConfig())
    offset = BoardOffset(x=0.003, y=0.0, yaw=np.radians(1.0))
    scene = SimScene(board=displace_board(task.cell, task.board, offset=offset))
    task_path, scene_path = tmp_path / "task.json", tmp_path / "scene.json"
    task_path.write_text(task.model_dump_json())
    scene_path.write_text(scene.model_dump_json())

    result = run_place_task(
        task_path, out_dir=tmp_path / "out", settings=sim_settings, scene_path=scene_path
    )

    assert result.board == scene.board
    (part,) = check_result(task, result).parts
    assert part.passed, part
    assert part.position_error_mm < 1.0
