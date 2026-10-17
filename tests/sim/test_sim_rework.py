"""Isaac Sim에서 부품의 결함과 재작업 실행을 확인하는 테스트.

`pytest -m sim`으로만 실행한다. 테스트마다 캡처 1회와 실행 1~2회를 한다. 합해서 약 3분이 걸린다.
Isaac 설정(`ISAAC_SIM_PATH`)이나 SO-101 USD가 없으면 skip한다.
"""

from pathlib import Path

import numpy as np
import pytest
from PIL import Image

from pixel2place.cell.camera import PinholeCamera
from pixel2place.cell.capture import CAPTURE_FILE, compile_capture
from pixel2place.cell.check import check_result
from pixel2place.cell.config import CellConfig
from pixel2place.cell.image_check import ImageCheckReport, PartStatus, check_image
from pixel2place.cell.result import SimResult
from pixel2place.cell.rework import rework_scene, reworkable
from pixel2place.cell.scene import SceneFaults, SimScene, WrongColor
from pixel2place.cell.sim import run_capture, run_place_task
from pixel2place.cell.task import Task, compile_task
from pixel2place.config import SimSettings
from pixel2place.generation.synthetic import COLOR_RGB
from pixel2place.layout.schema import Board, Color, PartType
from pixel2place.pipeline import write_json
from tests.factories import make_layout, make_part

CELL = CellConfig()
# 복원한 부품이 재작업 실행 동안 움직여도 되는 거리(mm)
MAX_RESTORED_DRIFT_MM = 0.1
MM_PER_M = 1000.0


def _camera(settings: SimSettings, directory: Path) -> PinholeCamera:
    """빈 셀을 캡처해서 공칭 top-down 카메라를 얻는다."""
    directory.mkdir()
    capture_path = directory / CAPTURE_FILE
    write_json(capture_path, compile_capture(CELL, Board()))
    return run_capture(capture_path, out_dir=directory, settings=settings).camera


def _run(
    task: Task,
    directory: Path,
    *,
    scene: SimScene,
    settings: SimSettings,
    camera: PinholeCamera,
) -> tuple[SimResult, ImageCheckReport]:
    directory.mkdir()
    task_path, scene_path = directory / "task.json", directory / "scene.json"
    write_json(task_path, task)
    write_json(scene_path, scene)
    out_dir = directory / "sim"
    result = run_place_task(task_path, out_dir=out_dir, settings=settings, scene_path=scene_path)
    with Image.open(out_dir / "topdown_final.png") as image:
        return result, check_image(task, image.convert("RGB"), camera)


@pytest.mark.sim
def test_rework_places_missing_part_next_to_placed_part(sim_settings, tmp_path):
    """공급하지 않은 부품을 재작업으로 이웃 부품 옆에 놓는다. 놓인 부품은 움직이지 않는다."""
    layout = make_layout(
        make_part("P001", PartType.SQUARE_TILE, origin=(3, 3)),
        make_part("P002", PartType.SQUARE_TILE, origin=(4, 3)),
    )
    task = compile_task(layout, CELL)
    camera = _camera(sim_settings, tmp_path / "capture")
    first_scene = SimScene(faults=SceneFaults(missing=(1,)))

    first, first_report = _run(
        task, tmp_path / "first", scene=first_scene, settings=sim_settings, camera=camera
    )

    assert [part.passed for part in check_result(task, first).parts] == [False, True]
    assert first.parts[0].fell
    assert first_report.parts[0].status is PartStatus.MISMATCH
    assert first_report.parts[0].empty
    assert reworkable(first_report) == (1,)

    scene = rework_scene(first_scene, task=task, result=first, sequences=reworkable(first_report))
    second, report = _run(
        task, tmp_path / "second", scene=scene, settings=sim_settings, camera=camera
    )

    check = check_result(task, second)
    assert check.passed, check.parts
    assert report.passed, report.parts
    drift = np.linalg.norm(np.array(second.parts[1].position) - np.array(first.parts[1].position))
    assert drift * MM_PER_M < MAX_RESTORED_DRIFT_MM


@pytest.mark.sim
def test_wrong_color_part_passes_pose_check_and_fails_image_check(sim_settings, tmp_path):
    """다른 색 부품은 자세 판정을 통과한다. 카메라 판독은 불일치다. 칸은 덮여 있다."""
    layout = make_layout(make_part("P001", PartType.SQUARE_TILE, origin=(3, 3), color=Color.BLUE))
    task = compile_task(layout, CELL)
    camera = _camera(sim_settings, tmp_path / "capture")
    orange = WrongColor(sequence=1, color=Color.ORANGE, color_rgb=COLOR_RGB[Color.ORANGE])
    scene = SimScene(faults=SceneFaults(wrong_color=(orange,)))

    result, report = _run(task, tmp_path / "run", scene=scene, settings=sim_settings, camera=camera)

    assert check_result(task, result).passed
    (part,) = report.parts
    assert part.status is PartStatus.MISMATCH
    assert not part.empty
    assert reworkable(report) == ()
