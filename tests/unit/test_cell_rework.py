import numpy as np
import pytest
from PIL import Image

from pixel2place.cell.check import check_result
from pixel2place.cell.config import CellConfig
from pixel2place.cell.image_check import (
    ImageCheckReport,
    PartStatus,
    check_image,
)
from pixel2place.cell.result import SimResult
from pixel2place.cell.rework import ReworkError, rework_scene, reworkable
from pixel2place.cell.scene import (
    BoardOffset,
    SceneFaults,
    SimScene,
    WrongColor,
    displace_board,
)
from pixel2place.cell.sim import run_place_task
from pixel2place.cell.task import Task, compile_task
from pixel2place.config import SimSettings
from pixel2place.generation.synthetic import COLOR_RGB
from pixel2place.layout.schema import Board, Color, PartType
from pixel2place.pipeline import write_json
from tests.factories import (
    make_image_check_report,
    make_layout,
    make_part,
    make_part_image_check,
    make_sim_result,
)
from tests.fakes.topdown_renderer import nominal_camera

CELL = CellConfig()
NOMINAL = nominal_camera()
LAYOUT = make_layout(
    make_part("P001", PartType.SQUARE_TILE, origin=(1, 1), color=Color.BLUE),
    make_part("P002", PartType.RECT_TILE, origin=(3, 2), color=Color.GRAY),
    make_part("P003", PartType.SQUARE_TILE, origin=(5, 5), color=Color.BLUE),
)
ORANGE = WrongColor(sequence=1, color=Color.ORANGE, color_rgb=COLOR_RGB[Color.ORANGE])
MOVED_BOARD = displace_board(CELL, Board(), offset=BoardOffset(x=0.002, y=-0.003, yaw=0.01))
# 가짜 실행기가 복원한 부품의 위치를 그대로 쓰는지 보는 허용 차이(m)
POSITION_TOLERANCE_M = 1e-12


@pytest.fixture
def task() -> Task:
    return compile_task(LAYOUT, CELL)


@pytest.fixture
def run_cell(fake_isaac, task, tmp_path):
    """가짜 실행기로 작업을 실행한다. `(결과, 카메라 판독)`을 반환한다."""
    task_path = tmp_path / "task.json"
    write_json(task_path, task)

    def run(name: str, scene: SimScene | None = None) -> tuple[SimResult, ImageCheckReport]:
        out_dir = tmp_path / name
        scene_path = None
        if scene is not None:
            scene_path = tmp_path / f"{name}.scene.json"
            write_json(scene_path, scene)
        result = run_place_task(
            task_path,
            out_dir=out_dir,
            settings=SimSettings(),  # pyright: ignore[reportCallIssue]
            scene_path=scene_path,
        )
        with Image.open(out_dir / "topdown_final.png") as image:
            report = check_image(task, image.convert("RGB"), NOMINAL)
        return result, report

    return run


@pytest.mark.unit
class TestReworkable:
    """카메라 판독에서 재작업의 대상을 고르는 테스트."""

    def test_empty_mismatch_parts_are_targets(self):
        """불일치이고 칸이 빈 부품만 대상이다."""
        report = make_image_check_report(
            make_part_image_check(1, status=PartStatus.MATCH, empty=False),
            make_part_image_check(2, status=PartStatus.MISMATCH, empty=True),
            make_part_image_check(3, status=PartStatus.MISMATCH, empty=False),
            make_part_image_check(4, status=PartStatus.SHADOWED, empty=False),
        )
        assert reworkable(report) == (2,)

    def test_unexpected_cell_has_no_target(self):
        """계획에 없는 점유 칸이 있으면 대상이 없다."""
        report = make_image_check_report(
            make_part_image_check(1, status=PartStatus.MISMATCH, empty=True), unexpected=((6, 6),)
        )
        assert reworkable(report) == ()

    def test_passed_report_has_no_target(self):
        """통과한 판독에는 대상이 없다."""
        assert (
            reworkable(
                make_image_check_report(
                    make_part_image_check(1, status=PartStatus.MATCH, empty=False)
                )
            )
            == ()
        )


@pytest.mark.unit
class TestReworkScene:
    """재작업 실행의 장면 테스트."""

    def _result(self, task: Task) -> SimResult:
        """모든 부품에 quaternion이 있는 결과."""
        base = make_sim_result(task)
        parts = [
            part.model_copy(update={"orientation": (1.0, 0.0, 0.0, 0.0)}) for part in base.parts
        ]
        return base.model_copy(update={"parts": parts})

    def test_other_parts_keep_result_pose(self, task):
        """대상이 아닌 부품은 결과의 위치와 quaternion에 놓인다."""
        result = self._result(task)
        scene = rework_scene(None, task=task, result=result, sequences=[2])
        assert scene.preplaced is not None
        assert [part.sequence for part in scene.preplaced] == [1, 3]
        for placed in scene.preplaced:
            pose = result.parts[placed.sequence - 1]
            assert placed.position == pose.position
            assert placed.orientation == pose.orientation

    def test_missing_fault_is_removed(self, task):
        """공급 누락은 뺀다. 다른 색 결함과 보드 자세는 그대로다."""
        first = SimScene(board=MOVED_BOARD, faults=SceneFaults(missing=(2,), wrong_color=(ORANGE,)))
        result = self._result(task).model_copy(update={"board": MOVED_BOARD})
        scene = rework_scene(first, task=task, result=result, sequences=[2])
        assert scene.board == MOVED_BOARD
        assert scene.faults == SceneFaults(wrong_color=(ORANGE,))

    def test_scene_without_other_fault_has_no_faults(self, task):
        """공급 누락만 있던 장면은 결함이 없는 장면이 된다."""
        first = SimScene(faults=SceneFaults(missing=(2,)))
        scene = rework_scene(first, task=task, result=self._result(task), sequences=[2])
        assert scene.faults is None

    def test_all_parts_as_targets_have_no_preplaced(self, task):
        """모든 부품이 대상이면 놓인 부품이 없다."""
        scene = rework_scene(None, task=task, result=self._result(task), sequences=[1, 2, 3])
        assert scene.preplaced is None

    @pytest.mark.parametrize("sequences", [[], [4], [1, 9]])
    def test_rejects_parts_not_in_task(self, task, sequences):
        """작업에 없는 부품과 빈 목록을 거부한다."""
        with pytest.raises(ReworkError, match="not parts of the task"):
            rework_scene(None, task=task, result=self._result(task), sequences=sequences)

    def test_rejects_result_without_orientation(self, task):
        """quaternion이 없는 결과로는 장면을 만들지 못한다."""
        with pytest.raises(ReworkError, match="no orientation"):
            rework_scene(None, task=task, result=make_sim_result(task), sequences=[2])

    def test_rejects_result_of_other_board(self, task):
        """결과의 보드가 장면의 보드와 다르면 거부한다. 자세는 그 보드에서 잰 값이다."""
        moved = self._result(task).model_copy(update={"board": MOVED_BOARD})
        with pytest.raises(ReworkError, match="different from the scene board"):
            rework_scene(None, task=task, result=moved, sequences=[2])
        with pytest.raises(ReworkError, match="different from the scene board"):
            rework_scene(
                SimScene(board=MOVED_BOARD), task=task, result=self._result(task), sequences=[2]
            )

    @pytest.mark.parametrize(
        "orientation", [(float("nan"), 0.0, 0.0, 0.0), (0.0, 0.0, 0.0, 0.0), (2.0, 0.0, 0.0, 0.0)]
    )
    def test_rejects_pose_that_is_not_usable(self, task, orientation):
        """유한하지 않은 값과 단위 quaternion이 아닌 값을 거부한다."""
        result = self._result(task)
        broken = result.parts[0].model_copy(update={"orientation": orientation})
        result = result.model_copy(update={"parts": [broken, *result.parts[1:]]})
        with pytest.raises(ReworkError, match="not usable"):
            rework_scene(None, task=task, result=result, sequences=[3])

    def test_rejects_result_of_other_task(self, task):
        """부품이 작업과 다른 결과를 거부한다."""
        result = self._result(task)
        shorter = result.model_copy(update={"parts": result.parts[:2]})
        with pytest.raises(ReworkError, match="different from the task"):
            rework_scene(None, task=task, result=shorter, sequences=[2])


@pytest.mark.unit
class TestFaultAndReworkRuns:
    """가짜 실행기의 결함 실행과 재작업 실행 테스트."""

    def test_missing_part_leaves_empty_cells(self, run_cell, task):
        """공급하지 않은 부품은 판정에서 실패하고 판독에서 빈 칸으로 읽힌다."""
        result, report = run_cell("first", SimScene(faults=SceneFaults(missing=(2,))))
        check = check_result(task, result)
        assert [part.passed for part in check.parts] == [True, False, True]
        assert result.parts[1].fell
        missing = report.parts[1]
        assert missing.status is PartStatus.MISMATCH
        assert missing.empty
        assert reworkable(report) == (2,)

    def test_rework_places_missing_part(self, run_cell, task):
        """재작업 실행 뒤에는 작업 전체의 판정과 판독이 통과한다."""
        first_scene = SimScene(faults=SceneFaults(missing=(2,)))
        first, first_report = run_cell("first", first_scene)
        scene = rework_scene(
            first_scene, task=task, result=first, sequences=reworkable(first_report)
        )
        second, report = run_cell("second", scene)
        assert check_result(task, second).passed
        assert report.passed
        assert [part.status for part in report.parts] == [PartStatus.MATCH] * 3
        for before, after in zip(first.parts, second.parts, strict=True):
            if before.fell:
                continue
            drift = np.linalg.norm(np.array(after.position) - np.array(before.position))
            assert drift < POSITION_TOLERANCE_M
            assert after.yaw == pytest.approx(before.yaw)

    def test_wrong_color_part_is_not_a_target(self, run_cell, task):
        """다른 색 부품은 자세 판정을 통과한다. 판독은 불일치이고 칸은 덮여 있다."""
        result, report = run_cell("first", SimScene(faults=SceneFaults(wrong_color=(ORANGE,))))
        assert check_result(task, result).passed
        wrong = report.parts[0]
        assert wrong.status is PartStatus.MISMATCH
        assert not wrong.empty
        assert reworkable(report) == ()

    def test_rework_keeps_wrong_color_part(self, run_cell, task):
        """재작업은 다른 색 부품을 고치지 않는다. 그 부품은 재작업 뒤에도 불일치다."""
        first_scene = SimScene(faults=SceneFaults(missing=(3,), wrong_color=(ORANGE,)))
        first, _ = run_cell("first", first_scene)
        scene = rework_scene(first_scene, task=task, result=first, sequences=[3])
        _, report = run_cell("second", scene)
        assert [part.status for part in report.parts] == [
            PartStatus.MISMATCH,
            PartStatus.MATCH,
            PartStatus.MATCH,
        ]

    def test_result_has_orientation(self, run_cell):
        """결과는 부품마다 quaternion을 준다."""
        result, _ = run_cell("first")
        for part in result.parts:
            assert part.orientation is not None
            assert np.linalg.norm(part.orientation) == pytest.approx(1.0)
