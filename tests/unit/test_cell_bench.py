import json
from pathlib import Path

import numpy as np
import pytest
from typer.testing import CliRunner

from pixel2place.cell import bench
from pixel2place.cell.bench import (
    ALL_LAYOUTS,
    CALIBRATED_CELL_FILE,
    CAPTURE_DIR,
    CHECK_FILE,
    ERROR_FILE,
    FEEDER_JITTER_M,
    POLICY_CONTEXT_FILE,
    SCENE_FILE,
    SUMMARY_FILE,
    TASK_FILE,
    BenchLayout,
    BenchPolicy,
    RunErrorKind,
    Shard,
    jitter_feeder,
    neighbor_sides,
    run_cell_benchmark,
    sample_board_offset,
    sample_camera_offset,
)
from pixel2place.cell.camera import PinholeCamera
from pixel2place.cell.check import CheckFailure, FailureStage
from pixel2place.cell.config import CellConfig
from pixel2place.cell.policy_link import DOF_NAMES, PolicyAddress, PolicyServer
from pixel2place.cell.result import SimResult
from pixel2place.cell.scene import (
    JOINT_ERROR_ROWS,
    BoardOffset,
    CameraOffset,
    JointErrorLevel,
    SimScene,
    placement_of,
)
from pixel2place.cell.sim import (
    CAMERA_FILE,
    CAMERA_TRUTH_FILE,
    CAPTURE_IMAGE_FILE,
    RECORD_OUTPUTS,
    SimSetupError,
)
from pixel2place.cell.task import Task, compile_task
from pixel2place.cli import EXIT_INFEASIBLE, EXIT_USAGE, app
from pixel2place.config import SimSettings
from pixel2place.layout.schema import Board, Layout, PartType
from tests.factories import make_layout, make_part

POCKET_LAYOUT = Path(__file__).parents[2] / "examples" / "layouts" / "repetitive_panel_pocket.json"
# 셀 설정에서 보드가 로봇 작업 범위 밖에 있다. 컴파일이 실패한다.
UNREACHABLE_CELL = CellConfig(board_origin=(1.0, 0.0))
BOARD_ERROR = BoardOffset(x=0.003, y=0.005, yaw=np.radians(1.5))
# 이동만 있는 보드 오차. 모든 셀 중심이 같은 거리만큼 움직인다.
SHIFT_ONLY_ERROR = BoardOffset(x=0.008, y=0.0, yaw=0.0)
CAMERA_ERROR = CameraOffset(position=(0.01, 0.01, 0.01), rotation=(0.017, 0.017, 0.017))
JOINT_ERROR = JointErrorLevel(sigma=np.radians(1.0), tau=1.0)
# 가짜 실행기는 부품을 task의 목표에 놓는다. 보정 task의 목표와 실제 목표의 차이는 추정 오차다.
MAX_FAKE_CALIBRATION_ERROR_MM = 0.1
runner = CliRunner()


def _bench_layouts() -> list[BenchLayout]:
    """부품 1개와 2개의 레이아웃."""
    return [
        BenchLayout("one", make_layout(make_part("P001", PartType.SQUARE_TILE, origin=(3, 3)))),
        BenchLayout(
            "two",
            make_layout(
                make_part("P001", PartType.RECT_TILE, origin=(0, 0)),
                make_part("P002", PartType.SQUARE_TILE, origin=(0, 2)),
            ),
        ),
    ]


def _settings() -> SimSettings:
    return SimSettings()  # pyright: ignore[reportCallIssue]


def _run(out_dir: Path, **options):
    options.setdefault("cell", CellConfig())
    return run_cell_benchmark(
        _bench_layouts(),
        out_dir=out_dir,
        sim_settings=_settings(),
        start_interval_s=0.0,
        **options,
    )


@pytest.mark.unit
class TestNeighborSides:
    """놓는 시점의 이웃 면 수 테스트."""

    def test_pocket_part(self):
        """pocket 레이아웃에서 포켓으로 내려가는 L(P005)은 이웃 면 7개와 맞닿는다."""
        layout = Layout.model_validate_json(POCKET_LAYOUT.read_text())
        task = compile_task(layout, CellConfig())
        sides = neighbor_sides(layout, task)
        assert sides["P005"] == 7
        assert sides[task.parts[0].part_id] == 0

    def test_counts_only_placed_parts(self):
        """뒤에 놓는 부품의 면은 세지 않는다. 맞닿은 두 부품 중 나중 부품만 센다."""
        layout = _bench_layouts()[1].layout
        task = compile_task(layout, CellConfig())
        sides = neighbor_sides(layout, task)
        assert sides[task.parts[0].part_id] == 0
        assert sides[task.parts[1].part_id] == 1


@pytest.mark.unit
class TestJitterFeeder:
    """공급대 위치 섭동 테스트."""

    @pytest.fixture
    def task(self) -> Task:
        return compile_task(_bench_layouts()[1].layout, CellConfig())

    def test_moves_each_part_within_limit(self, task):
        """부품마다 셀 전체를 같은 거리만큼 옮긴다. x, y 각각 한계 안이다."""
        moved = jitter_feeder(task, np.random.default_rng(0), max_offset=FEEDER_JITTER_M)
        for before, after in zip(task.parts, moved.parts, strict=True):
            offsets = np.array(after.feeder_cells) - np.array(before.feeder_cells)
            assert np.allclose(offsets, offsets[0])
            assert np.all(np.abs(offsets) <= FEEDER_JITTER_M)
            assert np.any(offsets != 0)
            assert after.waypoints == before.waypoints

    def test_same_seed_same_task(self, task):
        """같은 seed는 같은 task를 만든다."""
        first = jitter_feeder(task, np.random.default_rng([0, 1, 1]), max_offset=FEEDER_JITTER_M)
        second = jitter_feeder(task, np.random.default_rng([0, 1, 1]), max_offset=FEEDER_JITTER_M)
        assert first == second


@pytest.mark.unit
class TestSampleBoardOffset:
    """보드 오차 뽑기 테스트."""

    def test_within_limit(self):
        """x, y, yaw는 각각 한계 안에 있다. 한계가 0인 필드는 0이다."""
        rng = np.random.default_rng(0)
        limit = BoardOffset(x=0.003, y=0.0, yaw=0.02)
        offsets = [sample_board_offset(rng, limit=limit) for _ in range(50)]
        assert all(abs(offset.x) <= limit.x for offset in offsets)
        assert all(offset.y == 0.0 for offset in offsets)
        assert all(abs(offset.yaw) <= limit.yaw for offset in offsets)
        assert len({offset.x for offset in offsets}) == len(offsets)


@pytest.mark.unit
class TestSampleCameraOffset:
    """카메라 오차 뽑기 테스트."""

    def test_within_limit(self):
        """위치와 회전의 성분은 각각 한계 안에 있다. 한계가 0인 성분은 0이다."""
        rng = np.random.default_rng(0)
        limit = CameraOffset(position=(0.01, 0.01, 0.0), rotation=(0.02, 0.0, 0.02))
        offsets = [sample_camera_offset(rng, limit=limit) for _ in range(50)]
        for offset in offsets:
            assert np.all(np.abs(offset.position) <= limit.position)
            assert np.all(np.abs(offset.rotation) <= limit.rotation)
            assert offset.position[2] == 0.0
            assert offset.rotation[1] == 0.0
        assert len({offset.position[0] for offset in offsets}) == len(offsets)


@pytest.mark.unit
class TestShard:
    """shard 형식과 선택 테스트."""

    def test_parse(self):
        """K/N을 읽고 번호를 N으로 나눈 나머지가 K인 레이아웃을 고른다."""
        shard = Shard.parse("1/3")
        assert [index for index in range(7) if shard.includes(index)] == [1, 4]

    @pytest.mark.parametrize("text", ["3/3", "a/2", "2", "-1/2"])
    def test_invalid(self, text):
        """K/N 형식이 아니거나 K가 N 이상이면 ValueError다."""
        with pytest.raises(ValueError, match="Shard"):
            Shard.parse(text)


class _HoldPolicy:
    """관절 목표 0을 주는 제어기. 받은 문맥을 기억한다."""

    def __init__(self) -> None:
        self.contexts: list[dict] = []

    def reset(self, context):
        self.contexts.append(dict(context))
        return None

    def act(self, observation):
        return np.zeros((2, len(DOF_NAMES)), dtype=np.float32)


@pytest.mark.unit
class TestRunCellBenchmark:
    """셀 벤치마크 실행 테스트. 가짜 Isaac 실행기를 쓴다."""

    def test_writes_runs_and_summary(self, fake_isaac, tmp_path):
        """레이아웃마다 task, 결과, 판정을 쓰고 요약을 만든다."""
        out_dir = tmp_path / "bench"
        summary = _run(out_dir)
        run_dir = out_dir / "001_two" / "r0"
        for path in (TASK_FILE, "sim/result.json", CHECK_FILE):
            assert (run_dir / path).is_file(), path
        assert (out_dir / SUMMARY_FILE).is_file()
        assert summary.total.parts == 3
        assert summary.total.failed == 0
        assert summary.passed_runs == 2
        assert summary.by_type[PartType.RECT_TILE].parts == 1
        assert summary.by_neighbor_sides["0-2"].parts == 3
        assert list(summary.by_clearance) == ["1.9+", "none"]
        assert summary.by_clearance["1.9+"].parts == 1
        assert summary.by_clearance["none"].parts == 2
        clearances = [part.clearance_mm for part in summary.runs[1].parts]
        assert clearances == [None, pytest.approx(3.0)]

    def test_first_repeat_is_compiled_task(self, fake_isaac, tmp_path):
        """반복 0은 컴파일한 task 그대로다. 반복 1은 공급대 위치만 다르다."""
        out_dir = tmp_path / "bench"
        summary = _run(out_dir, repeats=2)
        compiled = compile_task(_bench_layouts()[0].layout, CellConfig())
        first = Task.model_validate_json((out_dir / "000_one/r0" / TASK_FILE).read_text())
        second = Task.model_validate_json((out_dir / "000_one/r1" / TASK_FILE).read_text())
        assert first == compiled
        assert second.parts[0].feeder_cells != compiled.parts[0].feeder_cells
        assert second.parts[0].waypoints == compiled.parts[0].waypoints
        assert len(summary.runs) == 4

    def test_shard_runs_selected_layouts(self, fake_isaac, tmp_path):
        """shard에 든 레이아웃만 실행하고 요약한다."""
        out_dir = tmp_path / "bench"
        summary = _run(out_dir, shard=Shard(index=1, count=2))
        assert [run.layout for run in summary.runs] == ["001_two"]
        assert not (out_dir / "000_one").exists()

    def test_skips_finished_runs(self, fake_isaac, tmp_path, monkeypatch):
        """판정이 있고 task가 같은 실행은 다시 하지 않는다. 합친 shard는 요약만 쓴다."""
        out_dir = tmp_path / "bench"
        _run(out_dir, shard=Shard(index=0, count=2))
        _run(out_dir, shard=Shard(index=1, count=2))
        monkeypatch.setenv("FAKE_ISAAC_MODE", "crash")
        summary = _run(out_dir, shard=ALL_LAYOUTS)
        assert summary.errors == 0
        assert summary.total.parts == 3

    def test_reruns_changed_task(self, fake_isaac, tmp_path, monkeypatch):
        """셀 설정이 바뀌어 task가 다르면 다시 실행한다."""
        out_dir = tmp_path / "bench"
        _run(out_dir)
        monkeypatch.setenv("FAKE_ISAAC_MODE", "crash")
        summary = _run(out_dir, cell=CellConfig(part_inset=0.001))
        assert summary.errors == 2
        assert not (out_dir / "000_one/r0" / CHECK_FILE).exists()

    def test_runner_failure_is_run_error(self, fake_isaac, tmp_path, monkeypatch):
        """실행기가 결과를 쓰지 못하면 그 실행의 오류로 적고 다음 실행으로 간다."""
        monkeypatch.setenv("FAKE_ISAAC_MODE", "crash")
        out_dir = tmp_path / "bench"
        summary = _run(out_dir, jobs=2)
        assert summary.errors == 2
        assert summary.errors_by_kind == {RunErrorKind.RUNNER_ERROR: 2}
        assert summary.total.parts == 0
        assert summary.runs[0].error is not None
        assert "Fake runner failed" in summary.runs[0].error
        assert (out_dir / "000_one/r0" / ERROR_FILE).is_file()

    def test_part_failure_has_stage(self, fake_isaac, tmp_path, monkeypatch):
        """허용 오차 밖 부품은 실패 항목과 단계로 센다."""
        monkeypatch.setenv("FAKE_ISAAC_MODE", "miss")
        summary = _run(tmp_path / "bench")
        assert summary.total.failed == 2
        assert summary.by_stage == {FailureStage.PLACE: 2}
        failed = [part for run in summary.runs for part in run.parts if not part.passed]
        assert all(part.failures == [CheckFailure.POSITION] for part in failed)

    def test_compile_failure_is_error_for_each_repeat(self, fake_isaac, tmp_path):
        """컴파일할 수 없는 레이아웃은 반복마다 오류로 센다."""
        out_dir = tmp_path / "bench"
        summary = _run(out_dir, cell=UNREACHABLE_CELL, repeats=2)
        assert summary.errors == 4
        assert summary.errors_by_kind == {RunErrorKind.COMPILE_FAILED: 4}
        assert summary.runs[0].error is not None
        assert "P001" in summary.runs[0].error
        assert (out_dir / "000_one" / ERROR_FILE).is_file()

    def test_board_error_writes_scene(self, fake_isaac, tmp_path):
        """보드 오차를 주면 실행마다 다른 scene을 쓴다. 결과의 board는 그 scene의 보드다."""
        out_dir = tmp_path / "bench"
        summary = _run(out_dir, repeats=2, board_error=BOARD_ERROR)
        scenes = []
        for run in summary.runs:
            run_dir = out_dir / run.layout / f"r{run.repeat}"
            scene = SimScene.model_validate_json((run_dir / SCENE_FILE).read_text())
            result = SimResult.model_validate_json((run_dir / "sim/result.json").read_text())
            assert result.board == scene.board
            assert run.board_shift_mm is not None
            # 이동 한계 hypot(3, 5) mm + 모서리 셀 중심의 1.5° 회전 거리 약 2.6 mm
            assert 0 < run.board_shift_mm < 8.5
            scenes.append(scene)
        assert len({scene.model_dump_json() for scene in scenes}) == len(scenes)
        again = _run(tmp_path / "again", repeats=2, board_error=BOARD_ERROR)
        assert [run.board_shift_mm for run in again.runs] == [
            run.board_shift_mm for run in summary.runs
        ]

    def test_no_board_error_has_no_scene(self, fake_isaac, tmp_path):
        """보드 오차가 없으면 scene을 쓰지 않는다."""
        out_dir = tmp_path / "bench"
        summary = _run(out_dir)
        assert not (out_dir / "000_one/r0" / SCENE_FILE).exists()
        assert all(run.board_shift_mm is None for run in summary.runs)
        assert all(run.calibration_error_mm is None for run in summary.runs)
        assert all(run.order_changed is None for run in summary.runs)
        assert summary.max_calibration_error_mm is None

    def test_joint_error_writes_scene_rows(self, fake_isaac, tmp_path):
        """관절 센서 오차를 주면 실행마다 다른 값을 scene에 쓴다. 같은 seed는 같은 값이다."""
        out_dir = tmp_path / "bench"
        summary = _run(out_dir, repeats=2, joint_error=JOINT_ERROR)
        scenes = []
        for run in summary.runs:
            run_dir = out_dir / run.layout / f"r{run.repeat}"
            scene = SimScene.model_validate_json((run_dir / SCENE_FILE).read_text())
            assert scene.board is None
            assert scene.joint_error is not None
            assert len(scene.joint_error) == JOINT_ERROR_ROWS
            scenes.append(scene)
        assert len({scene.model_dump_json() for scene in scenes}) == len(scenes)
        _run(tmp_path / "again", repeats=2, joint_error=JOINT_ERROR)
        again = SimScene.model_validate_json(
            (tmp_path / "again/000_one/r0" / SCENE_FILE).read_text()
        )
        assert again == scenes[0]

    def test_joint_error_keeps_board_error(self, fake_isaac, tmp_path):
        """관절 센서 오차를 더해도 같은 seed의 보드 오차는 같다."""
        plain = _run(tmp_path / "plain", board_error=BOARD_ERROR)
        noisy = _run(tmp_path / "noisy", board_error=BOARD_ERROR, joint_error=JOINT_ERROR)
        assert [run.board_shift_mm for run in noisy.runs] == [
            run.board_shift_mm for run in plain.runs
        ]

    def test_reruns_changed_joint_error(self, fake_isaac, tmp_path, monkeypatch):
        """관절 센서 오차가 바뀌면 다시 실행한다. 같으면 건너뛴다."""
        out_dir = tmp_path / "bench"
        _run(out_dir, joint_error=JOINT_ERROR)
        monkeypatch.setenv("FAKE_ISAAC_MODE", "crash")
        assert _run(out_dir, joint_error=JOINT_ERROR).errors == 0
        assert _run(out_dir, joint_error=JointErrorLevel(sigma=np.radians(1.0))).errors == 2

    def test_calibrate_keeps_joint_error_in_scene(self, fake_isaac, tmp_path):
        """보정 실행의 scene에도 관절 센서 오차가 있다. 캡처와 추정은 그대로 된다."""
        out_dir = tmp_path / "bench"
        summary = _run(out_dir, board_error=BOARD_ERROR, calibrate=True, joint_error=JOINT_ERROR)
        scene = SimScene.model_validate_json((out_dir / "000_one/r0" / SCENE_FILE).read_text())
        assert scene.board is not None
        assert scene.joint_error is not None
        assert summary.errors == 0
        assert summary.total.failed == 0

    def test_parts_on_planned_board_miss_by_shift(self, fake_isaac, tmp_path):
        """가짜 실행기는 믿는 보드에 놓는다. 판정의 위치 오차는 보드 이동 거리다."""
        out_dir = tmp_path / "bench"
        summary = _run(out_dir, board_error=SHIFT_ONLY_ERROR)
        for run in summary.runs:
            check = json.loads((out_dir / run.layout / "r0" / CHECK_FILE).read_text())
            for part in check["parts"]:
                assert part["position_error_mm"] == pytest.approx(run.board_shift_mm)

    def test_reruns_changed_scene(self, fake_isaac, tmp_path, monkeypatch):
        """task가 같아도 scene이 바뀌면 다시 실행한다. 같은 scene은 건너뛴다."""
        out_dir = tmp_path / "bench"
        _run(out_dir, board_error=BOARD_ERROR)
        monkeypatch.setenv("FAKE_ISAAC_MODE", "crash")
        assert _run(out_dir, board_error=BOARD_ERROR).errors == 0
        assert _run(out_dir, board_error=SHIFT_ONLY_ERROR).errors == 2

    def test_reruns_without_board_error(self, fake_isaac, tmp_path, monkeypatch):
        """scene이 있던 실행은 보드 오차를 빼면 다시 실행한다."""
        out_dir = tmp_path / "bench"
        _run(out_dir, board_error=BOARD_ERROR)
        monkeypatch.setenv("FAKE_ISAAC_MODE", "crash")
        assert _run(out_dir).errors == 2
        assert not (out_dir / "000_one/r0" / SCENE_FILE).exists()

    def test_calibrate_places_parts_on_actual_board(self, fake_isaac, tmp_path, monkeypatch):
        """보정하면 보드가 8 mm 벗어나도 부품이 실제 보드의 목표에 놓인다. 캡처와 보정 셀을 쓴다."""
        monkeypatch.setattr(bench, "sample_board_offset", lambda rng, *, limit: SHIFT_ONLY_ERROR)
        out_dir = tmp_path / "bench"
        summary = _run(out_dir, board_error=SHIFT_ONLY_ERROR, calibrate=True)
        assert summary.errors == 0
        assert summary.total.failed == 0
        for run in summary.runs:
            assert run.board_shift_mm == pytest.approx(8.0)
            assert run.calibration_error_mm is not None
            assert run.calibration_error_mm < MAX_FAKE_CALIBRATION_ERROR_MM
            assert run.order_changed is False
        run_dir = out_dir / "000_one/r0"
        capture_dir = run_dir / CAPTURE_DIR
        for name in (CAPTURE_IMAGE_FILE, CAMERA_FILE, CALIBRATED_CELL_FILE, "cell.report.json"):
            assert (capture_dir / name).is_file(), name
        calibrated = CellConfig.model_validate_json(
            (capture_dir / CALIBRATED_CELL_FILE).read_text()
        )
        task = Task.model_validate_json((run_dir / TASK_FILE).read_text())
        scene = SimScene.model_validate_json((run_dir / SCENE_FILE).read_text())
        assert scene.board is not None
        assert task.cell == calibrated
        assert calibrated.board_origin == pytest.approx(scene.board.board_origin, abs=1e-4)
        assert summary.max_calibration_error_mm == max(
            run.calibration_error_mm for run in summary.runs if run.calibration_error_mm
        )

    def test_calibrate_without_board_error_writes_actual_board(self, fake_isaac, tmp_path):
        """보드 오차가 없어도 보정하면 scene에 셀 설정의 보드를 적는다.

        실행기가 보드를 보정 셀 설정이 아니라 그 자세에 만든다.
        """
        out_dir = tmp_path / "bench"
        summary = _run(out_dir, calibrate=True)
        scene = SimScene.model_validate_json((out_dir / "000_one/r0" / SCENE_FILE).read_text())
        assert scene.board == placement_of(CellConfig(), Board())
        assert scene.camera is None
        assert summary.total.failed == 0
        assert all(run.board_shift_mm == 0.0 for run in summary.runs)

    def test_camera_error_moves_capture_camera(self, fake_isaac, tmp_path):
        """카메라 오차를 주면 실행마다 캡처 카메라가 공칭 자세에서 벗어난다. 추정은 그래도 맞다."""
        out_dir = tmp_path / "bench"
        summary = _run(out_dir, board_error=BOARD_ERROR, camera_error=CAMERA_ERROR, calibrate=True)
        cameras = []
        for run in summary.runs:
            run_dir = out_dir / run.layout / "r0"
            scene = SimScene.model_validate_json((run_dir / SCENE_FILE).read_text())
            assert scene.camera is not None
            truth = PinholeCamera.model_validate_json(
                (run_dir / CAPTURE_DIR / CAMERA_TRUTH_FILE).read_text()
            )
            nominal = PinholeCamera.model_validate_json(
                (run_dir / CAPTURE_DIR / CAMERA_FILE).read_text()
            )
            assert truth.position != nominal.position
            cameras.append(scene.camera)
            assert run.calibration_error_mm is not None
            assert run.calibration_error_mm < MAX_FAKE_CALIBRATION_ERROR_MM
        assert cameras[0] != cameras[1]
        assert summary.total.failed == 0

    def test_calibrate_skips_finished_captures_and_runs(self, fake_isaac, tmp_path, monkeypatch):
        """캡처와 판정이 있고 scene과 task가 같으면 Isaac을 다시 실행하지 않는다."""
        out_dir = tmp_path / "bench"
        first = _run(out_dir, board_error=BOARD_ERROR, calibrate=True)
        monkeypatch.setenv("FAKE_ISAAC_MODE", "crash")
        again = _run(out_dir, board_error=BOARD_ERROR, calibrate=True)
        assert again.errors == 0
        assert again.runs == first.runs

    def test_calibrate_recaptures_changed_scene(self, fake_isaac, tmp_path, monkeypatch):
        """scene이 바뀌면 다시 캡처한다. 이전 판정은 지운다."""
        out_dir = tmp_path / "bench"
        _run(out_dir, board_error=BOARD_ERROR, calibrate=True)
        monkeypatch.setenv("FAKE_ISAAC_MODE", "crash")
        summary = _run(out_dir, board_error=SHIFT_ONLY_ERROR, calibrate=True)
        assert summary.errors_by_kind == {RunErrorKind.RUNNER_ERROR: 2}
        assert not (out_dir / "000_one/r0" / CHECK_FILE).exists()

    def test_calibrate_recaptures_changed_capture_task(self, fake_isaac, tmp_path, monkeypatch):
        """셀 설정이 바뀌어 캡처 작업이 다르면 다시 캡처한다."""
        out_dir = tmp_path / "bench"
        _run(out_dir, calibrate=True)
        monkeypatch.setenv("FAKE_ISAAC_MODE", "crash")
        summary = _run(out_dir, cell=CellConfig(part_inset=0.001), calibrate=True)
        assert summary.errors_by_kind == {RunErrorKind.RUNNER_ERROR: 2}

    def test_calibrate_skips_layout_that_does_not_compile(self, fake_isaac, tmp_path):
        """셀 설정으로 컴파일하지 못한 레이아웃은 캡처하지 않는다."""
        out_dir = tmp_path / "bench"
        summary = _run(out_dir, cell=UNREACHABLE_CELL, calibrate=True)
        assert summary.errors_by_kind == {RunErrorKind.COMPILE_FAILED: 2}
        assert not (out_dir / "000_one/r0" / CAPTURE_DIR).exists()

    def test_capture_failure_is_runner_error(self, fake_isaac, tmp_path, monkeypatch):
        """캡처 실행기가 파일을 쓰지 못하면 실행기 오류다. 부품 실행은 하지 않는다."""
        monkeypatch.setenv("FAKE_ISAAC_MODE", "crash")
        out_dir = tmp_path / "bench"
        summary = _run(out_dir, board_error=BOARD_ERROR, calibrate=True)
        assert summary.errors_by_kind == {RunErrorKind.RUNNER_ERROR: 2}
        assert summary.runs[0].error is not None
        assert "Fake runner failed" in summary.runs[0].error
        assert summary.runs[0].calibration_error_mm is None
        assert not (out_dir / "000_one/r0" / TASK_FILE).exists()

    def test_calibration_failure_is_run_error(self, fake_isaac, tmp_path):
        """보드 자세를 구하지 못하면 calibration_failed다. 부품 실행은 하지 않는다."""
        out_dir = tmp_path / "bench"
        cell = CellConfig(markers=CellConfig().markers[:3])
        summary = _run(out_dir, cell=cell, board_error=BOARD_ERROR, calibrate=True)
        assert summary.errors_by_kind == {RunErrorKind.CALIBRATION_FAILED: 2}
        assert summary.runs[0].error is not None
        assert "markers" in summary.runs[0].error
        assert (out_dir / "000_one/r0" / ERROR_FILE).is_file()
        assert not (out_dir / "000_one/r0" / TASK_FILE).exists()

    def test_unreachable_calibrated_board_is_out_of_reach(self, fake_isaac, tmp_path, monkeypatch):
        """보정한 보드가 팔에 닿지 않으면 out_of_reach다. 추정 오차는 적는다."""
        near = BoardOffset(x=-0.010, y=0.0, yaw=0.0)
        monkeypatch.setattr(bench, "sample_board_offset", lambda rng, *, limit: near)
        layout = make_layout(make_part("P001", PartType.SQUARE_TILE, origin=(0, 3)))
        summary = run_cell_benchmark(
            [BenchLayout("near", layout)],
            out_dir=tmp_path / "bench",
            cell=CellConfig(),
            sim_settings=_settings(),
            start_interval_s=0.0,
            board_error=BOARD_ERROR,
            calibrate=True,
        )
        (run,) = summary.runs
        assert run.error_kind == RunErrorKind.OUT_OF_REACH
        assert run.error is not None
        assert "pre_place" in run.error
        assert run.calibration_error_mm is not None
        assert run.order_changed is None

    def test_order_changed_run(self, fake_isaac, tmp_path, monkeypatch):
        """보정 보드로 컴파일한 순서가 셀 설정의 순서와 다르면 센다.

        두 정사각은 pan 축에서 거리가 같다. 보드가 +y로 가면 P002가 더 멀어져서 먼저 놓인다.
        """
        shifted = BoardOffset(x=0.0, y=0.005, yaw=0.0)
        monkeypatch.setattr(bench, "sample_board_offset", lambda rng, *, limit: shifted)
        layout = make_layout(
            make_part("P001", PartType.SQUARE_TILE, origin=(3, 1)),
            make_part("P002", PartType.SQUARE_TILE, origin=(3, 6)),
        )
        out_dir = tmp_path / "bench"
        summary = run_cell_benchmark(
            [BenchLayout("mirror", layout)],
            out_dir=out_dir,
            cell=CellConfig(),
            sim_settings=_settings(),
            start_interval_s=0.0,
            board_error=BOARD_ERROR,
            calibrate=True,
        )
        assert summary.order_changed_runs == 1
        task = Task.model_validate_json((out_dir / "000_mirror/r0" / TASK_FILE).read_text())
        assert [part.part_id for part in task.parts] == ["P002", "P001"]
        assert summary.total.failed == 0

    def test_calibrated_repeat_moves_feeder(self, fake_isaac, tmp_path):
        """보정해도 반복 1 이상은 공급대 위치만 옮긴다. 반복 0은 보정 task 그대로다."""
        out_dir = tmp_path / "bench"
        _run(out_dir, repeats=2, calibrate=True)
        run_dir = out_dir / "000_one"
        first = Task.model_validate_json((run_dir / "r0" / TASK_FILE).read_text())
        second = Task.model_validate_json((run_dir / "r1" / TASK_FILE).read_text())
        calibrated = CellConfig.model_validate_json(
            (run_dir / "r0" / CAPTURE_DIR / CALIBRATED_CELL_FILE).read_text()
        )
        assert first == compile_task(_bench_layouts()[0].layout, calibrated)
        assert second.parts[0].feeder_cells != first.parts[0].feeder_cells

    def test_actual_board_plan_places_parts_on_actual_board(
        self, fake_isaac, tmp_path, monkeypatch
    ):
        """실제 보드 계획은 캡처 없이 scene의 보드 자세로 컴파일한다. 부품이 실제 목표에 놓인다."""
        monkeypatch.setattr(bench, "sample_board_offset", lambda rng, *, limit: SHIFT_ONLY_ERROR)
        out_dir = tmp_path / "bench"
        summary = _run(out_dir, board_error=SHIFT_ONLY_ERROR, plan_on_actual_board=True)
        assert summary.errors == 0
        assert summary.total.failed == 0
        run_dir = out_dir / "000_one/r0"
        task = Task.model_validate_json((run_dir / TASK_FILE).read_text())
        scene = SimScene.model_validate_json((run_dir / SCENE_FILE).read_text())
        assert scene.board is not None
        assert task.cell.board_origin == scene.board.board_origin
        assert not (run_dir / CAPTURE_DIR).exists()
        for run in summary.runs:
            assert run.board_shift_mm == pytest.approx(8.0)
            assert run.calibration_error_mm is None

    def test_actual_board_plan_keeps_feeder(self, fake_isaac, tmp_path):
        """실제 보드 계획에서 보드가 돌아도 공급대의 부품 위치는 셀 설정의 위치다."""
        out_dir = tmp_path / "bench"
        _run(out_dir, board_error=BOARD_ERROR, plan_on_actual_board=True)
        task = Task.model_validate_json((out_dir / "000_one/r0" / TASK_FILE).read_text())
        planned = compile_task(_bench_layouts()[0].layout, CellConfig())
        assert task.parts[0].feeder_cells == pytest.approx(planned.parts[0].feeder_cells)
        assert task.parts[0].target_center != pytest.approx(planned.parts[0].target_center)

    def test_actual_board_plan_without_board_error_writes_scene(self, fake_isaac, tmp_path):
        """보드 오차가 없어도 실제 보드 계획은 scene에 보드를 적는다."""
        out_dir = tmp_path / "bench"
        summary = _run(out_dir, plan_on_actual_board=True)
        scene = SimScene.model_validate_json((out_dir / "000_one/r0" / SCENE_FILE).read_text())
        assert scene.board == placement_of(CellConfig(), Board())
        assert summary.total.failed == 0

    def test_actual_board_plan_excludes_calibrate(self, fake_isaac, tmp_path):
        """실제 보드 계획과 보정을 함께 주면 실행 전에 ValueError다."""
        with pytest.raises(ValueError, match="exclusive"):
            _run(tmp_path / "bench", calibrate=True, plan_on_actual_board=True)

    def test_unreachable_actual_board_is_out_of_reach(self, fake_isaac, tmp_path, monkeypatch):
        """실제 보드가 팔에 닿지 않으면 out_of_reach다. 실행기를 실행하지 않는다."""
        near = BoardOffset(x=-0.010, y=0.0, yaw=0.0)
        monkeypatch.setattr(bench, "sample_board_offset", lambda rng, *, limit: near)
        layout = make_layout(make_part("P001", PartType.SQUARE_TILE, origin=(0, 3)))
        out_dir = tmp_path / "bench"
        summary = run_cell_benchmark(
            [BenchLayout("near", layout)],
            out_dir=out_dir,
            cell=CellConfig(),
            sim_settings=_settings(),
            start_interval_s=0.0,
            board_error=BOARD_ERROR,
            plan_on_actual_board=True,
        )
        (run,) = summary.runs
        assert run.error_kind == RunErrorKind.OUT_OF_REACH
        assert run.error is not None
        assert "pre_place" in run.error
        assert not (out_dir / "000_near/r0" / TASK_FILE).exists()

    def test_record_writes_episode_files(self, fake_isaac, tmp_path):
        """기록을 켜면 실행마다 시연 기록 파일을 쓴다."""
        out_dir = tmp_path / "bench"
        _run(out_dir, record=True)
        for name in RECORD_OUTPUTS:
            assert (out_dir / "000_one/r0/sim" / name).is_file(), name

    def test_record_reruns_run_without_episode_files(self, fake_isaac, tmp_path, monkeypatch):
        """판정이 있어도 시연 기록 파일이 없으면 기록 실행은 다시 한다. 있으면 건너뛴다."""
        out_dir = tmp_path / "bench"
        _run(out_dir)
        assert _run(out_dir, record=True).errors == 0
        monkeypatch.setenv("FAKE_ISAAC_MODE", "crash")
        assert _run(out_dir, record=True).errors == 0
        (out_dir / "000_one/r0/sim" / RECORD_OUTPUTS[-1]).unlink()
        assert _run(out_dir, record=True).errors == 1

    def test_policy_writes_context_and_episode(self, fake_isaac, tmp_path):
        """제어기 서버를 주면 실행마다 문맥 파일을 쓰고 서버의 관절 목표를 기록한다."""
        policy = _HoldPolicy()
        out_dir = tmp_path / "bench"
        with PolicyServer(policy) as server:
            bench_policy = BenchPolicy(
                address=server.address, max_steps=4, contexts=[{"name": "a"}, {"name": "b"}]
            )
            summary = _run(out_dir, repeats=2, policy=bench_policy)
        assert summary.errors == 0
        context = json.loads((out_dir / "001_two/r1" / POLICY_CONTEXT_FILE).read_text())
        assert context == {"name": "b", "layout": "001_two", "repeat": 1}
        assert [seen["layout"] for seen in policy.contexts] == [
            "000_one",
            "000_one",
            "001_two",
            "001_two",
        ]
        for name in RECORD_OUTPUTS:
            assert (out_dir / "000_one/r0/sim" / name).is_file(), name

    def test_policy_needs_context_for_each_layout(self, fake_isaac, tmp_path):
        """문맥의 수가 레이아웃의 수와 다르면 실행 전에 ValueError다."""
        policy = BenchPolicy(
            address=PolicyAddress(host="127.0.0.1", port=1), max_steps=4, contexts=[{}]
        )
        with pytest.raises(ValueError, match="one context"):
            _run(tmp_path / "bench", policy=policy)

    def test_policy_failure_is_runner_error(self, fake_isaac, tmp_path):
        """제어기 서버에 접속할 수 없으면 그 실행은 runner_error다."""
        with PolicyServer(_HoldPolicy()) as server:
            closed = server.address
        policy = BenchPolicy(address=closed, max_steps=4, contexts=[{}, {}])
        summary = _run(tmp_path / "bench", policy=policy)
        assert summary.errors == 2
        assert summary.errors_by_kind == {RunErrorKind.RUNNER_ERROR: 2}

    def test_missing_isaac_raises_before_runs(self, fake_isaac, tmp_path):
        """Isaac 실행 파일이 없으면 레이아웃을 쓰기 전에 SimSetupError다."""
        (fake_isaac / "python.sh").unlink()
        with pytest.raises(SimSetupError):
            _run(tmp_path / "bench")
        assert not (tmp_path / "bench").exists()


@pytest.mark.unit
class TestBenchCellCommand:
    """bench-cell 명령 테스트."""

    def test_generated_and_file_layouts(self, fake_isaac, tmp_path):
        """무작위 레이아웃과 디렉터리의 레이아웃을 실행하고 요약 줄을 출력한다."""
        layout_dir = tmp_path / "layouts"
        layout_dir.mkdir()
        (layout_dir / "one.json").write_text(_bench_layouts()[0].layout.model_dump_json())
        out_dir = tmp_path / "bench"
        args = ["bench-cell", "-o", str(out_dir), "--count", "1", "--layouts", str(layout_dir)]
        result = runner.invoke(app, args)
        assert result.exit_code == 0, result.output
        assert "runs_passed=2/2 errors=0" in result.output
        assert (out_dir / "000_random_00").is_dir()
        assert (out_dir / "001_one").is_dir()
        summary = json.loads((out_dir / SUMMARY_FILE).read_text())
        assert summary["total"]["failure_rate"] == 0.0

    def test_failed_part_lines(self, fake_isaac, tmp_path, monkeypatch):
        """실패한 부품마다 이웃 면 수, 틈, 실패 항목, 단계를 출력한다. 부품 실패는 exit 0이다."""
        monkeypatch.setenv("FAKE_ISAAC_MODE", "miss")
        result = runner.invoke(app, ["bench-cell", "-o", str(tmp_path / "b"), "--count", "1"])
        assert result.exit_code == 0, result.output
        assert "000_random_00 r0 P" in result.output
        assert "clearance_mm=none fail position stage=place" in result.output

    def test_run_error_exit_one(self, fake_isaac, tmp_path, monkeypatch):
        """실행하지 못한 실행이 있으면 오류 줄을 쓰고 exit 1이다."""
        monkeypatch.setenv("FAKE_ISAAC_MODE", "crash")
        result = runner.invoke(app, ["bench-cell", "-o", str(tmp_path / "b"), "--count", "1"])
        assert result.exit_code == EXIT_INFEASIBLE
        assert "000_random_00 r0 error runner_error" in result.output

    def test_board_error_option(self, fake_isaac, tmp_path):
        """--board-error를 주면 실행마다 scene.json을 쓴다."""
        out_dir = tmp_path / "bench"
        args = ["bench-cell", "-o", str(out_dir), "--count", "1", "--board-error", "3,5,1.5"]
        result = runner.invoke(app, args)
        assert result.exit_code == 0, result.output
        assert (out_dir / "000_random_00/r0" / SCENE_FILE).is_file()
        summary = json.loads((out_dir / SUMMARY_FILE).read_text())
        assert summary["runs"][0]["board_shift_mm"] > 0

    def test_calibrate_option(self, fake_isaac, tmp_path):
        """--calibrate와 --camera-error를 주면 캡처한다. 요약 줄에 추정 오차와 순서 변경이 있다."""
        out_dir = tmp_path / "bench"
        args = ["bench-cell", "-o", str(out_dir), "--count", "1", "--board-error", "3,5,1.5"]
        args += ["--camera-error", "10,1", "--calibrate"]
        result = runner.invoke(app, args)
        assert result.exit_code == 0, result.output
        assert "calibration_error_max_mm=0.0" in result.output
        assert "order_changed_runs=" in result.output
        assert (out_dir / "000_random_00/r0" / CAPTURE_DIR / CALIBRATED_CELL_FILE).is_file()
        scene = SimScene.model_validate_json(
            (out_dir / "000_random_00/r0" / SCENE_FILE).read_text()
        )
        assert scene.camera is not None
        assert max(np.abs(scene.camera.position)) <= 0.01

    def test_camera_error_needs_calibrate(self, fake_isaac, tmp_path):
        """--camera-error만 주면 exit 2다."""
        args = ["bench-cell", "-o", str(tmp_path / "b"), "--count", "1", "--camera-error", "10,1"]
        result = runner.invoke(app, args)
        assert result.exit_code == EXIT_USAGE
        assert "--calibrate" in result.output
        assert not (tmp_path / "b").exists()

    @pytest.mark.parametrize("value", ["10", "10,x", "10,-1"])
    def test_invalid_camera_error_exit_two(self, fake_isaac, tmp_path, value):
        """--camera-error가 POS_MM,ANGLE_DEG가 아니거나 음수가 있으면 exit 2다."""
        args = ["bench-cell", "-o", str(tmp_path / "b"), "--count", "1", "--calibrate"]
        result = runner.invoke(app, [*args, "--camera-error", value])
        assert result.exit_code == EXIT_USAGE

    @pytest.mark.parametrize("value", ["3,5", "3,5,x", "3,-5,1.5"])
    def test_invalid_board_error_exit_two(self, fake_isaac, tmp_path, value):
        """--board-error가 X_MM,Y_MM,YAW_DEG가 아니거나 음수가 있으면 exit 2다."""
        args = ["bench-cell", "-o", str(tmp_path / "b"), "--count", "1", "--board-error", value]
        result = runner.invoke(app, args)
        assert result.exit_code == EXIT_USAGE
        assert not (tmp_path / "b").exists()

    def test_invalid_shard_exit_two(self, fake_isaac, tmp_path):
        """shard 형식이 틀리면 exit 2다."""
        args = ["bench-cell", "-o", str(tmp_path / "b"), "--shard", "2/2"]
        result = runner.invoke(app, args)
        assert result.exit_code == EXIT_USAGE

    def test_no_layouts_exit_two(self, fake_isaac, tmp_path):
        """레이아웃이 없으면 exit 2다."""
        result = runner.invoke(app, ["bench-cell", "-o", str(tmp_path / "b"), "--count", "0"])
        assert result.exit_code == EXIT_USAGE
        assert "no layouts" in result.output

    def test_invalid_layout_file_exit_two(self, fake_isaac, tmp_path):
        """레이아웃 파일 형식이 틀리면 exit 2다."""
        layout_dir = tmp_path / "layouts"
        layout_dir.mkdir()
        (layout_dir / "bad.json").write_text('{"parts": 1}')
        args = ["bench-cell", "-o", str(tmp_path / "b"), "--layouts", str(layout_dir)]
        result = runner.invoke(app, args)
        assert result.exit_code == EXIT_USAGE

    def test_missing_python_sh_exit_two(self, fake_isaac, tmp_path):
        """python.sh가 없으면 exit 2다."""
        (fake_isaac / "python.sh").unlink()
        result = runner.invoke(app, ["bench-cell", "-o", str(tmp_path / "b"), "--count", "1"])
        assert result.exit_code == EXIT_USAGE
        assert "ISAAC_SIM_PATH" in result.output

    def test_missing_settings_exit_two(self, fake_isaac, tmp_path, monkeypatch):
        """ISAAC_SIM_PATH가 없으면 exit 2다."""
        monkeypatch.delenv("ISAAC_SIM_PATH")
        result = runner.invoke(app, ["bench-cell", "-o", str(tmp_path / "b"), "--count", "1"])
        assert result.exit_code == EXIT_USAGE
        assert "isaac_sim_path" in result.output
