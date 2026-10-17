import json
import time
from pathlib import Path

import numpy as np
import pytest
from typer.testing import CliRunner

from pixel2place import cli
from pixel2place.cell import sim
from pixel2place.cell.config import CellConfig
from pixel2place.cell.policy_link import DOF_NAMES, PolicyAddress, PolicyServer
from pixel2place.cell.scene import BoardOffset, SceneFaults, SimScene, displace_board
from pixel2place.cell.sim import (
    EPISODE_FILE,
    LOG_FILE,
    PRESENT_VIDEO_FILE,
    RECORD_OUTPUTS,
    PolicyRun,
    SimRunError,
    SimSetupError,
    run_place_task,
)
from pixel2place.cell.task import Task, compile_task
from pixel2place.cli import EXIT_INFEASIBLE, EXIT_USAGE, app
from pixel2place.config import SimSettings
from pixel2place.layout.schema import PartType
from pixel2place.reasoner.client import CosmosReasoner
from tests.factories import (
    make_layout,
    make_part,
    make_planner_response_json,
    make_sim_result,
)
from tests.fakes.fake_openai import FakeOpenAI

runner = CliRunner()
# 가짜 실행기는 hang에서 60초 멈춘다. timeout이 실행기를 끝내면 이 시간 안에 돌아온다.
MAX_KILL_WAIT_S = 10.0


@pytest.fixture
def task_path(tmp_path) -> Path:
    layout = make_layout(make_part("P001", PartType.SQUARE_TILE, origin=(3, 3)))
    path = tmp_path / "task.json"
    path.write_text(compile_task(layout, CellConfig()).model_dump_json())
    return path


def _settings(**overrides) -> SimSettings:
    return SimSettings(**overrides)  # pyright: ignore[reportCallIssue]


@pytest.fixture
def scene_path(task_path, tmp_path) -> Path:
    """보드를 x로 5 mm 옮긴 scene.json."""
    task = Task.model_validate_json(task_path.read_text())
    offset = BoardOffset(x=0.005, y=0.0, yaw=0.0)
    path = tmp_path / "scene.json"
    path.write_text(
        SimScene(board=displace_board(task.cell, task.board, offset=offset)).model_dump_json()
    )
    return path


class _ConstantPolicy:
    """같은 관절 목표 3개를 주는 제어기."""

    VALUE = 0.25

    def __init__(self, *, max_steps: int | None) -> None:
        self.max_steps = max_steps
        self.contexts: list = []

    def reset(self, context):
        self.contexts.append(dict(context))
        return self.max_steps

    def act(self, observation):
        return np.full((3, len(DOF_NAMES)), self.VALUE, dtype=np.float32)


@pytest.mark.unit
class TestRunPlaceTask:
    """Isaac 실행기 subprocess 테스트."""

    def test_reads_result(self, fake_isaac, task_path, tmp_path):
        """실행기가 쓴 result.json을 읽고 전체 출력을 로그 파일에 쓴다."""
        out_dir = tmp_path / "out"
        result = run_place_task(task_path, out_dir=out_dir, settings=_settings())
        assert [part.part_id for part in result.parts] == ["P001"]
        log = (out_dir / LOG_FILE).read_text()
        assert "omni.fake" in log
        assert "[place_task] Task finished." in log

    def test_passes_scene(self, fake_isaac, task_path, scene_path, tmp_path):
        """scene.json을 실행기에 넘긴다. result.json의 board는 scene의 보드 자세다."""
        result = run_place_task(
            task_path, out_dir=tmp_path / "out", settings=_settings(), scene_path=scene_path
        )
        assert result.board == SimScene.model_validate_json(scene_path.read_text()).board

    def test_record_writes_episode_files(self, fake_isaac, task_path, tmp_path):
        """기록을 켜면 실행기가 영상 2개와 프레임별 관절 값을 쓴다."""
        out_dir = tmp_path / "out"
        run_place_task(task_path, out_dir=out_dir, settings=_settings(), record=True)
        for name in RECORD_OUTPUTS:
            assert (out_dir / name).is_file(), name

    def test_no_record_has_no_episode_files(self, fake_isaac, task_path, tmp_path):
        """기록을 켜지 않으면 실행기에 --record를 주지 않는다."""
        out_dir = tmp_path / "out"
        run_place_task(task_path, out_dir=out_dir, settings=_settings())
        assert not any((out_dir / name).exists() for name in RECORD_OUTPUTS)

    def test_present_writes_presentation_video(self, fake_isaac, task_path, tmp_path):
        """발표 영상을 켜면 실행기가 발표 영상을 쓴다. 시연 기록 파일은 쓰지 않는다."""
        out_dir = tmp_path / "out"
        run_place_task(task_path, out_dir=out_dir, settings=_settings(), present=True)
        assert (out_dir / PRESENT_VIDEO_FILE).is_file()
        assert not any((out_dir / name).exists() for name in RECORD_OUTPUTS)

    def test_no_present_has_no_presentation_video(self, fake_isaac, task_path, tmp_path):
        """발표 영상을 켜지 않으면 실행기에 --present를 주지 않는다."""
        out_dir = tmp_path / "out"
        run_place_task(task_path, out_dir=out_dir, settings=_settings(), record=True)
        assert not (out_dir / PRESENT_VIDEO_FILE).exists()

    def test_no_present_removes_earlier_presentation_video(self, fake_isaac, task_path, tmp_path):
        """발표 영상을 켜지 않은 실행은 같은 디렉터리의 이전 발표 영상을 지운다."""
        out_dir = tmp_path / "out"
        run_place_task(task_path, out_dir=out_dir, settings=_settings(), present=True)
        result = run_place_task(task_path, out_dir=out_dir, settings=_settings())
        assert result.parts
        assert not (out_dir / PRESENT_VIDEO_FILE).exists()

    def test_present_with_record_writes_all_files(self, fake_isaac, task_path, tmp_path):
        """발표 영상과 기록을 함께 켜면 실행기가 두 쪽의 파일을 모두 쓴다."""
        out_dir = tmp_path / "out"
        run_place_task(task_path, out_dir=out_dir, settings=_settings(), record=True, present=True)
        for name in (*RECORD_OUTPUTS, PRESENT_VIDEO_FILE):
            assert (out_dir / name).is_file(), name

    def test_missing_presentation_video_is_an_error(
        self, fake_isaac, task_path, tmp_path, monkeypatch
    ):
        """실행기가 발표 영상을 쓰지 못하면 result.json이 있어도 SimRunError다.

        이전 실행의 발표 영상은 읽지 않는다.
        """
        out_dir = tmp_path / "out"
        run_place_task(task_path, out_dir=out_dir, settings=_settings(), present=True)
        monkeypatch.setenv("FAKE_ISAAC_MODE", "no_present_video")
        with pytest.raises(SimRunError, match=PRESENT_VIDEO_FILE):
            run_place_task(task_path, out_dir=out_dir, settings=_settings(), present=True)
        assert (out_dir / "result.json").is_file()

    def test_policy_drives_runner(self, fake_isaac, task_path, tmp_path):
        """제어기 서버를 주면 실행기가 문맥을 보내고 서버의 관절 목표를 따른다. 기록 파일도 쓴다."""
        policy = _ConstantPolicy(max_steps=5)
        context_path = tmp_path / "context.json"
        context_path.write_text(json.dumps({"slot": 1}))
        out_dir = tmp_path / "out"
        with PolicyServer(policy) as server:
            run_place_task(
                task_path,
                out_dir=out_dir,
                settings=_settings(),
                policy=PolicyRun(address=server.address, context_path=context_path, max_steps=9),
            )
        assert policy.contexts == [{"slot": 1}]
        for name in RECORD_OUTPUTS:
            assert (out_dir / name).is_file(), name
        with np.load(out_dir / EPISODE_FILE) as arrays:
            action = arrays["action"]
        # 첫 프레임은 처음 자세다. 그 뒤 4프레임이 제어기의 목표다. 서버의 한도 5가 9보다 작다.
        assert action.shape == (5, len(DOF_NAMES))
        np.testing.assert_allclose(action[1:], _ConstantPolicy.VALUE)

    def test_policy_step_limit_of_runner(self, fake_isaac, task_path, tmp_path):
        """서버가 한도를 주지 않으면 실행기의 프레임 수 한도를 쓴다."""
        context_path = tmp_path / "context.json"
        context_path.write_text("{}")
        out_dir = tmp_path / "out"
        with PolicyServer(_ConstantPolicy(max_steps=None)) as server:
            run_place_task(
                task_path,
                out_dir=out_dir,
                settings=_settings(),
                policy=PolicyRun(address=server.address, context_path=context_path, max_steps=6),
            )
        with np.load(out_dir / EPISODE_FILE) as arrays:
            assert len(arrays["action"]) == 6

    def test_policy_server_error_has_no_result(self, fake_isaac, task_path, tmp_path):
        """제어기 서버에 접속할 수 없으면 실행기는 결과를 쓰지 못한다."""
        context_path = tmp_path / "context.json"
        context_path.write_text("{}")
        with PolicyServer(_ConstantPolicy(max_steps=5)) as server:
            closed = PolicyAddress(host=server.address.host, port=server.address.port)
        with pytest.raises(SimRunError, match="result.json"):
            run_place_task(
                task_path,
                out_dir=tmp_path / "out",
                settings=_settings(),
                policy=PolicyRun(address=closed, context_path=context_path, max_steps=5),
            )

    def test_removes_old_result(self, fake_isaac, task_path, tmp_path, monkeypatch):
        """실행기가 결과를 쓰지 못하면 이전 result.json을 읽지 않는다."""
        out_dir = tmp_path / "out"
        run_place_task(task_path, out_dir=out_dir, settings=_settings())
        monkeypatch.setenv("FAKE_ISAAC_MODE", "crash")
        with pytest.raises(SimRunError, match="Fake runner failed"):
            run_place_task(task_path, out_dir=out_dir, settings=_settings())

    def test_timeout_kills_silent_runner(self, fake_isaac, task_path, tmp_path, monkeypatch):
        """출력 없이 멈춘 실행기는 timeout 뒤에 끝낸다."""
        monkeypatch.setenv("FAKE_ISAAC_MODE", "hang")
        with pytest.raises(SimRunError, match="did not finish"):
            run_place_task(
                task_path, out_dir=tmp_path / "out", settings=_settings(sim_timeout_s=0.5)
            )

    def test_timeout_kills_runner_under_wrapper(self, fake_isaac, task_path, tmp_path, monkeypatch):
        """python.sh가 실행기를 자식 프로세스로 실행해도 timeout 뒤에 실행기를 끝낸다.

        Isaac의 python.sh는 Python을 자식으로 실행한다. 자식이 남으면 출력 파이프가 닫히지 않는다.
        """
        python_sh = fake_isaac / "python.sh"
        python_sh.write_text(python_sh.read_text().replace(" exec ", " "))
        monkeypatch.setenv("FAKE_ISAAC_MODE", "hang")
        started = time.monotonic()
        with pytest.raises(SimRunError, match="did not finish"):
            run_place_task(
                task_path, out_dir=tmp_path / "out", settings=_settings(sim_timeout_s=0.5)
            )
        assert time.monotonic() - started < MAX_KILL_WAIT_S

    def test_runner_that_does_not_exit_after_result(
        self, fake_isaac, task_path, tmp_path, monkeypatch
    ):
        """실행기가 결과와 끝 로그를 쓴 뒤 끝나지 않으면 프로세스를 끝내고 결과를 읽는다."""
        python_sh = fake_isaac / "python.sh"
        python_sh.write_text(python_sh.read_text().replace(" exec ", " "))
        monkeypatch.setenv("FAKE_ISAAC_MODE", "linger")
        monkeypatch.setattr(sim, "EXIT_GRACE_S", 0.5)
        started = time.monotonic()
        result = run_place_task(task_path, out_dir=tmp_path / "out", settings=_settings())
        assert [part.part_id for part in result.parts] == ["P001"]
        assert time.monotonic() - started < MAX_KILL_WAIT_S

    def test_missing_python_sh(self, fake_isaac, task_path, tmp_path):
        """python.sh가 없으면 SimSetupError다."""
        (fake_isaac / "python.sh").unlink()
        with pytest.raises(SimSetupError, match="ISAAC_SIM_PATH"):
            run_place_task(task_path, out_dir=tmp_path / "out", settings=_settings())

    def test_missing_usd(self, fake_isaac, task_path, tmp_path):
        """SO-101 USD가 없으면 만드는 방법을 알려 준다."""
        with pytest.raises(SimSetupError, match="import_so101"):
            run_place_task(
                task_path,
                out_dir=tmp_path / "out",
                settings=_settings(),
                usd_path=tmp_path / "missing.usda",
            )


@pytest.mark.unit
class TestSimPlaceCommand:
    """sim-place 명령 테스트."""

    def test_success(self, fake_isaac, task_path, tmp_path):
        """놓은 부품 수와 출력 디렉터리를 출력한다."""
        out_dir = tmp_path / "out"
        result = runner.invoke(app, ["sim-place", str(task_path), "-o", str(out_dir)])
        assert result.exit_code == 0, result.output
        assert f"parts=1/1 output={out_dir}" in result.output
        assert not (out_dir / PRESENT_VIDEO_FILE).exists()

    def test_present_option(self, fake_isaac, task_path, tmp_path):
        """--present를 주면 출력 디렉터리에 발표 영상이 있다."""
        out_dir = tmp_path / "out"
        result = runner.invoke(app, ["sim-place", str(task_path), "-o", str(out_dir), "--present"])
        assert result.exit_code == 0, result.output
        assert (out_dir / PRESENT_VIDEO_FILE).is_file()

    def test_missing_presentation_video_exit_one(
        self, fake_isaac, task_path, tmp_path, monkeypatch
    ):
        """--present에서 실행기가 발표 영상을 쓰지 못하면 exit 1이다."""
        monkeypatch.setenv("FAKE_ISAAC_MODE", "no_present_video")
        args = ["sim-place", str(task_path), "-o", str(tmp_path / "o"), "--present"]
        result = runner.invoke(app, args)
        assert result.exit_code == EXIT_INFEASIBLE
        assert PRESENT_VIDEO_FILE in result.output

    def test_scene_option(self, fake_isaac, task_path, scene_path, tmp_path):
        """--scene의 보드 자세가 result.json에 있다. 믿는 보드의 목표는 5 mm 벗어나서 실패다."""
        out_dir = tmp_path / "out"
        args = ["sim-place", str(task_path), "-o", str(out_dir), "--scene", str(scene_path)]
        result = runner.invoke(app, args)
        assert result.exit_code == 0, result.output
        assert json.loads((out_dir / "result.json").read_text())["board"] is not None
        check = runner.invoke(app, ["sim-check", str(task_path), str(out_dir / "result.json")])
        assert check.exit_code == EXIT_INFEASIBLE

    def test_scene_part_not_in_task_exit_two(self, fake_isaac, task_path, tmp_path):
        """장면이 작업에 없는 부품을 가리키면 Isaac을 실행하지 않고 exit 2다."""
        scene = tmp_path / "scene.json"
        scene.write_text(SimScene(faults=SceneFaults(missing=(2,))).model_dump_json())
        args = ["sim-place", str(task_path), "-o", str(tmp_path / "o"), "--scene", str(scene)]
        result = runner.invoke(app, args)
        assert result.exit_code == EXIT_USAGE
        assert "does not have" in result.output
        assert not (tmp_path / "o").exists()

    def test_invalid_scene_exit_two(self, fake_isaac, task_path, tmp_path):
        """scene.json 형식이 틀리면 Isaac을 실행하지 않고 exit 2다."""
        bad = tmp_path / "scene.json"
        bad.write_text('{"board": {"board_origin": [0.1, 0.0]}}')
        args = ["sim-place", str(task_path), "-o", str(tmp_path / "o"), "--scene", str(bad)]
        result = runner.invoke(app, args)
        assert result.exit_code == EXIT_USAGE
        assert not (tmp_path / "o").exists()

    def test_runner_failure_exit_one(self, fake_isaac, task_path, tmp_path, monkeypatch):
        """실행기가 결과를 쓰지 못하면 exit 1이고 로그 끝을 보여 준다."""
        monkeypatch.setenv("FAKE_ISAAC_MODE", "crash")
        result = runner.invoke(app, ["sim-place", str(task_path), "-o", str(tmp_path / "o")])
        assert result.exit_code == EXIT_INFEASIBLE
        assert "Fake runner failed" in result.output

    def test_missing_settings_exit_two(self, fake_isaac, task_path, tmp_path, monkeypatch):
        """ISAAC_SIM_PATH가 없으면 exit 2다."""
        monkeypatch.delenv("ISAAC_SIM_PATH")
        result = runner.invoke(app, ["sim-place", str(task_path), "-o", str(tmp_path / "o")])
        assert result.exit_code == EXIT_USAGE
        assert "isaac_sim_path" in result.output

    def test_invalid_task_exit_two(self, fake_isaac, tmp_path):
        """task.json 형식이 틀리면 Isaac을 실행하지 않고 exit 2다."""
        bad = tmp_path / "bad.json"
        bad.write_text('{"task_version": "0.1"}')
        result = runner.invoke(app, ["sim-place", str(bad), "-o", str(tmp_path / "o")])
        assert result.exit_code == EXIT_USAGE

    def test_missing_python_sh_exit_two(self, fake_isaac, task_path, tmp_path):
        """python.sh가 없으면 exit 2다."""
        (fake_isaac / "python.sh").unlink()
        result = runner.invoke(app, ["sim-place", str(task_path), "-o", str(tmp_path / "o")])
        assert result.exit_code == EXIT_USAGE


def _write_result(
    task_path: Path, path: Path, *, x_offset: float = 0.0, z_offset: float = 0.0
) -> Path:
    """부품의 최종 자세를 목표에서 x로 `x_offset`, z로 `z_offset`(m) 옮긴 result.json을 쓴다."""
    task = Task.model_validate_json(task_path.read_text())
    result = make_sim_result(task)
    parts = []
    for pose in result.parts:
        x, y, z = pose.position
        parts.append(pose.model_copy(update={"position": (x + x_offset, y, z + z_offset)}))
    path.write_text(result.model_copy(update={"parts": parts}).model_dump_json())
    return path


@pytest.mark.unit
class TestSimReworkCommand:
    """sim-rework 명령 테스트."""

    @pytest.fixture
    def two_part_task(self, tmp_path) -> Path:
        layout = make_layout(
            make_part("P001", PartType.SQUARE_TILE, origin=(1, 1)),
            make_part("P002", PartType.SQUARE_TILE, origin=(5, 5)),
        )
        path = tmp_path / "two.task.json"
        path.write_text(compile_task(layout, CellConfig()).model_dump_json())
        return path

    @pytest.fixture
    def first_run(self, fake_isaac, two_part_task, tmp_path) -> tuple[Path, Path]:
        """둘째 부품을 공급하지 않은 첫 실행. `(result.json, scene.json)`을 반환한다."""
        scene_path = tmp_path / "first.scene.json"
        scene_path.write_text(SimScene(faults=SceneFaults(missing=(2,))).model_dump_json())
        out_dir = tmp_path / "first"
        args = ["sim-place", str(two_part_task), "-o", str(out_dir), "--scene", str(scene_path)]
        assert runner.invoke(app, args).exit_code == 0
        return out_dir / "result.json", scene_path

    def test_places_missing_part(self, two_part_task, first_run, tmp_path):
        """빠진 부품을 다시 놓으면 작업 전체의 판정이 통과한다."""
        result_path, scene_path = first_run
        failed = runner.invoke(app, ["sim-check", str(two_part_task), str(result_path)])
        assert failed.exit_code == EXIT_INFEASIBLE
        out_dir = tmp_path / "rework"
        args = ["sim-rework", str(two_part_task), str(result_path), "-o", str(out_dir)]
        result = runner.invoke(app, [*args, "--part", "2", "--scene", str(scene_path)])
        assert result.exit_code == 0, result.output
        assert f"parts=2/2 reworked=1 output={out_dir}" in result.output
        scene = SimScene.model_validate_json((out_dir / cli.REWORK_SCENE_FILE).read_text())
        assert scene.faults is None
        assert scene.preplaced is not None
        assert [part.sequence for part in scene.preplaced] == [1]
        check = runner.invoke(app, ["sim-check", str(two_part_task), str(out_dir / "result.json")])
        assert check.exit_code == 0, check.output

    def test_unknown_part_exit_two(self, two_part_task, first_run, tmp_path):
        """작업에 없는 부품이면 Isaac을 실행하지 않고 exit 2다."""
        result_path, _ = first_run
        out_dir = tmp_path / "rework"
        args = ["sim-rework", str(two_part_task), str(result_path), "-o", str(out_dir)]
        result = runner.invoke(app, [*args, "--part", "3"])
        assert result.exit_code == EXIT_USAGE
        assert not out_dir.exists()

    def test_output_in_first_run_directory_exit_two(self, two_part_task, first_run):
        """출력 디렉터리가 첫 결과의 디렉터리이면 exit 2다. 첫 결과는 그대로다."""
        result_path, scene_path = first_run
        before = result_path.read_bytes()
        args = ["sim-rework", str(two_part_task), str(result_path), "--part", "2"]
        for output in (result_path.parent, scene_path.parent):
            result = runner.invoke(app, [*args, "-o", str(output), "--scene", str(scene_path)])
            assert result.exit_code == EXIT_USAGE
            assert "Use a different directory" in result.output
        assert result_path.read_bytes() == before
        assert SimScene.model_validate_json(scene_path.read_text()).faults is not None

    def test_result_without_orientation_exit_two(self, fake_isaac, two_part_task, tmp_path):
        """quaternion이 없는 결과로는 재작업하지 못한다. exit 2다."""
        task = Task.model_validate_json(two_part_task.read_text())
        result_path = tmp_path / "result.json"
        result_path.write_text(make_sim_result(task).model_dump_json())
        args = ["sim-rework", str(two_part_task), str(result_path), "-o", str(tmp_path / "o")]
        result = runner.invoke(app, [*args, "--part", "2"])
        assert result.exit_code == EXIT_USAGE
        assert "no orientation" in result.output

    def test_runner_failure_exit_one(self, two_part_task, first_run, tmp_path, monkeypatch):
        """실행기가 결과를 쓰지 못하면 exit 1이다."""
        result_path, _ = first_run
        monkeypatch.setenv("FAKE_ISAAC_MODE", "crash")
        args = ["sim-rework", str(two_part_task), str(result_path), "-o", str(tmp_path / "o")]
        result = runner.invoke(app, [*args, "--part", "2"])
        assert result.exit_code == EXIT_INFEASIBLE


@pytest.mark.unit
class TestSimCheckCommand:
    """sim-check 명령 테스트."""

    def test_success(self, task_path, tmp_path):
        """부품별 판정 줄과 요약 줄을 출력한다."""
        result_path = _write_result(task_path, tmp_path / "result.json", x_offset=0.001)
        result = runner.invoke(app, ["sim-check", str(task_path), str(result_path)])
        assert result.exit_code == 0, result.output
        assert (
            "P001 seq=1 square_tile position_mm=1.0 yaw_deg=+0.0 tilt_deg=0.0 height_mm=+0.0 pass"
            in result.output
        )
        assert result.output.strip().endswith("passed=1/1 success")

    def test_out_of_tolerance_exit_one(self, task_path, tmp_path):
        """허용 오차 밖 부품이 있으면 실패 항목을 쓰고 exit 1이다."""
        result_path = _write_result(task_path, tmp_path / "result.json", x_offset=0.005)
        result = runner.invoke(app, ["sim-check", str(task_path), str(result_path)])
        assert result.exit_code == EXIT_INFEASIBLE
        assert "position_mm=5.0" in result.output
        assert "fail position stage=later" in result.output
        assert "passed=0/1 failure" in result.output

    def test_tolerance_options(self, task_path, tmp_path):
        """옵션으로 허용 오차를 바꾼다."""
        result_path = _write_result(task_path, tmp_path / "result.json", x_offset=0.005)
        args = ["sim-check", str(task_path), str(result_path), "--position-ratio", "0.3"]
        result = runner.invoke(app, args)
        assert result.exit_code == 0, result.output

    def test_height_option(self, task_path, tmp_path):
        """기본 높이 한계는 0.5 mm다. --height-mm으로 한계를 바꾼다."""
        result_path = _write_result(task_path, tmp_path / "result.json", z_offset=0.001)
        args = ["sim-check", str(task_path), str(result_path)]
        failed = runner.invoke(app, args)
        assert failed.exit_code == EXIT_INFEASIBLE
        assert "height_mm=+1.0 fail height stage=later" in failed.output
        passed = runner.invoke(app, [*args, "--height-mm", "1.5"])
        assert passed.exit_code == 0, passed.output

    def test_writes_report(self, task_path, tmp_path):
        """-o로 판정 JSON을 쓴다."""
        result_path = _write_result(task_path, tmp_path / "result.json", x_offset=0.005)
        output = tmp_path / "check.json"
        args = ["sim-check", str(task_path), str(result_path), "-o", str(output)]
        runner.invoke(app, args)
        report = json.loads(output.read_text())
        assert report["passed"] is False
        assert report["position_limit_mm"] == pytest.approx(4.0)
        assert report["parts"][0]["failures"] == ["position"]

    def test_report_makes_missing_parent_dir(self, task_path, tmp_path):
        """-o 경로의 부모 디렉터리가 없으면 그 디렉터리를 만든다."""
        result_path = _write_result(task_path, tmp_path / "result.json", x_offset=0.001)
        output = tmp_path / "new" / "dir" / "check.json"
        args = ["sim-check", str(task_path), str(result_path), "-o", str(output)]
        result = runner.invoke(app, args)
        assert result.exit_code == 0, result.output
        assert json.loads(output.read_text())["passed"] is True

    def test_mismatched_result_exit_two(self, task_path, tmp_path):
        """result.json의 부품이 task와 다르면 exit 2다."""
        task = Task.model_validate_json(task_path.read_text())
        result = make_sim_result(task)
        other = result.model_copy(
            update={"parts": [result.parts[0].model_copy(update={"part_id": "P999"})]}
        )
        result_path = tmp_path / "result.json"
        result_path.write_text(other.model_dump_json())
        result = runner.invoke(app, ["sim-check", str(task_path), str(result_path)])
        assert result.exit_code == EXIT_USAGE
        assert "P999" in result.output

    def test_invalid_result_exit_two(self, task_path, tmp_path):
        """result.json 형식이 틀리면 exit 2다."""
        result_path = tmp_path / "result.json"
        result_path.write_text('{"result_version": "0.1"}')
        result = runner.invoke(app, ["sim-check", str(task_path), str(result_path)])
        assert result.exit_code == EXIT_USAGE

    def test_checks_runner_output(self, fake_isaac, task_path, tmp_path):
        """sim-place가 쓴 result.json을 그대로 판정한다."""
        out_dir = tmp_path / "out"
        runner.invoke(app, ["sim-place", str(task_path), "-o", str(out_dir)])
        result = runner.invoke(app, ["sim-check", str(task_path), str(out_dir / "result.json")])
        assert result.exit_code == 0, result.output


# run-cell 결과 파일. 셀 실행 결과는 `sim/` 아래에 있다.
RUN_CELL_FILES = [
    "plan_layout.json",
    "raw_image.png",
    "layout.json",
    "normalized_image.png",
    "report.json",
    "task.json",
    "sim/result.json",
    "sim/place_task.log",
    "check.json",
]


@pytest.fixture
def fake_planner(monkeypatch) -> FakeOpenAI:
    """정사각 부품 1개를 계획하는 Fake Reasoner. 호출 기록으로 VLM 호출 여부를 본다."""
    layout = make_layout(make_part("P001", PartType.SQUARE_TILE, origin=(3, 3)))
    fake = FakeOpenAI([make_planner_response_json(layout)])
    monkeypatch.setattr(cli, "_reasoner", lambda: CosmosReasoner(client=fake, model="m"))
    return fake


@pytest.mark.unit
class TestRunCellCommand:
    """run-cell 명령 테스트."""

    def test_success(self, fake_isaac, fake_planner, tmp_path):
        """모든 결과 파일을 쓰고 판정 줄과 요약 줄을 출력한다."""
        out_dir = tmp_path / "out"
        result = runner.invoke(app, ["run-cell", "one square", "-o", str(out_dir)])
        assert result.exit_code == 0, result.output
        for name in RUN_CELL_FILES:
            assert (out_dir / name).is_file(), name
        assert "P001 seq=1 square_tile position_mm=0.0" in result.output
        assert f"cell_accuracy=1.000 passed=1/1 output={out_dir}" in result.output
        assert json.loads((out_dir / "check.json").read_text())["passed"] is True

    def test_out_of_tolerance_exit_one(self, fake_isaac, fake_planner, tmp_path, monkeypatch):
        """허용 오차 밖 부품이 있으면 exit 1이고 check.json에 실패 항목이 있다."""
        monkeypatch.setenv("FAKE_ISAAC_MODE", "miss")
        out_dir = tmp_path / "out"
        result = runner.invoke(app, ["run-cell", "one square", "-o", str(out_dir)])
        assert result.exit_code == EXIT_INFEASIBLE
        assert "passed=0/1" in result.output
        report = json.loads((out_dir / "check.json").read_text())
        assert report["parts"][0]["failures"] == ["position"]

    def test_runner_failure_exit_one(self, fake_isaac, fake_planner, tmp_path, monkeypatch):
        """실행기가 결과를 쓰지 못하면 exit 1이다."""
        monkeypatch.setenv("FAKE_ISAAC_MODE", "crash")
        result = runner.invoke(app, ["run-cell", "one square", "-o", str(tmp_path / "out")])
        assert result.exit_code == EXIT_INFEASIBLE
        assert "Fake runner failed" in result.output

    def test_compile_failure_exit_one(self, fake_isaac, fake_planner, tmp_path):
        """셀 설정에서 로봇이 보드에 닿지 않으면 컴파일 실패로 exit 1이다."""
        cell_path = tmp_path / "cell.json"
        cell_path.write_text(CellConfig(board_origin=(1.0, 0.0)).model_dump_json())
        out_dir = tmp_path / "out"
        args = ["run-cell", "one square", "-o", str(out_dir), "--cell", str(cell_path)]
        result = runner.invoke(app, args)
        assert result.exit_code == EXIT_INFEASIBLE
        assert "P001" in result.output
        assert (out_dir / "layout.json").is_file()
        assert not (out_dir / "task.json").exists()

    def test_missing_settings_skips_reasoner(self, fake_isaac, fake_planner, tmp_path, monkeypatch):
        """ISAAC_SIM_PATH가 없으면 Reasoner를 부르지 않고 exit 2다."""
        monkeypatch.delenv("ISAAC_SIM_PATH")
        result = runner.invoke(app, ["run-cell", "one square", "-o", str(tmp_path / "out")])
        assert result.exit_code == EXIT_USAGE
        assert "isaac_sim_path" in result.output
        assert fake_planner.calls == []

    def test_missing_usd_skips_reasoner(self, fake_isaac, fake_planner, tmp_path, monkeypatch):
        """SO-101 USD가 없으면 Reasoner를 부르지 않고 exit 2다."""
        monkeypatch.setenv("SO101_USD_PATH", str(tmp_path / "missing.usda"))
        result = runner.invoke(app, ["run-cell", "one square", "-o", str(tmp_path / "out")])
        assert result.exit_code == EXIT_USAGE
        assert "import_so101" in result.output
        assert fake_planner.calls == []
