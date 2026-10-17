import itertools
import math
from pathlib import Path

import numpy as np
import pytest
from typer.testing import CliRunner

from pixel2place.agent.fixture import (
    FIXTURE_X_LIMIT_MM,
    FIXTURE_Y_LIMIT_MM,
    FIXTURE_YAW_LIMIT_DEG,
)
from pixel2place.cell.bench import CHECK_FILE, LAYOUT_FILE, SCENE_FILE, TASK_FILE
from pixel2place.cell.config import CellConfig
from pixel2place.cell.episodes import (
    CONTROLLER_CALIBRATED,
    CONTROLLER_SCRIPTED,
    EVALUATION_FILE,
    REPLAY_PREFIX,
    SLOT_BOARD_ERROR,
    SLOTS,
    TAIL_IDLE_FRAMES,
    TRAIN_CONSTANT_ERROR_SHARE,
    TRAIN_TAU_RANGE_S,
    Episode,
    EpisodeError,
    PolicyEvaluation,
    ReplayPolicy,
    evaluate_policy,
    load_episode,
    perturb_episode,
    record_episodes,
    sample_training_joint_error,
    slot_instruction,
    slot_layouts,
    slot_name,
    training_steps,
)
from pixel2place.cell.policy_link import DOF_NAMES, Observation, PolicyServer
from pixel2place.cell.result import SimResult
from pixel2place.cell.scene import (
    ARM_JOINT_COUNT,
    JOINT_ERROR_ROWS,
    BoardOffset,
    JointErrorLevel,
    SimScene,
    displace_board,
    placed_cell,
    sample_joint_error,
    to_placed_board,
)
from pixel2place.cell.sim import EPISODE_FILE, RECORD_OUTPUTS, RESULT_FILE
from pixel2place.cell.task import Task, compile_task
from pixel2place.cli import EXIT_INFEASIBLE, EXIT_USAGE, app
from pixel2place.config import SimSettings
from pixel2place.feasibility.checker import check_layout
from pixel2place.layout.schema import Board, PartType
from pixel2place.pipeline import write_json
from tests.factories import make_layout, make_part
from tests.fakes.fake_episode import (
    FAKE_DOF_NAMES,
    FAKE_EPISODE_FRAMES,
    FAKE_FPS,
    FAKE_IDLE_FRAMES,
    fake_action,
)

runner = CliRunner()
FIRST_RUN = "000_slot_r2c2/r0"
# 빠르게 바뀌는 오차. 행이 1칸 어긋나면 값이 달라진다.
JOINT_ERROR = JointErrorLevel(sigma=math.radians(1.0), tau=0.1)
# float32 관절 값의 반올림 오차(rad)
FLOAT32_ATOL = 1e-6


def _record(out_dir: Path, **options):
    options.setdefault("per_slot", 1)
    options.setdefault("seed", 0)
    options.setdefault("board_error", SLOT_BOARD_ERROR)
    return record_episodes(
        out_dir=out_dir,
        cell=CellConfig(),
        sim_settings=SimSettings(),  # pyright: ignore[reportCallIssue]
        **options,
    )


def _evaluate(controller: str, out_dir: Path, **options) -> PolicyEvaluation:
    options.setdefault("per_slot", 2)
    options.setdefault("seed", 1)
    options.setdefault("board_error", SLOT_BOARD_ERROR)
    return evaluate_policy(
        controller,
        out_dir=out_dir,
        cell=CellConfig(),
        sim_settings=SimSettings(),  # pyright: ignore[reportCallIssue]
        **options,
    )


class _ZeroPolicy:
    """관절 목표 0을 주는 제어기. 받은 문맥을 기억한다."""

    def __init__(self) -> None:
        self.contexts: list[dict] = []

    def reset(self, context):
        self.contexts.append(dict(context))
        return None

    def act(self, observation):
        return np.zeros((4, len(DOF_NAMES)), dtype=np.float32)


def _episode(action: np.ndarray) -> Episode:
    """관절 목표만 정한 에피소드. 다른 값은 `training_steps`가 쓰지 않는다."""
    return Episode(
        slot=0,
        instruction=slot_instruction(0),
        passed=True,
        action=action.astype(np.float32),
        state=action.astype(np.float32),
        phase=("move",) * len(action),
        dof_names=FAKE_DOF_NAMES,
        fps=FAKE_FPS,
        front_video=Path("cell.mp4"),
        wrist_video=Path("wrist.mp4"),
    )


@pytest.mark.unit
class TestSlots:
    """슬롯 과제의 레이아웃과 지시문 테스트."""

    def test_one_square_for_each_slot(self):
        """슬롯마다 정사각 부품 1개의 레이아웃이 있다. 배치 규칙을 통과한다."""
        layouts = slot_layouts()
        assert [bench_layout.name for bench_layout in layouts] == [
            "slot_r2c2",
            "slot_r2c5",
            "slot_r5c2",
            "slot_r5c5",
        ]
        for bench_layout, slot in zip(layouts, SLOTS, strict=True):
            (part,) = bench_layout.layout.parts
            assert part.type == PartType.SQUARE_TILE
            assert part.cells == [slot]
            assert check_layout(bench_layout.layout).feasible

    def test_slots_compile_at_board_error_limits(self):
        """보드 오차의 모든 극단 조합에서 네 슬롯이 컴파일된다."""
        cell = CellConfig()
        limit = SLOT_BOARD_ERROR
        for x, y, yaw in itertools.product(
            (-limit.x, 0.0, limit.x), (-limit.y, 0.0, limit.y), (-limit.yaw, 0.0, limit.yaw)
        ):
            placement = displace_board(cell, Board(), offset=BoardOffset(x=x, y=y, yaw=yaw))
            actual = placed_cell(cell, placement=placement)
            for bench_layout in slot_layouts():
                assert len(compile_task(bench_layout.layout, actual).parts) == 1

    def test_instruction_names_row_and_col(self):
        """지시문은 슬롯의 row와 col을 쓴다. 슬롯마다 다르다."""
        assert slot_instruction(1) == "Place the blue square tile at row 2 col 5."
        assert len({slot_instruction(slot) for slot in range(len(SLOTS))}) == len(SLOTS)

    def test_board_error_is_twice_the_jig_range(self):
        """보드 오차의 기본 한계는 x 6 mm, y 10 mm, yaw 3°다. 웹 시연의 지그 오차 범위의 2배다."""
        assert SLOT_BOARD_ERROR.x == pytest.approx(2 * FIXTURE_X_LIMIT_MM / 1000)
        assert SLOT_BOARD_ERROR.y == pytest.approx(2 * FIXTURE_Y_LIMIT_MM / 1000)
        assert math.degrees(SLOT_BOARD_ERROR.yaw) == pytest.approx(2 * FIXTURE_YAW_LIMIT_DEG)


@pytest.mark.unit
class TestRecordEpisodes:
    """시연 기록 테스트. 가짜 Isaac 실행기를 쓴다."""

    def test_writes_one_episode_for_each_slot(self, fake_isaac, tmp_path):
        """슬롯마다 실행 디렉터리를 만들고 시연 기록 파일을 쓴다. 모든 시연이 통과한다."""
        out_dir = tmp_path / "episodes"
        summary = _record(out_dir, per_slot=2)
        assert len(summary.runs) == 2 * len(SLOTS)
        assert summary.passed_runs == len(summary.runs)
        for run in summary.runs:
            sim_dir = out_dir / run.layout / f"r{run.repeat}" / "sim"
            for name in RECORD_OUTPUTS:
                assert (sim_dir / name).is_file(), name

    def test_plans_on_actual_board(self, fake_isaac, tmp_path):
        """시연의 task는 scene의 실제 보드 자세로 컴파일한다. 보드는 실행마다 다르다."""
        out_dir = tmp_path / "episodes"
        summary = _record(out_dir)
        origins = set()
        for run in summary.runs:
            run_dir = out_dir / run.layout / f"r{run.repeat}"
            task = Task.model_validate_json((run_dir / TASK_FILE).read_text())
            scene = SimScene.model_validate_json((run_dir / SCENE_FILE).read_text())
            assert scene.board is not None
            assert task.cell.board_origin == scene.board.board_origin
            assert run.board_shift_mm is not None
            assert run.board_shift_mm > 0
            origins.add(scene.board.board_origin)
        assert len(origins) == len(SLOTS)

    def test_seed_changes_board_errors(self, fake_isaac, tmp_path):
        """seed가 다르면 보드 오차가 다르다. 같은 seed는 같다."""
        first = _record(tmp_path / "a", seed=0)
        again = _record(tmp_path / "b", seed=0)
        other = _record(tmp_path / "c", seed=1)
        shifts = [run.board_shift_mm for run in first.runs]
        assert shifts == [run.board_shift_mm for run in again.runs]
        assert shifts != [run.board_shift_mm for run in other.runs]


@pytest.mark.unit
class TestLoadEpisode:
    """에피소드 읽기 테스트."""

    def test_reads_slot_arrays_and_videos(self, fake_isaac, tmp_path):
        """실행 디렉터리에서 슬롯, 지시문, 관절 값과 영상 경로를 읽는다."""
        out_dir = tmp_path / "episodes"
        _record(out_dir)
        episode = load_episode(out_dir / "003_slot_r5c5/r0")
        assert episode.slot == 3
        assert episode.instruction == slot_instruction(3)
        assert episode.passed
        assert episode.action.shape == (FAKE_EPISODE_FRAMES, len(FAKE_DOF_NAMES))
        assert episode.action.dtype == np.float32
        assert episode.state.shape == episode.action.shape
        assert len(episode.phase) == FAKE_EPISODE_FRAMES
        assert episode.dof_names == FAKE_DOF_NAMES
        assert episode.fps == FAKE_FPS
        assert episode.front_video.name == "cell.mp4"
        assert episode.wrist_video.name == "wrist.mp4"

    def test_failed_check_is_not_passed(self, fake_isaac, tmp_path, monkeypatch):
        """판정이 실패한 시연은 passed가 거짓이다."""
        monkeypatch.setenv("FAKE_ISAAC_MODE", "miss")
        out_dir = tmp_path / "episodes"
        _record(out_dir)
        assert not load_episode(out_dir / FIRST_RUN).passed

    def test_missing_file(self, fake_isaac, tmp_path):
        """기록 파일이 없으면 EpisodeError다. 메시지에 없는 파일이 있다."""
        out_dir = tmp_path / "episodes"
        _record(out_dir)
        (out_dir / FIRST_RUN / "sim" / EPISODE_FILE).unlink()
        with pytest.raises(EpisodeError, match=EPISODE_FILE):
            load_episode(out_dir / FIRST_RUN)

    def test_layout_is_not_slot_task(self, fake_isaac, tmp_path):
        """레이아웃이 슬롯 과제가 아니면 EpisodeError다."""
        out_dir = tmp_path / "episodes"
        _record(out_dir)
        other = make_layout(make_part("P001", PartType.SQUARE_TILE, origin=(0, 0)))
        write_json(out_dir / "000_slot_r2c2" / LAYOUT_FILE, other)
        with pytest.raises(EpisodeError, match="not a slot task"):
            load_episode(out_dir / FIRST_RUN)

    def test_arrays_do_not_agree(self, fake_isaac, tmp_path):
        """관절 목표와 측정값의 프레임 수가 다르면 EpisodeError다."""
        out_dir = tmp_path / "episodes"
        _record(out_dir)
        path = out_dir / FIRST_RUN / "sim" / EPISODE_FILE
        with np.load(path) as arrays:
            changed = {name: arrays[name] for name in arrays.files}
        changed["state"] = changed["state"][:-1]
        np.savez(path, **changed)
        with pytest.raises(EpisodeError, match="do not agree"):
            load_episode(out_dir / FIRST_RUN)

    def test_check_file_marks_finished_episode(self, fake_isaac, tmp_path):
        """판정 파일이 없는 실행은 읽지 않는다."""
        out_dir = tmp_path / "episodes"
        _record(out_dir)
        (out_dir / FIRST_RUN / CHECK_FILE).unlink()
        with pytest.raises(EpisodeError, match=CHECK_FILE):
            load_episode(out_dir / FIRST_RUN)


@pytest.mark.unit
class TestTrainingSteps:
    """학습 프레임 선택 테스트."""

    def test_trims_idle_frames_at_both_ends(self):
        """처음의 정지 프레임은 1개, 끝의 정지 프레임은 tail_idle개만 남긴다."""
        ramp = np.concatenate([np.zeros(5), np.linspace(0.1, 1.0, 10), np.ones(20)])
        episode = _episode(np.outer(ramp, np.ones(6)))
        steps = training_steps(episode, tail_idle=3)
        # 프레임 4의 다음 행동이 처음으로 변한다. 프레임 13의 다음 행동이 마지막 변화다.
        # 그 뒤의 정지 프레임 3개(14, 15, 16)를 남긴다.
        assert steps == list(range(4, 17))

    def test_last_frame_has_no_next_action(self):
        """마지막 프레임은 다음 행동이 없다. 그래서 뺀다."""
        ramp = np.linspace(0.0, 1.0, 8)
        steps = training_steps(_episode(np.outer(ramp, np.ones(6))))
        assert steps == list(range(7))

    def test_keeps_idle_frames_inside(self):
        """움직임 사이의 정지 프레임은 남긴다."""
        ramp = np.array([0.0, 0.1, 0.1, 0.1, 0.2, 0.2])
        steps = training_steps(_episode(np.outer(ramp, np.ones(6))), tail_idle=0)
        assert steps == [0, 1, 2, 3]

    def test_no_motion_has_no_steps(self):
        """움직임이 없는 에피소드는 학습 프레임이 없다."""
        assert training_steps(_episode(np.zeros((10, 6)))) == []

    def test_fake_runner_episode(self, fake_isaac, tmp_path):
        """가짜 실행기의 에피소드는 처음의 정지 프레임을 1개만 남긴다."""
        out_dir = tmp_path / "episodes"
        _record(out_dir)
        steps = training_steps(load_episode(out_dir / FIRST_RUN))
        assert steps[0] == FAKE_IDLE_FRAMES
        assert len(steps) <= FAKE_EPISODE_FRAMES - FAKE_IDLE_FRAMES
        assert TAIL_IDLE_FRAMES > FAKE_IDLE_FRAMES


@pytest.mark.unit
class TestPerturbEpisode:
    """관절 센서 오차를 아는 제어기의 시연 테스트."""

    def test_subtracts_error_from_arm_joints(self):
        """프레임 i의 팔 관절 목표와 측정값에서 오차의 i행을 뺀다. 그리퍼와 영상은 그대로다."""
        clean = _episode(fake_action())
        rows = sample_joint_error(
            np.random.default_rng(0), level=JOINT_ERROR, rows=FAKE_EPISODE_FRAMES
        )
        noisy = perturb_episode(clean, rows)
        expected = fake_action()[:, :ARM_JOINT_COUNT] - np.array(rows)
        np.testing.assert_allclose(noisy.action[:, :ARM_JOINT_COUNT], expected, atol=FLOAT32_ATOL)
        np.testing.assert_allclose(noisy.state[:, :ARM_JOINT_COUNT], expected, atol=FLOAT32_ATOL)
        np.testing.assert_array_equal(noisy.action[:, ARM_JOINT_COUNT:], fake_action()[:, -1:])
        assert noisy.action.dtype == np.float32
        assert (noisy.phase, noisy.front_video, noisy.slot) == (
            clean.phase,
            clean.front_video,
            clean.slot,
        )
        np.testing.assert_array_equal(clean.action, fake_action())

    def test_one_row_shifts_all_frames(self):
        """1행의 오차는 모든 프레임에서 같은 값을 뺀다. 프레임 사이의 변화는 그대로다."""
        clean = _episode(fake_action())
        row = (0.01, -0.02, 0.0, 0.03, 0.005)
        noisy = perturb_episode(clean, (row,))
        np.testing.assert_allclose(
            noisy.action[:, :ARM_JOINT_COUNT],
            fake_action()[:, :ARM_JOINT_COUNT] - np.array(row),
            atol=FLOAT32_ATOL,
        )
        assert training_steps(noisy) == training_steps(clean)

    def test_error_shorter_than_episode(self):
        """여러 행의 오차가 에피소드보다 짧으면 ValueError다."""
        rows = sample_joint_error(np.random.default_rng(0), level=JOINT_ERROR, rows=3)
        with pytest.raises(ValueError, match="3 rows"):
            perturb_episode(_episode(fake_action()), rows)


@pytest.mark.unit
class TestSampleTrainingJointError:
    """학습 사본의 관절 센서 오차 테스트."""

    SIGMA_MAX = math.radians(1.5)
    DRAWS = 2000

    def test_levels_cover_sigma_and_tau_ranges(self):
        """크기는 0과 한계 사이다. 고정 오차의 비율과 지속 시간의 범위는 상수와 같다."""
        rng = np.random.default_rng(0)
        levels = [
            sample_training_joint_error(rng, sigma_max=self.SIGMA_MAX)[0] for _ in range(self.DRAWS)
        ]
        sigmas = np.array([level.sigma for level in levels])
        taus = np.array([level.tau for level in levels if level.tau is not None])
        assert 0.0 <= sigmas.min() < 0.1 * self.SIGMA_MAX
        assert 0.9 * self.SIGMA_MAX < sigmas.max() <= self.SIGMA_MAX
        constant_share = 1.0 - len(taus) / self.DRAWS
        assert constant_share == pytest.approx(TRAIN_CONSTANT_ERROR_SHARE, abs=0.04)
        low, high = TRAIN_TAU_RANGE_S
        assert low <= taus.min() < 2 * low
        assert high / 2 < taus.max() <= high
        # log-uniform이면 중앙값은 두 끝의 기하 평균이다.
        assert np.median(taus) == pytest.approx(math.sqrt(low * high), rel=0.2)

    def test_values_follow_the_level(self):
        """고정 오차는 1행이다. 바뀌는 오차는 실행 1회의 행 수다. 같은 seed는 같은 값이다."""
        for seed in range(20):
            level, values = sample_training_joint_error(
                np.random.default_rng([0, seed, 0]), sigma_max=self.SIGMA_MAX
            )
            assert len(values) == (1 if level.tau is None else JOINT_ERROR_ROWS)
            again = sample_training_joint_error(
                np.random.default_rng([0, seed, 0]), sigma_max=self.SIGMA_MAX
            )
            assert again == (level, values)

    def test_copy_changes_the_error(self):
        """사본 번호가 다르면 오차가 다르다."""
        first = sample_training_joint_error(
            np.random.default_rng([0, 3, 0]), sigma_max=self.SIGMA_MAX
        )
        second = sample_training_joint_error(
            np.random.default_rng([0, 3, 1]), sigma_max=self.SIGMA_MAX
        )
        assert first != second

    def test_constant_share_one_gives_only_constant_errors(self):
        """비율이 1이면 모든 사본의 오차가 고정이다."""
        rng = np.random.default_rng(0)
        for _ in range(200):
            level, values = sample_training_joint_error(
                rng, sigma_max=self.SIGMA_MAX, constant_share=1.0
            )
            assert level.tau is None
            assert len(values) == 1

    def test_constant_share_zero_gives_only_changing_errors(self):
        """비율이 0이면 모든 사본의 오차가 실행 동안 바뀐다."""
        rng = np.random.default_rng(0)
        for _ in range(50):
            level, values = sample_training_joint_error(
                rng, sigma_max=self.SIGMA_MAX, constant_share=0.0
            )
            assert level.tau is not None
            assert len(values) == JOINT_ERROR_ROWS

    def test_default_constant_share_is_the_constant(self):
        """비율을 주지 않은 호출은 상수의 비율을 준 호출과 같다."""
        for seed in range(20):
            default = sample_training_joint_error(
                np.random.default_rng([0, seed, 0]), sigma_max=self.SIGMA_MAX
            )
            explicit = sample_training_joint_error(
                np.random.default_rng([0, seed, 0]),
                sigma_max=self.SIGMA_MAX,
                constant_share=TRAIN_CONSTANT_ERROR_SHARE,
            )
            assert explicit == default

    def test_constant_share_keeps_the_error_size(self):
        """비율은 오차의 크기를 바꾸지 않는다. 같은 seed의 크기는 같다."""
        for seed in range(20):
            default, _ = sample_training_joint_error(
                np.random.default_rng([0, seed, 0]), sigma_max=self.SIGMA_MAX
            )
            constant, _ = sample_training_joint_error(
                np.random.default_rng([0, seed, 0]),
                sigma_max=self.SIGMA_MAX,
                constant_share=1.0,
            )
            assert constant.sigma == default.sigma


@pytest.mark.unit
class TestRecordEpisodesCommand:
    """record-episodes 명령 테스트."""

    def test_success(self, fake_isaac, tmp_path):
        """통과한 에피소드 수와 출력 디렉터리를 출력한다."""
        out_dir = tmp_path / "episodes"
        result = runner.invoke(app, ["record-episodes", "-o", str(out_dir), "--per-slot", "2"])
        assert result.exit_code == 0, result.output
        assert "episodes_passed=8/8 slots=4 errors=0" in result.output
        assert (out_dir / FIRST_RUN / "sim" / EPISODE_FILE).is_file()

    def test_shard_runs_selected_slots(self, fake_isaac, tmp_path):
        """shard는 슬롯을 나눈다."""
        out_dir = tmp_path / "episodes"
        result = runner.invoke(
            app, ["record-episodes", "-o", str(out_dir), "--per-slot", "1", "--shard", "1/2"]
        )
        assert result.exit_code == 0, result.output
        assert "episodes_passed=2/2" in result.output
        assert not (out_dir / "000_slot_r2c2").exists()

    def test_board_error_option(self, fake_isaac, tmp_path):
        """보드 오차 한계를 0으로 주면 보드가 움직이지 않는다."""
        out_dir = tmp_path / "episodes"
        result = runner.invoke(
            app,
            ["record-episodes", "-o", str(out_dir), "--per-slot", "1", "--board-error", "0,0,0"],
        )
        assert result.exit_code == 0, result.output
        task = Task.model_validate_json((out_dir / FIRST_RUN / TASK_FILE).read_text())
        assert task.cell.board_origin == pytest.approx(CellConfig().board_origin)

    def test_run_error_exit_one(self, fake_isaac, tmp_path, monkeypatch):
        """끝나지 못한 에피소드가 있으면 exit 1이다."""
        monkeypatch.setenv("FAKE_ISAAC_MODE", "crash")
        result = runner.invoke(
            app, ["record-episodes", "-o", str(tmp_path / "episodes"), "--per-slot", "1"]
        )
        assert result.exit_code == EXIT_INFEASIBLE
        assert "runner_error" in result.output

    def test_missing_python_sh_exit_two(self, fake_isaac, tmp_path):
        """python.sh가 없으면 exit 2다."""
        (fake_isaac / "python.sh").unlink()
        result = runner.invoke(
            app, ["record-episodes", "-o", str(tmp_path / "episodes"), "--per-slot", "1"]
        )
        assert result.exit_code == EXIT_USAGE

    def test_missing_settings_exit_two(self, fake_isaac, tmp_path, monkeypatch):
        """ISAAC_SIM_PATH가 없으면 exit 2다."""
        monkeypatch.delenv("ISAAC_SIM_PATH")
        result = runner.invoke(
            app, ["record-episodes", "-o", str(tmp_path / "episodes"), "--per-slot", "1"]
        )
        assert result.exit_code == EXIT_USAGE
        assert "Isaac settings are not valid" in result.output


@pytest.mark.unit
class TestReplayPolicy:
    """재생 제어기 테스트."""

    @pytest.fixture
    def recorded(self, fake_isaac, tmp_path) -> Path:
        out_dir = tmp_path / "episodes"
        _record(out_dir)
        return out_dir

    def _observation(self, step: int) -> Observation:
        image = np.zeros((2, 2, 3), dtype=np.uint8)
        state = np.zeros(len(DOF_NAMES), dtype=np.float32)
        return Observation(step=step, state=state, front=image, wrist=image)

    def test_reset_finds_episode_and_gives_frame_count(self, recorded):
        """문맥의 레이아웃과 반복으로 에피소드를 찾는다. 프레임 수를 한도로 준다."""
        policy = ReplayPolicy(recorded)
        assert policy.reset({"layout": "001_slot_r2c5", "repeat": 0}) == FAKE_EPISODE_FRAMES

    def test_act_gives_next_recorded_actions(self, recorded):
        """프레임 i의 관측에는 프레임 i+1부터의 관절 목표를 준다."""
        policy = ReplayPolicy(recorded, chunk=3)
        policy.reset({"layout": "000_slot_r2c2", "repeat": 0})
        np.testing.assert_array_equal(policy.act(self._observation(9)), fake_action()[10:13])

    def test_act_after_the_end_holds_last_action(self, recorded):
        """기록이 끝난 뒤에는 마지막 관절 목표 1개를 준다."""
        policy = ReplayPolicy(recorded)
        policy.reset({"layout": "000_slot_r2c2", "repeat": 0})
        actions = policy.act(self._observation(FAKE_EPISODE_FRAMES + 5))
        np.testing.assert_array_equal(actions, fake_action()[-1:])

    def test_subtracts_joint_error_of_scene(self, recorded, tmp_path):
        """평가의 scene에 관절 센서 오차가 있으면 목표에서 같은 프레임의 오차를 뺀다."""
        rows = sample_joint_error(np.random.default_rng(0), level=JOINT_ERROR)
        scene_root = tmp_path / "eval"
        (scene_root / FIRST_RUN).mkdir(parents=True)
        write_json(scene_root / FIRST_RUN / SCENE_FILE, SimScene(joint_error=rows))
        policy = ReplayPolicy(recorded, chunk=3, scene_root=scene_root)
        policy.reset({"layout": "000_slot_r2c2", "repeat": 0})
        expected = fake_action()[10:13].astype(np.float64)
        expected[:, :ARM_JOINT_COUNT] -= np.array(rows[10:13])
        np.testing.assert_allclose(policy.act(self._observation(9)), expected, atol=FLOAT32_ATOL)

    def test_scene_without_joint_error_keeps_record(self, recorded, tmp_path):
        """scene이 없거나 scene에 관절 센서 오차가 없으면 기록 그대로 준다."""
        scene_root = tmp_path / "eval"
        (scene_root / FIRST_RUN).mkdir(parents=True)
        for scene in (None, SimScene()):
            if scene is not None:
                write_json(scene_root / FIRST_RUN / SCENE_FILE, scene)
            policy = ReplayPolicy(recorded, chunk=3, scene_root=scene_root)
            policy.reset({"layout": "000_slot_r2c2", "repeat": 0})
            np.testing.assert_array_equal(policy.act(self._observation(9)), fake_action()[10:13])

    def test_reset_without_episode(self, recorded):
        """기록에 없는 실행은 EpisodeError다."""
        with pytest.raises(EpisodeError, match="missing"):
            ReplayPolicy(recorded).reset({"layout": "000_slot_r2c2", "repeat": 9})


@pytest.mark.unit
class TestEvaluatePolicy:
    """제어기 평가 테스트. 가짜 Isaac 실행기를 쓴다."""

    def test_scripted_misses_by_slot_shift(self, fake_isaac, tmp_path):
        """보정 없는 스크립트는 믿는 보드에 놓는다. 슬롯 이동이 위치 한계보다 크면 실패다."""
        out_dir = tmp_path / "eval"
        evaluation = _evaluate(CONTROLLER_SCRIPTED, out_dir)
        assert evaluation.controller == CONTROLLER_SCRIPTED
        assert evaluation.total.episodes == 2 * len(SLOTS)
        assert evaluation.errors == 0
        for episode in evaluation.episodes:
            assert episode.passed == (episode.slot_shift_mm <= 4.0)
        assert 0 < evaluation.total.passed < evaluation.total.episodes
        assert (out_dir / EVALUATION_FILE).is_file()

    def test_calibrated_passes_all(self, fake_isaac, tmp_path):
        """보정한 스크립트는 실제 보드에 놓는다. 모든 에피소드가 통과한다."""
        evaluation = _evaluate(CONTROLLER_CALIBRATED, tmp_path / "eval")
        assert evaluation.total.passed == evaluation.total.episodes
        assert evaluation.wrong_slot == 0
        assert all(episode.placed_slot == episode.slot for episode in evaluation.episodes)

    def test_same_seed_has_same_scenes(self, fake_isaac, tmp_path):
        """제어기가 달라도 같은 seed의 슬롯 이동량은 같다. 기록의 seed와 같으면 기록과도 같다."""
        scripted = _evaluate(CONTROLLER_SCRIPTED, tmp_path / "a", per_slot=1)
        calibrated = _evaluate(CONTROLLER_CALIBRATED, tmp_path / "b", per_slot=1)
        other = _evaluate(CONTROLLER_SCRIPTED, tmp_path / "c", per_slot=1, seed=2)
        shifts = [episode.slot_shift_mm for episode in scripted.episodes]
        assert shifts == [episode.slot_shift_mm for episode in calibrated.episodes]
        assert shifts != [episode.slot_shift_mm for episode in other.episodes]

    def test_groups_by_slot_and_shift_band(self, fake_isaac, tmp_path):
        """슬롯별 수와 슬롯 이동량 구간별 수의 합은 전체 수다."""
        evaluation = _evaluate(CONTROLLER_SCRIPTED, tmp_path / "eval", per_slot=3)
        assert list(evaluation.by_slot) == [slot_name(slot) for slot in range(len(SLOTS))]
        assert all(count.episodes == 3 for count in evaluation.by_slot.values())
        assert set(evaluation.by_shift) <= {"0-4", "4-8", "8+"}
        assert sum(count.episodes for count in evaluation.by_shift.values()) == 12
        assert (
            sum(count.passed for count in evaluation.by_shift.values()) == evaluation.total.passed
        )

    def test_replay_follows_recorded_actions(self, fake_isaac, tmp_path):
        """재생은 기록 디렉터리의 관절 목표를 실행기에 준다."""
        recorded = tmp_path / "episodes"
        _record(recorded, seed=1)
        out_dir = tmp_path / "eval"
        evaluation = _evaluate(f"{REPLAY_PREFIX}{recorded}", out_dir, per_slot=1)
        assert evaluation.errors == 0
        with np.load(out_dir / FIRST_RUN / "sim" / EPISODE_FILE) as arrays:
            np.testing.assert_array_equal(arrays["action"], fake_action())

    def test_joint_error_goes_to_scene_and_summary(self, fake_isaac, tmp_path):
        """관절 센서 오차를 주면 scene에 값을 쓰고 요약에 크기와 지속 시간을 적는다."""
        out_dir = tmp_path / "eval"
        evaluation = _evaluate(CONTROLLER_SCRIPTED, out_dir, per_slot=1, joint_error=JOINT_ERROR)
        assert evaluation.joint_error == JOINT_ERROR
        saved = PolicyEvaluation.model_validate_json((out_dir / EVALUATION_FILE).read_text())
        assert saved.joint_error == JOINT_ERROR
        scene = SimScene.model_validate_json((out_dir / FIRST_RUN / SCENE_FILE).read_text())
        assert scene.board is not None
        assert scene.joint_error is not None
        assert len(scene.joint_error) == JOINT_ERROR_ROWS

    def test_no_joint_error_by_default(self, fake_isaac, tmp_path):
        """관절 센서 오차를 주지 않으면 scene과 요약에 오차가 없다."""
        out_dir = tmp_path / "eval"
        evaluation = _evaluate(CONTROLLER_SCRIPTED, out_dir, per_slot=1)
        scene = SimScene.model_validate_json((out_dir / FIRST_RUN / SCENE_FILE).read_text())
        assert evaluation.joint_error is None
        assert scene.joint_error is None

    def test_replay_subtracts_joint_error_of_run(self, fake_isaac, tmp_path):
        """관절 센서 오차가 있는 재생은 실행마다 그 실행의 오차를 목표에서 뺀다."""
        recorded = tmp_path / "episodes"
        _record(recorded, seed=1)
        out_dir = tmp_path / "eval"
        evaluation = _evaluate(
            f"{REPLAY_PREFIX}{recorded}", out_dir, per_slot=1, joint_error=JOINT_ERROR
        )
        assert evaluation.errors == 0
        scene = SimScene.model_validate_json((out_dir / FIRST_RUN / SCENE_FILE).read_text())
        assert scene.joint_error is not None
        expected = perturb_episode(load_episode(recorded / FIRST_RUN), scene.joint_error)
        with np.load(out_dir / FIRST_RUN / "sim" / EPISODE_FILE) as arrays:
            # 가짜 실행기의 프레임 0은 제어기의 목표가 아니다.
            np.testing.assert_array_equal(arrays["action"][1:], expected.action[1:])
        assert not np.array_equal(expected.action[1:], fake_action()[1:])

    def test_step_limit_must_fit_joint_error_rows(self, fake_isaac, tmp_path):
        """바뀌는 오차에서 제어기의 프레임 한도가 오차의 행 수를 넘으면 실행 전에 ValueError다."""
        recorded = tmp_path / "episodes"
        _record(recorded, seed=1)
        with pytest.raises(ValueError, match="Step limit"):
            _evaluate(
                f"{REPLAY_PREFIX}{recorded}",
                tmp_path / "eval",
                per_slot=1,
                joint_error=JOINT_ERROR,
                max_steps=JOINT_ERROR_ROWS,
            )

    def test_constant_joint_error_has_no_step_limit(self, fake_isaac, tmp_path):
        """실행 동안 같은 오차는 1행이다. 프레임 한도를 막지 않는다."""
        recorded = tmp_path / "episodes"
        _record(recorded, seed=1)
        evaluation = _evaluate(
            f"{REPLAY_PREFIX}{recorded}",
            tmp_path / "eval",
            per_slot=1,
            joint_error=JointErrorLevel(sigma=math.radians(1.0)),
            max_steps=JOINT_ERROR_ROWS,
        )
        assert evaluation.errors == 0

    def test_replay_directory_must_exist(self, fake_isaac, tmp_path):
        """재생 디렉터리가 없으면 실행 전에 ValueError다."""
        with pytest.raises(ValueError, match="does not exist"):
            _evaluate(f"{REPLAY_PREFIX}{tmp_path / 'missing'}", tmp_path / "eval")

    def test_policy_server_gets_slot_and_instruction(self, fake_isaac, tmp_path):
        """제어기 서버 주소를 주면 실행마다 슬롯 번호와 지시문을 보낸다."""
        policy = _ZeroPolicy()
        with PolicyServer(policy) as server:
            evaluation = _evaluate(str(server.address), tmp_path / "eval", per_slot=1, max_steps=5)
        assert evaluation.errors == 0
        assert [context["slot"] for context in policy.contexts] == list(range(len(SLOTS)))
        assert policy.contexts[3]["instruction"] == slot_instruction(3)
        assert policy.contexts[3]["layout"] == "003_slot_r5c5"

    def test_invalid_controller(self, fake_isaac, tmp_path):
        """제어기 이름이 형식과 다르면 ValueError다."""
        with pytest.raises(ValueError, match="HOST:PORT"):
            _evaluate("learned", tmp_path / "eval")

    def test_runner_error_is_not_passed(self, fake_isaac, tmp_path, monkeypatch):
        """끝나지 못한 에피소드는 통과가 아니다. 놓인 슬롯은 None이다."""
        monkeypatch.setenv("FAKE_ISAAC_MODE", "crash")
        evaluation = _evaluate(CONTROLLER_SCRIPTED, tmp_path / "eval", per_slot=1)
        assert evaluation.errors == len(SLOTS)
        assert evaluation.total.passed == 0
        assert all(episode.placed_slot is None for episode in evaluation.episodes)

    def test_counts_part_in_other_slot(self, fake_isaac, tmp_path):
        """부품이 다른 슬롯의 셀 안에 있으면 wrong_slot으로 센다."""
        out_dir = tmp_path / "eval"
        _evaluate(CONTROLLER_CALIBRATED, out_dir, per_slot=1)
        run_dir = out_dir / FIRST_RUN
        scene = SimScene.model_validate_json((run_dir / SCENE_FILE).read_text())
        assert scene.board is not None
        cell = CellConfig()
        nominal = compile_task(slot_layouts()[3].layout, cell).parts[0].target_center
        other = to_placed_board(nominal, cell=cell, placement=scene.board)
        result_path = run_dir / "sim" / RESULT_FILE
        result = SimResult.model_validate_json(result_path.read_text())
        pose = result.parts[0]
        moved = pose.model_copy(
            update={"position": (float(other[0]), float(other[1]), pose.position[2])}
        )
        write_json(result_path, result.model_copy(update={"parts": [moved]}))
        evaluation = _evaluate(CONTROLLER_CALIBRATED, out_dir, per_slot=1)
        first = evaluation.episodes[0]
        assert not first.passed
        assert first.placed_slot == 3
        assert evaluation.wrong_slot == 1


@pytest.mark.unit
class TestEvalPolicyCommand:
    """eval-policy 명령 테스트."""

    def test_prints_bands_and_summary(self, fake_isaac, tmp_path):
        """실패한 에피소드, 슬롯별 수, 슬롯 이동량 구간별 수와 요약을 출력한다."""
        out_dir = tmp_path / "eval"
        result = runner.invoke(
            app, ["eval-policy", "-o", str(out_dir), "--policy", "scripted", "--per-slot", "2"]
        )
        assert result.exit_code == 0, result.output
        assert "slot slot_r2c2 passed=" in result.output
        assert "slot_shift_mm " in result.output
        assert " fail slot_shift_mm=" in result.output
        assert "controller=scripted passed=" in result.output
        assert "/8 wrong_slot=0 errors=0" in result.output
        assert (out_dir / EVALUATION_FILE).is_file()

    def test_default_seed_differs_from_recording(self, fake_isaac, tmp_path):
        """평가의 기본 seed는 기록의 기본 seed와 다르다. 그래서 평가 장면은 학습 장면과 다르다."""
        recorded = tmp_path / "episodes"
        runner.invoke(app, ["record-episodes", "-o", str(recorded), "--per-slot", "1"])
        out_dir = tmp_path / "eval"
        runner.invoke(
            app, ["eval-policy", "-o", str(out_dir), "--policy", "scripted", "--per-slot", "1"]
        )
        assert (recorded / FIRST_RUN / SCENE_FILE).read_text() != (
            out_dir / FIRST_RUN / SCENE_FILE
        ).read_text()

    @pytest.mark.parametrize(
        ("value", "tau", "rows"), [("0.5,1", 1.0, JOINT_ERROR_ROWS), ("0.5,inf", None, 1)]
    )
    def test_joint_error_option(self, fake_isaac, tmp_path, value, tau, rows):
        """--joint-error는 표준편차(deg)와 지속 시간(s)이다. inf는 실행 동안 같은 오차다."""
        out_dir = tmp_path / "eval"
        result = runner.invoke(
            app,
            [
                "eval-policy",
                "-o",
                str(out_dir),
                "--policy",
                "scripted",
                "--per-slot",
                "1",
                "--joint-error",
                value,
            ],
        )
        assert result.exit_code == 0, result.output
        evaluation = PolicyEvaluation.model_validate_json((out_dir / EVALUATION_FILE).read_text())
        assert evaluation.joint_error == JointErrorLevel(sigma=math.radians(0.5), tau=tau)
        scene = SimScene.model_validate_json((out_dir / FIRST_RUN / SCENE_FILE).read_text())
        assert scene.joint_error is not None
        assert len(scene.joint_error) == rows

    @pytest.mark.parametrize("value", ["1", "a,b", "-1,1", "1,0", "1,nan", "inf,1"])
    def test_invalid_joint_error_exit_two(self, fake_isaac, tmp_path, value):
        """--joint-error의 형식이나 범위가 틀리면 실행 전에 exit 2다."""
        out_dir = tmp_path / "eval"
        result = runner.invoke(
            app,
            ["eval-policy", "-o", str(out_dir), "--policy", "scripted", "--joint-error", value],
        )
        assert result.exit_code == EXIT_USAGE, result.output
        assert not out_dir.exists()

    def test_max_steps_option(self, fake_isaac, tmp_path):
        """--max-steps는 제어기가 제어하는 에피소드의 프레임 수 한도다."""
        out_dir = tmp_path / "eval"
        with PolicyServer(_ZeroPolicy()) as server:
            result = runner.invoke(
                app,
                [
                    "eval-policy",
                    "-o",
                    str(out_dir),
                    "--policy",
                    str(server.address),
                    "--per-slot",
                    "1",
                    "--max-steps",
                    "6",
                ],
            )
        assert result.exit_code == 0, result.output
        with np.load(out_dir / FIRST_RUN / "sim" / EPISODE_FILE) as arrays:
            assert len(arrays["action"]) == 6

    def test_run_error_exit_one(self, fake_isaac, tmp_path, monkeypatch):
        """끝나지 못한 에피소드가 있으면 exit 1이다."""
        monkeypatch.setenv("FAKE_ISAAC_MODE", "crash")
        result = runner.invoke(
            app,
            [
                "eval-policy",
                "-o",
                str(tmp_path / "eval"),
                "--policy",
                "scripted",
                "--per-slot",
                "1",
            ],
        )
        assert result.exit_code == EXIT_INFEASIBLE
        assert "error runner_error" in result.output

    @pytest.mark.parametrize("policy", ["learned", "replay:/missing/episodes"])
    def test_invalid_policy_exit_two(self, fake_isaac, tmp_path, policy):
        """제어기 형식이 틀리거나 재생 디렉터리가 없으면 exit 2다."""
        result = runner.invoke(
            app, ["eval-policy", "-o", str(tmp_path / "eval"), "--policy", policy]
        )
        assert result.exit_code == EXIT_USAGE

    def test_missing_settings_exit_two(self, fake_isaac, tmp_path, monkeypatch):
        """ISAAC_SIM_PATH가 없으면 exit 2다."""
        monkeypatch.delenv("ISAAC_SIM_PATH")
        result = runner.invoke(
            app, ["eval-policy", "-o", str(tmp_path / "eval"), "--policy", "scripted"]
        )
        assert result.exit_code == EXIT_USAGE
        assert "Isaac settings are not valid" in result.output
