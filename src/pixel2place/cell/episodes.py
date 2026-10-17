"""슬롯 과제의 시연 에피소드.

과제: 정사각 부품 1개를 공급대에서 집어 보드의 슬롯 1곳에 놓는다. 슬롯은 `SLOTS`의 셀이다.
에피소드 1개는 `bench-cell` 형식의 실행 디렉터리 1개다. 실행기가 `--record`로 기록한다.

- 시연은 실제 보드 자세로 컴파일한 task의 실행이다. 그래서 보드 오차가 있어도 부품이 슬롯에 간다.
- 제어기는 프레임 i의 관측에서 프레임 i+1의 행동을 낸다. `training_steps`가 이 짝을 정한다.
- 평가는 같은 장면에서 제어기를 바꿔서 실행한다.
  스크립트 제어기(보정 없음, 보정)와 학습한 제어기를 비교한다.
- 평가는 관절 센서 오차를 넣을 수 있다. 스크립트 제어기는 이 오차를 모른다.
  `perturb_episode`는 이 오차를 아는 제어기의 시연을 만든다.
"""

import math
from collections.abc import Mapping
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any

import numpy as np
from numpy.typing import NDArray
from pydantic import BaseModel, ConfigDict

from pixel2place.cell.bench import (
    ALL_LAYOUTS,
    CHECK_FILE,
    LAYOUT_FILE,
    RUN_DIR_FORMAT,
    SCENE_FILE,
    SIM_DIR,
    TASK_FILE,
    BenchLayout,
    BenchPolicy,
    CellBenchmark,
    RunErrorKind,
    RunOutcome,
    Shard,
    run_cell_benchmark,
)
from pixel2place.cell.check import check_result
from pixel2place.cell.config import CellConfig
from pixel2place.cell.policy_link import Observation, PolicyAddress, PolicyServer
from pixel2place.cell.result import SimResult
from pixel2place.cell.scene import (
    ARM_JOINT_COUNT,
    JOINT_ERROR_ROWS,
    SIM_FPS,
    BoardOffset,
    JointErrorLevel,
    JointErrorRow,
    SimScene,
    joint_error_frames,
    sample_joint_error,
    to_placed_board,
)
from pixel2place.cell.sim import EPISODE_FILE, FRONT_VIDEO_FILE, RESULT_FILE, WRIST_VIDEO_FILE
from pixel2place.cell.task import MM_PER_M, BoardFrame, Task
from pixel2place.config import SimSettings
from pixel2place.layout.schema import Board, Cell, Color, Layout, Part, PartType
from pixel2place.pipeline import write_json

# 슬롯의 (row, col). 네 곳 모두 보드 오차의 한계에서 팔이 닿는다. 슬롯 사이는 60 mm다.
SLOTS: tuple[Cell, ...] = ((2, 2), (2, 5), (5, 2), (5, 5))
SLOT_PART_TYPE = PartType.SQUARE_TILE
SLOT_COLOR = Color.BLUE
SLOT_PART_ID = "P001"
SLOT_NAME_FORMAT = "slot_r{row}c{col}"
# 보드 놓기 오차의 한계. x 6 mm, y 10 mm, yaw 3°다. 웹 시연의 지그 오차 범위의 2배다.
# 1배 범위에서는 위치결정 핀이 보드 오차의 대부분을 받아들인다.
# 그래서 보드를 보지 않는 제어기도 대부분 성공한다.
SLOT_BOARD_ERROR = BoardOffset(x=0.006, y=0.010, yaw=math.radians(3.0))
# 언어 조건 제어기가 받는 지시문. 작업 지시의 row, col 문구와 같은 형식이다.
SLOT_INSTRUCTION_FORMAT = "Place the blue square tile at row {row} col {col}."
# 제어기가 멈추는 법을 배우도록 끝의 정지 프레임을 이 수만큼 남긴다.
TAIL_IDLE_FRAMES = 10
# 관절 목표의 변화가 이 값(rad)보다 작으면 정지 프레임이다.
IDLE_TOLERANCE_RAD = 1e-6
# 평가의 제어기 이름. 제어기 서버는 `HOST:PORT`, 재생은 `replay:DIR`로 준다.
CONTROLLER_SCRIPTED = "scripted"
CONTROLLER_CALIBRATED = "scripted-calibrated"
REPLAY_PREFIX = "replay:"
# 폐루프 실행의 프레임 수 한도. 시연 1개는 약 207 프레임이다.
EVAL_MAX_STEPS = 300
# 실행기는 제어기의 프레임 뒤에 이 시간(s)을 더 진행하고 판정한다.
FINAL_SETTLE_S = 1.0
# 재생 제어기가 한 번에 주는 프레임 수
REPLAY_CHUNK_FRAMES = 30
# 학습 사본의 관절 센서 오차. 이 비율의 사본은 실행 동안 오차가 같다.
TRAIN_CONSTANT_ERROR_SHARE = 0.3
# 나머지 사본의 지속 시간 범위(s). log-uniform으로 뽑는다.
TRAIN_TAU_RANGE_S = (0.1, 10.0)
EVALUATION_FILE = "evaluation.json"
# 슬롯 이동량 구간의 (이름, 최솟값 mm). 최솟값이 큰 구간부터 찾는다.
SHIFT_BANDS_MM: tuple[tuple[str, float], ...] = (("8+", 8.0), ("4-8", 4.0), ("0-4", 0.0))


class EpisodeError(ValueError):
    """에피소드 디렉터리를 읽을 수 없다."""


@dataclass(frozen=True)
class Episode:
    """실행 디렉터리 1개에서 읽은 시연."""

    slot: int
    """`SLOTS`의 번호."""
    instruction: str
    passed: bool
    """판정이 통과했다. 통과하지 않은 시연은 학습에 쓰지 않는다."""
    action: NDArray[np.float32]
    """프레임마다 준 관절 목표. 모양은 (프레임, 6)이다. 팔 5개와 그리퍼 순서다. 단위는 rad다."""
    state: NDArray[np.float32]
    """프레임 뒤의 측정 관절 값. 모양과 순서는 `action`과 같다."""
    phase: tuple[str, ...]
    dof_names: tuple[str, ...]
    fps: int
    front_video: Path
    """외부 사선 카메라 영상. 프레임 i는 `state[i]`와 같은 시점이다."""
    wrist_video: Path


def slot_layouts() -> list[BenchLayout]:
    """슬롯마다 정사각 부품 1개를 놓는 레이아웃."""
    return [
        BenchLayout(
            slot_name(slot),
            Layout(
                palette=list(Color),
                parts=[
                    Part(
                        part_id=SLOT_PART_ID,
                        type=SLOT_PART_TYPE,
                        color=SLOT_COLOR,
                        cells=[cell],
                    )
                ],
            ),
        )
        for slot, cell in enumerate(SLOTS)
    ]


def slot_name(slot: int) -> str:
    row, col = SLOTS[slot]
    return SLOT_NAME_FORMAT.format(row=row, col=col)


def slot_instruction(slot: int) -> str:
    row, col = SLOTS[slot]
    return SLOT_INSTRUCTION_FORMAT.format(row=row, col=col)


def record_episodes(
    *,
    out_dir: Path,
    cell: CellConfig,
    sim_settings: SimSettings,
    per_slot: int,
    seed: int,
    board_error: BoardOffset,
    shard: Shard = ALL_LAYOUTS,
    jobs: int = 1,
) -> CellBenchmark:
    """슬롯마다 `per_slot`개의 시연을 기록한다. 요약은 `bench-cell`과 같다.

    실행마다 seed로 보드 오차를 뽑고 실제 보드 자세로 컴파일한다.
    반복 1 이상은 공급대의 부품 위치도 옮긴다.
    """
    return run_cell_benchmark(
        slot_layouts(),
        out_dir=out_dir,
        cell=cell,
        sim_settings=sim_settings,
        repeats=per_slot,
        seed=seed,
        shard=shard,
        jobs=jobs,
        board_error=board_error,
        plan_on_actual_board=True,
        record=True,
    )


def load_episode(run_dir: Path) -> Episode:
    """`record_episodes`가 쓴 실행 디렉터리 1개를 읽는다.

    판정 파일은 완료 표시로만 쓴다. 판정은 task와 결과로 다시 계산한다.

    Raises:
        EpisodeError: 파일이 없거나 레이아웃이 슬롯 과제가 아닐 때.
    """
    sim_dir = run_dir / SIM_DIR
    layout_path = run_dir.parent / LAYOUT_FILE
    task_path = run_dir / TASK_FILE
    result_path = sim_dir / RESULT_FILE
    paths = [layout_path, task_path, run_dir / CHECK_FILE, result_path]
    paths += [sim_dir / name for name in (EPISODE_FILE, FRONT_VIDEO_FILE, WRIST_VIDEO_FILE)]
    missing = [str(path) for path in paths if not path.is_file()]
    if missing:
        raise EpisodeError(f"Episode files are missing: {', '.join(missing)}.")
    layout = Layout.model_validate_json(layout_path.read_text(encoding="utf-8"))
    check = check_result(
        Task.model_validate_json(task_path.read_text(encoding="utf-8")),
        SimResult.model_validate_json(result_path.read_text(encoding="utf-8")),
    )
    cells = [tuple(part.cells) for part in layout.parts]
    if len(cells) != 1 or len(cells[0]) != 1 or cells[0][0] not in SLOTS:
        raise EpisodeError(f"Layout of {run_dir} is not a slot task.")
    slot = SLOTS.index(cells[0][0])
    with np.load(sim_dir / EPISODE_FILE) as arrays:
        action = arrays["action"].astype(np.float32)
        state = arrays["state"].astype(np.float32)
        phase = tuple(str(name) for name in arrays["phase"])
        dof_names = tuple(str(name) for name in arrays["dof_names"])
        fps = int(arrays["fps"])
    if not len(action) == len(state) == len(phase) or action.shape[1:] != (len(dof_names),):
        raise EpisodeError(f"Episode arrays of {run_dir} do not agree.")
    return Episode(
        slot=slot,
        instruction=slot_instruction(slot),
        passed=check.passed,
        action=action,
        state=state,
        phase=phase,
        dof_names=dof_names,
        fps=fps,
        front_video=sim_dir / FRONT_VIDEO_FILE,
        wrist_video=sim_dir / WRIST_VIDEO_FILE,
    )


def training_steps(episode: Episode, *, tail_idle: int = TAIL_IDLE_FRAMES) -> list[int]:
    """학습에 쓰는 프레임 번호. 프레임 i의 관측은 `action[i + 1]`과 짝이다.

    - 마지막 프레임은 다음 행동이 없어서 뺀다.
    - 처음의 정지 프레임은 마지막 1개만 남긴다. 실행기는 첫 구간에서 움직이지 않는다.
    - 끝의 정지 프레임은 `tail_idle`개만 남긴다.
    """
    steps = len(episode.action) - 1
    moving = [
        index
        for index in range(steps)
        if float(np.max(np.abs(episode.action[index + 1] - episode.action[index])))
        > IDLE_TOLERANCE_RAD
    ]
    if not moving:
        return []
    return list(range(moving[0], min(moving[-1] + 1 + tail_idle, steps)))


def perturb_episode(episode: Episode, joint_error: tuple[JointErrorRow, ...]) -> Episode:
    """관절 센서 오차 `b`를 아는 제어기의 시연으로 바꾼다.

    그 제어기는 관절 목표 `action - b`를 준다. 그러면 실제 관절 각도는 기록과 같다.
    그래서 영상은 기록 그대로 쓴다. 측정값은 `state - b`다.
    프레임 i의 `action`과 `state`에서 `b`의 i행을 뺀다. 그리퍼의 값은 그대로다.

    Raises:
        ValueError: 여러 행의 오차가 에피소드보다 짧을 때.
    """
    error = joint_error_frames(joint_error, frames=len(episode.action))
    action = episode.action.copy()
    state = episode.state.copy()
    # float64로 뺀 뒤에 float32로 바꾼다. 실행기는 float32 목표에 float64 오차를 더한다.
    action[:, :ARM_JOINT_COUNT] = action[:, :ARM_JOINT_COUNT].astype(np.float64) - error
    state[:, :ARM_JOINT_COUNT] = state[:, :ARM_JOINT_COUNT].astype(np.float64) - error
    return replace(episode, action=action, state=state)


def sample_training_joint_error(
    rng: np.random.Generator,
    *,
    sigma_max: float,
    constant_share: float = TRAIN_CONSTANT_ERROR_SHARE,
) -> tuple[JointErrorLevel, tuple[JointErrorRow, ...]]:
    """학습 사본 1개의 관절 센서 오차를 뽑는다. 크기와 지속 시간, 그리고 프레임별 값을 준다.

    - 크기는 0과 `sigma_max`(rad) 사이의 균등 분포다.
    - 지속 시간은 `constant_share`의 확률로 None이다. 실행 동안 오차가 같다.
    - 그 밖의 지속 시간은 `TRAIN_TAU_RANGE_S`의 log-uniform이다.
    - `constant_share`가 1이면 모든 사본의 오차가 고정이다.
    """
    sigma = float(rng.uniform(0.0, sigma_max))
    tau = None
    if rng.uniform() >= constant_share:
        low, high = TRAIN_TAU_RANGE_S
        tau = math.exp(float(rng.uniform(math.log(low), math.log(high))))
    level = JointErrorLevel(sigma=sigma, tau=tau)
    return level, sample_joint_error(rng, level=level)


class ReplayPolicy:
    """기록한 시연의 관절 목표를 그대로 돌려주는 제어기. 폐루프 실행의 배선을 확인하는 데 쓴다.

    문맥의 `layout`과 `repeat`으로 `root`의 실행 디렉터리를 찾는다.
    평가의 seed와 보드 오차가 기록과 같아야 장면이 같다.

    `scene_root`는 평가의 출력 디렉터리다. 그 실행의 `scene.json`에 관절 센서 오차가 있으면
    목표에서 오차를 뺀다(`perturb_episode`). 이때 이 제어기는 오차를 아는 제어기다.
    """

    def __init__(
        self, root: Path, *, chunk: int = REPLAY_CHUNK_FRAMES, scene_root: Path | None = None
    ) -> None:
        self.root = root
        self.chunk = chunk
        self.scene_root = scene_root
        self._action = np.zeros((0, 0), dtype=np.float32)

    def reset(self, context: Mapping[str, Any]) -> int:
        run = Path(str(context["layout"])) / RUN_DIR_FORMAT.format(repeat=context["repeat"])
        episode = load_episode(self.root / run)
        joint_error = self._joint_error(run)
        if joint_error is not None:
            episode = perturb_episode(episode, joint_error)
        self._action = episode.action
        return len(self._action)

    def _joint_error(self, run: Path) -> tuple[JointErrorRow, ...] | None:
        if self.scene_root is None:
            return None
        scene_path = self.scene_root / run / SCENE_FILE
        if not scene_path.is_file():
            return None
        return SimScene.model_validate_json(scene_path.read_text(encoding="utf-8")).joint_error

    def act(self, observation: Observation) -> NDArray[np.float32]:
        start = observation.step + 1
        actions = self._action[start : start + self.chunk]
        # 기록이 끝나면 마지막 목표를 유지한다.
        return actions if len(actions) else self._action[-1:]


class _EvaluationModel(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)


class EpisodeOutcome(_EvaluationModel):
    """평가 에피소드 1개의 결과."""

    layout: str
    repeat: int
    slot: int
    passed: bool
    error_kind: RunErrorKind | None = None
    slot_shift_mm: float
    """슬롯 중심이 셀 설정의 자리에서 실제 자리까지 움직인 거리(mm)."""
    placed_slot: int | None = None
    """부품의 최종 위치가 든 슬롯. 어느 슬롯에도 없으면 None이다."""


class PassCount(_EvaluationModel):
    episodes: int = 0
    passed: int = 0


class PolicyEvaluation(_EvaluationModel):
    """평가 요약. `evaluation.json`에 쓴다. 끝나지 못한 에피소드는 통과가 아니다."""

    controller: str
    episodes: list[EpisodeOutcome]
    total: PassCount
    errors: int
    wrong_slot: int
    """부품이 지시와 다른 슬롯에 놓인 에피소드 수."""
    by_slot: dict[str, PassCount]
    by_shift: dict[str, PassCount]
    """슬롯 이동량 구간별 수. 구간은 `SHIFT_BANDS_MM`이다."""
    joint_error: JointErrorLevel | None = None
    """관절 센서 오차의 크기와 지속 시간. None이면 오차가 없다."""


def evaluate_policy(
    controller: str,
    *,
    out_dir: Path,
    cell: CellConfig,
    sim_settings: SimSettings,
    per_slot: int,
    seed: int,
    board_error: BoardOffset,
    shard: Shard = ALL_LAYOUTS,
    jobs: int = 1,
    max_steps: int = EVAL_MAX_STEPS,
    joint_error: JointErrorLevel | None = None,
) -> PolicyEvaluation:
    """슬롯마다 `per_slot`개의 장면에서 제어기 1개를 실행하고 판정한다.

    `controller`:
    - `scripted`: 셀 설정으로 컴파일한 task를 실행한다. 보드 오차를 모른다.
    - `scripted-calibrated`: 실행마다 캡처와 보드 캘리브레이션을 하고 컴파일한다.
    - `replay:DIR`: `record_episodes`의 출력 `DIR`에서 같은 실행의 관절 목표를 재생한다.
      `joint_error`가 있으면 목표에서 그 실행의 오차를 뺀다.
    - `HOST:PORT`: 그 주소의 제어기 서버가 관절 목표를 낸다.

    장면은 seed, `board_error`와 `joint_error`가 정한다. 제어기가 달라도 같은 seed의 장면은 같다.
    모든 제어기를 시연 기록(`--record`)으로 실행한다.

    Raises:
        ValueError: `controller`의 형식이 틀렸을 때. `max_steps`가 관절 센서 오차의 행 수를 넘을 때.
    """
    layouts = slot_layouts()
    common: dict[str, Any] = {
        "out_dir": out_dir,
        "cell": cell,
        "sim_settings": sim_settings,
        "repeats": per_slot,
        "seed": seed,
        "shard": shard,
        "jobs": jobs,
        "board_error": board_error,
        "record": True,
        "joint_error": joint_error,
    }
    if controller in (CONTROLLER_SCRIPTED, CONTROLLER_CALIBRATED):
        summary = run_cell_benchmark(
            layouts, calibrate=controller == CONTROLLER_CALIBRATED, **common
        )
    else:
        _check_step_limit(max_steps, joint_error=joint_error)
        contexts = [
            {"slot": slot, "instruction": slot_instruction(slot)} for slot in range(len(SLOTS))
        ]
        if controller.startswith(REPLAY_PREFIX):
            root = Path(controller.removeprefix(REPLAY_PREFIX))
            if not root.is_dir():
                raise ValueError(f"Replay directory {root} does not exist.")
            with PolicyServer(ReplayPolicy(root, scene_root=out_dir)) as server:
                policy = BenchPolicy(address=server.address, max_steps=max_steps, contexts=contexts)
                summary = run_cell_benchmark(layouts, policy=policy, **common)
        else:
            address = PolicyAddress.parse(controller)
            policy = BenchPolicy(address=address, max_steps=max_steps, contexts=contexts)
            summary = run_cell_benchmark(layouts, policy=policy, **common)
    evaluation = summarize_evaluation(
        summary, controller=controller, out_dir=out_dir, cell=cell, joint_error=joint_error
    )
    write_json(out_dir / EVALUATION_FILE, evaluation)
    return evaluation


def _check_step_limit(max_steps: int, *, joint_error: JointErrorLevel | None) -> None:
    """바뀌는 관절 센서 오차는 행 수가 정해져 있다. 제어기 실행의 프레임이 그 안에 있어야 한다."""
    if joint_error is None or joint_error.tau is None:
        return
    limit = JOINT_ERROR_ROWS - round(FINAL_SETTLE_S * SIM_FPS)
    if max_steps > limit:
        raise ValueError(
            f"Step limit {max_steps} is more than {limit}. "
            f"The joint error has {JOINT_ERROR_ROWS} rows."
        )


def summarize_evaluation(
    summary: CellBenchmark,
    *,
    controller: str,
    out_dir: Path,
    cell: CellConfig,
    joint_error: JointErrorLevel | None = None,
) -> PolicyEvaluation:
    """벤치마크 요약과 실행 디렉터리의 결과로 슬롯별, 슬롯 이동량별 통과 수를 센다."""
    episodes = [_episode_outcome(run, out_dir=out_dir, cell=cell) for run in summary.runs]
    by_slot: dict[str, list[EpisodeOutcome]] = {}
    by_shift: dict[str, list[EpisodeOutcome]] = {}
    for episode in episodes:
        by_slot.setdefault(slot_name(episode.slot), []).append(episode)
        by_shift.setdefault(_shift_band(episode.slot_shift_mm), []).append(episode)
    bands = [name for name, _ in reversed(SHIFT_BANDS_MM)]
    return PolicyEvaluation(
        controller=controller,
        episodes=episodes,
        total=_pass_count(episodes),
        errors=sum(episode.error_kind is not None for episode in episodes),
        wrong_slot=sum(
            episode.placed_slot is not None and episode.placed_slot != episode.slot
            for episode in episodes
        ),
        by_slot={name: _pass_count(members) for name, members in sorted(by_slot.items())},
        by_shift={name: _pass_count(by_shift[name]) for name in bands if name in by_shift},
        joint_error=joint_error,
    )


def _episode_outcome(run: RunOutcome, *, out_dir: Path, cell: CellConfig) -> EpisodeOutcome:
    run_dir = out_dir / run.layout / RUN_DIR_FORMAT.format(repeat=run.repeat)
    layout = Layout.model_validate_json((run_dir.parent / LAYOUT_FILE).read_text(encoding="utf-8"))
    slot = SLOTS.index(layout.parts[0].cells[0])
    centers = _slot_centers(cell, layout.board)
    actual = centers
    scene_path = run_dir / SCENE_FILE
    if scene_path.is_file():
        placement = SimScene.model_validate_json(scene_path.read_text(encoding="utf-8")).board
        if placement is not None:
            actual = [
                to_placed_board((float(x), float(y)), cell=cell, placement=placement)
                for x, y in centers
            ]
    shift_mm = float(np.hypot(*(np.asarray(actual[slot]) - np.asarray(centers[slot])))) * MM_PER_M
    placed_slot = None
    result_path = run_dir / SIM_DIR / RESULT_FILE
    if run.error is None and result_path.is_file():
        result = SimResult.model_validate_json(result_path.read_text(encoding="utf-8"))
        position = np.array(result.parts[0].position[:2])
        half_cell = layout.board.cell_size_mm / MM_PER_M / 2
        distances = [float(np.hypot(*(position - np.asarray(center)))) for center in actual]
        nearest = int(np.argmin(distances))
        if distances[nearest] < half_cell:
            placed_slot = nearest
    return EpisodeOutcome(
        layout=run.layout,
        repeat=run.repeat,
        slot=slot,
        passed=run.error is None and all(part.passed for part in run.parts),
        error_kind=run.error_kind,
        slot_shift_mm=shift_mm,
        placed_slot=placed_slot,
    )


def _slot_centers(cell: CellConfig, board: Board) -> list[NDArray[np.float64]]:
    """셀 설정이 믿는 보드의 슬롯 중심(m)."""
    frame = BoardFrame(cell=cell, cell_size=board.cell_size_mm / MM_PER_M)
    return [frame.point(np.array([col + 0.5, row + 0.5])) for row, col in SLOTS]


def _pass_count(episodes: list[EpisodeOutcome]) -> PassCount:
    return PassCount(episodes=len(episodes), passed=sum(episode.passed for episode in episodes))


def _shift_band(shift_mm: float) -> str:
    return next(name for name, minimum in SHIFT_BANDS_MM if shift_mm >= minimum)
