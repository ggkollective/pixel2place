"""Isaac 셀 벤치마크. 여러 레이아웃을 셀에서 실행하고 부품 실패율을 잰다.

같은 task의 Isaac 실행은 결과가 같다.
그러나 공급대 부품 위치가 0.1 mm만 달라도 실패하는 부품이 바뀐다.
그래서 실행 1회는 표본 1개다. 레이아웃 수와 반복 수로 표본을 늘린다.

실행 단위는 (레이아웃, 반복)이다.
- 반복 0은 컴파일한 task 그대로다. `sim-place`와 결과가 같다.
- 반복 1 이상은 부품마다 공급대 위치를 옮긴다(`jitter_feeder`). waypoint는 그대로다.
- 보드 오차 한계를 주면 실행마다 보드 놓기 오차를 뽑는다. task는 그대로고 실제 보드만 옮긴다.
- 보정을 켜면 실행마다 빈 셀을 캡처하고 보드 자세를 추정한다. 그 다음 보정 셀 설정으로 컴파일한다.
- 실제 보드 계획을 켜면 실행마다 실제 보드 자세로 컴파일한다. 시연 데이터의 제어기다.
- 기록을 켜면 실행기가 손목 카메라 영상과 프레임별 관절 값을 더 쓴다.
- 제어기 서버를 주면 실행기가 waypoint 대신 제어기의 관절 목표를 따른다. 기록도 켠다.
- 판정이 이미 있고 task가 같은 실행은 다시 하지 않는다. 그래서 중단한 측정을 이어서 할 수 있다.
  캡처도 같은 방식으로 다시 하지 않는다. 추정과 보정 컴파일은 저장한 캡처로 매번 다시 계산한다.
  노드마다 shard를 실행하고 출력 디렉터리를 합친 뒤, shard 없이 다시 실행하면 요약만 쓴다.
"""

import json
import logging
import time
from collections import Counter
from collections.abc import Callable, Mapping, Sequence
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from enum import StrEnum
from pathlib import Path
from typing import Any, NamedTuple

import numpy as np
from PIL import Image
from pydantic import BaseModel, ConfigDict, computed_field

from pixel2place.cell.calibration import CalibrationError, calibrate_board, write_calibration
from pixel2place.cell.capture import CAPTURE_FILE, CaptureTask, compile_capture
from pixel2place.cell.check import CheckFailure, FailureStage, check_result
from pixel2place.cell.clearance import placement_clearances
from pixel2place.cell.config import CellConfig
from pixel2place.cell.policy_link import PolicyAddress
from pixel2place.cell.result import SimResult
from pixel2place.cell.scene import (
    BoardOffset,
    BoardPlacement,
    CameraOffset,
    JointErrorLevel,
    SimScene,
    board_shift_mm,
    displace_board,
    placed_cell,
    placement_of,
    sample_joint_error,
)
from pixel2place.cell.sim import (
    CAPTURE_OUTPUTS,
    RECORD_OUTPUTS,
    RESULT_FILE,
    PolicyRun,
    SimRunError,
    read_capture,
    require_sim_files,
    run_capture,
    run_place_task,
)
from pixel2place.cell.task import Task, TaskCompileError, compile_task
from pixel2place.config import SimSettings
from pixel2place.layout.schema import Cell, Layout, PartType
from pixel2place.pipeline import JSON_INDENT, write_json

logger = logging.getLogger(__name__)

LAYOUT_DIR_FORMAT = "{index:03d}_{name}"
RUN_DIR_FORMAT = "r{repeat}"
LAYOUT_FILE = "layout.json"
TASK_FILE = "task.json"
SCENE_FILE = "scene.json"
SIM_DIR = "sim"
# 보정 실행의 캡처 출력, 보정 셀 설정, 보고, overlay가 이 디렉터리에 있다.
CAPTURE_DIR = "capture"
CALIBRATED_CELL_FILE = "cell.json"
CHECK_FILE = "check.json"
# 폐루프 실행의 문맥. 실행기가 제어기 서버에 그대로 보낸다.
POLICY_CONTEXT_FILE = "policy_context.json"
ERROR_FILE = "error.txt"
SUMMARY_FILE = "summary.json"
# 반복마다 공급대 부품 위치를 x, y 각각 이 거리 안에서 옮긴다(m).
FEEDER_JITTER_M = 0.0005
# 보드 오차, 카메라 오차와 관절 센서 오차 rng의 마지막 seed 값. 공급대 섭동 rng와 다른 수열을 쓴다.
BOARD_ERROR_STREAM = 1
CAMERA_ERROR_STREAM = 2
JOINT_ERROR_STREAM = 3
# Isaac 기동이 겹치지 않도록 처음 `jobs`개 실행의 시작 간격을 둔다(초).
START_INTERVAL_S = 20.0
NEIGHBOR_OFFSETS: tuple[Cell, ...] = ((1, 0), (-1, 0), (0, 1), (0, -1))
# 이웃 면 수 구간의 (이름, 최솟값). 최솟값이 큰 구간부터 찾는다.
NEIGHBOR_BANDS: tuple[tuple[str, int], ...] = (("5+", 5), ("3-4", 3), ("0-2", 0))
# 놓을 때 최소 틈 구간의 (이름, 최솟값 mm). 최솟값이 큰 구간부터 찾는다.
CLEARANCE_BANDS_MM: tuple[tuple[str, float], ...] = (
    ("1.9+", 1.9),
    ("1.5-1.9", 1.5),
    ("1.0-1.5", 1.0),
    ("0-1.0", 0.0),
)
# 반 셀 안에 먼저 놓인 이웃이 없는 부품의 구간
NO_NEIGHBOR_BAND = "none"


class BenchLayout(NamedTuple):
    """이름을 붙인 레이아웃. 이름은 출력 디렉터리 이름에 쓴다."""

    name: str
    layout: Layout


@dataclass(frozen=True)
class Shard:
    """레이아웃 번호를 `count`로 나눈 나머지가 `index`인 레이아웃만 실행한다."""

    index: int = 0
    count: int = 1

    @classmethod
    def parse(cls, text: str) -> "Shard":
        """`K/N` 형식을 읽는다. 0 <= K < N이어야 한다."""
        index, separator, count = text.partition("/")
        if not separator or not index.isdigit() or not count.isdigit():
            raise ValueError(f"Shard {text!r} is not K/N.")
        shard = cls(index=int(index), count=int(count))
        if shard.index >= shard.count:
            raise ValueError(f"Shard {text!r} needs 0 <= K < N.")
        return shard

    def includes(self, layout_index: int) -> bool:
        return layout_index % self.count == self.index


ALL_LAYOUTS = Shard()


class RunErrorKind(StrEnum):
    """실행이 부품 판정까지 가지 못한 이유."""

    COMPILE_FAILED = "compile_failed"
    """레이아웃을 셀 설정으로 컴파일하지 못했다. 배치 규칙 위반 같은 레이아웃 문제다."""
    CALIBRATION_FAILED = "calibration_failed"
    """캡처 이미지에서 보드 자세를 구하지 못했다."""
    OUT_OF_REACH = "out_of_reach"
    """보정한 보드의 waypoint에 팔이 닿지 않는다. 셀 설정으로는 컴파일된다."""
    RUNNER_ERROR = "runner_error"
    """Isaac 실행기가 캡처 파일이나 결과를 쓰지 못했다. 실행기 오류나 timeout이다."""


class _BenchModel(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)


class PartOutcome(_BenchModel):
    """실행 1회에서 부품 1개의 판정."""

    sequence: int
    part_id: str
    type: PartType
    neighbor_sides: int
    """놓는 시점에 이미 놓인 부품과 맞닿는 셀 면 수."""
    clearance_mm: float | None
    """놓는 시점에 먼저 놓인 이웃과의 최소 틈(mm). 반 셀 안에 이웃이 없으면 None이다."""
    failures: list[CheckFailure]
    failed_stage: FailureStage | None

    @computed_field
    @property
    def passed(self) -> bool:
        return not self.failures


class RunOutcome(_BenchModel):
    """(레이아웃, 반복) 실행 1회의 결과.

    실행하지 못했으면 `error`에 이유가 있고 `parts`는 비어 있다.
    """

    layout: str
    """출력 디렉터리 이름(`NNN_이름`)."""
    repeat: int
    parts: list[PartOutcome]
    error: str | None = None
    error_kind: RunErrorKind | None = None
    board_shift_mm: float | None = None
    """보드 셀 중심이 셀 설정의 자리에서 실제 자리까지 움직인 거리의 최댓값(mm).

    scene에 보드가 없으면 None이다.
    """
    calibration_error_mm: float | None = None
    """보정한 보드의 셀 중심과 실제 보드의 셀 중심 사이 거리의 최댓값(mm).

    보정하지 않았거나 보드 자세를 구하지 못했으면 None이다.
    """
    order_changed: bool | None = None
    """보정 task의 놓는 순서가 셀 설정으로 컴파일한 순서와 다르다.

    보정 task가 없으면 None이다.
    """


class FailureCount(_BenchModel):
    """부품 수와 실패한 부품 수."""

    parts: int = 0
    failed: int = 0

    @computed_field
    @property
    def failure_rate(self) -> float:
        return self.failed / self.parts if self.parts else 0.0


class CellBenchmark(_BenchModel):
    """벤치마크 요약. `summary.json`에 쓴다. 실행 오류가 난 실행의 부품은 수에 넣지 않는다."""

    runs: list[RunOutcome]
    total: FailureCount
    passed_runs: int
    """모든 부품이 통과한 실행 수."""
    errors: int
    """실행하지 못한 실행 수."""
    errors_by_kind: dict[RunErrorKind, int]
    order_changed_runs: int
    """보정 task의 놓는 순서가 셀 설정의 순서와 다른 실행 수."""
    max_calibration_error_mm: float | None
    """실행별 `calibration_error_mm`의 최댓값. 값이 있는 실행이 없으면 None이다."""
    by_stage: dict[FailureStage, int]
    by_type: dict[PartType, FailureCount]
    by_neighbor_sides: dict[str, FailureCount]
    """놓는 시점의 이웃 면 수 구간별 수. 구간은 `NEIGHBOR_BANDS`다."""
    by_clearance: dict[str, FailureCount]
    """놓을 때 최소 틈 구간별 수. 구간은 `CLEARANCE_BANDS_MM`과 `NO_NEIGHBOR_BAND`다."""
    elapsed_s: float
    """이번 명령의 실행 시간. 건너뛴 실행의 시간은 들어가지 않는다."""


@dataclass(frozen=True)
class BenchPolicy:
    """폐루프 실행의 제어기 서버와 문맥."""

    address: PolicyAddress
    max_steps: int
    contexts: Sequence[Mapping[str, Any]]
    """레이아웃마다의 문맥. 순서는 레이아웃의 순서다.

    실행기는 `layout`과 `repeat`을 더해서 제어기 서버에 보낸다.
    """


class _Run(NamedTuple):
    layout: Layout
    layout_name: str
    layout_index: int
    repeat: int
    planned: Task | None
    """셀 설정으로 컴파일한 task. 공급대 섭동 전이다. 컴파일하지 못했으면 None이다."""
    scene: SimScene | None
    """보드 오차, 카메라 오차, 보정이 모두 없으면 None이다."""
    run_dir: Path
    task: Task | None = None
    """실행할 task. 실행 전에 오류가 났으면 None이다."""
    error_kind: RunErrorKind | None = None
    """실행 전에 난 오류. 부품 실행의 오류는 판정 파일이 없는 것으로 안다."""
    calibrated: CellConfig | None = None
    """보드 자세를 고친 셀 설정. 보정하지 않았거나 추정에 실패했으면 None이다."""


def neighbor_sides(layout: Layout, task: Task) -> dict[str, int]:
    """부품마다 놓는 시점에 이미 놓인 부품과 맞닿는 셀 면 수를 센다. 키는 `part_id`다.

    값이 크면 부품이 이웃 사이로 내려간다. 3면이 막힌 포켓에서 이웃 모서리에 걸리기 쉽다.
    """
    cells_by_id = {part.part_id: set(part.cells) for part in layout.parts}
    placed: set[Cell] = set()
    sides: dict[str, int] = {}
    for part in task.parts:
        cells = cells_by_id[part.part_id]
        sides[part.part_id] = sum(
            (row + d_row, col + d_col) in placed
            for row, col in cells
            for d_row, d_col in NEIGHBOR_OFFSETS
        )
        placed |= cells
    return sides


def jitter_feeder(task: Task, rng: np.random.Generator, *, max_offset: float) -> Task:
    """부품마다 공급대 위치를 x, y 각각 `max_offset`(m) 안에서 옮긴 task를 반환한다.

    waypoint는 그대로다. 그래서 jaw가 부품을 잡는 위치가 조금 달라진다.
    """
    parts = []
    for part in task.parts:
        dx, dy = rng.uniform(-max_offset, max_offset, size=2)
        cells = [(x + float(dx), y + float(dy)) for x, y in part.feeder_cells]
        parts.append(part.model_copy(update={"feeder_cells": cells}))
    return task.model_copy(update={"parts": parts})


def sample_board_offset(rng: np.random.Generator, *, limit: BoardOffset) -> BoardOffset:
    """x, y, yaw를 각각 ±`limit`의 같은 필드 안에서 균등 분포로 뽑는다."""
    return BoardOffset(
        x=float(rng.uniform(-limit.x, limit.x)),
        y=float(rng.uniform(-limit.y, limit.y)),
        yaw=float(rng.uniform(-limit.yaw, limit.yaw)),
    )


def sample_camera_offset(rng: np.random.Generator, *, limit: CameraOffset) -> CameraOffset:
    """위치와 회전 벡터의 성분을 각각 ±`limit`의 같은 성분 안에서 균등 분포로 뽑는다."""
    position = rng.uniform(-np.array(limit.position), limit.position)
    rotation = rng.uniform(-np.array(limit.rotation), limit.rotation)
    return CameraOffset(
        position=(float(position[0]), float(position[1]), float(position[2])),
        rotation=(float(rotation[0]), float(rotation[1]), float(rotation[2])),
    )


def run_cell_benchmark(
    layouts: Sequence[BenchLayout],
    *,
    out_dir: Path,
    cell: CellConfig,
    sim_settings: SimSettings,
    repeats: int = 1,
    seed: int = 0,
    shard: Shard = ALL_LAYOUTS,
    jobs: int = 1,
    start_interval_s: float = START_INTERVAL_S,
    board_error: BoardOffset | None = None,
    camera_error: CameraOffset | None = None,
    calibrate: bool = False,
    plan_on_actual_board: bool = False,
    record: bool = False,
    policy: BenchPolicy | None = None,
    joint_error: JointErrorLevel | None = None,
) -> CellBenchmark:
    """레이아웃마다 `repeats`회 셀에서 실행하고 판정한다. 요약을 `summary.json`에 쓴다.

    `board_error`는 보드 놓기 오차의 한계다. 실행마다 seed로 오차를 뽑아서 실제 보드를 옮긴다.
    `camera_error`는 top-down 카메라 자세 오차의 한계다. 실행마다 seed로 오차를 뽑는다.
    `calibrate`가 참이면 실행마다 캡처와 추정을 하고 보정 셀 설정으로 컴파일한다.
    `plan_on_actual_board`가 참이면 실행마다 scene의 실제 보드 자세로 컴파일한다. 캡처는 없다.
    `calibrate`와 함께 쓰지 않는다.
    `record`가 참이면 실행기가 시연 기록(`RECORD_OUTPUTS`)을 쓴다.
    `policy`가 있으면 실행기가 task의 waypoint 대신 제어기 서버의 관절 목표를 따른다. 기록도 켠다.
    `joint_error`는 관절 센서 오차의 크기와 지속 시간이다. 실행마다 seed로 오차를 뽑는다.
    캡처는 이 오차를 쓰지 않는다.

    단계:
    1. Isaac 실행 파일을 확인한다. 없으면 `SimSetupError`를 낸다.
    2. shard에 든 레이아웃을 셀 설정으로 컴파일한다. 실행마다 scene을 정한다.
    3. 보정을 켜면 캡처가 없거나 바뀐 실행만 `jobs`개씩 동시에 캡처한다.
       그 다음 실행마다 보드 자세를 추정하고 보정 셀 설정으로 컴파일한다.
       캡처 자세에 팔이 닿지 않으면 `TaskCompileError`를 낸다.
    4. 반복 1 이상은 seed로 공급대 위치를 옮긴다.
    5. 판정이 없거나 task나 scene이 바뀐 실행만 `jobs`개씩 동시에 실행하고 판정한다.
    6. shard의 모든 실행을 요약한다.

    컴파일 실패, 추정 실패, 실행기 오류는 그 실행의 `error`로 적고 다음 실행으로 간다.
    """
    if calibrate and plan_on_actual_board:
        raise ValueError("Calibration and the actual board plan are exclusive.")
    if policy is not None and len(policy.contexts) != len(layouts):
        raise ValueError("The controller needs one context for each layout.")
    record = record or policy is not None
    started = time.monotonic()
    require_sim_files(sim_settings)
    runs = [
        run
        for index, bench_layout in enumerate(layouts)
        if shard.includes(index)
        for run in _prepare_runs(
            bench_layout,
            index=index,
            out_dir=out_dir,
            cell=cell,
            repeats=repeats,
            seed=seed,
            board_error=board_error,
            camera_error=camera_error,
            calibrate=calibrate,
            plan_on_actual_board=plan_on_actual_board,
            joint_error=joint_error,
        )
    ]
    if calibrate:
        captures = [
            run for run in runs if run.planned is not None and not _is_captured(run, cell=cell)
        ]
        logger.info(
            "Cell benchmark captures started.", extra={"runs": len(runs), "pending": len(captures)}
        )
        _run_all(
            captures,
            lambda run: _capture(run, cell=cell, sim_settings=sim_settings),
            jobs=jobs,
            start_interval_s=start_interval_s,
        )
        runs = [_calibrate(run, cell=cell) for run in runs]
    runs = [_with_jitter(run, seed=seed) for run in runs]
    pending = [run for run in runs if run.task is not None and not _is_done(run, record=record)]
    logger.info("Cell benchmark started.", extra={"runs": len(runs), "pending": len(pending)})
    _run_all(
        pending,
        lambda run: _execute(run, sim_settings=sim_settings, record=record, policy=policy),
        jobs=jobs,
        start_interval_s=start_interval_s,
    )
    outcomes = [_outcome(run, cell=cell) for run in runs]
    summary = _summarize(outcomes, elapsed_s=time.monotonic() - started)
    write_json(out_dir / SUMMARY_FILE, summary)
    return summary


def _run_all(
    runs: Sequence[_Run],
    action: Callable[[_Run], None],
    *,
    jobs: int,
    start_interval_s: float,
) -> None:
    """실행마다 `action`을 `jobs`개씩 동시에 부른다.

    Isaac 기동이 겹치지 않도록 처음 `jobs`개만 간격을 두고 시작한다.
    뒤 실행은 앞 실행이 끝나야 시작한다.
    """

    def start(position: int) -> None:
        if position < jobs:
            time.sleep(position * start_interval_s)
        action(runs[position])

    with ThreadPoolExecutor(max_workers=jobs) as executor:
        list(executor.map(start, range(len(runs))))


def _summarize(runs: list[RunOutcome], *, elapsed_s: float) -> CellBenchmark:
    """실행 결과를 부품 수와 실패 수로 모은다."""
    parts = [part for run in runs for part in run.parts]
    stages = Counter(part.failed_stage for part in parts if part.failed_stage is not None)
    error_kinds = Counter(run.error_kind for run in runs if run.error_kind is not None)
    by_band = _count_by(parts, key=lambda part: _neighbor_band(part.neighbor_sides))
    by_clearance = _count_by(parts, key=lambda part: _clearance_band(part.clearance_mm))
    clearance_bands = [name for name, _ in CLEARANCE_BANDS_MM] + [NO_NEIGHBOR_BAND]
    calibration_errors = [
        run.calibration_error_mm for run in runs if run.calibration_error_mm is not None
    ]
    return CellBenchmark(
        runs=runs,
        total=_count(parts),
        passed_runs=sum(run.error is None and all(p.passed for p in run.parts) for run in runs),
        errors=sum(run.error is not None for run in runs),
        errors_by_kind={kind: error_kinds[kind] for kind in RunErrorKind if kind in error_kinds},
        order_changed_runs=sum(run.order_changed is True for run in runs),
        max_calibration_error_mm=max(calibration_errors, default=None),
        by_stage=dict(stages),
        by_type=dict(sorted(_count_by(parts, key=lambda part: part.type).items())),
        by_neighbor_sides={name: by_band[name] for name, _ in NEIGHBOR_BANDS if name in by_band},
        by_clearance={name: by_clearance[name] for name in clearance_bands if name in by_clearance},
        elapsed_s=elapsed_s,
    )


def _count(parts: Sequence[PartOutcome]) -> FailureCount:
    return FailureCount(parts=len(parts), failed=sum(not part.passed for part in parts))


def _count_by[K](
    parts: Sequence[PartOutcome], *, key: Callable[[PartOutcome], K]
) -> dict[K, FailureCount]:
    groups: dict[K, list[PartOutcome]] = {}
    for part in parts:
        groups.setdefault(key(part), []).append(part)
    return {name: _count(members) for name, members in groups.items()}


def _prepare_runs(
    bench_layout: BenchLayout,
    *,
    index: int,
    out_dir: Path,
    cell: CellConfig,
    repeats: int,
    seed: int,
    board_error: BoardOffset | None,
    camera_error: CameraOffset | None,
    calibrate: bool,
    plan_on_actual_board: bool,
    joint_error: JointErrorLevel | None,
) -> list[_Run]:
    """레이아웃을 셀 설정으로 컴파일하고 반복마다 scene을 정한다.

    보정하지 않으면 컴파일한 task를 그대로 실행한다. 보정하면 task는 추정 뒤에 정한다.
    실제 보드 계획은 scene의 보드 자세로 다시 컴파일한다.
    """
    layout_name = LAYOUT_DIR_FORMAT.format(index=index, name=bench_layout.name)
    layout_dir = out_dir / layout_name
    layout_dir.mkdir(parents=True, exist_ok=True)
    write_json(layout_dir / LAYOUT_FILE, bench_layout.layout)
    try:
        planned: Task | None = compile_task(bench_layout.layout, cell)
    except TaskCompileError as exc:
        logger.warning(
            "Cell benchmark layout does not compile.",
            extra={"layout": layout_name, "error": str(exc)},
        )
        (layout_dir / ERROR_FILE).write_text(str(exc) + "\n", encoding="utf-8")
        planned = None
    board = bench_layout.layout.board
    runs = []
    for repeat in range(repeats):
        placement: BoardPlacement | None = None
        if board_error is not None:
            rng = np.random.default_rng([seed, index, repeat, BOARD_ERROR_STREAM])
            offset = sample_board_offset(rng, limit=board_error)
            placement = displace_board(cell, board, offset=offset)
        elif calibrate or plan_on_actual_board:
            # 보정 task의 셀 설정은 실제 보드와 조금 다르다.
            # scene에 보드가 없으면 실행기가 보드를 보정 셀 설정에 만든다.
            # 그래서 실제 보드를 scene에 적는다.
            placement = placement_of(cell, board)
        camera = None
        if camera_error is not None:
            rng = np.random.default_rng([seed, index, repeat, CAMERA_ERROR_STREAM])
            camera = sample_camera_offset(rng, limit=camera_error)
        joint_values = None
        if joint_error is not None:
            rng = np.random.default_rng([seed, index, repeat, JOINT_ERROR_STREAM])
            joint_values = sample_joint_error(rng, level=joint_error)
        scene = None
        if placement is not None or camera is not None or joint_values is not None:
            scene = SimScene(board=placement, camera=camera, joint_error=joint_values)
        run = _Run(
            layout=bench_layout.layout,
            layout_name=layout_name,
            layout_index=index,
            repeat=repeat,
            planned=planned,
            scene=scene,
            run_dir=layout_dir / RUN_DIR_FORMAT.format(repeat=repeat),
            task=None if calibrate else planned,
            error_kind=RunErrorKind.COMPILE_FAILED if planned is None else None,
        )
        if plan_on_actual_board and placement is not None and planned is not None:
            run = _plan_on_board(run, cell=cell, placement=placement)
        runs.append(run)
    return runs


def _plan_on_board(run: _Run, *, cell: CellConfig, placement: BoardPlacement) -> _Run:
    """실제 보드 자세로 컴파일한다. 팔이 닿지 않으면 이유를 `error.txt`에 쓴다."""
    try:
        task = compile_task(run.layout, placed_cell(cell, placement=placement))
    except TaskCompileError as exc:
        run.run_dir.mkdir(parents=True, exist_ok=True)
        return _failed(run._replace(task=None), kind=RunErrorKind.OUT_OF_REACH, error=exc)
    return run._replace(task=task)


def _is_captured(run: _Run, *, cell: CellConfig) -> bool:
    """캡처 파일이 있고 저장한 캡처 작업과 scene이 이번 실행과 같으면 캡처를 건너뛴다."""
    capture_dir = run.run_dir / CAPTURE_DIR
    capture_path = capture_dir / CAPTURE_FILE
    if not capture_path.is_file() or not all((capture_dir / n).is_file() for n in CAPTURE_OUTPUTS):
        return False
    saved = CaptureTask.model_validate_json(capture_path.read_text(encoding="utf-8"))
    if saved != compile_capture(cell, run.layout.board):
        return False
    return _saved_scene(run.run_dir) == run.scene


def _capture(run: _Run, *, cell: CellConfig, sim_settings: SimSettings) -> None:
    """빈 셀을 scene의 보드와 카메라로 캡처한다. 이전 판정은 지운다."""
    assert run.scene is not None
    capture_dir = run.run_dir / CAPTURE_DIR
    capture_dir.mkdir(parents=True, exist_ok=True)
    for name in (CHECK_FILE, ERROR_FILE):
        (run.run_dir / name).unlink(missing_ok=True)
    scene_path = run.run_dir / SCENE_FILE
    write_json(scene_path, run.scene)
    capture_path = capture_dir / CAPTURE_FILE
    write_json(capture_path, compile_capture(cell, run.layout.board))
    try:
        run_capture(capture_path, out_dir=capture_dir, settings=sim_settings, scene_path=scene_path)
    except SimRunError as exc:
        logger.warning(
            "Cell benchmark capture failed.",
            extra={"layout": run.layout_name, "repeat": run.repeat, "error": str(exc)},
        )
        (run.run_dir / ERROR_FILE).write_text(str(exc) + "\n", encoding="utf-8")


def _calibrate(run: _Run, *, cell: CellConfig) -> _Run:
    """저장한 캡처로 보드 자세를 추정하고 보정 셀 설정으로 컴파일한다.

    캡처 파일이 없으면 캡처의 실행기 오류다. 추정이나 컴파일에 실패하면 이유를 `error.txt`에 쓴다.
    """
    if run.planned is None:
        return run
    capture_dir = run.run_dir / CAPTURE_DIR
    if not all((capture_dir / name).is_file() for name in CAPTURE_OUTPUTS):
        return run._replace(error_kind=RunErrorKind.RUNNER_ERROR)
    captured = read_capture(capture_dir)
    with Image.open(captured.image_path) as image:
        rgb = image.convert("RGB")
    board = run.layout.board
    try:
        calibration = calibrate_board(rgb, captured.camera, cell=cell, board=board)
    except CalibrationError as exc:
        return _failed(run, kind=RunErrorKind.CALIBRATION_FAILED, error=exc)
    write_calibration(calibration, rgb, output=capture_dir / CALIBRATED_CELL_FILE, board=board)
    run = run._replace(calibrated=calibration.cell)
    try:
        task = compile_task(run.layout, calibration.cell)
    except TaskCompileError as exc:
        return _failed(run, kind=RunErrorKind.OUT_OF_REACH, error=exc)
    return run._replace(task=task)


def _failed(run: _Run, *, kind: RunErrorKind, error: Exception) -> _Run:
    logger.warning(
        "Cell benchmark run cannot start.",
        extra={"layout": run.layout_name, "repeat": run.repeat, "kind": kind, "error": str(error)},
    )
    (run.run_dir / ERROR_FILE).write_text(str(error) + "\n", encoding="utf-8")
    return run._replace(error_kind=kind)


def _with_jitter(run: _Run, *, seed: int) -> _Run:
    """반복 1 이상은 seed로 공급대 위치를 옮긴다. 반복 0은 task 그대로다."""
    if run.task is None or run.repeat == 0:
        return run
    rng = np.random.default_rng([seed, run.layout_index, run.repeat])
    return run._replace(task=jitter_feeder(run.task, rng, max_offset=FEEDER_JITTER_M))


def _is_done(run: _Run, *, record: bool) -> bool:
    """판정이 있고 저장한 task와 scene이 이번 실행과 같으면 실행을 건너뛴다.

    기록을 켰으면 시연 기록 파일도 모두 있어야 한다.
    """
    task_path = run.run_dir / TASK_FILE
    if run.task is None or not (run.run_dir / CHECK_FILE).is_file() or not task_path.is_file():
        return False
    if record and not all((run.run_dir / SIM_DIR / name).is_file() for name in RECORD_OUTPUTS):
        return False
    if Task.model_validate_json(task_path.read_text(encoding="utf-8")) != run.task:
        return False
    return _saved_scene(run.run_dir) == run.scene


def _saved_scene(run_dir: Path) -> SimScene | None:
    scene_path = run_dir / SCENE_FILE
    if not scene_path.is_file():
        return None
    return SimScene.model_validate_json(scene_path.read_text(encoding="utf-8"))


def _execute(
    run: _Run, *, sim_settings: SimSettings, record: bool, policy: BenchPolicy | None
) -> None:
    assert run.task is not None
    run.run_dir.mkdir(parents=True, exist_ok=True)
    (run.run_dir / CHECK_FILE).unlink(missing_ok=True)
    (run.run_dir / ERROR_FILE).unlink(missing_ok=True)
    task_path = run.run_dir / TASK_FILE
    write_json(task_path, run.task)
    scene_path = run.run_dir / SCENE_FILE
    scene_path.unlink(missing_ok=True)
    if run.scene is not None:
        write_json(scene_path, run.scene)
    extra = {"layout": run.layout_name, "repeat": run.repeat}
    policy_run = None
    if policy is not None:
        context_path = run.run_dir / POLICY_CONTEXT_FILE
        context = {**policy.contexts[run.layout_index], **extra}
        context_path.write_text(json.dumps(context, indent=JSON_INDENT) + "\n", encoding="utf-8")
        policy_run = PolicyRun(
            address=policy.address, context_path=context_path, max_steps=policy.max_steps
        )
    try:
        result = run_place_task(
            task_path,
            out_dir=run.run_dir / SIM_DIR,
            settings=sim_settings,
            scene_path=scene_path if run.scene is not None else None,
            record=record,
            policy=policy_run,
        )
    except SimRunError as exc:
        logger.warning("Cell benchmark run failed.", extra={**extra, "error": str(exc)})
        (run.run_dir / ERROR_FILE).write_text(str(exc) + "\n", encoding="utf-8")
        return
    check = check_result(run.task, result)
    write_json(run.run_dir / CHECK_FILE, check)
    logger.info(
        "Cell benchmark run finished.",
        extra={**extra, "passed": check.passed_count, "parts": len(check.parts)},
    )


def _outcome(run: _Run, *, cell: CellConfig) -> RunOutcome:
    """실행 디렉터리의 task와 결과로 판정을 다시 계산한다. 판정 파일은 완료 표시로만 쓴다."""
    board = run.layout.board
    placement = run.scene.board if run.scene is not None else None
    shift = calibration_error = order_changed = None
    if placement is not None:
        shift = board_shift_mm(cell, board, placement=placement)
        if run.calibrated is not None:
            calibration_error = board_shift_mm(run.calibrated, board, placement=placement)
    if run.calibrated is not None and run.task is not None and run.planned is not None:
        order_changed = _part_order(run.task) != _part_order(run.planned)
    outcome = RunOutcome(
        layout=run.layout_name,
        repeat=run.repeat,
        parts=[],
        board_shift_mm=shift,
        calibration_error_mm=calibration_error,
        order_changed=order_changed,
    )
    if run.error_kind is not None:
        error_dir = (
            run.run_dir.parent if run.error_kind == RunErrorKind.COMPILE_FAILED else run.run_dir
        )
        error = _read_error(error_dir / ERROR_FILE)
        return outcome.model_copy(update={"error": error, "error_kind": run.error_kind})
    assert run.task is not None
    if not (run.run_dir / CHECK_FILE).is_file():
        error = _read_error(run.run_dir / ERROR_FILE)
        return outcome.model_copy(update={"error": error, "error_kind": RunErrorKind.RUNNER_ERROR})
    result_path = run.run_dir / SIM_DIR / RESULT_FILE
    result = SimResult.model_validate_json(result_path.read_text(encoding="utf-8"))
    check = check_result(run.task, result)
    sides = neighbor_sides(run.layout, run.task)
    clearances = placement_clearances(run.layout, run.task, result)
    parts = [
        PartOutcome(
            sequence=part.sequence,
            part_id=part.part_id,
            type=part.type,
            neighbor_sides=sides[part.part_id],
            clearance_mm=clearances[part.part_id],
            failures=part.failures,
            failed_stage=part.failed_stage,
        )
        for part in check.parts
    ]
    return outcome.model_copy(update={"parts": parts})


def _part_order(task: Task) -> list[str]:
    return [part.part_id for part in task.parts]


def _read_error(path: Path) -> str:
    if path.is_file():
        return path.read_text(encoding="utf-8").strip()
    return "Run did not finish."


def _neighbor_band(sides: int) -> str:
    return next(name for name, minimum in NEIGHBOR_BANDS if sides >= minimum)


def _clearance_band(clearance_mm: float | None) -> str:
    if clearance_mm is None:
        return NO_NEIGHBOR_BAND
    return next(name for name, minimum in CLEARANCE_BANDS_MM if clearance_mm >= minimum)
