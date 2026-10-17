"""Isaac Sim에서 시연 기록, 폐루프 재생과 관절 센서 오차의 smoke 테스트.

`pytest -m sim`으로만 실행한다. 부품 실행이 5회, 캡처가 1회 있다(부품 실행은 각 약 40초).
Isaac 설정(`ISAAC_SIM_PATH`)이나 SO-101 USD가 없으면 skip한다.
"""

import math
import subprocess
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

from pixel2place.cell.bench import SCENE_FILE, Shard
from pixel2place.cell.config import CellConfig
from pixel2place.cell.episodes import (
    CONTROLLER_CALIBRATED,
    REPLAY_PREFIX,
    SLOT_BOARD_ERROR,
    evaluate_policy,
    load_episode,
    perturb_episode,
    record_episodes,
    slot_name,
    training_steps,
)
from pixel2place.cell.result import SimResult
from pixel2place.cell.scene import ARM_JOINT_COUNT, JointErrorLevel, SimScene
from pixel2place.cell.sim import FRONT_VIDEO_FILE, RECORD_OUTPUTS, RESULT_FILE, WRIST_VIDEO_FILE

# 슬롯 4개 중 첫 슬롯만 실행한다.
FIRST_SLOT = Shard(index=0, count=4)
SEED = 3
# 재생의 관절 목표는 float32로 오간다. 최종 부품 위치의 차이는 이 값보다 작다.
MAX_REPLAY_POSITION_DIFFERENCE_MM = 0.05
# 부품 1개의 시연은 약 200 프레임이다.
MIN_EPISODE_FRAMES = 150
FIRST_RUN = Path(f"000_{slot_name(0)}") / "r0"
# 빠르게 바뀌는 오차. 프레임 번호가 1칸 어긋나면 관절 값이 약 0.01 rad 달라진다.
FAST_JOINT_ERROR = JointErrorLevel(sigma=math.radians(1.0), tau=0.1)
# 오차를 아는 재생의 drive는 float32 반올림만큼 기록과 다르다.
# 부품을 잡는 접촉에서 이 차이가 커진다.
# 측정 관절 값의 차이 한도(rad). 약 0.06°다. 프레임 번호가 1칸 어긋나면 약 0.04 rad다.
MAX_RELABEL_STATE_DIFFERENCE_RAD = 1e-3
# 관절 목표의 차이 한도(rad). 목표는 float32로 오간다.
MAX_RELABEL_ACTION_DIFFERENCE_RAD = 1e-6
# 최종 부품 위치의 차이 한도(mm). 핀과 구멍의 틈 안에서 달라진다.
MAX_RELABEL_POSITION_DIFFERENCE_MM = 1.0
# 실행 동안 같은 오차. 2°는 끝점을 수용 범위(약 3 mm)보다 멀리 옮긴다.
CONSTANT_JOINT_ERROR = JointErrorLevel(sigma=math.radians(2.0))


# 제어기의 첫 관측에 부품이 있는지 보는 기준. 부품은 파란색이다. 장면의 다른 것은 파란색이 아니다.
# 파란 화소는 B 값이 R 값과 G 값보다 이 값 넘게 큰 화소다.
BLUE_MARGIN = 40
# 첫 프레임의 파란 화소 수를 둘째 프레임의 수로 나눈 값의 하한.
# 첫 프레임에 부품이 없으면 이 값은 0이다.
# 부품이 있으면 사선 카메라는 약 0.7~1.0이다. 손목 카메라는 약 1.0이다.
# 부품이 있는 첫 프레임은 부품의 윤곽이 조금 흐리다. 그래서 사선 카메라의 값이 1보다 작다.
MIN_FIRST_FRAME_BLUE_RATIO = 0.5


def _blue_pixels(frame: np.ndarray) -> int:
    return int(np.count_nonzero(frame[..., 2] - frame[..., :2].max(axis=-1) > BLUE_MARGIN))


def _first_frames(video: Path, out_dir: Path) -> list[np.ndarray]:
    """영상의 처음 2프레임."""
    out_dir.mkdir()
    subprocess.run(
        ["ffmpeg", "-v", "error", "-i", str(video), "-frames:v", "2", str(out_dir / "%d.png")],
        check=True,
    )
    return [
        np.asarray(Image.open(out_dir / f"{index}.png").convert("RGB"), dtype=np.int16)
        for index in (1, 2)
    ]


def _part_position(root: Path) -> np.ndarray:
    result = SimResult.model_validate_json((root / FIRST_RUN / "sim" / RESULT_FILE).read_text())
    return np.array(result.parts[0].position)


@pytest.mark.sim
def test_replay_reproduces_recorded_episode(sim_settings, tmp_path):
    """보드 오차가 있는 시연을 기록하고, 제어기 서버로 재생한 실행이 같은 자리에 부품을 놓는다."""
    recorded = tmp_path / "episodes"
    options = {
        "cell": CellConfig(),
        "sim_settings": sim_settings,
        "per_slot": 1,
        "seed": SEED,
        "board_error": SLOT_BOARD_ERROR,
        "shard": FIRST_SLOT,
    }
    summary = record_episodes(out_dir=recorded, **options)
    (run,) = summary.runs
    assert run.error is None, run.error
    assert run.parts[0].passed, run.parts[0]

    run_dir = recorded / f"000_{slot_name(0)}" / "r0"
    for name in RECORD_OUTPUTS:
        assert (run_dir / "sim" / name).stat().st_size > 0, name
    episode = load_episode(run_dir)
    assert len(episode.action) > MIN_EPISODE_FRAMES
    assert len(training_steps(episode)) < len(episode.action)
    # 그리퍼의 닫기 목표는 부품 폭보다 좁다. 측정값은 부품에서 멈춘다.
    assert episode.action[:, 5].min() < episode.state[:, 5].min()

    replayed = tmp_path / "replay"
    evaluation = evaluate_policy(f"{REPLAY_PREFIX}{recorded}", out_dir=replayed, **options)
    (outcome,) = evaluation.episodes
    assert outcome.error_kind is None
    assert outcome.passed
    assert outcome.placed_slot == 0

    recorded_position, replayed_position = (
        np.array(
            SimResult.model_validate_json(
                (root / f"000_{slot_name(0)}" / "r0" / "sim" / RESULT_FILE).read_text()
            )
            .parts[0]
            .position
        )
        for root in (recorded, replayed)
    )
    difference_mm = 1000 * float(np.linalg.norm(recorded_position - replayed_position))
    assert difference_mm < MAX_REPLAY_POSITION_DIFFERENCE_MM

    # 제어기의 첫 관측은 영상의 첫 프레임이다. 두 카메라의 첫 프레임에 부품이 있다.
    for name in (FRONT_VIDEO_FILE, WRIST_VIDEO_FILE):
        first, second = _first_frames(
            replayed / f"000_{slot_name(0)}" / "r0" / "sim" / name, tmp_path / f"frames-{name}"
        )
        assert _blue_pixels(second) > 0, name
        assert _blue_pixels(first) > MIN_FIRST_FRAME_BLUE_RATIO * _blue_pixels(second), name


@pytest.mark.sim
def test_replay_with_known_joint_error_matches_relabeled_episode(sim_settings, tmp_path):
    """관절 센서 오차를 아는 재생은 기록과 같은 슬롯에 부품을 놓는다.

    그 실행이 기록하는 관절 목표는 `perturb_episode`의 값과 프레임마다 같다.
    측정값은 접촉에서 커진 반올림 오차만큼만 다르다.
    """
    recorded = tmp_path / "episodes"
    options = {
        "cell": CellConfig(),
        "sim_settings": sim_settings,
        "per_slot": 1,
        "seed": SEED,
        "board_error": SLOT_BOARD_ERROR,
        "shard": FIRST_SLOT,
    }
    (run,) = record_episodes(out_dir=recorded, **options).runs
    assert run.error is None, run.error
    clean = load_episode(recorded / FIRST_RUN)

    replayed = tmp_path / "replay"
    evaluation = evaluate_policy(
        f"{REPLAY_PREFIX}{recorded}", out_dir=replayed, joint_error=FAST_JOINT_ERROR, **options
    )
    (outcome,) = evaluation.episodes
    assert outcome.error_kind is None
    assert outcome.passed

    difference_mm = 1000 * float(
        np.linalg.norm(_part_position(recorded) - _part_position(replayed))
    )
    assert difference_mm < MAX_RELABEL_POSITION_DIFFERENCE_MM, f"부품 위치 차이 {difference_mm} mm"

    scene = SimScene.model_validate_json((replayed / FIRST_RUN / SCENE_FILE).read_text())
    assert scene.joint_error is not None
    relabeled = perturb_episode(clean, scene.joint_error)
    followed = load_episode(replayed / FIRST_RUN)
    frames = len(clean.action)
    for name, logged, expected, limit in (
        ("action", followed.action, relabeled.action, MAX_RELABEL_ACTION_DIFFERENCE_RAD),
        ("state", followed.state, relabeled.state, MAX_RELABEL_STATE_DIFFERENCE_RAD),
    ):
        difference = float(np.max(np.abs(logged[:frames] - expected)))
        assert difference < limit, f"{name}의 차이 {difference} rad"
    # 오차가 실제로 들어갔다. 다시 라벨링한 목표는 기록한 목표와 다르다.
    shift = np.abs(relabeled.action - clean.action)[:, :ARM_JOINT_COUNT]
    assert float(shift.max()) > 10 * MAX_RELABEL_STATE_DIFFERENCE_RAD


@pytest.mark.sim
def test_calibrated_script_misses_with_joint_error(sim_settings, tmp_path):
    """보드 캘리브레이션은 팔의 오차를 재지 않는다. 관절 센서 오차 2°에서 스크립트는 실패한다."""
    evaluation = evaluate_policy(
        CONTROLLER_CALIBRATED,
        out_dir=tmp_path / "eval",
        cell=CellConfig(),
        sim_settings=sim_settings,
        per_slot=1,
        seed=SEED,
        board_error=SLOT_BOARD_ERROR,
        shard=FIRST_SLOT,
        joint_error=CONSTANT_JOINT_ERROR,
    )
    (outcome,) = evaluation.episodes
    assert outcome.error_kind is None
    assert not outcome.passed
