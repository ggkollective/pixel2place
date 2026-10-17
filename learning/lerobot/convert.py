"""`record-episodes`의 출력을 LeRobot 데이터셋으로 바꾼다.

사용법: uv run python convert.py --episodes DIR --root DATASET_DIR [--repo-id ID]
  [--joint-error-copies K --joint-error-sigma-max DEG [--joint-error-seed S]
   [--joint-error-constant-share F]] [--drop-quantile-stats]

- 통과한 시연만 넣는다.
- 프레임은 `training_steps`가 고른다. 프레임 i의 관측은 프레임 i+1의 관절 목표와 짝이다.
- 관절 값의 단위는 rad다. 실물 SO-101의 LeRobot 데이터셋과 단위가 다르다.
- 영상은 H.264로 쓴다. GR00T의 영상 디코더가 H.264를 요구한다.

`--joint-error-copies K`는 시연 1개에서 관절 센서 오차가 있는 사본 K개를 만든다.
- 사본마다 오차의 크기와 지속 시간을 뽑는다. 크기의 한계는 `--joint-error-sigma-max`(deg)다.
- 사본의 관절 목표와 측정값은 오차를 아는 제어기의 값이다(`perturb_episode`). 영상은 기록과 같다.
- 프레임은 기록한 시연으로 고른다. 그래서 사본의 프레임 번호는 기록과 같다.
- 사본마다 오차의 크기와 지속 시간을 `DATASET_DIR.joint_error.json`에 쓴다.
- 기록 그대로의 사본은 넣지 않는다. 크기가 0에 가까운 사본이 그 역할을 한다.
- `--joint-error-constant-share F`는 오차가 실행 동안 같은 사본의 비율이다. 1이면 모든 사본이다.

`--drop-quantile-stats`는 `meta/stats.json`의 관절 값에서 분위수를 지운다.
- LeRobot의 분위수는 에피소드별 분위수의 평균이다. 그래서 데이터의 범위보다 좁다.
- GR00T의 상대 행동 학습은 분위수가 있으면 그 범위로 측정값을 정규화하고 자른다.
- 분위수가 없으면 GR00T는 최솟값과 최댓값을 쓴다.
"""

import argparse
import json
import logging
import math
from pathlib import Path

import av
import numpy as np
from lerobot.configs.video import RGBEncoderConfig
from lerobot.datasets.lerobot_dataset import LeRobotDataset
from lerobot.datasets.utils import STATS_PATH

from pixel2place.cell.bench import CHECK_FILE
from pixel2place.cell.episodes import (
    SLOTS,
    TRAIN_CONSTANT_ERROR_SHARE,
    Episode,
    EpisodeError,
    load_episode,
    perturb_episode,
    sample_training_joint_error,
    slot_name,
    training_steps,
)
from pixel2place.cell.scene import JointErrorLevel

logger = logging.getLogger("convert")

ROBOT_TYPE = "so101_isaac_sim"
FRONT_KEY = "observation.images.front"
WRIST_KEY = "observation.images.wrist"
STATE_KEY = "observation.state"
SLOT_KEY = "observation.environment_state"
ACTION_KEY = "action"
IMAGE_SHAPE = (480, 640, 3)
VIDEO_CODEC = "h264"
# 기록 영상의 품질(crf 17)과 비슷하게 둔다. 기본값 30은 핀의 가장자리를 흐리게 한다.
VIDEO_CRF = 18
# 사본마다의 관절 센서 오차를 쓰는 파일. 데이터셋 디렉터리 이름 뒤에 붙인다.
JOINT_ERROR_SUFFIX = ".joint_error.json"
# `--drop-quantile-stats`가 지우는 통계의 이름과 그 대상 피처
QUANTILE_STATS = ("q01", "q10", "q50", "q90", "q99")
JOINT_FEATURES = (STATE_KEY, ACTION_KEY)


def read_frames(path: Path) -> np.ndarray:
    """영상의 모든 프레임을 (프레임, 높이, 너비, 3)의 uint8 배열로 읽는다."""
    with av.open(str(path)) as container:
        frames = [frame.to_ndarray(format="rgb24") for frame in container.decode(video=0)]
    return np.stack(frames)


def features(dof_names: tuple[str, ...]) -> dict:
    joints = [f"{name}.pos" for name in dof_names]
    image = {"dtype": "video", "shape": IMAGE_SHAPE, "names": ["height", "width", "channels"]}
    return {
        FRONT_KEY: image,
        WRIST_KEY: image,
        STATE_KEY: {"dtype": "float32", "shape": (len(joints),), "names": joints},
        SLOT_KEY: {
            "dtype": "float32",
            "shape": (len(SLOTS),),
            "names": [slot_name(slot) for slot in range(len(SLOTS))],
        },
        ACTION_KEY: {"dtype": "float32", "shape": (len(joints),), "names": joints},
    }


def episode_copies(
    episode: Episode,
    *,
    source: int,
    copies: int,
    sigma_max: float,
    seed: int,
    constant_share: float,
) -> list[tuple[Episode, JointErrorLevel | None]]:
    """데이터셋에 쓸 에피소드와 그 관절 센서 오차. `copies`가 0이면 기록 그대로 1개다.

    `source`는 입력 시연의 번호다. 사본의 난수는 seed, 시연 번호와 사본 번호가 정한다.
    `constant_share`는 오차가 실행 동안 같은 사본의 비율이다.
    """
    if copies == 0:
        return [(episode, None)]
    result: list[tuple[Episode, JointErrorLevel | None]] = []
    for copy in range(copies):
        rng = np.random.default_rng([seed, source, copy])
        level, values = sample_training_joint_error(
            rng, sigma_max=sigma_max, constant_share=constant_share
        )
        result.append((perturb_episode(episode, values), level))
    return result


def drop_quantile_stats(root: Path) -> None:
    """데이터셋의 `meta/stats.json`에서 관절 값의 분위수를 지운다."""
    path = root / STATS_PATH
    stats = json.loads(path.read_text(encoding="utf-8"))
    for feature in JOINT_FEATURES:
        for name in QUANTILE_STATS:
            stats[feature].pop(name, None)
    path.write_text(json.dumps(stats, indent=4) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--episodes", type=Path, required=True)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--repo-id", default="local/pixel2place_slots")
    parser.add_argument("--joint-error-copies", type=int, default=0)
    parser.add_argument("--joint-error-sigma-max", type=float, default=0.0)
    parser.add_argument("--joint-error-seed", type=int, default=0)
    parser.add_argument(
        "--joint-error-constant-share", type=float, default=TRAIN_CONSTANT_ERROR_SHARE
    )
    parser.add_argument("--drop-quantile-stats", action="store_true")
    args = parser.parse_args()
    if args.joint_error_copies < 0 or args.joint_error_sigma_max < 0:
        parser.error("Joint error options must be 0 or more.")
    if args.joint_error_copies > 0 and args.joint_error_sigma_max == 0:
        parser.error("--joint-error-copies needs --joint-error-sigma-max.")
    if not 0.0 <= args.joint_error_constant_share <= 1.0:
        parser.error("--joint-error-constant-share must be between 0 and 1.")
    # lerobot의 import가 root logger에 handler를 붙인다. force로 그 설정을 바꾼다.
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s", force=True
    )

    run_dirs = sorted(path.parent for path in args.episodes.glob(f"*/r*/{CHECK_FILE}"))
    if not run_dirs:
        raise SystemExit(f"No finished episodes are in {args.episodes}.")
    dataset = None
    counts = {"used": 0, "failed": 0, "unreadable": 0, "frames": 0, "written": 0}
    joint_errors = []
    for source, run_dir in enumerate(run_dirs):
        try:
            episode = load_episode(run_dir)
        except EpisodeError as exc:
            counts["unreadable"] += 1
            logger.warning("Episode is not readable. run_dir=%s error=%s", run_dir, exc)
            continue
        if not episode.passed:
            counts["failed"] += 1
            continue
        if dataset is None:
            dataset = LeRobotDataset.create(
                repo_id=args.repo_id,
                fps=episode.fps,
                features=features(episode.dof_names),
                root=args.root,
                robot_type=ROBOT_TYPE,
                use_videos=True,
                rgb_encoder=RGBEncoderConfig(vcodec=VIDEO_CODEC, crf=VIDEO_CRF),
            )
        front = read_frames(episode.front_video)
        wrist = read_frames(episode.wrist_video)
        if not len(front) == len(wrist) == len(episode.action):
            raise SystemExit(f"Frame counts of {run_dir} do not agree.")
        slot = np.zeros(len(SLOTS), dtype=np.float32)
        slot[episode.slot] = 1.0
        steps = training_steps(episode)
        variants = episode_copies(
            episode,
            source=source,
            copies=args.joint_error_copies,
            sigma_max=math.radians(args.joint_error_sigma_max),
            seed=args.joint_error_seed,
            constant_share=args.joint_error_constant_share,
        )
        for copy, (variant, level) in enumerate(variants):
            for index in steps:
                dataset.add_frame(
                    {
                        FRONT_KEY: front[index],
                        WRIST_KEY: wrist[index],
                        STATE_KEY: variant.state[index],
                        SLOT_KEY: slot,
                        ACTION_KEY: variant.action[index + 1],
                        "task": episode.instruction,
                    }
                )
            dataset.save_episode()
            if level is not None:
                joint_errors.append(
                    {
                        "episode_index": counts["written"],
                        "source": str(run_dir.relative_to(args.episodes)),
                        "copy": copy,
                        "sigma_deg": math.degrees(level.sigma),
                        "tau_s": level.tau,
                    }
                )
            counts["written"] += 1
            counts["frames"] += len(steps)
        counts["used"] += 1
        if counts["used"] % 20 == 0:
            logger.info("Episodes converted. used=%d", counts["used"])
    if dataset is None:
        raise SystemExit("No passed episodes are in the input.")
    dataset.finalize()
    if args.drop_quantile_stats:
        drop_quantile_stats(args.root)
    if joint_errors:
        joint_error_path = args.root.with_name(f"{args.root.name}{JOINT_ERROR_SUFFIX}")
        joint_error_path.write_text(json.dumps(joint_errors, indent=2) + "\n", encoding="utf-8")
    logger.info(
        "Dataset written. root=%s sources=%d episodes=%d frames=%d failed=%d unreadable=%d",
        args.root,
        counts["used"],
        counts["written"],
        counts["frames"],
        counts["failed"],
        counts["unreadable"],
    )


if __name__ == "__main__":
    main()
