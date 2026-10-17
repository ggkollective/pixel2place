"""가짜 Isaac 실행기가 쓰는 에피소드 배열.

가짜 시연의 관절 목표는 처음과 끝의 `FAKE_IDLE_FRAMES`개 프레임에서 멈춘다.
그 사이에서는 프레임마다 변한다. 측정 관절 값은 목표와 같다.
"""

from pathlib import Path

import numpy as np

FAKE_EPISODE_FRAMES = 40
FAKE_IDLE_FRAMES = 5
FAKE_FPS = 30
FAKE_DOF_NAMES = (
    "shoulder_pan",
    "shoulder_lift",
    "elbow_flex",
    "wrist_flex",
    "wrist_roll",
    "gripper",
)


def fake_action() -> np.ndarray:
    """가짜 시연의 관절 목표. 모양은 (`FAKE_EPISODE_FRAMES`, 6)이다."""
    moving = FAKE_EPISODE_FRAMES - 2 * FAKE_IDLE_FRAMES
    ramp = np.concatenate(
        [np.zeros(FAKE_IDLE_FRAMES), np.linspace(0.0, 1.0, moving), np.ones(FAKE_IDLE_FRAMES)]
    )
    return np.outer(ramp, np.ones(len(FAKE_DOF_NAMES))).astype(np.float32)


def write_episode(path: Path, action: np.ndarray, *, phase: str = "move") -> None:
    """관절 목표로 `episode.npz`를 쓴다. 측정 관절 값은 목표와 같다."""
    np.savez(
        path,
        action=action,
        state=action,
        phase=np.array([phase] * len(action)),
        dof_names=np.array(FAKE_DOF_NAMES),
        fps=np.array(FAKE_FPS),
    )
