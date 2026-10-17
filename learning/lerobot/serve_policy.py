"""학습한 LeRobot 제어기를 pixel2place의 제어기 서버로 제공한다.

사용법: uv run python serve_policy.py --checkpoint PRETRAINED_MODEL_DIR [--host 127.0.0.1]
    [--port 5555] [--action-steps 15] [--device cuda]

`eval-policy --policy HOST:PORT`의 실행기가 이 서버에 접속한다.
- 서버는 관측마다 행동 묶음을 계산하고 앞의 `--action-steps`개를 준다.
- 관측의 키와 슬롯 one-hot은 `convert.py`의 데이터셋과 같다.
"""

import argparse
import logging
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import numpy as np
import torch
from convert import FRONT_KEY, ROBOT_TYPE, SLOT_KEY, STATE_KEY, WRIST_KEY
from lerobot.configs.policies import PreTrainedConfig
from lerobot.policies.factory import get_policy_class, make_pre_post_processors
from lerobot.policies.utils import prepare_observation_for_inference

from pixel2place.cell.episodes import SLOTS
from pixel2place.cell.policy_link import LOOPBACK_HOST, Observation, PolicyServer

logger = logging.getLogger("serve_policy")

DEFAULT_PORT = 5555
# 묶음에서 실행하는 프레임 수. 30 fps에서 15프레임은 0.5초다.
DEFAULT_ACTION_STEPS = 15


class LeRobotPolicy:
    """체크포인트의 제어기로 관절 목표의 묶음을 낸다."""

    def __init__(self, checkpoint: Path, *, device: str, action_steps: int) -> None:
        config = PreTrainedConfig.from_pretrained(checkpoint)
        config.device = device
        self.device = torch.device(device)
        self.action_steps = action_steps
        self.policy = get_policy_class(config.type).from_pretrained(checkpoint, config=config)
        self.policy.to(self.device).eval()
        self.preprocess, self.postprocess = make_pre_post_processors(
            config, pretrained_path=str(checkpoint)
        )
        self.slot = np.zeros(len(SLOTS), dtype=np.float32)
        self.instruction = ""

    def reset(self, context: Mapping[str, Any]) -> int | None:
        self.policy.reset()
        self.slot = np.zeros(len(SLOTS), dtype=np.float32)
        self.slot[int(context["slot"])] = 1.0
        self.instruction = str(context["instruction"])
        logger.info("Episode started. context=%s", dict(context))
        return None

    def act(self, observation: Observation) -> np.ndarray:
        # 받은 배열은 읽기 전용이다. torch 텐서로 바꾸기 전에 복사한다.
        frame = prepare_observation_for_inference(
            {
                FRONT_KEY: observation.front.copy(),
                WRIST_KEY: observation.wrist.copy(),
                STATE_KEY: observation.state.copy(),
                SLOT_KEY: self.slot.copy(),
            },
            self.device,
            task=self.instruction,
            robot_type=ROBOT_TYPE,
        )
        with torch.inference_mode():
            chunk = self.policy.predict_action_chunk(self.preprocess(frame))
            actions = self.postprocess(chunk)
        return actions[0, : self.action_steps].to(torch.float32).cpu().numpy()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--host", default=LOOPBACK_HOST)
    parser.add_argument("--port", type=int, default=DEFAULT_PORT)
    parser.add_argument("--action-steps", type=int, default=DEFAULT_ACTION_STEPS)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()
    # lerobot의 import가 root logger에 handler를 붙인다. force로 그 설정을 바꾼다.
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s", force=True
    )
    policy = LeRobotPolicy(args.checkpoint, device=args.device, action_steps=args.action_steps)
    server = PolicyServer(policy, host=args.host, port=args.port)
    logger.info(
        "Controller server started. address=%s checkpoint=%s", server.address, args.checkpoint
    )
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        server.close()


if __name__ == "__main__":
    main()
