"""Isaac 실행기(`sim/place_task.py`)를 흉내 내는 스크립트.

가짜 `python.sh`가 실제 실행기 경로와 인자를 그대로 넘긴다. 첫 인자(실행기 경로)는 쓰지 않는다.
`FAKE_ISAAC_MODE` 환경 변수로 동작을 고른다.

- `ok`: 모든 부품을 목표 자세에 놓은 `result.json`을 쓴다.
- `miss`: 첫 부품을 목표에서 x로 `MISS_OFFSET_M`만큼 옮긴 `result.json`을 쓴다.
  놓은 직후 자세도 같다.
- `crash`: traceback을 출력하고 결과 없이 exit 0으로 끝난다. Isaac `app.close()`와 같은 동작이다.
- `hang`: 출력 없이 오래 멈춘다.
- `linger`: 결과와 끝 로그를 쓴 뒤 오래 멈춘다. Isaac이 종료 단계에서 멈춘 경우와 같다.
- `wrong_version`: 작업 버전이 다른 배치 결과를 쓴다.
- `missing_part`: 첫 부품이 없는 배치 결과를 쓴다.
- `invalid_result`: 배치 결과에 유효하지 않은 JSON을 쓴다.
- `no_final_image`: `ok`와 같은 결과를 쓴다. 최종 이미지는 쓰지 않는다.
- `bad_final_image`: `ok`와 같은 결과를 쓴다. 최종 이미지 파일에는 이미지가 아닌 내용을 쓴다.
- `no_orientation`: `ok`와 같은 결과에서 부품의 `orientation`을 뺀다. 결과 버전 0.3의 실행기와 같다.
- `no_present_video`: `ok`와 같은 결과를 쓴다. `--present`가 있어도 발표 영상은 쓰지 않는다.

`--scene`이 있으면 `scene.json`의 `board`를 `result.json`에 그대로 쓴다.
부품은 task의 `target_center`(믿는 보드의 목표)에 놓는다. `scene.json`이 아래를 바꾼다.
- `faults.missing`의 부품은 작업대 아래의 대기 위치에 있다. 최종 이미지에 그리지 않는다.
- `faults.wrong_color`의 부품은 목표에 있다. 최종 이미지에 그 색으로 그린다.
- `preplaced`의 부품은 주어진 위치와 quaternion에 있다. 실제 실행기의 재작업 실행과 같다.

`ok`와 `miss`는 배치가 끝난 보드의 합성 이미지를 `topdown_final.png`로 쓴다.
`result.json`보다 먼저 쓴다. 실제 실행기와 같은 순서다.
- 보드는 `scene.json`의 `board`, 카메라는 공칭 자세에 `camera`를 적용한 자세다.
- 부품은 결과의 위치와 yaw에 `block_cells`의 모양으로 그린다.
- 부품의 색은 판독기의 태양광 기준 색이다. task의 `color_rgb`가 아니다.
- 떨어진 부품(`fell`)은 그리지 않는다.
- 팔과 팔 그림자는 그리지 않는다.

`--record`가 있으면 `episode.npz`와 영상 파일 2개도 쓴다. 영상 파일은 내용이 없다.
`episode.npz`의 내용은 `fake_episode`가 정한다.
`--present`가 있으면 `present.mp4`도 쓴다. 이 파일도 내용이 없다.

`--policy`가 있으면 제어기 서버에 접속한다. 첫 프레임의 목표는 0이다.
그 뒤로 프레임 수의 한도까지 서버가 준 관절 목표를 따른다. 관측의 이미지는 작은 검은 이미지다.
`episode.npz`에는 따른 목표를 쓴다. 제어기가 준 목표는 부품의 자세를 바꾸지 않는다.
서버가 오류로 답하면 결과를 쓰지 않고 끝난다.

`--capture`는 `topdown_renderer`의 합성 이미지를 `topdown_start.png`로 쓴다.
보드는 `scene.json`의 `board`, 카메라는 공칭 자세에 `camera`를 적용한 자세다.
`camera.json`에는 공칭 카메라를, `camera_truth.json`에는 실제 카메라를 쓴다.
"""

import argparse
import json
import os
import sys
import time
from pathlib import Path

import numpy as np
from PIL import Image

from pixel2place.cell.capture import CaptureTask
from pixel2place.cell.config import CellConfig
from pixel2place.cell.image_check import SHADOW_RGB, SUNLIT_RGB
from pixel2place.cell.policy_link import DOF_NAMES, Observation, PolicyAddress, PolicyClient
from pixel2place.cell.scene import SceneFaults, SimScene
from pixel2place.layout.schema import Board, Color
from tests.fakes.fake_episode import fake_action, write_episode
from tests.fakes.topdown_renderer import PartFace, nominal_camera, render_topdown, shaken

HANG_S = 60
# 20 mm 셀의 위치 한계(4 mm)를 넘는 거리(m)
MISS_OFFSET_M = 0.01
FAKE_IMAGE_SIZE = 4
POLICY_TIMEOUT_S = 10.0
FINAL_IMAGE_FILE = "topdown_final.png"
# 최종 이미지를 그리는 모드
FINAL_IMAGE_MODES = ("ok", "miss", "no_orientation", "no_present_video")
BAD_IMAGE_BYTES = b"This is not an image."
# 공급되지 않은 부품의 자리. 작업대 밖의 대기 위치에서 떨어진 높이다.
PARKED_XY = (-0.6, -0.6)
FALLEN_Z = -1.0
FELL_BELOW_M = -0.05
IDENTITY_WXYZ = [1.0, 0.0, 0.0, 0.0]


def final_image(task: dict, poses: list[dict], *, scene: SimScene) -> Image.Image:
    """배치가 끝난 보드의 합성 top-down 이미지. 부품은 `poses`의 자세에 그린다.

    자세의 yaw는 실제 보드의 u축에서 잰 값이다. 실행기의 `result.json`과 같은 기준이다.
    렌더는 화소마다 광선 1개를 쓴다. 가짜 실행을 쓰는 테스트가 많아서 시간을 줄인다.
    """
    board_yaw = task["board_yaw"] if scene.board is None else scene.board.board_yaw
    supplied = {item.sequence: item.color for item in (scene.faults or SceneFaults()).wrong_color}
    faces = []
    for part, pose in zip(task["parts"], poses, strict=True):
        if pose["fell"]:
            continue
        color = supplied.get(part["sequence"], Color(part["color"]))
        faces.append(
            PartFace(
                center=(pose["position"][0], pose["position"][1]),
                yaw=board_yaw + pose["yaw"],
                block_cells=tuple((i, j) for i, j in part["block_cells"]),
                rgb=SUNLIT_RGB[color],
                shadow_rgb=SHADOW_RGB[color],
            )
        )
    return render_topdown(
        shaken(nominal_camera(), scene.camera),
        cell=CellConfig.model_validate(task["cell"]),
        board=Board.model_validate(task["board"]),
        placement=scene.board,
        supersample=1,
        faces=faces,
    )


def yaw_quaternion(yaw: float) -> list[float]:
    """z축 회전의 quaternion (w, x, y, z)."""
    return [float(np.cos(yaw / 2)), 0.0, 0.0, float(np.sin(yaw / 2))]


def quaternion_yaw_tilt(wxyz: list[float]) -> tuple[float, float]:
    """quaternion의 z축 둘레 회전(rad)과 부품 z축의 기울기(rad)."""
    w, x, y, z = wxyz
    yaw = float(np.arctan2(2 * (w * z + x * y), 1 - 2 * (y * y + z * z)))
    tilt = float(np.arccos(np.clip(1 - 2 * (x * x + y * y), -1.0, 1.0)))
    return yaw, tilt


def part_poses(task: dict, scene: SimScene) -> list[dict]:
    """부품마다의 최종 자세. `result.json`의 부품 형식에서 `part_id`와 `placed`를 뺀 값이다."""
    board_yaw = task["board_yaw"] if scene.board is None else scene.board.board_yaw
    # 보드에 앉은 부품의 중심 높이
    seated_z = task["cell"]["table_z"] + task["cell"]["part_height"] / 2
    missing = set((scene.faults or SceneFaults()).missing)
    preplaced = {part.sequence: part for part in scene.preplaced or ()}
    poses = []
    for part in task["parts"]:
        restored = preplaced.get(part["sequence"])
        if restored is not None:
            yaw, tilt = quaternion_yaw_tilt(list(restored.orientation))
            position, orientation = list(restored.position), list(restored.orientation)
            yaw -= board_yaw
        elif part["sequence"] in missing:
            position, yaw, tilt = [*PARKED_XY, FALLEN_Z], 0.0, 0.0
            orientation = IDENTITY_WXYZ
        else:
            position, yaw, tilt = [*part["target_center"], seated_z], part["target_yaw"], 0.0
            orientation = yaw_quaternion(board_yaw + yaw)
        poses.append(
            {
                "position": position,
                "yaw": yaw,
                "tilt": tilt,
                "fell": position[2] < FELL_BELOW_M,
                "orientation": orientation,
            }
        )
    return poses


parser = argparse.ArgumentParser()
parser.add_argument("script")
mode_group = parser.add_mutually_exclusive_group(required=True)
mode_group.add_argument("--task", type=Path)
mode_group.add_argument("--capture", type=Path)
parser.add_argument("--usd", type=Path, required=True)
parser.add_argument("--out-dir", type=Path, required=True)
parser.add_argument("--scene", type=Path)
parser.add_argument("--record", action="store_true")
parser.add_argument("--present", action="store_true")
parser.add_argument("--policy")
parser.add_argument("--policy-context", type=Path)
parser.add_argument("--max-steps", type=int, default=300)
args = parser.parse_args()

mode = os.environ.get("FAKE_ISAAC_MODE", "ok")
print("[ext: omni.fake] startup", flush=True)
if mode == "crash":
    print("Traceback (most recent call last):\nRuntimeError: Fake runner failed.", flush=True)
    sys.exit(0)
if mode == "hang":
    time.sleep(HANG_S)
if args.capture is not None:
    capture = CaptureTask.model_validate_json(args.capture.read_text())
    scene = SimScene.model_validate_json(args.scene.read_text()) if args.scene else SimScene()
    nominal = nominal_camera()
    truth = shaken(nominal, scene.camera)
    image = render_topdown(truth, cell=capture.cell, board=capture.board, placement=scene.board)
    image.save(args.out_dir / "topdown_start.png")
    (args.out_dir / "camera.json").write_text(nominal.model_dump_json())
    (args.out_dir / "camera_truth.json").write_text(truth.model_dump_json())
    print("[place_task] Capture finished.", flush=True)
    sys.exit(0)
task = json.loads(args.task.read_text())
scene = json.loads(args.scene.read_text()) if args.scene else {}
poses = part_poses(task, SimScene.model_validate(scene))
if mode == "miss":
    poses[0]["position"][0] += MISS_OFFSET_M
if mode == "no_orientation":
    poses = [{key: value for key, value in pose.items() if key != "orientation"} for pose in poses]
parts = [
    {"part_id": part["part_id"], **pose, "placed": pose}
    for part, pose in zip(task["parts"], poses, strict=True)
]
result = {
    "result_version": "0.4",
    "task_version": task["task_version"],
    "sim_time_s": 1.0,
    "max_arm_error_rad": 0.01,
    "board": scene.get("board"),
    "parts": parts,
}
if mode == "wrong_version":
    result["task_version"] = "wrong"
if mode == "missing_part":
    result["parts"] = parts[1:]
episode_action = fake_action()
if args.policy is not None:
    image = np.zeros((FAKE_IMAGE_SIZE, FAKE_IMAGE_SIZE, 3), dtype=np.uint8)
    followed = [np.zeros(len(DOF_NAMES), dtype=np.float32)]
    context = json.loads(args.policy_context.read_text()) if args.policy_context else {}
    with PolicyClient(PolicyAddress.parse(args.policy), timeout_s=POLICY_TIMEOUT_S) as client:
        client.hello()
        limit = min(args.max_steps, client.reset(context) or args.max_steps)
        while len(followed) < limit:
            observation = Observation(
                step=len(followed) - 1, state=followed[-1], front=image, wrist=image
            )
            followed += list(client.act(observation)[: limit - len(followed)])
    episode_action = np.array(followed)
if mode in FINAL_IMAGE_MODES:
    final_image(task, poses, scene=SimScene.model_validate(scene)).save(
        args.out_dir / FINAL_IMAGE_FILE
    )
if mode == "bad_final_image":
    (args.out_dir / FINAL_IMAGE_FILE).write_bytes(BAD_IMAGE_BYTES)
payload = "invalid json" if mode == "invalid_result" else json.dumps(result)
(args.out_dir / "result.json").write_text(payload)
if args.record or args.policy is not None:
    phase = "move" if args.policy is None else "policy"
    write_episode(args.out_dir / "episode.npz", episode_action, phase=phase)
    for name in ("cell.mp4", "wrist.mp4"):
        (args.out_dir / name).write_bytes(b"")
if args.present and mode != "no_present_video":
    (args.out_dir / "present.mp4").write_bytes(b"")
print(f"[place_task] Task finished. parts={len(parts)}", flush=True)
if mode == "linger":
    time.sleep(HANG_S)
