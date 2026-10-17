"""Isaac Sim 셀에서 `task.json`의 pick-and-place를 실행한다.

Isaac `python.sh`로 실행한다. pixel2place 패키지를 import하지 않는다. `task.json` 형식은
pixel2place의 `cell.task.Task` 모델이 정한다.

단계:
1. 작업대, 보드, 보드 핀, 공급대, SO-101, 조명, 카메라와 배경 소품으로 장면을 만든다.
2. 부품 블록을 모두 만들고 카메라 밖에 둔다.
3. 부품마다 블록을 공급대로 옮기고 waypoint를 차례로 따라간다.
   공급대에 앞 부품이 남아 있으면 먼저 대기 위치로 치운다.
4. 부품을 놓을 때마다 top-down 이미지를 찍고 단계별 오차를 로그에 쓴다.
5. 모든 부품의 최종 자세와 놓은 직후 자세를 `result.json`에 쓴다.

사선 카메라 프레임은 ffmpeg로 `cell.mp4`에 쓴다.

`--record`는 시연 기록이다. 손목 카메라를 더하고 `wrist.mp4`와 `episode.npz`를 쓴다.
`episode.npz`는 프레임마다 준 관절 목표(`action`), 측정 관절 값(`state`)과 단계(`phase`)다.
`--record`가 없으면 실행기는 손목 카메라를 만들지 않고 관절 값을 프레임마다 읽지 않는다.

`--present`는 발표 영상이다. 실행기는 발표용 카메라를 더한다. 이 카메라의 1280×720 영상을
`present.mp4`에 쓴다. 제어기의 관측과 판정은 이 카메라를 쓰지 않는다.

`--policy HOST:PORT`는 폐루프 실행이다. waypoint 대신 제어기 서버가 낸 관절 목표를 프레임마다 준다.
부품이 1개인 작업만 받는다. `--record`를 함께 켠다. 메시지 형식은 pixel2place의
`cell.policy_link`가 정한다. 시뮬레이션은 제어기의 응답을 기다리는 동안 멈춘다.

`scene.json`의 `joint_error`는 관절 센서 오차다. 행은 프레임, 열은 `ARM_DOFS`의 관절이다.
측정 관절 값은 실제 각도에서 오차를 뺀 값이다. drive는 관절 목표에 오차를 더한 각도로 간다.
작업을 시작할 때 팔의 실제 자세는 접근 자세다. 캡처는 이 오차를 쓰지 않는다.
형식은 pixel2place의 `cell.scene` 모델이 정한다.

`scene.json`의 `faults`는 부품의 결함이다. 부품은 `sequence`로 지정한다.
- `missing`: 부품을 공급대로 옮기지 않는다. 팔은 빈 공급대에서 waypoint를 따라간다.
- `wrong_color`: 부품을 `color_rgb`의 색으로 만든다.

`scene.json`의 `preplaced`는 이미 놓인 부품이다. 재작업 실행이 쓴다.
실행기는 그 부품을 주어진 위치와 quaternion에 두고 건너뛴다. 다른 부품만 공급대에서 놓는다.
이 실행은 공급대 위에 있는 다른 부품을 모두 치운다. 놓인 부품의 `placed`는 복원한 자세다.
실행기는 `faults`와 `preplaced`를 위해 prim을 더 만들지 않는다.

`--capture`는 보드 캘리브레이션 캡처다. 부품 없이 장면을 만들고 팔을 `capture.json`의 자세에 둔다.
조명은 돔 조명만 쓴다. 그 다음 top-down 이미지 1장과 카메라 파일 2개를 쓴다.
형식은 pixel2place의 `cell.capture`, `cell.camera` 모델이 정한다.

사용법:
- python.sh sim/place_task.py --task TASK_JSON --usd USD --out-dir DIR [--scene SCENE_JSON]
  [--record] [--present] [--policy HOST:PORT [--policy-context CONTEXT_JSON] [--max-steps N]]
- python.sh sim/place_task.py --capture CAPTURE_JSON --usd USD --out-dir DIR [--scene SCENE_JSON]
출력:
- 작업: DIR/result.json, DIR/cell.mp4, DIR/topdown_<순서>.png, DIR/topdown_final.png
- 기록: 작업의 출력과 DIR/wrist.mp4, DIR/episode.npz
- 발표: 작업의 출력과 DIR/present.mp4
- 캡처: DIR/topdown_start.png, DIR/camera.json(공칭 카메라), DIR/camera_truth.json(실제 카메라)
"""

import argparse
import json
import socket
import struct
import subprocess
import time
import traceback
from pathlib import Path
from typing import NamedTuple

from isaacsim import SimulationApp

STARTED = time.monotonic()
parser = argparse.ArgumentParser()
mode = parser.add_mutually_exclusive_group(required=True)
mode.add_argument("--task", type=Path)
# 캘리브레이션 캡처(`capture.json`). 부품 없이 top-down 이미지 1장을 찍는다.
mode.add_argument("--capture", type=Path)
parser.add_argument("--usd", type=Path, required=True)
parser.add_argument("--out-dir", type=Path, required=True)
# 보드와 top-down 카메라의 실제 자세, 그리고 관절 센서 오차(`scene.json`).
# 없으면 보드를 셀 설정에, 카메라를 공칭 자세에 만든다. 관절 센서 오차는 없다.
parser.add_argument("--scene", type=Path)
parser.add_argument("--width", type=int, default=640)
parser.add_argument("--height", type=int, default=480)
parser.add_argument("--fps", type=int, default=30)
parser.add_argument("--settle-frames", type=int, default=60)
# 시연 기록. 손목 카메라 영상과 프레임별 관절 값을 더 쓴다.
parser.add_argument("--record", action="store_true")
# 발표 영상. 발표용 카메라의 영상을 더 쓴다.
parser.add_argument("--present", action="store_true")
# 폐루프 실행. 제어기 서버의 주소, 에피소드의 문맥(JSON)과 프레임 수의 한도다.
parser.add_argument("--policy")
parser.add_argument("--policy-context", type=Path)
parser.add_argument("--max-steps", type=int, default=300)
args = parser.parse_args()
if args.policy is not None:
    args.record = True
if (args.record or args.present) and args.task is None:
    parser.error("--record, --present and --policy need --task.")

# SimulationApp은 다른 omni·pxr import보다 먼저 만들어야 한다.
app = SimulationApp(
    {"headless": True, "width": args.width, "height": args.height, "renderer": "RayTracedLighting"}
)

import numpy as np  # noqa: E402
import omni.replicator.core as rep  # noqa: E402
import omni.timeline  # noqa: E402
import omni.usd  # noqa: E402
from isaacsim.core.experimental.prims import Articulation, RigidPrim, XformPrim  # noqa: E402
from isaacsim.core.simulation_manager import SimulationManager  # noqa: E402
from PIL import Image  # noqa: E402
from pxr import Gf, PhysxSchema, Sdf, UsdGeom, UsdLux, UsdPhysics, UsdShade  # noqa: E402

RESULT_VERSION = "0.4"
ROBOT_PATH = "/World/so101"
ARTICULATION_ROOT_PATH = f"{ROBOT_PATH}/Geometry/base_link"
GRIPPER_PATH = (
    f"{ARTICULATION_ROOT_PATH}/shoulder_link/upper_arm_link/lower_arm_link/wrist_link/gripper_link"
)
# 툴 끝점 link. 단계별 오차를 잴 때 툴 자세를 읽는다.
TOOL_PATH = f"{GRIPPER_PATH}/gripper_frame_link"
MOVING_JAW_PATH = f"{GRIPPER_PATH}/moving_jaw_so101_v1_link"
PHYSICS_VARIANT = "physx"
PHYSICS_DT_S = 1.0 / 120.0
ARM_DOFS = ("shoulder_pan", "shoulder_lift", "elbow_flex", "wrist_flex", "wrist_roll")
GRIPPER_DOF = "gripper"

# 관절 drive. 변환한 USD에는 stiffness와 damping이 없다. 단위는 N·m/rad, N·m·s/rad다.
ARM_STIFFNESS = 200.0
ARM_DAMPING = 10.0
GRIPPER_STIFFNESS = 20.0
GRIPPER_DAMPING = 1.0
# 그리퍼 drive의 최대 토크(N·m). 닫기 목표가 부품 폭보다 좁아서 이 값이 쥐는 힘을 정한다.
GRIPPER_MAX_EFFORT = 0.3

# 동작 시간. 구간 시간은 가장 크게 움직이는 관절이 정한다.
MAX_JOINT_SPEED = 1.5
MIN_SEGMENT_S = 0.4
# 닫기는 천천히 한다. 빨리 닫으면 jaw가 부품을 친다.
# 놓기는 빨리 한다. 천천히 열면 부품의 한쪽이 jaw에 오래 걸려서 부품이 기운다.
GRIPPER_CLOSE_S = 0.5
GRIPPER_OPEN_S = 0.15
GRIPPER_HOLD_S = 0.3
FINAL_SETTLE_S = 1.0
# 구간이 끝난 뒤 관절 오차가 이 값 아래로 줄 때까지 기다린다.
# 기다리는 시간은 최대 SETTLE_TIMEOUT_S다.
# 보드 위에서 내려갈 때 잡은 부품은 이웃 윗면보다 낮다. 그래서 옆으로 움직이면 이웃에 부딪힌다.
SETTLE_TOLERANCE_RAD = 0.0035
SETTLE_TIMEOUT_S = 1.0
# teleport한 자세가 렌더에 반영될 때까지 기록하지 않고 진행하는 update 수
TELEPORT_SETTLE_UPDATES = 10

# 마찰과 반발. 로봇, 부품, 공급대, 작업대에 같은 재질을 쓴다.
# jaw 마찰을 낮추면 빠른 이동 구간에서 부품이 미끄러져 빠진다.
STATIC_FRICTION = 1.0
DYNAMIC_FRICTION = 0.8
RESTITUTION = 0.0
# 보드와 핀의 재질. 마찰을 min으로 결합해서 부품과의 쌍이 이 값이 된다.
# 부품이 핀 끝에 닿으면 기울고 반대쪽 아래 모서리가 보드에 닿는다.
# 보드 마찰이 1.0이면 보드가 그 모서리를 붙잡아서 부품이 핀 끝에 걸린다.
BOARD_STATIC_FRICTION = 0.3
BOARD_DYNAMIC_FRICTION = 0.25
PLA_DENSITY = 1240.0
# contact offset. 두 collider 사이 거리가 두 값의 합보다 작으면 PhysX가 접촉을 미리 만든다.
# 부품 두 개의 합은 이웃 블록 사이 틈(기본 3 mm)보다 크면 안 된다.
# 합이 틈보다 크면 틈 사이로 내려가는 부품과 이웃이 서로 민다.
# 그러면 부품이 걸리거나 이웃이 튕겨 나간다.
PART_CONTACT_OFFSET_M = 0.001
JAW_CONTACT_OFFSET_M = 0.002
PIN_CONTACT_OFFSET_M = 0.0005
REST_OFFSET_M = 0.0
# articulation과 부품 강체의 solver 위치 반복 횟수. 잡은 동안 팔 처짐은 solver 잔차다.
# 반복을 늘리면 처짐이 준다.
SOLVER_POSITION_ITERATIONS = 128

# jaw collider. 변환한 USD는 jaw 메시를 convexHull로 근사한다. 고정 jaw 메시는 그리퍼 몸체 전체다.
# 그래서 hull의 비스듬한 면이 부품 영역 안으로 약 1 mm 들어와서 부품을 아래로 민다.
# 실행기는 jaw 메시 2개의 collision을 끄고 볼록 collider를 붙인다.
# (instanceable 부모 Xform, jaw 메시 이름)
JAW_MESHES = (
    (f"{GRIPPER_PATH}/wrist_roll_follower_so101_v1_1", "wrist_roll_follower_so101_v1"),
    (f"{MOVING_JAW_PATH}/moving_jaw_so101_v1_1", "moving_jaw_so101_v1"),
)
# 고정 jaw 손끝은 툴 좌표(그리퍼 0°)의 (x, z) 단면을 y 범위만큼 민 볼록 기둥이다(mm).
# 툴 z축은 jaw가 향하는 방향이다.
# - 안쪽 면(x = 0)은 메시 면과 같다. 면이 튀어나오면 내려갈 때 부품 윗면에 얹힌다.
# - 안쪽 아래 모서리는 메시의 둥근 모서리(반지름 약 0.8 mm)를 따라 깎는다.
#   모서리가 각지면 놓을 때 떨어지는 부품의 윗모서리가 여기에 걸린다.
#   그러면 부품이 돌면서 툴 +x로 밀린다.
FIXED_JAW_TIP_PROFILE_MM = (
    (0.0, -4.0),
    (6.5, -4.0),
    (6.5, 6.3),
    (0.75, 6.3),
    (0.3, 6.1),
    (0.0, 5.5),
)
FIXED_JAW_TIP_Y_MM = (-4.0, 4.0)
# 상자마다 툴 좌표(그리퍼 0°)의 ((x0, x1), (y0, y1), (z0, z1)) 범위(mm)다.
# - 움직이는 jaw의 첫 상자는 손끝이다. 안쪽 면은 메시 면과 같다.
# - 나머지 상자는 몸체다. 파지할 때 부품 윗면(툴 z 약 2 mm)보다 위에만 있다.
FIXED_JAW_BOXES_MM = (
    ((2.0, 13.1), (-7.4, 7.4), (-16.0, -4.0)),
    ((3.7, 20.8), (-9.0, 9.0), (-37.0, -16.0)),
    ((5.4, 27.3), (-14.2, 14.2), (-58.0, -37.0)),
    ((-37.9, 27.3), (-27.8, 24.2), (-99.1, -58.0)),
)
MOVING_JAW_BOXES_MM = (
    ((-24.0, -15.8), (-4.5, 4.5), (-3.0, 7.27)),
    ((-32.0, -17.8), (-7.7, 7.7), (-16.0, -3.0)),
    ((-35.7, -19.8), (-10.8, 10.8), (-49.0, -16.0)),
    ((-38.1, -18.1), (-23.9, 24.1), (-85.0, -49.0)),
)
MM_PER_M = 1000.0
# 다각형 자르기에서 경계 위의 점을 안쪽으로 보는 허용 오차(m)
CLIP_EPSILON_M = 1e-12
# 핀 collider는 이 수의 옆면을 가진 모따기 원기둥이다.
# 꼭짓점 48개는 GPU 볼록 hull 한도(64개) 안이다.
PIN_SIDES = 16

PLATE_THICKNESS_M = 0.002
# 작업대 상자의 중심 (x, y), 한 변과 두께
TABLE_CENTER_XY = (0.2, 0.0)
TABLE_SIZE_M = 1.0
TABLE_THICKNESS_M = 0.02
BOARD_MARGIN_M = 0.01
FEEDER_PAD_SIZE_M = 0.08
GRID_LINE_WIDTH_M = 0.001
GRID_LINE_HEIGHT_M = 0.0002
BOARD_RGB = (30, 30, 34)
PIN_RGB = (168, 170, 176)
GRID_LINE_RGB = (70, 70, 76)
FEEDER_RGB = (60, 62, 68)
TABLE_RGB = (150, 152, 156)
MARKER_PLATE_RGB = (235, 235, 235)
MARKER_RGB = (20, 20, 22)
# 표식 판의 한 변은 표식 원 지름의 이 배수다.
MARKER_PLATE_RATIO = 2.0
# 표식 원의 두께. 원의 윗면이 table_z다. 판은 원 아래에서 작업대 윗면까지다.
MARKER_THICKNESS_M = 0.0002


class Look(NamedTuple):
    """시각 재질(UsdPreviewSurface) 1개의 값. `name`은 재질 prim의 이름이다."""

    name: str
    color: tuple[int, int, int]
    roughness: float
    metallic: float = 0.0
    # 참이면 정반사(specular)가 없다.
    matte: bool = False


LOOKS_PATH = "/World/Looks"
# 판정 면의 재질. 색은 `displayColor`와 같다. 캘리브레이션과 카메라 판정이 이 면을 읽는다.
# - 이 재질은 정반사가 없다. 정반사가 있으면 빈 칸의 핀 밝기 비가 준다.
#   보드 자세의 추정 오차도 는다.
# - 핀은 금속 재질을 쓰지 않는다. 금속 핀은 핀 밝기 비를 판정 한계 아래로 줄인다.
TABLE_LOOK = Look("table", TABLE_RGB, roughness=0.6, matte=True)
BOARD_LOOK = Look("board", BOARD_RGB, roughness=0.8, matte=True)
PIN_LOOK = Look("pin", PIN_RGB, roughness=0.6, matte=True)
FEEDER_LOOK = Look("feeder", FEEDER_RGB, roughness=0.7, matte=True)
# 부품의 재질은 색마다 1개다. 이 재질은 정반사가 없다. 그래서 윗면의 색이 고르다.
PART_ROUGHNESS = 0.6
# 팔의 재질. 팔 전체가 이 색이다.
ARM_LOOK = Look("arm", (214, 216, 220), roughness=0.45)
# 배경 소품의 재질. 바닥, 벽, 어두운 띠(굽도리, 이음새, 손잡이), 알루미늄 프로파일과 제어반이다.
FLOOR_LOOK = Look("floor", (52, 54, 56), roughness=0.5)
WALL_LOOK = Look("wall", (96, 100, 106), roughness=0.7)
TRIM_LOOK = Look("trim", (40, 42, 46), roughness=0.6)
PROFILE_LOOK = Look("profile", (38, 40, 44), roughness=0.4, metallic=0.3)
CABINET_LOOK = Look("cabinet", (70, 74, 80), roughness=0.5)
# 배경 소품의 치수(m). 높이는 바닥에서 잰다.
# 작업대 윗면의 높이
BENCH_HEIGHT_M = 0.75
FLOOR_SIZE_M = 6.0
FLOOR_THICKNESS_M = 0.02
# 알루미늄 프로파일의 한 변. 작업대 다리, 가로대와 셀 프레임이 쓴다.
PROFILE_M = 0.04
# 프로파일의 중심선이 작업대 모서리에서 안쪽으로 들어간 거리
PROFILE_INSET_M = 0.06
# 작업대 아래 가로대의 높이
LOWER_RAIL_HEIGHT_M = 0.12
# 셀 프레임. 뒤 기둥의 길이와 뒤 가로대의 높이는 작업대 윗면에서 잰다.
POST_LENGTH_M = 0.9
BACK_RAIL_HEIGHT_M = 0.22
# 벽. 뒷벽은 로봇 뒤(-x)에 있다. 옆벽은 +y 쪽에 있다. 굽도리와 이음새는 벽 앞면에 있다.
WALL_HEIGHT_M = 2.05
WALL_THICKNESS_M = 0.05
BACK_WALL_X = -0.75
BACK_WALL_LENGTH_M = 4.0
BACK_WALL_SEAMS_Y = (-1.2, -0.6, 0.0, 0.6, 1.2)
SIDE_WALL_CENTER_XY = (0.9, 1.8)
SIDE_WALL_LENGTH_M = 3.3
SKIRTING_HEIGHT_M = 0.12
SKIRTING_THICKNESS_M = 0.01
# 뒷벽 패널 이음새의 두께와 폭. 두께는 벽 앞면에서 나온 길이다.
WALL_SEAM_THICKNESS_M = 0.004
WALL_SEAM_WIDTH_M = 0.01
# 제어반. 뒷벽 앞의 바닥에 있다. 중심 (x, y)와 크기 (x, y, z)다.
CABINET_CENTER_XY = (-0.55, 0.95)
CABINET_SIZE_M = (0.3, 0.6, 1.2)
# 제어반 앞면의 문 이음새. 두께, 폭과 위아래의 여백이다.
CABINET_SEAM_THICKNESS_M = 0.003
CABINET_SEAM_WIDTH_M = 0.006
CABINET_SEAM_MARGIN_M = 0.05
# 제어반 앞면의 손잡이. 크기 (x, y, z), 문 이음새에서 중심까지의 거리(y)와 중심의 높이다.
CABINET_HANDLE_SIZE_M = (0.02, 0.03, 0.12)
CABINET_HANDLE_OFFSET_M = 0.06
CABINET_HANDLE_HEIGHT_M = 0.95
# 대기 중인 부품을 두는 곳. 작업대 밖이다. 부품은 여기서 작업대 아래로 떨어진다.
# top-down 카메라는 이곳을 보지 않는다.
PARKING_ORIGIN = (-0.6, -0.6)
PARKING_STEP_M = 0.08
# 공급대 중심에서 이 거리 안에 남은 앞 부품은 다음 부품을 놓기 전에 치운다.
# 보드의 가장 가까운 셀 중심은 약 90 mm 떨어져 있다.
FEEDER_CLEAR_RADIUS_M = 0.06
# 이 높이보다 낮으면 부품이 작업대 밖으로 떨어졌다.
FELL_BELOW_M = -0.05

OBLIQUE_EYE = Gf.Vec3d(0.72, -0.42, 0.46)
OBLIQUE_TARGET = Gf.Vec3d(0.15, 0.04, 0.06)
TOPDOWN_EYE = Gf.Vec3d(0.19, 0.04, 0.6)
TOPDOWN_TARGET = Gf.Vec3d(0.19, 0.04, 0.0)
# 이미지 위쪽이 -x(로봇 쪽)다. 그러면 row가 아래로, col이 오른쪽으로 간다. 레이아웃 이미지와 같다.
TOPDOWN_UP = Gf.Vec3d(-1.0, 0.0, 0.0)
CAPTURE_SUBFRAMES = 16
# 부품 실행의 조명. 돔 조명(주변광)과 태양광(방향성)이다.
DOME_INTENSITY = 250.0
SUN_INTENSITY = 900.0
SUN_ROTATION_DEG = (-40.0, 20.0, 0.0)
# 캡처 조명. 돔 조명만 쓴다. 밝기는 부품 실행의 두 조명 합과 비슷하다.
# 태양광은 핀의 한쪽 모따기 면만 밝힌다. 그러면 핀 끝의 밝은 원 중심이 약 0.6 mm 치우친다.
CAPTURE_DOME_INTENSITY = 1000.0
FOCAL_LENGTH_MM = 24.0
HORIZONTAL_APERTURE_MM = 20.955
# USD 카메라 축(y 위, z 뒤)과 OpenCV 카메라 축(y 아래, z 앞) 사이의 변환. 역행렬이 자신이다.
USD_TO_OPENCV = np.diag([1.0, -1.0, -1.0])
CAPTURE_IMAGE_NAME = "start"
# 손목 카메라. 그리퍼 link에 붙는다. 위치와 보는 점은 툴 좌표(mm)다. 툴 z축은 jaw가 향하는 방향이다.
# 카메라는 jaw 옆에서 손끝과 그 아래의 보드를 본다. 이미지 위쪽이 jaw가 향하는 방향이다.
WRIST_CAMERA_PATH = f"{GRIPPER_PATH}/wrist_camera"
WRIST_EYE_MM = (-8.5, 60.0, -70.0)
WRIST_TARGET_MM = (-8.5, 0.0, 30.0)
WRIST_UP = Gf.Vec3d(0.0, 0.0, 1.0)
WRIST_FOCAL_LENGTH_MM = 12.0
WRIST_VIDEO_FILE = "wrist.mp4"
EPISODE_FILE = "episode.npz"
# 제어기 서버와의 통신. 형식은 pixel2place의 `cell.policy_link`와 같다.
POLICY_PROTOCOL = 1
POLICY_HEADER_FORMAT = ">I"
POLICY_CONNECT_TIMEOUT_S = 10.0
# 제어기의 응답을 기다리는 시간. 시뮬레이션은 그동안 멈춘다.
POLICY_REPLY_TIMEOUT_S = 120.0
# 제어기가 준 목표의 프레임당 변화 한도(rad). 시연의 가장 빠른 구간은 프레임당 약 0.09 rad다.
POLICY_MAX_STEP_RAD = 0.15
POLICY_PHASE = "policy"
# 기록 영상의 x264 품질. 학습 영상과 평가의 원본 렌더 사이의 차이를 줄인다.
RECORD_CRF = 17
CAMERA_FILE = "camera.json"
CAMERA_TRUTH_FILE = "camera_truth.json"
# 발표용 카메라. 셀 프레임과 벽이 함께 보이는 사선 자세다. 이미지 크기는 (가로, 세로)다.
PRESENT_CAMERA_PATH = "/World/PresentCamera"
PRESENT_SIZE = (1280, 720)
PRESENT_EYE = Gf.Vec3d(0.88, -0.60, 0.50)
PRESENT_TARGET = Gf.Vec3d(0.11, 0.06, 0.13)
PRESENT_FOCAL_LENGTH_MM = 28.0
PRESENT_VIDEO_FILE = "present.mp4"
# 발표 영상의 x264 품질. 편집이 이 영상을 다시 인코딩한다.
PRESENT_CRF = 18


def log(message: str) -> None:
    print(f"[place_task] {message}", flush=True)


def rgb(values) -> Gf.Vec3f:
    return Gf.Vec3f(*(value / 255.0 for value in values))


def yaw_quaternion(yaw: float) -> np.ndarray:
    """z축 회전의 wxyz 쿼터니언."""
    return np.array([np.cos(yaw / 2), 0.0, 0.0, np.sin(yaw / 2)])


def quaternion_matrix(quaternion_wxyz) -> np.ndarray:
    """wxyz 쿼터니언의 3×3 회전 행렬. 열은 물체 x, y, z축의 world 방향이다."""
    w, x, y, z = (float(value) for value in quaternion_wxyz)
    return np.array(
        [
            [1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y)],
            [2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x)],
            [2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y)],
        ]
    )


def matrix_yaw(rotation: np.ndarray) -> float:
    """회전 행렬의 yaw(rad). 물체 x축의 수평 방위각이다."""
    return float(np.arctan2(rotation[1, 0], rotation[0, 0]))


def quaternion_yaw_tilt(quaternion_wxyz) -> tuple[float, float]:
    """쿼터니언의 (yaw, 기울기) rad. yaw는 물체 x축의 수평 방위각, 기울기는 z축과 수직의 각도다."""
    rotation = quaternion_matrix(quaternion_wxyz)
    tilt = float(np.arccos(np.clip(rotation[2, 2], -1.0, 1.0)))
    return matrix_yaw(rotation), tilt


def wrap_angle(angle: float) -> float:
    return float(np.arctan2(np.sin(angle), np.cos(angle)))


def rotation_from_vector(vector) -> np.ndarray:
    """회전 벡터(축 × 각도, rad)의 3×3 회전 행렬(Rodrigues)."""
    vector = np.asarray(vector, dtype=np.float64)
    angle = float(np.linalg.norm(vector))
    if angle == 0.0:
        return np.eye(3)
    x, y, z = vector / angle
    cross = np.array([[0.0, -z, y], [z, 0.0, -x], [-y, x, 0.0]])
    return np.eye(3) + np.sin(angle) * cross + (1 - np.cos(angle)) * (cross @ cross)


def minimum_jerk(fraction: float) -> float:
    return fraction**3 * (10 - 15 * fraction + 6 * fraction**2)


def define_material(stage, path: str, *, static: float, dynamic: float, combine: str | None = None):
    """physics 재질. `combine`은 PhysX 마찰 결합 방식이다.

    두 재질의 결합 방식이 다르면 PhysX는 우선순위가 높은 방식을 쓴다. `min`은 `average`보다 높다.
    """
    material = UsdShade.Material.Define(stage, path)
    api = UsdPhysics.MaterialAPI.Apply(material.GetPrim())
    api.CreateStaticFrictionAttr(static)
    api.CreateDynamicFrictionAttr(dynamic)
    api.CreateRestitutionAttr(RESTITUTION)
    if combine is not None:
        physx = PhysxSchema.PhysxMaterialAPI.Apply(material.GetPrim())
        physx.CreateFrictionCombineModeAttr(combine)
    return material


def bind_material(prim, material) -> None:
    UsdShade.MaterialBindingAPI.Apply(prim).Bind(
        material, UsdShade.Tokens.weakerThanDescendants, "physics"
    )


def define_look(stage, look: Look):
    """시각 재질을 `LOOKS_PATH` 아래에 만든다."""
    path = f"{LOOKS_PATH}/{look.name}"
    material = UsdShade.Material.Define(stage, path)
    shader = UsdShade.Shader.Define(stage, f"{path}/PreviewSurface")
    shader.CreateIdAttr("UsdPreviewSurface")
    shader.CreateInput("diffuseColor", Sdf.ValueTypeNames.Color3f).Set(rgb(look.color))
    shader.CreateInput("roughness", Sdf.ValueTypeNames.Float).Set(look.roughness)
    shader.CreateInput("metallic", Sdf.ValueTypeNames.Float).Set(look.metallic)
    if look.matte:
        # specular workflow에서 정반사 색이 0이면 정반사가 없다.
        shader.CreateInput("useSpecularWorkflow", Sdf.ValueTypeNames.Int).Set(1)
        shader.CreateInput("specularColor", Sdf.ValueTypeNames.Color3f).Set(Gf.Vec3f(0.0, 0.0, 0.0))
    material.CreateSurfaceOutput().ConnectToSource(shader.ConnectableAPI(), "surface")
    return material


def bind_look(stage, prim, look: Look, *, strong: bool = False) -> None:
    """시각 재질을 `prim`에 바인딩한다. 같은 이름의 재질은 1개만 만든다.

    바인딩의 purpose는 기본값이다. 이 purpose는 물리 재질의 `physics` purpose와 다르다.
    `strong`이면 이 바인딩이 자손 prim의 바인딩보다 우선한다.
    """
    material = UsdShade.Material.Get(stage, f"{LOOKS_PATH}/{look.name}")
    if not material:
        material = define_look(stage, look)
    strength = (
        UsdShade.Tokens.strongerThanDescendants if strong else UsdShade.Tokens.weakerThanDescendants
    )
    UsdShade.MaterialBindingAPI.Apply(prim).Bind(material, strength)


def set_contact_offsets(prim, contact_offset: float) -> None:
    collision = PhysxSchema.PhysxCollisionAPI.Apply(prim)
    collision.CreateContactOffsetAttr(contact_offset)
    collision.CreateRestOffsetAttr(REST_OFFSET_M)


def define_box(stage, path: str, *, center, size, yaw: float, color, collision: bool):
    """축 방향 크기가 `size`인 상자. `yaw`는 z축 회전(rad)이다."""
    box = UsdGeom.Cube.Define(stage, path)
    box.CreateSizeAttr(1.0)
    box.AddTranslateOp().Set(Gf.Vec3d(*center))
    box.AddRotateZOp().Set(float(np.degrees(yaw)))
    box.AddScaleOp().Set(Gf.Vec3f(*size))
    box.CreateDisplayColorAttr([rgb(color)])
    if collision:
        UsdPhysics.CollisionAPI.Apply(box.GetPrim())
    return box


def define_convex(
    stage, path: str, points, counts, indices, *, contact_offset: float, material, color=None
):
    """볼록 껍질 collider 메시. `color`가 없으면 보이지 않는다.

    PhysX는 점들의 볼록 껍질을 collider로 쓴다. 면(`counts`, `indices`)은 렌더에만 쓴다.
    """
    mesh = UsdGeom.Mesh.Define(stage, path)
    mesh.CreatePointsAttr([Gf.Vec3f(*(float(value) for value in point)) for point in points])
    mesh.CreateFaceVertexCountsAttr(counts)
    mesh.CreateFaceVertexIndicesAttr(indices)
    if color is None:
        # visibility는 collision에 영향이 없다.
        mesh.MakeInvisible()
    else:
        mesh.CreateDisplayColorAttr([rgb(color)])
    UsdPhysics.CollisionAPI.Apply(mesh.GetPrim())
    UsdPhysics.MeshCollisionAPI.Apply(mesh.GetPrim()).CreateApproximationAttr("convexHull")
    set_contact_offsets(mesh.GetPrim(), contact_offset)
    bind_material(mesh.GetPrim(), material)
    return mesh


def clip_polygon(points: list[np.ndarray], normal, offset: float) -> list[np.ndarray]:
    """볼록 다각형을 반평면 `normal · p <= offset`으로 자른다."""
    normal = np.asarray(normal, dtype=np.float64)
    result = []
    for index, current in enumerate(points):
        following = points[(index + 1) % len(points)]
        current_inside = normal @ current <= offset + CLIP_EPSILON_M
        following_inside = normal @ following <= offset + CLIP_EPSILON_M
        if current_inside:
            result.append(current)
        if current_inside != following_inside:
            fraction = (offset - normal @ current) / (normal @ (following - current))
            result.append(current + fraction * (following - current))
    return result


def define_camera(
    stage, path: str, *, pose, focal_length: float = FOCAL_LENGTH_MM, size=None
) -> None:
    """`pose`는 카메라 좌표를 부모 좌표로 바꾸는 Gf 행렬이다. USD 카메라는 −z를 보고 y가 위다.

    `size`는 이미지의 (가로, 세로)다. None이면 `--width`와 `--height`다.
    화소가 정사각이 되도록 세로 aperture를 가로 aperture와 이미지 비율로 정한다.
    """
    width, height = (args.width, args.height) if size is None else size
    camera = UsdGeom.Camera.Define(stage, path)
    camera.CreateFocalLengthAttr(focal_length)
    camera.CreateHorizontalApertureAttr(HORIZONTAL_APERTURE_MM)
    camera.CreateVerticalApertureAttr(HORIZONTAL_APERTURE_MM * height / width)
    # 기본 clipping range는 (1, 1e6)이다. 1 m보다 가까운 물체가 잘리므로 near를 1 cm로 둔다.
    camera.CreateClippingRangeAttr(Gf.Vec2f(0.01, 100.0))
    camera.AddTransformOp().Set(pose)


def look_at(eye, target, up):
    """`eye`에서 `target`을 보는 카메라 자세.

    SetLookAt은 view 행렬을 만든다. 자세는 그 역행렬이다.
    """
    return Gf.Matrix4d().SetLookAt(eye, target, up).GetInverse()


def topdown_pose(offset: dict | None) -> tuple[np.ndarray, np.ndarray]:
    """top-down 카메라의 (OpenCV 축 회전 행렬, 중심).

    공칭 자세에 `scene.json`의 `camera`를 적용한다. `offset`이 None이면 공칭 자세다.
    회전 행렬의 열은 OpenCV 카메라 x, y, z축의 world 방향이다.
    """
    pose = look_at(TOPDOWN_EYE, TOPDOWN_TARGET, TOPDOWN_UP)
    # Gf 행렬은 행 벡터 규약이다. 3×3 블록의 행이 USD 카메라 축의 world 방향이다.
    rotation = np.array([[pose[row][col] for col in range(3)] for row in range(3)]).T
    rotation = rotation @ USD_TO_OPENCV
    center = np.array([pose[3][col] for col in range(3)])
    if offset is None:
        return rotation, center
    moved = rotation_from_vector(offset["rotation"]) @ rotation
    return moved, center + np.array(offset["position"])


def gf_pose(rotation: np.ndarray, center: np.ndarray):
    """`topdown_pose`의 (회전 행렬, 중심)을 USD 카메라의 Gf 자세 행렬로 바꾼다."""
    rows = (rotation @ USD_TO_OPENCV).T
    return Gf.Matrix4d(
        *(float(value) for value in rows[0]), 0.0,
        *(float(value) for value in rows[1]), 0.0,
        *(float(value) for value in rows[2]), 0.0,
        *(float(value) for value in center), 1.0,
    )  # fmt: skip


def camera_info(rotation: np.ndarray, center: np.ndarray) -> dict:
    """`camera.json`의 내용. 화소 좌표는 화소 중심이 정수다."""
    focal = FOCAL_LENGTH_MM / HORIZONTAL_APERTURE_MM * args.width
    return {
        "width": args.width,
        "height": args.height,
        "fx": focal,
        "fy": focal,
        "cx": (args.width - 1) / 2,
        "cy": (args.height - 1) / 2,
        "rotation": rotation.tolist(),
        "position": center.tolist(),
    }


class Cell:
    """`task.json`의 셀 기하와 보드의 실제 자세.

    `placement`는 `scene.json`의 `board`다. None이면 보드는 task의 셀 설정에 있다.
    보드 판, 격자, 핀, 부품 yaw 기준은 실제 보드를 쓴다. 공급대는 task의 셀 설정을 쓴다.
    """

    def __init__(self, task: dict, placement: dict | None) -> None:
        """`task`는 `task.json`이나 `capture.json`이다. 두 형식에 셀 기하 필드가 같이 있다."""
        cell = task["cell"]
        board = task["board"]
        self.cell_size = board["cell_size_mm"] / 1000.0
        self.rows, self.cols = board["rows"], board["cols"]
        self.placement = placement
        # 계획의 보드. 목표 위치를 실제 보드로 옮길 때 쓴다.
        self.planned_origin = np.array(cell["board_origin"])
        self.planned_axes = np.column_stack([cell["col_direction"], cell["row_direction"]])
        actual = cell if placement is None else placement
        self.origin = np.array(actual["board_origin"])
        self.row_direction = np.array(actual["row_direction"])
        self.col_direction = np.array(actual["col_direction"])
        self.feeder_center = np.array(cell["feeder_center"])
        self.table_z = cell["table_z"]
        self.part_height = cell["part_height"]
        self.part_inset = cell["part_inset"]
        self.part_chamfer = cell["part_chamfer"]
        self.locator = cell["locator"]
        if self.locator is None:
            self.hole_half_width = self.mouth_half_width = 0.0
        else:
            self.hole_half_width = self.locator["hole_width"] / 2
            self.mouth_half_width = self.hole_half_width + self.locator["hole_chamfer"]
        # 부품 아래 고리의 높이. 구멍과 바깥 모따기가 이 안에 있다. 0이면 부품 셀이 상자 1개다.
        self.ring_height = self.part_chamfer if self.locator is None else self.locator["hole_depth"]
        self.board_yaw = task["board_yaw"] if placement is None else placement["board_yaw"]
        self.feeder_yaw = task["feeder_yaw"]
        self.markers = [np.array(marker) for marker in cell["markers"]]
        self.marker_diameter = cell["marker_diameter"]

    def board_point(self, row: float, col: float) -> np.ndarray:
        return self.origin + (self.row_direction * row + self.col_direction * col) * self.cell_size

    def placed_point(self, point) -> np.ndarray:
        """계획의 보드 위 점을 실제 보드의 같은 격자 위치로 옮긴다."""
        point = np.asarray(point, dtype=np.float64)
        if self.placement is None:
            return point
        uv = np.linalg.solve(self.planned_axes, point - self.planned_origin)
        return self.origin + np.column_stack([self.col_direction, self.row_direction]) @ uv


def build_scene(stage, cell: Cell, material, board_material, *, topdown, diffuse: bool) -> None:
    """`topdown`은 top-down 카메라의 Gf 자세 행렬이다. `diffuse`이면 태양광 없이 돔 조명만 쓴다.

    격자선은 시각 재질이 없다. 격자선의 색은 `displayColor`다.
    """
    UsdGeom.SetStageUpAxis(stage, UsdGeom.Tokens.z)
    UsdGeom.SetStageMetersPerUnit(stage, 1.0)
    robot = stage.DefinePrim(ROBOT_PATH, "Xform")
    robot.GetReferences().AddReference(str(args.usd))
    # 변환한 USD는 Physics variant를 고르지 않는다. 고르지 않으면 물리와 articulation이 없다.
    robot.GetVariantSets().GetVariantSet("Physics").SetVariantSelection(PHYSICS_VARIANT)
    bind_material(robot, material)
    # 팔 부품의 재질은 USD의 instance 안에 있다. 실행기는 그 재질을 고칠 수 없다.
    # 그래서 팔 전체에 재질 1개를 바인딩한다.
    bind_look(stage, robot, ARM_LOOK, strong=True)
    replace_jaw_colliders(stage, material)
    articulation = PhysxSchema.PhysxArticulationAPI.Apply(
        stage.GetPrimAtPath(ARTICULATION_ROOT_PATH)
    )
    articulation.CreateSolverPositionIterationCountAttr(SOLVER_POSITION_ITERATIONS)

    # 작업대 윗면은 보드 판 아래에 있다. 보드와 공급대 판의 윗면이 table_z다.
    plate_z = cell.table_z - PLATE_THICKNESS_M / 2
    table_top = cell.table_z - PLATE_THICKNESS_M
    table = define_box(
        stage,
        "/World/Table",
        center=(*TABLE_CENTER_XY, table_top - TABLE_THICKNESS_M / 2),
        size=(TABLE_SIZE_M, TABLE_SIZE_M, TABLE_THICKNESS_M),
        yaw=0.0,
        color=TABLE_RGB,
        collision=True,
    )
    bind_material(table.GetPrim(), material)
    bind_look(stage, table.GetPrim(), TABLE_LOOK)

    center = cell.board_point(cell.rows / 2, cell.cols / 2)
    width_u = cell.cols * cell.cell_size + 2 * BOARD_MARGIN_M
    width_v = cell.rows * cell.cell_size + 2 * BOARD_MARGIN_M
    board = define_box(
        stage,
        "/World/Board",
        center=(*center, plate_z),
        size=(width_u, width_v, PLATE_THICKNESS_M),
        yaw=cell.board_yaw,
        color=BOARD_RGB,
        collision=True,
    )
    bind_material(board.GetPrim(), board_material)
    bind_look(stage, board.GetPrim(), BOARD_LOOK)
    line_z = cell.table_z + GRID_LINE_HEIGHT_M / 2
    for index in range(cell.cols + 1):
        middle = cell.board_point(cell.rows / 2, index)
        define_box(
            stage,
            f"/World/Grid/col_{index}",
            center=(*middle, line_z),
            size=(GRID_LINE_WIDTH_M, cell.rows * cell.cell_size, GRID_LINE_HEIGHT_M),
            yaw=cell.board_yaw,
            color=GRID_LINE_RGB,
            collision=False,
        )
    for index in range(cell.rows + 1):
        middle = cell.board_point(index, cell.cols / 2)
        define_box(
            stage,
            f"/World/Grid/row_{index}",
            center=(*middle, line_z),
            size=(cell.cols * cell.cell_size, GRID_LINE_WIDTH_M, GRID_LINE_HEIGHT_M),
            yaw=cell.board_yaw,
            color=GRID_LINE_RGB,
            collision=False,
        )
    feeder = define_box(
        stage,
        "/World/Feeder",
        center=(*cell.feeder_center, plate_z),
        size=(FEEDER_PAD_SIZE_M, FEEDER_PAD_SIZE_M, PLATE_THICKNESS_M),
        yaw=cell.feeder_yaw,
        color=FEEDER_RGB,
        collision=True,
    )
    bind_material(feeder.GetPrim(), material)
    bind_look(stage, feeder.GetPrim(), FEEDER_LOOK)
    if cell.locator is not None:
        # 보드 셀 중심마다 위치결정 핀. 정적 collider다.
        for row in range(cell.rows):
            for col in range(cell.cols):
                pin = define_convex(
                    stage,
                    f"/World/Pins/pin_{row}_{col}",
                    *pin_mesh(cell.board_point(row + 0.5, col + 0.5), cell=cell),
                    contact_offset=PIN_CONTACT_OFFSET_M,
                    material=board_material,
                    color=PIN_RGB,
                )
                bind_look(stage, pin.GetPrim(), PIN_LOOK)

    dome = UsdLux.DomeLight.Define(stage, "/World/DomeLight")
    dome.CreateIntensityAttr(CAPTURE_DOME_INTENSITY if diffuse else DOME_INTENSITY)
    if not diffuse:
        sun = UsdLux.DistantLight.Define(stage, "/World/SunLight")
        sun.CreateIntensityAttr(SUN_INTENSITY)
        sun.AddRotateXYZOp().Set(Gf.Vec3f(*SUN_ROTATION_DEG))
    define_camera(
        stage,
        "/World/ObliqueCamera",
        pose=look_at(OBLIQUE_EYE, OBLIQUE_TARGET, Gf.Vec3d(0, 0, 1)),
    )
    define_camera(stage, "/World/TopDownCamera", pose=topdown)
    if args.present:
        define_camera(
            stage,
            PRESENT_CAMERA_PATH,
            pose=look_at(PRESENT_EYE, PRESENT_TARGET, Gf.Vec3d(0, 0, 1)),
            focal_length=PRESENT_FOCAL_LENGTH_MM,
            size=PRESENT_SIZE,
        )


def build_markers(stage, cell: Cell) -> None:
    """로봇 기준 표식. 밝은 판 위의 어두운 원이다. 원의 윗면이 table_z다.

    표식은 collider가 없다. 실행기는 표식을 부품 뒤에 만든다.
    그래서 부품 실행의 물리가 바뀌지 않는다.
    표식은 시각 재질이 없다. 표식의 색은 `displayColor`다.
    """
    plate_size = cell.marker_diameter * MARKER_PLATE_RATIO
    plate_thickness = PLATE_THICKNESS_M - MARKER_THICKNESS_M
    table_top = cell.table_z - PLATE_THICKNESS_M
    for index, marker in enumerate(cell.markers):
        define_box(
            stage,
            f"/World/Markers/plate_{index}",
            center=(*marker, table_top + plate_thickness / 2),
            size=(plate_size, plate_size, plate_thickness),
            yaw=0.0,
            color=MARKER_PLATE_RGB,
            collision=False,
        )
        disc = UsdGeom.Cylinder.Define(stage, f"/World/Markers/marker_{index}")
        disc.CreateAxisAttr(UsdGeom.Tokens.z)
        disc.CreateRadiusAttr(cell.marker_diameter / 2)
        disc.CreateHeightAttr(MARKER_THICKNESS_M)
        disc.AddTranslateOp().Set(Gf.Vec3d(*marker, cell.table_z - MARKER_THICKNESS_M / 2))
        disc.CreateDisplayColorAttr([rgb(MARKER_RGB)])


def build_props(stage, cell: Cell) -> None:
    """배경 소품. 바닥, 작업대 프레임, 벽, 셀 프레임과 제어반이다.

    소품은 collider가 없다. 실행기는 소품을 표식 뒤에 만든다.
    그래서 부품 실행의 물리가 바뀌지 않는다.
    소품은 그림자를 만들지 않는다. 소품이 돔 조명을 가리면 핀 끝의 밝기가 한쪽으로 치우친다.
    그러면 보드 자세의 추정 오차가 는다.
    """
    table_top = cell.table_z - PLATE_THICKNESS_M
    floor_z = table_top - BENCH_HEIGHT_M
    table_x, table_y = TABLE_CENTER_XY
    # 프로파일 중심선의 x와 y. 작업대의 네 모서리 안쪽이다.
    reach = TABLE_SIZE_M / 2 - PROFILE_INSET_M
    back_x, front_x = table_x - reach, table_x + reach
    left_y, right_y = table_y - reach, table_y + reach
    span = 2 * reach
    leg_length = BENCH_HEIGHT_M - TABLE_THICKNESS_M
    rail_z = floor_z + LOWER_RAIL_HEIGHT_M
    wall_z = floor_z + WALL_HEIGHT_M / 2
    skirting_z = floor_z + SKIRTING_HEIGHT_M / 2
    # 굽도리의 중심과 이음새의 중심은 벽의 가운데 면에서 이 거리만큼 앞에 있다.
    skirting_offset = WALL_THICKNESS_M / 2 + SKIRTING_THICKNESS_M / 2
    seam_offset = WALL_THICKNESS_M / 2 + WALL_SEAM_THICKNESS_M / 2
    seam_length = WALL_HEIGHT_M - SKIRTING_HEIGHT_M
    side_x, side_y = SIDE_WALL_CENTER_XY
    cabinet_x, cabinet_y = CABINET_CENTER_XY
    cabinet_depth, _, cabinet_height = CABINET_SIZE_M
    cabinet_z = floor_z + cabinet_height / 2
    cabinet_front = cabinet_x + cabinet_depth / 2
    handle_depth = CABINET_HANDLE_SIZE_M[0]
    # (이름, 중심, 크기, 재질)
    boxes = [
        (
            "floor",
            (table_x, table_y, floor_z - FLOOR_THICKNESS_M / 2),
            (FLOOR_SIZE_M, FLOOR_SIZE_M, FLOOR_THICKNESS_M),
            FLOOR_LOOK,
        ),
        # 작업대 다리 4개와 아래 가로대 4개
        *(
            (
                f"leg_{index}",
                (x, y, floor_z + leg_length / 2),
                (PROFILE_M, PROFILE_M, leg_length),
                PROFILE_LOOK,
            )
            for index, (x, y) in enumerate(
                (x, y) for x in (back_x, front_x) for y in (left_y, right_y)
            )
        ),
        *(
            (f"rail_x_{index}", (table_x, y, rail_z), (span, PROFILE_M, PROFILE_M), PROFILE_LOOK)
            for index, y in enumerate((left_y, right_y))
        ),
        *(
            (f"rail_y_{index}", (x, table_y, rail_z), (PROFILE_M, span, PROFILE_M), PROFILE_LOOK)
            for index, x in enumerate((back_x, front_x))
        ),
        # 뒷벽, 굽도리와 패널 이음새
        (
            "wall",
            (BACK_WALL_X, 0.0, wall_z),
            (WALL_THICKNESS_M, BACK_WALL_LENGTH_M, WALL_HEIGHT_M),
            WALL_LOOK,
        ),
        (
            "wall_skirting",
            (BACK_WALL_X + skirting_offset, 0.0, skirting_z),
            (SKIRTING_THICKNESS_M, BACK_WALL_LENGTH_M, SKIRTING_HEIGHT_M),
            TRIM_LOOK,
        ),
        *(
            (
                f"wall_seam_{index}",
                (BACK_WALL_X + seam_offset, y, floor_z + SKIRTING_HEIGHT_M + seam_length / 2),
                (WALL_SEAM_THICKNESS_M, WALL_SEAM_WIDTH_M, seam_length),
                TRIM_LOOK,
            )
            for index, y in enumerate(BACK_WALL_SEAMS_Y)
        ),
        # 셀 프레임. 작업대 위의 뒤 기둥 2개, 위 가로대와 뒤 가로대다.
        *(
            (
                f"post_{index}",
                (back_x, y, table_top + POST_LENGTH_M / 2),
                (PROFILE_M, PROFILE_M, POST_LENGTH_M),
                PROFILE_LOOK,
            )
            for index, y in enumerate((left_y, right_y))
        ),
        (
            "beam",
            (back_x, table_y, table_top + POST_LENGTH_M - PROFILE_M / 2),
            (PROFILE_M, span, PROFILE_M),
            PROFILE_LOOK,
        ),
        (
            "rail_back",
            (back_x, table_y, table_top + BACK_RAIL_HEIGHT_M),
            (PROFILE_M, span, PROFILE_M),
            PROFILE_LOOK,
        ),
        # 옆벽과 굽도리
        (
            "side_wall",
            (side_x, side_y, wall_z),
            (SIDE_WALL_LENGTH_M, WALL_THICKNESS_M, WALL_HEIGHT_M),
            WALL_LOOK,
        ),
        (
            "side_wall_skirting",
            (side_x, side_y - skirting_offset, skirting_z),
            (SIDE_WALL_LENGTH_M, SKIRTING_THICKNESS_M, SKIRTING_HEIGHT_M),
            TRIM_LOOK,
        ),
        # 제어반. 앞면에 문 이음새와 손잡이가 있다.
        ("cabinet", (cabinet_x, cabinet_y, cabinet_z), CABINET_SIZE_M, CABINET_LOOK),
        (
            "cabinet_seam",
            (cabinet_front + CABINET_SEAM_THICKNESS_M / 2, cabinet_y, cabinet_z),
            (
                CABINET_SEAM_THICKNESS_M,
                CABINET_SEAM_WIDTH_M,
                cabinet_height - 2 * CABINET_SEAM_MARGIN_M,
            ),
            TRIM_LOOK,
        ),
        (
            "cabinet_handle",
            (
                cabinet_front + handle_depth / 2,
                cabinet_y + CABINET_HANDLE_OFFSET_M,
                floor_z + CABINET_HANDLE_HEIGHT_M,
            ),
            CABINET_HANDLE_SIZE_M,
            TRIM_LOOK,
        ),
    ]
    for name, center, size, look in boxes:
        prim = define_box(
            stage,
            f"/World/Props/{name}",
            center=center,
            size=size,
            yaw=0.0,
            color=look.color,
            collision=False,
        ).GetPrim()
        bind_look(stage, prim, look)
        UsdGeom.PrimvarsAPI(prim).CreatePrimvar("doNotCastShadows", Sdf.ValueTypeNames.Bool).Set(
            True
        )


def build_wrist_camera(stage) -> None:
    """손목 카메라를 그리퍼 link의 자식으로 만든다. collider가 없다."""
    # 저작된 USD 자세는 모든 관절이 0인 자세다.
    cache = UsdGeom.XformCache()
    tool = cache.GetLocalToWorldTransform(stage.GetPrimAtPath(TOOL_PATH))
    link = cache.GetLocalToWorldTransform(stage.GetPrimAtPath(GRIPPER_PATH))
    eye, target = (
        Gf.Vec3d(*(value / MM_PER_M for value in point))
        for point in (WRIST_EYE_MM, WRIST_TARGET_MM)
    )
    # Gf 행렬은 행 벡터 규약이다. 카메라 좌표 → 툴 좌표 → link 좌표 순서로 곱한다.
    define_camera(
        stage,
        WRIST_CAMERA_PATH,
        pose=look_at(eye, target, WRIST_UP) * tool * link.GetInverse(),
        focal_length=WRIST_FOCAL_LENGTH_MM,
    )


def pin_mesh(center, *, cell: Cell) -> tuple[list, list[int], list[int]]:
    """보드 윗면에 선 모따기 원기둥 핀의 (점, 면 꼭짓점 수, 면 인덱스)."""
    locator = cell.locator
    radius = locator["pin_diameter"] / 2
    top = cell.table_z + locator["pin_height"]
    # 아래 원, 모따기 시작 원, 끝 원
    rings = (
        (cell.table_z, radius),
        (top - locator["pin_chamfer"], radius),
        (top, radius - locator["pin_chamfer"]),
    )
    angles = np.arange(PIN_SIDES) * 2 * np.pi / PIN_SIDES
    points = [
        (center[0] + ring_radius * np.cos(angle), center[1] + ring_radius * np.sin(angle), z)
        for z, ring_radius in rings
        for angle in angles
    ]
    # 아래 면, 윗면, 원 사이 옆면 사각형들
    counts = [PIN_SIDES, PIN_SIDES]
    indices = [*reversed(range(PIN_SIDES)), *range(2 * PIN_SIDES, 3 * PIN_SIDES)]
    for ring in range(len(rings) - 1):
        for side in range(PIN_SIDES):
            first = ring * PIN_SIDES + side
            second = ring * PIN_SIDES + (side + 1) % PIN_SIDES
            counts.append(4)
            indices += [first, second, second + PIN_SIDES, first + PIN_SIDES]
    return points, counts, indices


def replace_jaw_colliders(stage, material) -> None:
    """jaw 메시 2개의 collision을 끄고 보이지 않는 볼록 collider를 jaw link에 붙인다."""
    for holder, name in JAW_MESHES:
        # instance 안의 prim은 고칠 수 없다. 그래서 instance를 먼저 푼다.
        stage.GetPrimAtPath(holder).SetInstanceable(False)
        mesh = stage.GetPrimAtPath(f"{holder}/{name}")
        UsdPhysics.CollisionAPI(mesh).CreateCollisionEnabledAttr(False)
    # 저작된 USD 자세는 모든 관절이 0인 자세다.
    cache = UsdGeom.XformCache()
    tool = cache.GetLocalToWorldTransform(stage.GetPrimAtPath(TOOL_PATH))
    for link_path, boxes in (
        (GRIPPER_PATH, FIXED_JAW_BOXES_MM),
        (MOVING_JAW_PATH, MOVING_JAW_BOXES_MM),
    ):
        link = cache.GetLocalToWorldTransform(stage.GetPrimAtPath(link_path))
        # Gf 행렬은 행 벡터 규약이다. 툴 좌표의 점에 이 행렬을 곱하면 link 좌표의 점이 된다.
        tool_to_link = tool * link.GetInverse()
        if link_path == GRIPPER_PATH:
            define_fixed_jaw_tip(stage, f"{link_path}/jaw_tip", tool_to_link, material)
        for index, ranges in enumerate(boxes):
            center = Gf.Vec3d(*((low + high) / 2 / MM_PER_M for low, high in ranges))
            size = Gf.Vec3f(*((high - low) / MM_PER_M for low, high in ranges))
            box = UsdGeom.Cube.Define(stage, f"{link_path}/jaw_box_{index}")
            box.CreateSizeAttr(1.0)
            box.AddTransformOp().Set(Gf.Matrix4d().SetTranslate(center) * tool_to_link)
            box.AddScaleOp().Set(size)
            # visibility는 collision에 영향이 없다.
            box.MakeInvisible()
            UsdPhysics.CollisionAPI.Apply(box.GetPrim())
            set_contact_offsets(box.GetPrim(), JAW_CONTACT_OFFSET_M)
            bind_material(box.GetPrim(), material)


def define_fixed_jaw_tip(stage, path: str, tool_to_link, material) -> None:
    """고정 jaw 손끝 collider를 만든다.

    `FIXED_JAW_TIP_PROFILE_MM` 단면을 `FIXED_JAW_TIP_Y_MM` 범위만큼 민 볼록 기둥이다.
    """
    profile = FIXED_JAW_TIP_PROFILE_MM
    count = len(profile)
    points = [
        Gf.Vec3f(tool_to_link.Transform(Gf.Vec3d(x / MM_PER_M, y / MM_PER_M, z / MM_PER_M)))
        for y in FIXED_JAW_TIP_Y_MM
        for x, z in profile
    ]
    # 양 끝 단면 2개와 옆면 사각형들. collider는 이 점들의 볼록 껍질이다.
    indices = [*reversed(range(count)), *range(count, 2 * count)]
    for first in range(count):
        second = (first + 1) % count
        indices += [first, second, count + second, count + first]
    mesh = UsdGeom.Mesh.Define(stage, path)
    mesh.CreatePointsAttr(points)
    mesh.CreateFaceVertexCountsAttr([count, count] + [4] * count)
    mesh.CreateFaceVertexIndicesAttr(indices)
    # visibility는 collision에 영향이 없다.
    mesh.MakeInvisible()
    UsdPhysics.CollisionAPI.Apply(mesh.GetPrim())
    UsdPhysics.MeshCollisionAPI.Apply(mesh.GetPrim()).CreateApproximationAttr("convexHull")
    set_contact_offsets(mesh.GetPrim(), JAW_CONTACT_OFFSET_M)
    bind_material(mesh.GetPrim(), material)


def build_part(stage, part: dict, *, cell: Cell, index: int, material) -> str:
    """부품 블록을 강체 1개로 만든다. 대기 위치에 둔다.

    부품의 바깥 면만 `part_inset`만큼 줄인다. 같은 부품의 셀 사이에는 틈을 두지 않는다.
    틈이 있으면 jaw가 셀 경계를 잡을 때 접촉 면이 작아진다.

    셀마다 보이는 상자가 있다. 구멍과 모따기가 없으면 이 상자가 collider다.
    구멍이나 모따기가 있으면 상자의 collision을 끈다. 그 대신 보이지 않는 볼록 collider 5개를
    붙인다.
    1. 윗부분 상자: 아래 고리 위부터 윗면까지
    2. 아래 고리 4조각: 바닥부터 `ring_height`까지. 구멍 깔때기와 바깥 모따기를 뺀 부분이다.
    구멍과 모따기는 바닥에 있어서 카메라에 거의 보이지 않는다. 그래서 보이는 형상은 상자로 둔다.
    """
    path = f"/World/Parts/{part['part_id']}"
    body = UsdGeom.Xform.Define(stage, path)
    body.AddTranslateOp().Set(Gf.Vec3d(*parking_position(index, cell)))
    body.AddOrientOp().Set(Gf.Quatf(1.0, 0.0, 0.0, 0.0))
    prim = body.GetPrim()
    UsdPhysics.RigidBodyAPI.Apply(prim)
    PhysxSchema.PhysxRigidBodyAPI.Apply(prim).CreateSolverPositionIterationCountAttr(
        SOLVER_POSITION_ITERATIONS
    )
    UsdPhysics.MassAPI.Apply(prim).CreateDensityAttr(PLA_DENSITY)
    bind_material(prim, material)
    cells = np.array(part["feeder_cells"])
    # 셀 위치를 블록 좌표로 바꾼다. 블록 x축이 공급대 u축이다.
    cosine, sine = np.cos(-cell.feeder_yaw), np.sin(-cell.feeder_yaw)
    local = (cells - cells.mean(axis=0)) @ np.array([[cosine, sine], [-sine, cosine]])
    # 셀 번호는 task가 준다. 셀 위치를 반올림하면 직사각의 두 셀(±0.5 셀)이 같은 번호가 된다.
    blocks = [tuple(index) for index in part["block_cells"]]
    grid = set(blocks)
    half = cell.cell_size / 2
    for cell_index, ((x, y), (i, j)) in enumerate(zip(local, blocks, strict=True)):
        # 축마다 (음의 방향 끝, 양의 방향 끝). 이웃 셀이 같은 부품이면 셀 경계까지 채운다.
        low_x = -half + (0.0 if (i - 1, j) in grid else cell.part_inset)
        high_x = half - (0.0 if (i + 1, j) in grid else cell.part_inset)
        low_y = -half + (0.0 if (i, j - 1) in grid else cell.part_inset)
        high_y = half - (0.0 if (i, j + 1) in grid else cell.part_inset)
        center_x, center_y = x + (low_x + high_x) / 2, y + (low_y + high_y) / 2
        size_x, size_y = high_x - low_x, high_y - low_y
        solid = cell.ring_height == 0.0
        box = define_box(
            stage,
            f"{path}/cell_{cell_index}",
            center=(center_x, center_y, 0.0),
            size=(size_x, size_y, cell.part_height),
            yaw=0.0,
            color=part["color_rgb"],
            collision=solid,
        )
        bind_look(stage, box.GetPrim(), part_look(part["color_rgb"]))
        if solid:
            set_contact_offsets(box.GetPrim(), PART_CONTACT_OFFSET_M)
            continue
        top = define_box(
            stage,
            f"{path}/cell_{cell_index}_top",
            center=(center_x, center_y, cell.ring_height / 2),
            size=(size_x, size_y, cell.part_height - cell.ring_height),
            yaw=0.0,
            color=part["color_rgb"],
            collision=True,
        )
        # visibility는 collision에 영향이 없다.
        top.MakeInvisible()
        set_contact_offsets(top.GetPrim(), PART_CONTACT_OFFSET_M)
        outer = tuple(
            neighbor not in grid for neighbor in ((i - 1, j), (i + 1, j), (i, j - 1), (i, j + 1))
        )
        bottom = -cell.part_height / 2
        pieces = ring_pieces((low_x, high_x, low_y, high_y), outer, cell=cell)
        for piece_index, sections in enumerate(pieces):
            points, counts, indices = [], [], []
            for z, polygon in sections:
                counts.append(len(polygon))
                indices += range(len(points), len(points) + len(polygon))
                points += [(x + px, y + py, bottom + z) for px, py in polygon]
            define_convex(
                stage,
                f"{path}/cell_{cell_index}_ring_{piece_index}",
                points,
                counts,
                indices,
                contact_offset=PART_CONTACT_OFFSET_M,
                material=material,
            )
    return path


def ring_pieces(bounds, outer, *, cell: Cell) -> list[list[tuple[float, list[np.ndarray]]]]:
    """셀 아래 고리 4조각의 높이별 단면 다각형.

    `bounds`는 셀 중심에서 잰 블록 끝 (low_x, high_x, low_y, high_y)(m)다.
    `outer`는 같은 순서의 면이 바깥 면인지다. 바깥 면만 아래 모서리를 깎는다.
    조각은 셀 중심을 지나는 두 대각선으로 나눈 +x, −x, +y, −y 방향이다.
    조각의 단면은 높이마다 반평면들의 교집합이라서 조각은 볼록이다. 조각은 서로 겹치지 않는다.
    반환 값은 조각마다 [(바닥에서 잰 높이 z, 단면 꼭짓점 목록)]이다.
    """
    lower, chamfer = cell.ring_height, cell.part_chamfer
    hole_half, mouth_half = cell.hole_half_width, cell.mouth_half_width
    # 꼭짓점은 고리 바닥, 바깥 모따기 끝, 입구 모따기 끝, 고리 위 높이에만 있다.
    levels = sorted({0.0, min(chamfer, lower), min(mouth_half - hole_half, lower), lower})
    pieces = []
    for direction in ((1.0, 0.0), (-1.0, 0.0), (0.0, 1.0), (0.0, -1.0)):
        u = np.array(direction)
        v = np.array([-u[1], u[0]])
        sections = []
        for z in levels:
            cut = max(0.0, chamfer - z)
            # 바깥 면만 모따기 높이 아래에서 안쪽으로 줄인다.
            low_x, high_x, low_y, high_y = (
                bound + (-cut if index % 2 else cut) * outer[index]
                for index, bound in enumerate(bounds)
            )
            corners = ((low_x, low_y), (high_x, low_y), (high_x, high_y), (low_x, high_y))
            polygon = [np.array(corner) for corner in corners]
            # 구멍 깔때기 바깥쪽이고 두 대각선 사이(|v·p| <= u·p)인 부분
            inner = max(hole_half, mouth_half - z)
            polygon = clip_polygon(polygon, -u, -inner)
            polygon = clip_polygon(polygon, v - u, 0.0)
            polygon = clip_polygon(polygon, -v - u, 0.0)
            sections.append((z, polygon))
        pieces.append(sections)
    return pieces


def part_look(color) -> Look:
    """색이 `color`(0-255)인 부품의 시각 재질."""
    red, green, blue = (int(value) for value in color)
    return Look(
        f"part_{red}_{green}_{blue}", (red, green, blue), roughness=PART_ROUGHNESS, matte=True
    )


def parking_position(index: int, cell: Cell) -> tuple[float, float, float]:
    x, y = PARKING_ORIGIN
    return (x, y + index * PARKING_STEP_M, cell.table_z + cell.part_height / 2)


def clear_feeder(
    parts: list[dict], prims: list[RigidPrim], cell: Cell, *, skip: int | None = None
) -> None:
    """공급대 위에 남은 부품을 대기 위치로 치운다. `skip` 번째 부품은 보지 않는다.

    파지에 실패한 부품은 공급대에 남는다. 다음 부품을 그 자리에 놓으면 두 블록이 겹쳐서 튕겨 나간다.
    치운 부품은 대기 위치에 남는다. 그래서 `result.json`에서 실패로 보인다.
    """
    for index, (part, prim) in enumerate(zip(parts, prims, strict=True)):
        if index == skip:
            continue
        position = prim.get_world_poses()[0].numpy()[0]
        if np.hypot(*(position[:2] - cell.feeder_center)) > FEEDER_CLEAR_RADIUS_M:
            continue
        prim.set_world_poses(
            positions=np.array([parking_position(index, cell)]),
            orientations=np.array([[1.0, 0.0, 0.0, 0.0]]),
        )
        prim.set_velocities(np.zeros((1, 3)), np.zeros((1, 3)))
        log(f"Part cleared from feeder. part_id={part['part_id']} position={millimeters(position)}")


class TopDownCamera:
    """top-down 정지 이미지를 `topdown_<이름>.png`로 쓴다.

    찍을 때만 렌더한다. 매 프레임 렌더하면 캡처 속도가 약 절반이 된다.
    """

    def __init__(self, out_dir: Path) -> None:
        self.out_dir = out_dir
        self.product = rep.create.render_product("/World/TopDownCamera", (args.width, args.height))
        self.annotator = rep.AnnotatorRegistry.get_annotator("rgb")
        self.annotator.attach([self.product])
        self.product.hydra_texture.set_updates_enabled(False)

    def capture(self, name: str) -> None:
        self.product.hydra_texture.set_updates_enabled(True)
        # 시간을 멈추고 RT 렌더를 수렴시킨다.
        rep.orchestrator.step(rt_subframes=CAPTURE_SUBFRAMES, delta_time=0.0, pause_timeline=False)
        frame = np.asarray(self.annotator.get_data())[:, :, :3]
        Image.fromarray(frame).save(self.out_dir / f"topdown_{name}.png")
        self.product.hydra_texture.set_updates_enabled(False)


def open_video(path: Path, *, size: tuple[int, int], crf: int | None) -> subprocess.Popen:
    """rawvideo 프레임을 stdin으로 받아 `path`에 x264 영상을 쓰는 ffmpeg.

    `size`는 프레임의 (가로, 세로)다. `crf`가 None이면 x264의 기본 품질이다.
    """
    width, height = size
    quality = [] if crf is None else ["-crf", str(crf)]
    return subprocess.Popen(
        [
            "ffmpeg",
            "-loglevel",
            "error",
            "-y",
            "-f",
            "rawvideo",
            "-pix_fmt",
            "rgb24",
            "-s",
            f"{width}x{height}",
            "-r",
            str(args.fps),
            "-i",
            "-",
            "-c:v",
            "libx264",
            *quality,
            "-pix_fmt",
            "yuv420p",
            str(path),
        ],
        stdin=subprocess.PIPE,
    )


def write_frame(annotator, video: subprocess.Popen) -> np.ndarray:
    """카메라의 현재 프레임을 영상에 쓴다. 그 프레임을 반환한다."""
    frame = np.ascontiguousarray(np.asarray(annotator.get_data())[:, :, :3])
    assert video.stdin is not None
    video.stdin.write(frame.tobytes())
    return frame


class Recorder:
    """사선 카메라 영상과 top-down 정지 이미지를 기록한다.

    `--record`에서는 손목 카메라 영상도 쓰고, 프레임마다 `observer`를 부른다.
    `--present`에서는 발표용 카메라 영상도 쓴다.
    """

    def __init__(self, out_dir: Path) -> None:
        size = (args.width, args.height)
        crf = RECORD_CRF if args.record else None
        self.oblique = rep.AnnotatorRegistry.get_annotator("rgb")
        self.oblique.attach([rep.create.render_product("/World/ObliqueCamera", size)])
        self.topdown = TopDownCamera(out_dir)
        self.frames = 0
        self.videos = [(self.oblique, open_video(out_dir / "cell.mp4", size=size, crf=crf))]
        # 프레임을 진행한 뒤에 부르는 함수. 시연 기록이 관절 값을 쌓는다.
        self.observer = None
        # 마지막 프레임의 카메라 이미지. 순서는 사선 카메라, 손목 카메라다. 제어기의 관측에 쓴다.
        self.images: list[np.ndarray] = []
        if args.record:
            wrist = rep.AnnotatorRegistry.get_annotator("rgb")
            wrist.attach([rep.create.render_product(WRIST_CAMERA_PATH, size)])
            self.videos.append((wrist, open_video(out_dir / WRIST_VIDEO_FILE, size=size, crf=crf)))
        # 발표 영상. 제어기의 관측(`images`)에 넣지 않는다.
        self.present = None
        if args.present:
            present = rep.AnnotatorRegistry.get_annotator("rgb")
            present.attach([rep.create.render_product(PRESENT_CAMERA_PATH, PRESENT_SIZE)])
            self.present = (
                present,
                open_video(out_dir / PRESENT_VIDEO_FILE, size=PRESENT_SIZE, crf=PRESENT_CRF),
            )

    def step(self) -> None:
        """시간을 1프레임 진행하고 카메라 프레임을 영상에 쓴다."""
        rep.orchestrator.step(delta_time=1.0 / args.fps, pause_timeline=False)
        self.images = [write_frame(annotator, video) for annotator, video in self.videos]
        if self.present is not None:
            write_frame(*self.present)
        self.frames += 1
        if self.observer is not None:
            self.observer()

    def close(self) -> None:
        videos = [*self.videos, *([self.present] if self.present is not None else [])]
        for _, video in videos:
            assert video.stdin is not None
            video.stdin.close()
            if video.wait() != 0:
                raise RuntimeError("ffmpeg failed to write a video.")


class JointError:
    """`scene.json`의 관절 센서 오차. 프레임 번호로 팔 관절의 오차(rad)를 준다."""

    def __init__(self, rows) -> None:
        self.rows = np.asarray(rows, dtype=np.float64)
        if self.rows.ndim != 2 or self.rows.shape[1] != len(ARM_DOFS) or not len(self.rows):
            raise ValueError(
                "Joint error of the scene does not have one column for each arm joint."
            )

    def at(self, frame: int) -> np.ndarray:
        """프레임 `frame`의 오차. 1행이면 모든 프레임이 그 행이다."""
        if len(self.rows) == 1:
            return self.rows[0]
        if frame >= len(self.rows):
            raise RuntimeError(
                f"Joint error has {len(self.rows)} rows. Frame {frame} is out of range."
            )
        return self.rows[frame]


class EpisodeLog:
    """프레임마다 준 관절 목표, 측정 관절 값과 단계를 쌓는다.

    관절 순서는 `ARM_DOFS`와 그리퍼다. 단위는 rad다.
    프레임 i의 `action`은 그 프레임에 준 목표다. `state`는 그 프레임 뒤의 측정값이다.
    """

    def __init__(self, controller: "Controller") -> None:
        self.controller = controller
        self.order = [*controller.arm_indices, controller.gripper_index]
        self.actions: list[np.ndarray] = []
        self.states: list[np.ndarray] = []
        self.phases: list[str] = []

    def observe(self) -> None:
        self.actions.append(self.controller.target[self.order].copy())
        self.states.append(self.controller.measured()[self.order].copy())
        self.phases.append(self.controller.phase)

    def write(self, path: Path) -> None:
        np.savez(
            path,
            action=np.array(self.actions),
            state=np.array(self.states),
            phase=np.array(self.phases),
            dof_names=np.array([*ARM_DOFS, GRIPPER_DOF]),
            fps=np.array(args.fps),
        )


class PolicyLink:
    """제어기 서버의 클라이언트.

    메시지는 헤더 길이(4바이트), JSON 헤더와 배열의 원시 바이트다.
    """

    def __init__(self, address: str) -> None:
        host, _, port = address.rpartition(":")
        self.connection = socket.create_connection(
            (host, int(port)), timeout=POLICY_CONNECT_TIMEOUT_S
        )
        self.connection.settimeout(POLICY_REPLY_TIMEOUT_S)

    def request(self, header: dict, arrays: dict | None = None) -> tuple[dict, dict]:
        arrays = arrays or {}
        described = {
            **header,
            "arrays": [
                {"name": name, "dtype": array.dtype.str, "shape": list(array.shape)}
                for name, array in arrays.items()
            ],
        }
        encoded = json.dumps(described).encode("utf-8")
        payload = [struct.pack(POLICY_HEADER_FORMAT, len(encoded)), encoded]
        payload += [np.ascontiguousarray(array).tobytes() for array in arrays.values()]
        self.connection.sendall(b"".join(payload))
        (length,) = struct.unpack(
            POLICY_HEADER_FORMAT, self._receive(struct.calcsize(POLICY_HEADER_FORMAT))
        )
        reply = json.loads(self._receive(length).decode("utf-8"))
        reply_arrays = {}
        for spec in reply.pop("arrays"):
            dtype = np.dtype(spec["dtype"])
            shape = tuple(spec["shape"])
            data = self._receive(dtype.itemsize * int(np.prod(shape, dtype=np.int64)))
            reply_arrays[spec["name"]] = np.frombuffer(data, dtype=dtype).reshape(shape)
        if reply.get("type") == "error":
            raise RuntimeError(f"Controller server returned an error: {reply.get('message')}")
        return reply, reply_arrays

    def _receive(self, size: int) -> bytes:
        chunks = bytearray()
        while len(chunks) < size:
            chunk = self.connection.recv(size - len(chunks))
            if not chunk:
                raise RuntimeError("Controller server closed the connection.")
            chunks += chunk
        return bytes(chunks)

    def close(self) -> None:
        self.connection.close()


class Controller:
    """관절 목표를 프레임마다 보간해서 준다.

    `joint_error`가 있으면 팔 관절의 측정값은 실제 각도에서 그 프레임의 오차를 뺀 값이다.
    drive에는 목표에 오차를 더한 각도를 준다. 그래서 측정값이 목표를 따라간다.
    `target`, 기록과 제어기의 관측은 측정값의 좌표다.
    """

    def __init__(
        self, robot: Articulation, recorder: Recorder, joint_error: JointError | None = None
    ) -> None:
        self.robot = robot
        self.recorder = recorder
        self.joint_error = joint_error
        names = list(robot.dof_names)
        self.arm_indices = [names.index(name) for name in ARM_DOFS]
        self.gripper_index = names.index(GRIPPER_DOF)
        self.target = robot.get_dof_positions().numpy()[0].copy()
        self.max_arm_error = 0.0
        # 지금 실행하는 단계의 이름. 시연 기록이 프레임마다 읽는다.
        self.phase = "start"

    def teleport(self, arm, gripper: float) -> None:
        self.target[self.arm_indices] = arm
        self.target[self.gripper_index] = gripper
        self.robot.set_dof_positions(self.target.reshape(1, -1))
        self.robot.set_dof_position_targets(self.target.reshape(1, -1))
        if self.joint_error is not None:
            # 팔의 실제 자세는 `arm`이다. 그 자세의 측정값이 목표가 된다.
            self.target[self.arm_indices] -= self.joint_error.at(0)

    def measured(self) -> np.ndarray:
        """측정 관절 값. 관절 센서 오차가 있으면 방금 진행한 프레임의 오차를 실제 각도에서 뺀다."""
        positions = self.robot.get_dof_positions().numpy()[0]
        if self.joint_error is None:
            return positions
        positions = positions.copy()
        positions[self.arm_indices] -= self.joint_error.at(max(self.recorder.frames - 1, 0))
        return positions

    def _drive(self) -> None:
        """목표를 drive에 준다. 관절 센서 오차가 있으면 다음에 진행할 프레임의 오차를 더한다."""
        if self.joint_error is None:
            self.robot.set_dof_position_targets(self.target.reshape(1, -1))
            return
        drive = self.target.copy()
        drive[self.arm_indices] += self.joint_error.at(self.recorder.frames)
        self.robot.set_dof_position_targets(drive.reshape(1, -1))

    def _step(self) -> None:
        """목표를 바꾸지 않고 1프레임 진행한다. 오차가 프레임마다 바뀌므로 drive를 다시 준다."""
        if self.joint_error is not None:
            self._drive()
        self.recorder.step()

    def move_arm(self, arm) -> None:
        start = self.target[self.arm_indices].copy()
        goal = np.asarray(arm, dtype=np.float64)
        duration = max(float(np.max(np.abs(goal - start))) / MAX_JOINT_SPEED, MIN_SEGMENT_S)
        self._interpolate(self.arm_indices, start, goal, duration)
        error = self._arm_error(goal)
        self.max_arm_error = max(self.max_arm_error, error)
        for _ in range(frame_count(SETTLE_TIMEOUT_S)):
            if error <= SETTLE_TOLERANCE_RAD:
                break
            self._step()
            error = self._arm_error(goal)

    def _arm_error(self, goal) -> float:
        actual = self.measured()[self.arm_indices]
        return float(np.max(np.abs(actual - goal)))

    def move_gripper(self, angle: float) -> None:
        start = self.target[[self.gripper_index]].copy()
        duration = GRIPPER_CLOSE_S if angle < start[0] else GRIPPER_OPEN_S
        self._interpolate([self.gripper_index], start, np.array([angle]), duration)
        self.hold(GRIPPER_HOLD_S)

    def hold(self, seconds: float) -> None:
        for _ in range(frame_count(seconds)):
            self._step()

    def _interpolate(self, indices, start, goal, duration: float) -> None:
        frames = frame_count(duration)
        for frame in range(1, frames + 1):
            self.target[indices] = start + (goal - start) * minimum_jerk(frame / frames)
            self._drive()
            self.recorder.step()

    def follow_policy(self, link: PolicyLink, episode: "EpisodeLog", *, context: dict) -> int:
        """제어기가 낸 관절 목표를 프레임마다 준다. 진행한 프레임 수를 반환한다.

        단계:
        1. 목표를 바꾸지 않고 1프레임 진행한다. 이 프레임이 제어기의 첫 관측이다.
        2. 관측을 보내고 목표의 묶음을 받는다. 묶음의 목표를 프레임마다 1개씩 준다.
        3. 프레임 수가 한도에 닿을 때까지 2를 반복한다. 한도는 `--max-steps`와 서버가 준 값 중
           작은 값이다.

        목표의 프레임당 변화는 `POLICY_MAX_STEP_RAD`로 자른다.
        """
        self.phase = POLICY_PHASE
        order = episode.order
        link.request(
            {"type": "hello", "protocol": POLICY_PROTOCOL, "dof_names": [*ARM_DOFS, GRIPPER_DOF]}
        )
        reply, _ = link.request({"type": "reset", "context": context})
        limit = min(args.max_steps, reply.get("max_steps") or args.max_steps)
        self._step()
        steps = 1
        while steps < limit:
            front, wrist = self.recorder.images
            _, arrays = link.request(
                {"type": "act", "step": steps - 1},
                {"state": episode.states[-1].astype(np.float32), "front": front, "wrist": wrist},
            )
            for action in arrays["actions"][: limit - steps]:
                goal = np.asarray(action, dtype=np.float64)
                if goal.shape != (len(order),) or not np.all(np.isfinite(goal)):
                    raise RuntimeError("The controller returned an action that is not valid.")
                current = self.target[order]
                change = np.clip(goal - current, -POLICY_MAX_STEP_RAD, POLICY_MAX_STEP_RAD)
                self.target[order] = current + change
                self._drive()
                self.recorder.step()
                steps += 1
        return steps


def frame_count(seconds: float) -> int:
    return max(1, round(seconds * args.fps))


def part_pose(prim: RigidPrim, cell: Cell) -> dict:
    positions, orientations = prim.get_world_poses()
    return pose_values(positions.numpy()[0], orientations.numpy()[0], cell)


def pose_values(position, orientation, cell: Cell) -> dict:
    """위치(m)와 quaternion (w, x, y, z)에서 `result.json`의 자세 값을 만든다."""
    yaw, tilt = quaternion_yaw_tilt(orientation)
    return {
        "position": [float(value) for value in position],
        # 블록 x축은 회전 0°인 부품의 u축이다. 그래서 실제 보드 u축에서 잰 회전은
        # task의 target_yaw와 같은 기준이다.
        "yaw": wrap_angle(yaw - cell.board_yaw),
        "tilt": tilt,
        "fell": bool(position[2] < FELL_BELOW_M),
        # base_link 좌표의 quaternion (w, x, y, z). 재작업 실행이 이 값으로 부품을 복원한다.
        "orientation": [float(value) for value in orientation],
    }


def world_pose(prim) -> tuple[np.ndarray, np.ndarray]:
    """prim 1개의 world 위치(m)와 3×3 회전 행렬."""
    positions, orientations = prim.get_world_poses()
    return positions.numpy()[0].astype(np.float64), quaternion_matrix(orientations.numpy()[0])


def millimeters(vector) -> str:
    return ",".join(f"{value * 1000:.1f}" for value in vector)


class StageProbe:
    """부품 1개의 단계별 오차를 잰다. 결과는 `Part placed.` 로그 줄에 붙인다.

    측정 시점:
    1. pick에 도착한 뒤, 그리퍼를 닫기 전
    2. lift에 도착한 뒤
    3. place에 도착한 뒤, 그리퍼를 열기 전
    4. place에서 그리퍼를 열고 기다린 뒤

    xy 성분은 툴 축 기준이다. 툴 x축은 움직이는 jaw에서 고정 jaw 쪽을 향한다.
    world 변위는 3번 시점의 툴 축으로 바꾼다. yaw는 위에서 본 반시계 방향이다.
    """

    def __init__(self, *, part: dict, target, prim: RigidPrim, tool: XformPrim, tcp) -> None:
        """`target`은 실제 보드의 목표 위치(m)다. `pre_release_mm`을 이 점에서 잰다."""
        self.prim = prim
        self.tool = tool
        self.tcp = np.asarray(tcp, dtype=np.float64)
        self.target = target
        self.place = np.asarray(
            next(point["position"] for point in part["waypoints"] if point["name"] == "place")
        )
        self.fields: dict[str, str] = {}
        self._before_grasp: tuple[np.ndarray, float] | None = None
        self._before_release: tuple[np.ndarray, np.ndarray, np.ndarray] | None = None

    def arrived(self, waypoint: str) -> None:
        """waypoint에 도착한 뒤, 그리퍼가 움직이기 전에 부른다."""
        if waypoint == "pick":
            self._before_grasp = self._part_in_tool()
        elif waypoint == "lift" and self._before_grasp is not None:
            position, yaw = self._part_in_tool()
            self.fields["grasp_mm"] = millimeters(position - self._before_grasp[0])
            self.fields["grasp_yaw_deg"] = (
                f"{np.degrees(wrap_angle(yaw - self._before_grasp[1])):.1f}"
            )
        elif waypoint == "place":
            tool_position, tool_rotation = world_pose(self.tool)
            part_position, part_rotation = world_pose(self.prim)
            tcp = tool_position + tool_rotation @ self.tcp
            self.fields["tcp_mm"] = millimeters((tool_rotation.T @ (tcp - self.place))[:2])
            error = np.array([*(part_position[:2] - self.target), 0.0])
            self.fields["pre_release_mm"] = millimeters((tool_rotation.T @ error)[:2])
            self._before_release = (tool_rotation, part_position, part_rotation)

    def released(self) -> None:
        """place에서 그리퍼를 열고 기다린 뒤에 부른다."""
        if self._before_release is None:
            return
        tool_rotation, before_position, before_rotation = self._before_release
        position, rotation = world_pose(self.prim)
        self.fields["release_mm"] = millimeters(
            (tool_rotation.T @ (position - before_position))[:2]
        )
        change = wrap_angle(matrix_yaw(rotation) - matrix_yaw(before_rotation))
        self.fields["release_yaw_deg"] = f"{np.degrees(change):.1f}"

    def summary(self) -> str:
        return " ".join(f"{key}={value}" for key, value in self.fields.items())

    def _part_in_tool(self) -> tuple[np.ndarray, float]:
        """툴 좌표에서 본 부품 중심의 xy(m)와 툴 x축에서 잰 부품 yaw(rad)."""
        tool_position, tool_rotation = world_pose(self.tool)
        part_position, part_rotation = world_pose(self.prim)
        position = (tool_rotation.T @ (part_position - tool_position))[:2]
        # 툴 z축은 아래를 향한다. 그래서 yaw는 world에서 두 x축의 방위각 차이로 잰다.
        # 위에서 본 반시계 방향이다. release_yaw_deg와 같은 방향이다.
        return position, wrap_angle(matrix_yaw(part_rotation) - matrix_yaw(tool_rotation))


def setup_robot() -> Articulation:
    """SO-101 articulation. 관절 drive를 정하고 로봇 link의 중력을 끈다."""
    robot = Articulation(ROBOT_PATH)
    names = list(robot.dof_names)
    stiffness = np.full(len(names), ARM_STIFFNESS)
    damping = np.full(len(names), ARM_DAMPING)
    gripper_index = names.index(GRIPPER_DOF)
    stiffness[gripper_index], damping[gripper_index] = GRIPPER_STIFFNESS, GRIPPER_DAMPING
    robot.set_dof_gains(stiffness.reshape(1, -1), damping.reshape(1, -1))
    robot.set_dof_max_efforts(GRIPPER_MAX_EFFORT, dof_indices=[gripper_index])
    # 실물 서보는 자세를 유지한다. 중력으로 팔이 처지면 jaw가 이웃 부품에 닿는다.
    # 그래서 로봇 link의 중력을 끈다.
    robot.set_link_enabled_gravities(np.zeros((1, robot.num_links), dtype=bool))
    return robot


def part_changes(task: dict, scene: dict, *, policy: bool) -> tuple[set[int], dict, dict]:
    """`scene.json`의 결함과 이미 놓인 부품을 읽는다. 열쇠는 부품의 `sequence`다.

    `(공급하지 않는 부품, 부품의 다른 색, 이미 놓인 부품의 자세)`를 반환한다.
    """
    faults = scene.get("faults") or {}
    missing = set(faults.get("missing", ()))
    colors = {item["sequence"]: item["color_rgb"] for item in faults.get("wrong_color", ())}
    preplaced = {item["sequence"]: item for item in scene.get("preplaced") or ()}
    sequences = {part["sequence"] for part in task["parts"]}
    unknown = (missing | set(colors) | set(preplaced)) - sequences
    if unknown:
        raise ValueError(f"Scene names parts that the task does not have: {sorted(unknown)}.")
    if missing & set(preplaced):
        raise ValueError("A preplaced part cannot be a missing part.")
    if sequences and set(preplaced) == sequences:
        raise ValueError("All parts are preplaced. The task has no part to place.")
    if policy and (missing or colors or preplaced):
        raise ValueError("A controller run does not accept faults or preplaced parts.")
    return missing, colors, preplaced


def start_timeline() -> None:
    # 첫 캡처 전에 렌더러가 프레임을 몇 번 진행해야 한다. 바로 캡처하면 renderer가 멈춘다.
    for _ in range(args.settle_frames):
        app.update()
    omni.timeline.get_timeline_interface().play()
    app.update()


def main() -> None:
    args.out_dir.mkdir(parents=True, exist_ok=True)
    source = args.task if args.task is not None else args.capture
    task = json.loads(source.read_text(encoding="utf-8"))
    scene = json.loads(args.scene.read_text(encoding="utf-8")) if args.scene else {}
    cell = Cell(task, scene.get("board"))
    if cell.placement is not None:
        log(
            f"Board placed from scene. origin_mm={millimeters(cell.origin)} "
            f"yaw_deg={np.degrees(cell.board_yaw):.3f}"
        )
    camera_offset = scene.get("camera")
    topdown_rotation, topdown_center = topdown_pose(camera_offset)
    if camera_offset is not None:
        angles = ",".join(f"{np.degrees(value):.3f}" for value in camera_offset["rotation"])
        log(
            "Top-down camera moved from scene. "
            f"position_mm={millimeters(camera_offset['position'])} rotation_deg={angles}"
        )
    stage = omni.usd.get_context().get_stage()
    material = define_material(
        stage, "/World/Physics/Material", static=STATIC_FRICTION, dynamic=DYNAMIC_FRICTION
    )
    board_material = define_material(
        stage,
        "/World/Physics/BoardMaterial",
        static=BOARD_STATIC_FRICTION,
        dynamic=BOARD_DYNAMIC_FRICTION,
        combine="min",
    )
    build_scene(
        stage,
        cell,
        material,
        board_material,
        topdown=gf_pose(topdown_rotation, topdown_center),
        diffuse=args.capture is not None,
    )
    if args.capture is not None:
        capture(task, cell, stage, truth=camera_info(topdown_rotation, topdown_center))
        return
    missing, wrong_colors, preplaced = part_changes(task, scene, policy=args.policy is not None)
    part_paths = [
        build_part(
            stage,
            {**part, "color_rgb": wrong_colors.get(part["sequence"], part["color_rgb"])},
            cell=cell,
            index=index,
            material=material,
        )
        for index, part in enumerate(task["parts"])
    ]
    build_markers(stage, cell)
    build_props(stage, cell)
    if args.record:
        build_wrist_camera(stage)
    SimulationManager.setup_simulation(dt=PHYSICS_DT_S)

    recorder = Recorder(args.out_dir)
    start_timeline()
    robot = setup_robot()
    parts = [RigidPrim(path) for path in part_paths]
    # 렌더가 읽는 자세를 쓰는 뷰. 제어기 실행만 쓴다.
    render_parts = [XformPrim(path) for path in part_paths] if args.policy is not None else []
    tool = XformPrim(TOOL_PATH)
    joint_error = None
    if scene.get("joint_error") is not None:
        joint_error = JointError(scene["joint_error"])
        log(
            f"Joint sensor error from scene. rows={len(joint_error.rows)} "
            f"max_deg={np.degrees(np.max(np.abs(joint_error.rows))):.3f}"
        )
    controller = Controller(robot, recorder, joint_error)
    episode = EpisodeLog(controller) if args.record else None
    if episode is not None:
        recorder.observer = episode.observe
    angles = task["gripper_angles"]
    if not task["parts"]:
        raise ValueError("Task has no parts.")
    if args.policy is not None and len(task["parts"]) != 1:
        raise ValueError("A controller run needs a task with one part.")
    # 공급대 위 접근 자세. 이 자세에서는 팔이 보드를 가리지 않는다. top-down 이미지를 여기서 찍는다.
    clear_pose = task["parts"][0]["waypoints"][0]["joints"]
    controller.teleport(clear_pose, angles["open"])
    for _ in range(TELEPORT_SETTLE_UPDATES):
        app.update()
    startup_s = time.monotonic() - STARTED
    log(f"Scene is ready. parts={len(parts)} startup_s={startup_s:.1f}")

    height = cell.table_z + cell.part_height / 2
    # 이미 놓인 부품의 복원한 자세. 열쇠는 sequence다.
    restored_poses = {}
    for part, prim in zip(task["parts"], parts, strict=True):
        restored = preplaced.get(part["sequence"])
        if restored is None:
            continue
        position, orientation = np.array(restored["position"]), np.array(restored["orientation"])
        prim.set_world_poses(
            positions=position.reshape(1, -1), orientations=orientation.reshape(1, -1)
        )
        prim.set_velocities(np.zeros((1, 3)), np.zeros((1, 3)))
        restored_poses[part["sequence"]] = pose_values(position, orientation, cell)
        log(f"Part restored. part_id={part['part_id']} sequence={part['sequence']}")
    # 부품을 놓고 팔이 다음 접근 자세로 간 뒤의 자세. 뒤 부품이 민 부품을 찾는 데 쓴다.
    # 이미 놓인 부품은 복원한 자세다. 재작업의 부품이 그 부품을 밀면 최종 자세와 달라진다.
    placed_poses = []
    for index, (part, prim) in enumerate(zip(task["parts"], parts, strict=True)):
        if part["sequence"] in preplaced:
            placed_poses.append(restored_poses[part["sequence"]])
            continue
        part_started = time.monotonic()
        if preplaced:
            # 재작업 실행에서는 뒤 순서의 부품도 공급대 위에 복원될 수 있다.
            clear_feeder(task["parts"], parts, cell, skip=index)
        else:
            clear_feeder(task["parts"][:index], parts[:index], cell)
        feeder = np.array(part["feeder_cells"]).mean(axis=0)
        feeder_pose = {
            "positions": np.array([[*feeder, height]]),
            "orientations": yaw_quaternion(cell.feeder_yaw).reshape(1, -1),
        }
        if part["sequence"] in missing:
            log(f"Part not supplied. part_id={part['part_id']} sequence={part['sequence']}")
        else:
            prim.set_world_poses(**feeder_pose)
            prim.set_velocities(np.zeros((1, 3)), np.zeros((1, 3)))
        if args.policy is not None:
            # 제어기의 첫 관측에 부품이 있어야 한다. 물리에 준 자세는 1프레임 뒤에 렌더에 보인다.
            # 그래서 렌더가 읽는 자세에도 같은 값을 쓴다. 물리의 결과는 바뀌지 않는다.
            render_parts[index].set_world_poses(**feeder_pose)
        probe = StageProbe(
            part=part,
            target=cell.placed_point(part["target_center"]),
            prim=prim,
            tool=tool,
            tcp=task["tcp"],
        )
        if args.policy is not None:
            assert episode is not None
            context = {}
            if args.policy_context is not None:
                context = json.loads(args.policy_context.read_text(encoding="utf-8"))
            link = PolicyLink(args.policy)
            try:
                steps = controller.follow_policy(link, episode, context=context)
            finally:
                link.close()
            summary = f"policy_steps={steps}"
        else:
            for waypoint in part["waypoints"]:
                controller.phase = waypoint["name"]
                controller.move_arm(waypoint["joints"])
                probe.arrived(waypoint["name"])
                angle = angles[waypoint["gripper"]]
                if angle != controller.target[controller.gripper_index]:
                    controller.phase = f"{waypoint['name']}_gripper"
                    controller.move_gripper(angle)
                if waypoint["name"] == "place":
                    probe.released()
            # 다음 부품의 접근 자세로 가서 팔이 보드를 가리지 않을 때 찍는다.
            next_parts = task["parts"][index + 1 :]
            controller.phase = "clear"
            controller.move_arm(
                next_parts[0]["waypoints"][0]["joints"] if next_parts else clear_pose
            )
            summary = probe.summary()
        recorder.topdown.capture(f"{part['sequence']:02d}")
        pose = part_pose(prim, cell)
        placed_poses.append(pose)
        log(
            f"Part placed. part_id={part['part_id']} sequence={part['sequence']} "
            f"wall_s={time.monotonic() - part_started:.1f} z={pose['position'][2]:.4f} "
            f"{summary}"
        )

    controller.phase = "settle"
    controller.hold(FINAL_SETTLE_S)
    recorder.topdown.capture("final")
    recorder.close()
    if episode is not None:
        episode.write(args.out_dir / EPISODE_FILE)
    result = {
        "result_version": RESULT_VERSION,
        "task_version": task["task_version"],
        "sim_time_s": recorder.frames / args.fps,
        "max_arm_error_rad": controller.max_arm_error,
        "board": cell.placement,
        "parts": [
            {"part_id": part["part_id"], **part_pose(prim, cell), "placed": placed}
            for part, prim, placed in zip(task["parts"], parts, placed_poses, strict=True)
        ],
    }
    (args.out_dir / "result.json").write_text(json.dumps(result, indent=2) + "\n")
    log(
        f"Task finished. parts={len(parts)} frames={recorder.frames} "
        f"wall_s={time.monotonic() - STARTED:.1f}"
    )


def capture(task: dict, cell: Cell, stage, *, truth: dict) -> None:
    """캘리브레이션 캡처. 팔을 `capture.json`의 자세에 두고 top-down 이미지 1장을 찍는다.

    `truth`는 실제 top-down 카메라의 `camera.json` 내용이다. 공칭 카메라와 함께 파일로 쓴다.
    """
    build_markers(stage, cell)
    build_props(stage, cell)
    SimulationManager.setup_simulation(dt=PHYSICS_DT_S)
    topdown = TopDownCamera(args.out_dir)
    start_timeline()
    robot = setup_robot()
    names = list(robot.dof_names)
    positions = robot.get_dof_positions().numpy()[0].copy()
    positions[[names.index(name) for name in ARM_DOFS]] = task["joints"]
    positions[names.index(GRIPPER_DOF)] = task["gripper"]
    robot.set_dof_positions(positions.reshape(1, -1))
    robot.set_dof_position_targets(positions.reshape(1, -1))
    for _ in range(TELEPORT_SETTLE_UPDATES):
        app.update()
    topdown.capture(CAPTURE_IMAGE_NAME)
    nominal = camera_info(*topdown_pose(None))
    for name, info in ((CAMERA_FILE, nominal), (CAMERA_TRUTH_FILE, truth)):
        (args.out_dir / name).write_text(json.dumps(info, indent=2) + "\n")
    log(f"Capture finished. markers={len(cell.markers)} wall_s={time.monotonic() - STARTED:.1f}")


try:
    main()
except Exception:
    # app.close()가 프로세스를 끝낸다. 그래서 traceback을 먼저 출력한다.
    traceback.print_exc()
    raise
finally:
    app.close()
