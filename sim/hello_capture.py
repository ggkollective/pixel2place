"""Isaac Sim headless 동작 확인 장면.

Isaac `python.sh`로 실행한다. pixel2place 패키지를 import하지 않는다.

단계:
1. 지면, 큐브 1개, 조명, 카메라로 장면을 만든다.
2. 큐브를 공중에서 떨어뜨리고 물리를 실행한다.
3. 프레임마다 RGB PNG를 쓴다.
4. 기동 시간, 루프 시간, 큐브의 최종 높이를 `summary.json`에 쓴다.

사용법: python.sh sim/hello_capture.py --out-dir DIR [--frames 90]
"""

import argparse
import json
import time
import traceback
from pathlib import Path

from isaacsim import SimulationApp

STARTED = time.monotonic()
parser = argparse.ArgumentParser()
parser.add_argument("--out-dir", type=Path, required=True)
parser.add_argument("--frames", type=int, default=90)
parser.add_argument("--width", type=int, default=640)
parser.add_argument("--height", type=int, default=480)
parser.add_argument("--settle-frames", type=int, default=60)
args = parser.parse_args()

# SimulationApp은 다른 omni·pxr import보다 먼저 만들어야 한다.
app = SimulationApp(
    {
        "headless": True,
        "width": args.width,
        "height": args.height,
        "renderer": "RayTracedLighting",
    }
)

import numpy as np  # noqa: E402
import omni.replicator.core as rep  # noqa: E402
import omni.timeline  # noqa: E402
import omni.usd  # noqa: E402
from PIL import Image  # noqa: E402
from pxr import Gf, UsdGeom, UsdLux, UsdPhysics  # noqa: E402

FRAME_DT_S = 1.0 / 30.0
CUBE_SIZE_M = 0.1
CUBE_START_Z_M = 0.6
CAMERA_EYE = Gf.Vec3d(1.2, -1.2, 0.9)
CAMERA_TARGET = Gf.Vec3d(0.0, 0.0, 0.1)


def build_scene(stage) -> UsdGeom.Xformable:
    """물리 장면, 지면, 큐브, 조명, 카메라를 만든다. 큐브 prim을 반환한다."""
    UsdGeom.SetStageUpAxis(stage, UsdGeom.Tokens.z)
    UsdGeom.SetStageMetersPerUnit(stage, 1.0)
    UsdPhysics.Scene.Define(stage, "/World/PhysicsScene")

    ground = UsdGeom.Cube.Define(stage, "/World/Ground")
    ground.CreateSizeAttr(1.0)
    ground.AddScaleOp().Set(Gf.Vec3f(4.0, 4.0, 0.02))
    ground.AddTranslateOp(opSuffix="offset").Set(Gf.Vec3d(0.0, 0.0, -0.5))
    ground.CreateDisplayColorAttr([Gf.Vec3f(0.35, 0.36, 0.38)])
    UsdPhysics.CollisionAPI.Apply(ground.GetPrim())

    cube = UsdGeom.Cube.Define(stage, "/World/Cube")
    cube.CreateSizeAttr(CUBE_SIZE_M)
    cube.AddTranslateOp().Set(Gf.Vec3d(0.0, 0.0, CUBE_START_Z_M))
    cube.CreateDisplayColorAttr([Gf.Vec3f(0.95, 0.45, 0.1)])
    UsdPhysics.CollisionAPI.Apply(cube.GetPrim())
    UsdPhysics.RigidBodyAPI.Apply(cube.GetPrim())
    UsdPhysics.MassAPI.Apply(cube.GetPrim()).CreateMassAttr(0.2)

    UsdLux.DomeLight.Define(stage, "/World/DomeLight").CreateIntensityAttr(800.0)
    sun = UsdLux.DistantLight.Define(stage, "/World/SunLight")
    sun.CreateIntensityAttr(2500.0)
    sun.AddRotateXYZOp().Set(Gf.Vec3f(-45.0, 0.0, 30.0))

    camera = UsdGeom.Camera.Define(stage, "/World/Camera")
    camera.CreateFocalLengthAttr(24.0)
    # 기본 clipping range는 (1, 1e6)이다. 1 m보다 가까운 물체가 잘리므로 near를 1 cm로 둔다.
    camera.CreateClippingRangeAttr(Gf.Vec2f(0.01, 100.0))
    # SetLookAt은 view 행렬을 만든다. 카메라 자세는 그 역행렬이다.
    view = Gf.Matrix4d().SetLookAt(CAMERA_EYE, CAMERA_TARGET, Gf.Vec3d(0.0, 0.0, 1.0))
    camera.AddTransformOp().Set(view.GetInverse())
    return cube


def cube_height(cube: UsdGeom.Xformable) -> float:
    matrix = cube.ComputeLocalToWorldTransform(0.0)
    return float(matrix.ExtractTranslation()[2])


def main() -> None:
    args.out_dir.mkdir(parents=True, exist_ok=True)
    stage = omni.usd.get_context().get_stage()
    cube = build_scene(stage)

    render_product = rep.create.render_product("/World/Camera", (args.width, args.height))
    rgb = rep.AnnotatorRegistry.get_annotator("rgb")
    rgb.attach([render_product])
    # 첫 캡처 전에 렌더러가 프레임을 몇 번 진행해야 한다. 바로 캡처하면 renderer가 멈춘다.
    for _ in range(args.settle_frames):
        app.update()
    startup_s = time.monotonic() - STARTED
    print(f"[hello_capture] Scene is ready. startup_s={startup_s:.1f}", flush=True)

    omni.timeline.get_timeline_interface().play()
    loop_started = time.monotonic()
    for index in range(args.frames):
        rep.orchestrator.step(delta_time=FRAME_DT_S, pause_timeline=False)
        frame = np.asarray(rgb.get_data())[:, :, :3]
        Image.fromarray(frame).save(args.out_dir / f"frame_{index:05d}.png")
    loop_s = time.monotonic() - loop_started

    summary = {
        "frames": args.frames,
        "startup_s": round(startup_s, 1),
        "loop_s": round(loop_s, 1),
        "fps": round(args.frames / loop_s, 2),
        "cube_start_z_m": CUBE_START_Z_M,
        "cube_final_z_m": round(cube_height(cube), 4),
        "mean_pixel": round(float(frame.mean()), 1),
    }
    (args.out_dir / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(f"[hello_capture] Capture finished. {json.dumps(summary)}", flush=True)


try:
    main()
except Exception:
    # app.close()가 프로세스를 끝낸다. 그래서 traceback을 먼저 출력한다.
    traceback.print_exc()
    raise
finally:
    app.close()
