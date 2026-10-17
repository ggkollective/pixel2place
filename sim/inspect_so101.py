"""Isaac Sim에서 SO-101 모델을 측정한다.

Isaac `python.sh`로 실행한다. USD는 `sim/import_so101.py`가 만든다.

측정:
1. 관절 이름과 관절 한계
2. 관절 표본마다 툴 끝점(`gripper_frame_link`)의 world 자세. pixel2place FK와 비교하는 데 쓴다.
3. 그리퍼 각도마다 툴 끝점 근처의 jaw 간격
4. 자세별 PNG

중력은 끈다. 관절 값을 그대로 유지해서 FK를 비교하기 위해서다.

사용법: python.sh sim/inspect_so101.py --usd USD --out-dir DIR [--samples 24] [--seed 0]
출력: DIR/inspect.json, DIR/pose_<이름>.png
"""

import argparse
import json
import traceback
from pathlib import Path

from isaacsim import SimulationApp

parser = argparse.ArgumentParser()
parser.add_argument("--usd", type=Path, required=True)
parser.add_argument("--out-dir", type=Path, required=True)
parser.add_argument("--samples", type=int, default=24)
parser.add_argument("--seed", type=int, default=0)
parser.add_argument("--settle-frames", type=int, default=60)
args = parser.parse_args()

WIDTH, HEIGHT = 640, 480
app = SimulationApp(
    {"headless": True, "width": WIDTH, "height": HEIGHT, "renderer": "RayTracedLighting"}
)

import numpy as np  # noqa: E402
import omni.replicator.core as rep  # noqa: E402
import omni.timeline  # noqa: E402
import omni.usd  # noqa: E402
from isaacsim.core.experimental.prims import Articulation, XformPrim  # noqa: E402
from isaacsim.core.simulation_manager import SimulationManager  # noqa: E402
from PIL import Image  # noqa: E402
from pxr import Gf, Usd, UsdGeom, UsdLux, UsdPhysics  # noqa: E402

ROBOT_PATH = "/World/so101"
PHYSICS_VARIANT = "physx"
TOOL_LINK = "gripper_frame_link"
GRIPPER_LINK = "gripper_link"
MOVING_JAW_LINK = "moving_jaw_so101_v1_link"
# 고정 jaw는 gripper_link의 이 visual 메시에 있다. 움직이는 jaw는 MOVING_JAW_LINK의 메시다.
FIXED_JAW_MESH = "wrist_roll_follower_so101_v1"
MOVING_JAW_MESH = "moving_jaw_so101_v1"
# 툴 끝점에서 이 거리 안의 메시 점으로 jaw 간격을 잰다.
JAW_RADIUS_M = 0.04
GRIPPER_ANGLES = 12
CAPTURE_SUBFRAMES = 16
CAMERA_EYE = Gf.Vec3d(0.55, -0.45, 0.4)
CAMERA_TARGET = Gf.Vec3d(0.15, 0.0, 0.12)


def find_prim(stage, root: str, name: str) -> Usd.Prim:
    """`root` 아래에서 이름이 `name`인 첫 prim을 찾는다. instance proxy도 찾는다."""
    for prim in Usd.PrimRange(stage.GetPrimAtPath(root), Usd.TraverseInstanceProxies()):
        if prim.GetName() == name:
            return prim
    raise LookupError(f"Prim {name} is not under {root}.")


def mesh_points(prim: Usd.Prim) -> list[tuple[str, np.ndarray]]:
    """`prim` 아래 모든 Mesh의 점을 USD world 좌표로 반환한다."""
    meshes = []
    for child in Usd.PrimRange(prim, Usd.TraverseInstanceProxies()):
        if not child.IsA(UsdGeom.Mesh):
            continue
        points = np.asarray(UsdGeom.Mesh(child).GetPointsAttr().Get(), dtype=np.float64)
        matrix = np.asarray(
            UsdGeom.Xformable(child).ComputeLocalToWorldTransform(Usd.TimeCode.Default())
        )
        # USD 행렬은 행 벡터 규약이다. 점 p의 world 좌표는 [p, 1] @ M이다.
        homogeneous = np.hstack([points, np.ones((len(points), 1))])
        meshes.append((str(child.GetPath()), (homogeneous @ matrix)[:, :3]))
    return meshes


def pose_matrix(position, quaternion_wxyz) -> np.ndarray:
    """위치와 wxyz 쿼터니언을 열 벡터 규약의 4x4 행렬로 바꾼다."""
    w, x, y, z = (float(value) for value in quaternion_wxyz)
    rotation = np.array(
        [
            [1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y)],
            [2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x)],
            [2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y)],
        ]
    )
    matrix = np.eye(4)
    matrix[:3, :3] = rotation
    matrix[:3, 3] = [float(value) for value in position]
    return matrix


def world_pose(prim_view: XformPrim) -> np.ndarray:
    positions, orientations = prim_view.get_world_poses()
    return pose_matrix(positions.numpy()[0], orientations.numpy()[0])


def build_scene(stage) -> None:
    UsdGeom.SetStageUpAxis(stage, UsdGeom.Tokens.z)
    UsdGeom.SetStageMetersPerUnit(stage, 1.0)
    robot = stage.DefinePrim(ROBOT_PATH, "Xform")
    robot.GetReferences().AddReference(str(args.usd))
    # 변환한 USD는 Physics variant를 고르지 않는다. 고르지 않으면 물리와 articulation이 없다.
    # physx variant는 physics variant에 PhysX 설정을 더한다.
    robot.GetVariantSets().GetVariantSet("Physics").SetVariantSelection(PHYSICS_VARIANT)
    table = UsdGeom.Cube.Define(stage, "/World/Table")
    table.CreateSizeAttr(1.0)
    table.AddTranslateOp().Set(Gf.Vec3d(0.2, 0.0, -0.01))
    table.AddScaleOp().Set(Gf.Vec3f(1.0, 1.0, 0.02))
    table.CreateDisplayColorAttr([Gf.Vec3f(0.3, 0.32, 0.35)])
    UsdLux.DomeLight.Define(stage, "/World/DomeLight").CreateIntensityAttr(300.0)
    sun = UsdLux.DistantLight.Define(stage, "/World/SunLight")
    sun.CreateIntensityAttr(1200.0)
    sun.AddRotateXYZOp().Set(Gf.Vec3f(-50.0, 20.0, 0.0))
    camera = UsdGeom.Camera.Define(stage, "/World/Camera")
    camera.CreateFocalLengthAttr(24.0)
    # 기본 clipping range는 (1, 1e6)이다. 1 m보다 가까운 물체가 잘리므로 near를 1 cm로 둔다.
    camera.CreateClippingRangeAttr(Gf.Vec2f(0.01, 100.0))
    view = Gf.Matrix4d().SetLookAt(CAMERA_EYE, CAMERA_TARGET, Gf.Vec3d(0.0, 0.0, 1.0))
    camera.AddTransformOp().Set(view.GetInverse())


def disable_gravity(stage) -> None:
    for prim in stage.Traverse():
        if prim.IsA(UsdPhysics.Scene):
            UsdPhysics.Scene(prim).CreateGravityMagnitudeAttr(0.0)


def main() -> None:
    args.out_dir.mkdir(parents=True, exist_ok=True)
    stage = omni.usd.get_context().get_stage()
    build_scene(stage)
    SimulationManager.setup_simulation(dt=1.0 / 60.0)
    disable_gravity(stage)

    # 메시와 link의 상대 변환은 관절 값과 무관하다. 기본 자세의 USD 값으로 한 번 계산한다.
    gripper_prim = find_prim(stage, ROBOT_PATH, GRIPPER_LINK)
    jaw_prim = find_prim(stage, ROBOT_PATH, MOVING_JAW_LINK)
    tool_prim = find_prim(stage, ROBOT_PATH, TOOL_LINK)
    local_meshes = {}
    for link_prim, mesh_name in ((gripper_prim, FIXED_JAW_MESH), (jaw_prim, MOVING_JAW_MESH)):
        link_world = np.asarray(
            UsdGeom.Xformable(link_prim).ComputeLocalToWorldTransform(Usd.TimeCode.Default())
        ).T
        points = np.vstack(
            [p for _, p in mesh_points(find_prim(stage, str(link_prim.GetPath()), mesh_name))]
        )
        homogeneous = np.hstack([points, np.ones((len(points), 1))])
        local_meshes[mesh_name] = (np.linalg.inv(link_world) @ homogeneous.T).T

    render_product = rep.create.render_product("/World/Camera", (WIDTH, HEIGHT))
    rgb = rep.AnnotatorRegistry.get_annotator("rgb")
    rgb.attach([render_product])

    robot = Articulation(ROBOT_PATH)
    tool_view = XformPrim(str(tool_prim.GetPath()))
    gripper_view = XformPrim(str(gripper_prim.GetPath()))
    jaw_view = XformPrim(str(jaw_prim.GetPath()))
    for _ in range(args.settle_frames):
        app.update()
    omni.timeline.get_timeline_interface().play()
    app.update()

    dof_names = list(robot.dof_names)
    lower, upper = (limit.numpy()[0] for limit in robot.get_dof_limits())

    def apply(joints: np.ndarray) -> np.ndarray:
        robot.set_dof_positions(joints.reshape(1, -1))
        robot.set_dof_position_targets(joints.reshape(1, -1))
        for _ in range(3):
            app.update()
        return robot.get_dof_positions().numpy()[0]

    def capture(name: str) -> None:
        # 자세가 바뀐 직후에는 RT 렌더가 수렴하지 않았다. subframe을 여러 번 렌더한다.
        rep.orchestrator.step(rt_subframes=CAPTURE_SUBFRAMES, delta_time=0.0, pause_timeline=False)
        frame = np.asarray(rgb.get_data())[:, :, :3]
        Image.fromarray(frame).save(args.out_dir / f"pose_{name}.png")

    rng = np.random.default_rng(args.seed)
    samples = []
    fixed_poses = {"zero": np.zeros(len(dof_names))}
    for index in range(args.samples):
        # 관절 한계의 안쪽 80% 범위에서 고른다.
        middle, half = (lower + upper) / 2, (upper - lower) / 2 * 0.8
        fixed_poses[f"random_{index:02d}"] = rng.uniform(middle - half, middle + half)
    for name, joints in fixed_poses.items():
        measured = apply(joints)
        tool = world_pose(tool_view)
        samples.append(
            {
                "name": name,
                "commanded": joints.tolist(),
                "measured": measured.tolist(),
                "tool_matrix": tool.tolist(),
            }
        )
        if name in ("zero", "random_00", "random_01"):
            capture(name)

    # 그리퍼 각도별 jaw 간격. 팔은 zero 자세다.
    # 움직이는 jaw의 끝점은 힌지 축(link 좌표의 z축)에서 가장 먼 메시 점이다.
    # 간격은 이 끝점과 툴 끝점 근처 고정 jaw 점 사이의 최소 거리다.
    moving_local = local_meshes[MOVING_JAW_MESH]
    jaw_tip_local = moving_local[np.argmax(np.hypot(moving_local[:, 0], moving_local[:, 1]))]
    gripper_index = dof_names.index("gripper")
    jaw_gaps = []
    for angle in np.linspace(lower[gripper_index], upper[gripper_index], GRIPPER_ANGLES):
        joints = np.zeros(len(dof_names))
        joints[gripper_index] = angle
        measured = apply(joints)
        tool_origin = world_pose(tool_view)[:3, 3]
        fixed = (world_pose(gripper_view) @ local_meshes[FIXED_JAW_MESH].T).T[:, :3]
        fixed = fixed[np.linalg.norm(fixed - tool_origin, axis=1) < JAW_RADIUS_M]
        tip = world_pose(jaw_view) @ jaw_tip_local
        gap = float(np.min(np.linalg.norm(fixed - tip[:3], axis=1))) if len(fixed) else None
        jaw_gaps.append({"angle": float(measured[gripper_index]), "gap_m": gap})
    capture("gripper_open")

    result = {
        "dof_names": dof_names,
        "lower": lower.tolist(),
        "upper": upper.tolist(),
        "samples": samples,
        "jaw_gaps": jaw_gaps,
    }
    (args.out_dir / "inspect.json").write_text(json.dumps(result, indent=2) + "\n")
    print(f"[inspect_so101] Inspection finished. dofs={dof_names}", flush=True)


try:
    main()
except Exception:
    traceback.print_exc()
    raise
finally:
    app.close()
