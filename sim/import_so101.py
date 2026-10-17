"""SO-101 URDF를 USD로 바꾼다.

Isaac `python.sh`로 실행한다. URDF는 `sim/scripts/fetch_so101.sh`가 받는다.

사용법: python.sh sim/import_so101.py --urdf URDF --out-dir DIR
출력: 변환한 USD 경로를 마지막 줄에 쓴다.
"""

import argparse
import traceback
from pathlib import Path

from isaacsim import SimulationApp

parser = argparse.ArgumentParser()
parser.add_argument("--urdf", type=Path, required=True)
parser.add_argument("--out-dir", type=Path, required=True)
args = parser.parse_args()

app = SimulationApp({"headless": True})

import omni.kit.app  # noqa: E402

# importer가 이 확장들을 쓴다. 공식 예제와 같은 순서로 켠다.
extensions = omni.kit.app.get_app().get_extension_manager()
extensions.set_extension_enabled_immediate("omni.scene.optimizer.core", True)
extensions.set_extension_enabled_immediate("isaacsim.robot.schema", True)

from isaacsim.asset.importer.urdf.impl import URDFImporter, URDFImporterConfig  # noqa: E402


def main() -> None:
    args.out_dir.mkdir(parents=True, exist_ok=True)
    config = URDFImporterConfig()
    config.urdf_path = str(args.urdf.resolve())
    config.usd_path = str(args.out_dir.resolve())
    # 로봇 base를 world에 고정한다. 관절은 위치 목표로 구동한다.
    config.fix_base = True
    config.joint_target_type = "position"
    # gripper_frame_link(툴 끝점)를 남긴다. 고정 관절을 합치면 이 link가 없어진다.
    config.merge_fixed_joints = False
    usd_path = URDFImporter(config).import_urdf()
    print(f"[import_so101] USD is ready. usd={usd_path}", flush=True)


try:
    main()
except Exception:
    traceback.print_exc()
    raise
finally:
    app.close()
