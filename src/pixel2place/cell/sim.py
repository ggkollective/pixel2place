"""Isaac Sim 실행기(`sim/place_task.py`)를 subprocess로 실행한다.

실행기는 Isaac `python.sh`로 돈다. pixel2place의 uv 환경과 다르다.
두 프로세스는 파일로 데이터를 주고받는다.
- 부품 작업: `task.json`을 주고 `result.json`을 받는다.
- 시연 기록: 부품 작업에 손목 카메라 영상과 프레임별 관절 값(`episode.npz`)을 더 받는다.
- 발표 영상: 부품 작업에 발표용 카메라 영상(`present.mp4`)을 더 받는다.
- 폐루프 실행: 실행기가 제어기 서버에서 관절 목표를 받는다. 실행기는 시연 기록도 쓴다.
- 캘리브레이션 캡처: `capture.json`을 주고 top-down 이미지와 카메라 파일 2개를 받는다.
"""

import contextlib
import logging
import os
import signal
import subprocess
import threading
from collections import deque
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

from pixel2place.cell.camera import PinholeCamera
from pixel2place.cell.policy_link import PolicyAddress
from pixel2place.cell.result import SimResult
from pixel2place.cell.scene import SimScene, check_scene_parts
from pixel2place.cell.task import Task
from pixel2place.config import SimSettings

logger = logging.getLogger(__name__)

RUNNER_SCRIPT = Path(__file__).resolve().parents[3] / "sim" / "place_task.py"
RESULT_FILE = "result.json"
# 배치가 끝난 보드의 top-down 이미지. 실행기가 `result.json`보다 먼저 쓴다.
FINAL_IMAGE_FILE = "topdown_final.png"
# 시연 기록의 출력. 사선 카메라 영상, 손목 카메라 영상과 프레임별 관절 값이다.
FRONT_VIDEO_FILE = "cell.mp4"
WRIST_VIDEO_FILE = "wrist.mp4"
EPISODE_FILE = "episode.npz"
RECORD_OUTPUTS = (FRONT_VIDEO_FILE, WRIST_VIDEO_FILE, EPISODE_FILE)
# 발표용 카메라 영상. 크기는 1280×720이다. 제어기의 관측과 판정은 이 영상을 쓰지 않는다.
PRESENT_VIDEO_FILE = "present.mp4"
LOG_FILE = "place_task.log"
CAPTURE_IMAGE_FILE = "topdown_start.png"
# 공칭 카메라. 추정은 intrinsics와 공칭 자세만 쓴다.
CAMERA_FILE = "camera.json"
# 흔들림을 적용한 실제 카메라. 평가에만 쓴다.
CAMERA_TRUTH_FILE = "camera_truth.json"
CAPTURE_OUTPUTS = (CAPTURE_IMAGE_FILE, CAMERA_FILE, CAMERA_TRUTH_FILE)
CAPTURE_LOG_FILE = "capture.log"
# 실행기 로그 줄의 앞머리. 이 줄만 진행 상황으로 전달한다.
RUNNER_LOG_PREFIX = "[place_task]"
# 실패하면 로그의 마지막 줄들을 오류 메시지에 넣는다.
ERROR_TAIL_LINES = 20
# 실행기가 출력 파일을 모두 쓴 뒤에 쓰는 로그 줄
RUNNER_DONE_MARKERS = ("Task finished.", "Capture finished.")
# 실행기가 끝 로그를 쓴 뒤 프로세스가 끝나기를 기다리는 시간(초).
# Isaac은 종료 단계에서 멈출 수 있다. 그때는 프로세스를 끝내고 출력 파일로 성공을 판정한다.
EXIT_GRACE_S = 60.0
PROC_DIR = Path("/proc")


class SimSetupError(ValueError):
    """Isaac 실행의 입력이 맞지 않는다. 필요한 파일이 없거나 장면이 작업과 맞지 않는다."""


class SimRunError(RuntimeError):
    """Isaac 실행기가 결과를 쓰지 못했다."""


@dataclass(frozen=True)
class PolicyRun:
    """폐루프 실행의 설정."""

    address: PolicyAddress
    context_path: Path
    """에피소드의 문맥(JSON). 실행기가 제어기 서버의 `reset` 요청에 그대로 넣는다."""
    max_steps: int
    """프레임 수의 한도. 서버가 더 작은 값을 주면 그 값을 쓴다."""


def run_place_task(
    task_path: Path,
    *,
    out_dir: Path,
    settings: SimSettings,
    usd_path: Path | None = None,
    scene_path: Path | None = None,
    record: bool = False,
    policy: PolicyRun | None = None,
    present: bool = False,
) -> SimResult:
    """`task.json`을 Isaac Sim에서 실행하고 `result.json`을 읽어서 반환한다.

    `scene_path`가 있으면 실행기가 보드를 `scene.json`의 자세에 만든다.
    `record`가 참이면 실행기가 `RECORD_OUTPUTS`도 쓴다.
    `policy`가 있으면 실행기가 waypoint 대신 제어기 서버의 관절 목표를 따른다. 기록도 켠다.
    `present`가 참이면 실행기가 `PRESENT_VIDEO_FILE`도 쓴다. 그 실행은 시간이 더 든다.
    전체 출력은 `place_task.log`에 쓴다.

    장면의 결함이나 이미 놓인 부품이 작업과 맞지 않으면 실행기를 띄우지 않고
    `SimSetupError`를 낸다.
    """
    if scene_path is not None:
        task = Task.model_validate_json(task_path.read_text(encoding="utf-8"))
        scene = SimScene.model_validate_json(scene_path.read_text(encoding="utf-8"))
        try:
            check_scene_parts(scene, task, policy=policy is not None)
        except ValueError as exc:
            raise SimSetupError(str(exc)) from exc
    flags: tuple[str, ...] = ()
    if policy is not None:
        flags = (
            "--policy",
            str(policy.address),
            "--policy-context",
            str(policy.context_path.resolve()),
            "--max-steps",
            str(policy.max_steps),
        )
    elif record:
        flags = ("--record",)
    outputs: tuple[str, ...] = (RESULT_FILE, *RECORD_OUTPUTS) if flags else (RESULT_FILE,)
    stale: tuple[str, ...] = ()
    if present:
        flags += ("--present",)
        outputs += (PRESENT_VIDEO_FILE,)
    else:
        # 이전 실행의 발표 영상이 남으면 호출자가 그 영상을 이 실행의 영상으로 읽는다.
        stale = (PRESENT_VIDEO_FILE,)
    _run_runner(
        "task",
        task_path,
        out_dir=out_dir,
        settings=settings,
        usd_path=usd_path,
        scene_path=scene_path,
        outputs=outputs,
        stale=stale,
        log_name=LOG_FILE,
        flags=flags,
    )
    return SimResult.model_validate_json((out_dir / RESULT_FILE).read_text(encoding="utf-8"))


@dataclass(frozen=True)
class SimCapture:
    """캘리브레이션 캡처의 출력."""

    image_path: Path
    camera: PinholeCamera
    """intrinsics와 공칭 자세."""
    camera_truth: PinholeCamera
    """`scene.json`의 카메라 오차를 적용한 실제 자세."""


def run_capture(
    capture_path: Path,
    *,
    out_dir: Path,
    settings: SimSettings,
    usd_path: Path | None = None,
    scene_path: Path | None = None,
) -> SimCapture:
    """`capture.json`을 Isaac Sim에서 실행하고 캡처 파일을 읽어서 반환한다.

    `scene_path`가 있으면 실행기가 보드와 top-down 카메라를 `scene.json`의 자세에 만든다.
    전체 출력은 `capture.log`에 쓴다.
    """
    _run_runner(
        "capture",
        capture_path,
        out_dir=out_dir,
        settings=settings,
        usd_path=usd_path,
        scene_path=scene_path,
        outputs=CAPTURE_OUTPUTS,
        log_name=CAPTURE_LOG_FILE,
    )
    return read_capture(out_dir)


def read_capture(out_dir: Path) -> SimCapture:
    """`run_capture`가 `out_dir`에 쓴 캡처 파일을 읽는다."""
    return SimCapture(
        image_path=out_dir / CAPTURE_IMAGE_FILE,
        camera=_read_camera(out_dir / CAMERA_FILE),
        camera_truth=_read_camera(out_dir / CAMERA_TRUTH_FILE),
    )


def _run_runner(
    mode: Literal["task", "capture"],
    input_path: Path,
    *,
    out_dir: Path,
    settings: SimSettings,
    usd_path: Path | None,
    scene_path: Path | None,
    outputs: tuple[str, ...],
    log_name: str,
    stale: tuple[str, ...] = (),
    flags: tuple[str, ...] = (),
) -> None:
    """실행기를 `--<mode> <input_path>`와 `flags`로 실행하고 출력 파일이 모두 있는지 확인한다.

    `outputs`는 이 실행이 쓰는 파일이다. `stale`은 이 실행이 쓰지 않는 파일이다.

    단계:
    1. `require_sim_files`로 실행에 필요한 파일을 확인한다.
    2. 이전 실행의 `outputs`와 `stale`을 지우고 실행기를 실행한다.
       전체 출력은 `log_name` 파일에 쓴다.
    3. `outputs`의 파일이 하나라도 없으면 `SimRunError`를 낸다.
       Isaac은 예외가 나도 exit 0으로 끝날 수 있다. 그래서 출력 파일로 성공을 판정한다.
    """
    python_sh, usd = require_sim_files(settings, usd_path=usd_path)
    out_dir.mkdir(parents=True, exist_ok=True)
    for name in (*outputs, *stale):
        (out_dir / name).unlink(missing_ok=True)
    command = [
        str(python_sh),
        str(RUNNER_SCRIPT),
        f"--{mode}",
        str(input_path.resolve()),
        "--usd",
        str(usd.resolve()),
        "--out-dir",
        str(out_dir.resolve()),
    ]
    if scene_path is not None:
        command += ["--scene", str(scene_path.resolve())]
    command += flags
    logger.info(
        "Isaac runner started.",
        extra={"mode": mode, "input": str(input_path), "out_dir": str(out_dir)},
    )
    tail = _run(command, log_path=out_dir / log_name, timeout_s=settings.sim_timeout_s)
    missing = [name for name in outputs if not (out_dir / name).is_file()]
    if missing:
        raise SimRunError(
            f"Isaac runner did not write {', '.join(missing)}. Last log lines:\n" + "\n".join(tail)
        )


def _read_camera(path: Path) -> PinholeCamera:
    return PinholeCamera.model_validate_json(path.read_text(encoding="utf-8"))


def require_sim_files(settings: SimSettings, *, usd_path: Path | None = None) -> tuple[Path, Path]:
    """`python.sh`, 실행기 스크립트, SO-101 USD가 있는지 확인한다.

    `(python.sh 경로, USD 경로)`를 반환한다. 파일이 없으면 `SimSetupError`를 낸다.
    """
    python_sh = settings.isaac_sim_path / "python.sh"
    usd = usd_path or settings.so101_usd_path
    for path, hint in (
        (python_sh, "Set ISAAC_SIM_PATH to the Isaac Sim release directory."),
        (RUNNER_SCRIPT, "Run the command from the pixel2place repository."),
        (usd, "Make it with sim/import_so101.py."),
    ):
        if not path.is_file():
            raise SimSetupError(f"File {path} does not exist. {hint}")
    return python_sh, usd


def _run(command: list[str], *, log_path: Path, timeout_s: float) -> list[str]:
    """명령을 실행하고 출력을 로그 파일에 쓴다. 출력의 마지막 줄들을 반환한다.

    - 출력이 없어도 timeout이 지나면 프로세스를 끝낸다. 그래서 timer 스레드로 끝낸다.
    - 실행기가 끝 로그(`RUNNER_DONE_MARKERS`)를 쓴 뒤 `EXIT_GRACE_S` 안에 끝나지 않으면
      프로세스를 끝낸다. 이 경우는 timeout이 아니다. 출력 파일은 이미 있다.
    - `python.sh`는 Isaac Python을 자식 프로세스로 실행한다. 그래서 자손 프로세스를 모두 끝낸다.
      자손이 남으면 출력 파이프가 닫히지 않아서 이 함수가 돌아오지 않는다.
    """
    tail: deque[str] = deque(maxlen=ERROR_TAIL_LINES)
    timed_out = threading.Event()
    with (
        log_path.open("w", encoding="utf-8") as log_file,
        subprocess.Popen(
            command,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            errors="replace",
        ) as process,
    ):

        def kill() -> None:
            # 실행기는 호출한 프로세스의 그룹에 둔다. 데모의 중단이 그 그룹 전체를 끝내기 때문이다.
            # 그래서 그룹이 아니라 자손 프로세스를 찾아서 끝낸다.
            for pid in [*_descendants(process.pid), process.pid]:
                with contextlib.suppress(ProcessLookupError):
                    os.kill(pid, signal.SIGKILL)

        def kill_at_timeout() -> None:
            timed_out.set()
            kill()

        timers = [threading.Timer(timeout_s, kill_at_timeout)]
        timers[0].start()
        try:
            assert process.stdout is not None
            for line in process.stdout:
                log_file.write(line)
                tail.append(line.rstrip())
                if RUNNER_LOG_PREFIX in line:
                    logger.info(line.strip())
                    if len(timers) == 1 and any(marker in line for marker in RUNNER_DONE_MARKERS):
                        timers.append(threading.Timer(EXIT_GRACE_S, kill))
                        timers[1].start()
            process.wait()
        finally:
            for timer in timers:
                timer.cancel()
    if timed_out.is_set():
        raise SimRunError(f"Isaac runner did not finish in {timeout_s:.0f} s.")
    return list(tail)


def _descendants(pid: int) -> list[int]:
    """`pid`의 모든 자손 프로세스의 pid. `/proc`의 부모 pid로 찾는다."""
    parents: dict[int, int] = {}
    for stat_path in PROC_DIR.glob("[0-9]*/stat"):
        try:
            # 형식은 `pid (이름) 상태 부모pid ...`다. 이름에 공백과 괄호가 있을 수 있다.
            fields = stat_path.read_text(encoding="utf-8", errors="replace").rsplit(")", 1)[1]
            parents[int(stat_path.parent.name)] = int(fields.split()[1])
        except (OSError, IndexError, ValueError):
            continue
    found: list[int] = []
    frontier = [pid]
    while frontier:
        parent = frontier.pop()
        children = [child for child, parent_pid in parents.items() if parent_pid == parent]
        found += children
        frontier += children
    return found
