"""기존 CLI를 실행한다. 부모 서버가 사라지면 작업 그룹을 정리한다."""

import argparse
import os
import signal
from pathlib import Path
from threading import Event, Thread
from time import sleep

from pixel2place.cli import app
from pixel2place.demo.contracts import RunRequest

PARENT_POLL_S = 1.0
ORPHAN_GRACE_S = 5.0


def watch_parent(parent: int, stopped: Event) -> None:
    """부모가 사라지면 인터럽트를 보낸다. 남은 작업은 제한 시간 뒤 정리한다."""
    while os.getppid() == parent:
        if stopped.wait(PARENT_POLL_S):
            return
    os.killpg(os.getpgrp(), signal.SIGINT)
    sleep(ORPHAN_GRACE_S)
    os.killpg(os.getpgrp(), signal.SIGKILL)


def main() -> None:
    """파일에서 검증한 원문을 읽는다. CLI의 예산과 상태 검사를 그대로 쓴다."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--parent", type=int, required=True)
    parser.add_argument("--job", type=Path, required=True)
    args = parser.parse_args()
    directory: Path = args.job
    request = RunRequest.model_validate_json((directory / "request.json").read_text())
    stopped = Event()
    watcher = Thread(target=watch_parent, args=(args.parent, stopped))
    watcher.start()
    try:
        app(
            args=[
                "agent-cell",
                "-o",
                str(directory / "run"),
                "--scene",
                str(directory / "scene.json"),
                "--",
                request.prompt,
            ]
        )
    finally:
        stopped.set()
        watcher.join()


if __name__ == "__main__":
    main()
