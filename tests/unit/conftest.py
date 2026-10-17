"""단위 테스트 공용 픽스처."""

import stat
import sys
from pathlib import Path

import pytest

from pixel2place.agent import inference

REPO_ROOT = Path(__file__).parents[2]
FAKE_RUNNER = REPO_ROOT / "tests" / "fakes" / "fake_isaac_runner.py"


@pytest.fixture
def fake_isaac(tmp_path, monkeypatch) -> Path:
    """가짜 Isaac release 디렉터리. `python.sh`가 가짜 실행기를 실행한다.

    가짜 실행기는 `tests.fakes`를 import한다. 그래서 `python.sh`가 저장소 루트를 PYTHONPATH에 둔다.
    저장소 `.env`를 읽지 않도록 작업 디렉터리를 옮긴다.
    """
    isaac_dir = tmp_path / "isaac"
    isaac_dir.mkdir()
    python_sh = isaac_dir / "python.sh"
    python_sh.write_text(
        f'#!/bin/sh\nPYTHONPATH="{REPO_ROOT}" exec "{sys.executable}" "{FAKE_RUNNER}" "$@"\n'
    )
    python_sh.chmod(python_sh.stat().st_mode | stat.S_IEXEC)
    usd = tmp_path / "so101.usda"
    usd.write_text("#usda 1.0\n")
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("ISAAC_SIM_PATH", str(isaac_dir))
    monkeypatch.setenv("SO101_USD_PATH", str(usd))
    return isaac_dir


@pytest.fixture
def retry_waits(monkeypatch: pytest.MonkeyPatch) -> list[float]:
    """재전송 앞의 대기 함수를 기록 함수로 바꾼다. 테스트는 기다리지 않는다.

    목록에는 요청한 대기 초가 순서대로 있다.
    """
    waits: list[float] = []
    monkeypatch.setattr(inference, "sleep", waits.append)
    return waits
