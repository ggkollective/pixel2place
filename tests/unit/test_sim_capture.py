import json
from pathlib import Path

import numpy as np
import pytest
from typer.testing import CliRunner

from pixel2place.cell.camera import PinholeCamera
from pixel2place.cell.capture import compile_capture
from pixel2place.cell.config import CellConfig
from pixel2place.cell.scene import (
    BoardOffset,
    CameraOffset,
    SimScene,
    board_shift_mm,
    displace_board,
)
from pixel2place.cell.sim import (
    CAMERA_FILE,
    CAMERA_TRUTH_FILE,
    CAPTURE_IMAGE_FILE,
    CAPTURE_LOG_FILE,
    SimRunError,
    run_capture,
)
from pixel2place.cli import EXIT_INFEASIBLE, EXIT_USAGE, app
from pixel2place.config import SimSettings
from pixel2place.layout.schema import Board

runner = CliRunner()
CELL = CellConfig()
BOARD = Board()
OFFSET = BoardOffset(x=-0.002, y=0.004, yaw=np.radians(-1.0))
SHAKE = CameraOffset(
    position=(-0.006, 0.009, -0.004), rotation=tuple(np.radians([-0.7, 0.9, -1.0]))
)
# 합성 이미지에서 기대하는 셀 중심 최대 오차(mm)
SYNTHETIC_TOLERANCE_MM = 0.1


def _settings() -> SimSettings:
    return SimSettings()  # pyright: ignore[reportCallIssue]


@pytest.fixture
def capture_path(tmp_path) -> Path:
    path = tmp_path / "capture.json"
    path.write_text(compile_capture(CELL, BOARD).model_dump_json())
    return path


@pytest.fixture
def scene() -> SimScene:
    return SimScene(board=displace_board(CELL, BOARD, offset=OFFSET), camera=SHAKE)


@pytest.fixture
def scene_path(scene, tmp_path) -> Path:
    path = tmp_path / "scene.json"
    path.write_text(scene.model_dump_json())
    return path


@pytest.mark.unit
class TestRunCapture:
    """캡처 실행기 subprocess 테스트."""

    def test_reads_capture_files(self, fake_isaac, capture_path, tmp_path):
        """이미지와 카메라 파일을 읽는다. 전체 출력은 capture.log에 쓴다."""
        out_dir = tmp_path / "out"
        captured = run_capture(capture_path, out_dir=out_dir, settings=_settings())
        assert captured.image_path == out_dir / CAPTURE_IMAGE_FILE
        assert captured.camera == captured.camera_truth
        assert "Capture finished." in (out_dir / CAPTURE_LOG_FILE).read_text()

    def test_scene_moves_camera(self, fake_isaac, capture_path, scene_path, tmp_path):
        """scene.json의 카메라 오차는 실제 카메라에만 있다. 공칭 카메라는 그대로다."""
        captured = run_capture(
            capture_path, out_dir=tmp_path / "out", settings=_settings(), scene_path=scene_path
        )
        moved = np.linalg.norm(captured.camera_truth.center - captured.camera.center)
        assert moved == pytest.approx(np.linalg.norm(SHAKE.position))

    def test_runner_failure(self, fake_isaac, capture_path, tmp_path, monkeypatch):
        """실행기가 캡처 파일을 쓰지 못하면 없는 파일 이름을 알린다. 이전 파일은 지운다."""
        out_dir = tmp_path / "out"
        run_capture(capture_path, out_dir=out_dir, settings=_settings())
        monkeypatch.setenv("FAKE_ISAAC_MODE", "crash")
        with pytest.raises(SimRunError, match=CAMERA_TRUTH_FILE):
            run_capture(capture_path, out_dir=out_dir, settings=_settings())
        assert not (out_dir / CAPTURE_IMAGE_FILE).exists()


@pytest.mark.unit
class TestSimCaptureCommand:
    """sim-capture 명령 테스트."""

    def test_success(self, fake_isaac, tmp_path):
        """캡처 작업과 캡처 파일을 쓰고 경로를 출력한다."""
        out_dir = tmp_path / "out"
        result = runner.invoke(app, ["sim-capture", "-o", str(out_dir)])
        assert result.exit_code == 0, result.output
        for name in ("capture.json", CAPTURE_IMAGE_FILE, CAMERA_FILE, CAMERA_TRUTH_FILE):
            assert (out_dir / name).is_file(), name
        assert f"output={out_dir}" in result.output

    def test_unreachable_feeder_exit_one(self, fake_isaac, tmp_path):
        cell_path = tmp_path / "cell.json"
        cell_path.write_text(CellConfig(feeder_center=(0.6, 0.6)).model_dump_json())
        args = ["sim-capture", "-o", str(tmp_path / "out"), "--cell", str(cell_path)]
        result = runner.invoke(app, args)
        assert result.exit_code == EXIT_INFEASIBLE
        assert "Capture pose" in result.output

    def test_invalid_scene_exit_two(self, fake_isaac, tmp_path):
        bad = tmp_path / "scene.json"
        bad.write_text('{"camera": {"position": [0.0, 0.0]}}')
        args = ["sim-capture", "-o", str(tmp_path / "out"), "--scene", str(bad)]
        result = runner.invoke(app, args)
        assert result.exit_code == EXIT_USAGE
        assert not (tmp_path / "out").exists()

    def test_runner_failure_exit_one(self, fake_isaac, tmp_path, monkeypatch):
        monkeypatch.setenv("FAKE_ISAAC_MODE", "crash")
        result = runner.invoke(app, ["sim-capture", "-o", str(tmp_path / "out")])
        assert result.exit_code == EXIT_INFEASIBLE
        assert "Fake runner failed" in result.output


@pytest.mark.unit
class TestCalibrateBoardCommand:
    """calibrate-board 명령 테스트. 입력은 가짜 실행기의 합성 캡처다."""

    @pytest.fixture
    def capture_dir(self, fake_isaac, scene_path, tmp_path) -> Path:
        out_dir = tmp_path / "capture"
        args = ["sim-capture", "-o", str(out_dir), "--scene", str(scene_path)]
        assert runner.invoke(app, args).exit_code == 0
        return out_dir

    def _args(self, capture_dir: Path, output: Path) -> list[str]:
        image, camera = capture_dir / CAPTURE_IMAGE_FILE, capture_dir / CAMERA_FILE
        return ["calibrate-board", str(image), str(camera), "-o", str(output)]

    def test_writes_calibrated_cell(self, capture_dir, scene, tmp_path):
        """보정한 셀 설정, 보고 JSON, overlay를 쓰고 요약 줄을 출력한다."""
        output = tmp_path / "cell.json"
        result = runner.invoke(app, self._args(capture_dir, output))
        assert result.exit_code == 0, result.output
        assert result.output.startswith("markers=6/6 pins=64/64 ")
        calibrated = CellConfig.model_validate_json(output.read_text())
        assert board_shift_mm(calibrated, BOARD, placement=scene.board) < SYNTHETIC_TOLERANCE_MM
        report = json.loads((tmp_path / "cell.report.json").read_text())
        assert report["pins_used"] == 64
        assert (tmp_path / "cell.overlay.png").is_file()

    def test_makes_missing_parent_dir(self, capture_dir, tmp_path):
        """출력 경로의 부모 디렉터리가 없으면 그 디렉터리를 만든다."""
        output = tmp_path / "new" / "dir" / "cell.json"
        result = runner.invoke(app, self._args(capture_dir, output))
        assert result.exit_code == 0, result.output
        assert output.is_file()
        assert output.with_name("cell.report.json").is_file()
        assert output.with_name("cell.overlay.png").is_file()

    def test_uses_cell_option(self, capture_dir, tmp_path):
        """--cell의 표식과 보드를 믿는 값으로 쓴다. 표식이 3개이면 exit 1이다."""
        cell_path = tmp_path / "few.json"
        cell_path.write_text(CellConfig(markers=CELL.markers[:3]).model_dump_json())
        output = tmp_path / "cell.json"
        result = runner.invoke(app, [*self._args(capture_dir, output), "--cell", str(cell_path)])
        assert result.exit_code == EXIT_INFEASIBLE
        assert "fewer than 4 markers" in result.output
        assert not output.exists()

    def test_invalid_camera_exit_two(self, capture_dir, tmp_path):
        camera = PinholeCamera.model_validate_json((capture_dir / CAMERA_FILE).read_text())
        bad = tmp_path / "camera.json"
        bad.write_text(camera.model_dump_json().replace('"fx":', '"focal":'))
        image = capture_dir / CAPTURE_IMAGE_FILE
        args = ["calibrate-board", str(image), str(bad), "-o", str(tmp_path / "cell.json")]
        result = runner.invoke(app, args)
        assert result.exit_code == EXIT_USAGE
