"""실제 도메인 함수와 Fake Cosmos·Isaac으로 조립 도구 경계를 검증한다."""

import json
import logging
import math
import subprocess
from pathlib import Path
from typing import Any

import httpx2 as httpx
import pytest
from openai import DefaultHttpxClient, OpenAI
from pydantic import ValidationError

from pixel2place.agent.artifacts import ArtifactError, ArtifactStore, Manifest
from pixel2place.agent.contracts import (
    CalibrateBoardArguments,
    CalibrationResult,
    ErrorCode,
    ErrorResult,
    ExecuteCellArguments,
    ExecutionResult,
    LayoutResult,
    PrepareLayoutArguments,
    ToolState,
)
from pixel2place.agent.tools import AssemblyTools, ToolLedger
from pixel2place.cell.config import CellConfig
from pixel2place.cell.image_check import ImageCheckReport, PartStatus
from pixel2place.cell.scene import (
    BoardOffset,
    CameraOffset,
    SimScene,
    board_shift_mm,
    displace_board,
    placement_of,
)
from pixel2place.cell.sim import CAMERA_TRUTH_FILE, PRESENT_VIDEO_FILE, SimSetupError
from pixel2place.cell.task import Task, compile_task
from pixel2place.config import SimSettings
from pixel2place.layout.schema import Board, Layout
from pixel2place.reasoner.client import CosmosReasoner
from tests.factories import make_planner_response_json
from tests.fakes.fake_openai import FakeOpenAI

pytestmark = [pytest.mark.unit, pytest.mark.usefixtures("fake_isaac")]
GOAL = "Place all four parts on the board. Keep their colors."
IMAGE_CHECK_PATH = "execution/image_check.json"
FINAL_IMAGE_PATH = "execution/sim/topdown_final.png"
# 카메라 판정이 없는 실행의 등록. 순서는 등록한 순서다.
CHECKED_KINDS = ["inputs", "layout", "calibration", "check"]


def _tools(
    root: Path,
    layout: Layout,
    *,
    responses: list[str | None] | None = None,
    finish_reason: str = "stop",
    cell: CellConfig | None = None,
    scene: SimScene | None = None,
) -> tuple[AssemblyTools, FakeOpenAI]:
    client = FakeOpenAI(
        responses if responses is not None else [make_planner_response_json(layout)],
        finish_reason=finish_reason,
    )
    tools = AssemblyTools(
        out_dir=root,
        prompt=GOAL,
        cell=cell or CellConfig(),
        scene=scene or SimScene(),
        reasoner=CosmosReasoner(client=client, model="fake"),
        sim_settings=SimSettings(),  # pyright: ignore[reportCallIssue]
    )
    return tools, client


def _prepare(tools: AssemblyTools) -> str:
    result = tools.prepare_layout(PrepareLayoutArguments(retry=False), tool_call_id="prepare")
    assert isinstance(result, LayoutResult), result
    return result.artifacts[0].id


def _calibrate(tools: AssemblyTools, layout_id: str) -> str:
    result = tools.calibrate_board(
        CalibrateBoardArguments(layout_id=layout_id), tool_call_id="calibrate"
    )
    assert isinstance(result, CalibrationResult), result
    return result.artifacts[0].id


def _execute(tools: AssemblyTools, layout_id: str, calibration_id: str) -> ExecutionResult:
    result = tools.execute_cell(
        ExecuteCellArguments(layout_id=layout_id, calibration_id=calibration_id),
        tool_call_id="execute",
    )
    assert isinstance(result, ExecutionResult), result
    return result


def _run_to_check(root: Path, layout: Layout) -> tuple[AssemblyTools, ExecutionResult]:
    """준비, 보정과 배치를 차례로 실행한다. 배치는 현재 `FAKE_ISAAC_MODE`를 따른다."""
    tools, _ = _tools(root, layout)
    layout_id = _prepare(tools)
    return tools, _execute(tools, layout_id, _calibrate(tools, layout_id))


def _manifest(tools: AssemblyTools) -> Manifest:
    return Manifest.model_validate_json((tools.root / "manifest.json").read_text())


def _image_check(tools: AssemblyTools) -> ImageCheckReport:
    return ImageCheckReport.model_validate_json((tools.root / IMAGE_CHECK_PATH).read_text())


@pytest.fixture
def isaac_calls(monkeypatch: pytest.MonkeyPatch) -> list[tuple[str, str, ToolLedger]]:
    """subprocess 경계에서 실제 Fake 실행을 관찰한다. 도메인 함수는 교체하지 않는다."""
    calls: list[tuple[str, str, ToolLedger]] = []
    real_popen = subprocess.Popen

    def record(args: list[str], **kwargs: Any) -> subprocess.Popen[str]:
        scene_path = Path(args[args.index("--scene") + 1])
        root = scene_path.parent.parent
        ledger = ToolLedger.model_validate_json((root / "tool_state.json").read_text())
        mode = "capture" if "--capture" in args else "task"
        calls.append((mode, str(scene_path), ledger))
        return real_popen(args, **kwargs)

    monkeypatch.setattr(subprocess, "Popen", record)
    return calls


def test_scene_keeps_joint_error(tmp_path: Path, valid_layout: Layout) -> None:
    """scene에 보드 자세를 채워도 관절 센서 오차는 그대로 남는다."""
    row = (0.01, -0.01, 0.0, 0.02, 0.0)
    _tools(tmp_path / "run", valid_layout, scene=SimScene(joint_error=(row,)))
    saved = SimScene.model_validate_json(
        (tmp_path / "run" / "evaluation" / "scene.json").read_text()
    )
    assert saved.joint_error == (row,)
    assert saved.board == placement_of(CellConfig(), Board())


@pytest.mark.parametrize("offset", [False, True], ids=["nominal", "board-and-camera-error"])
def test_calibration_to_execution_uses_fixed_scene(
    tmp_path: Path,
    valid_layout: Layout,
    isaac_calls: list[tuple[str, str, ToolLedger]],
    offset: bool,
) -> None:
    """이미지 보정과 재컴파일로 판정을 통과한다. 참값은 실행과 평가에만 쓴다."""
    cell = CellConfig()
    scene = SimScene()
    if offset:
        scene = SimScene(
            board=displace_board(
                cell, Board(), offset=BoardOffset(x=-0.002, y=0.004, yaw=math.radians(-1))
            ),
            camera=CameraOffset(
                position=(-0.006, 0.009, -0.004),
                rotation=(math.radians(-0.7), math.radians(0.9), math.radians(-1.0)),
            ),
        )
    tools, client = _tools(tmp_path / "run", valid_layout, scene=scene)
    layout_id = _prepare(tools)
    calibration_id = _calibrate(tools, layout_id)
    result = _execute(tools, layout_id, calibration_id)
    assert result.summary.passed
    assert result.summary.part_count == result.summary.passed_count == 4
    assert tools.ledger.state == ToolState.CHECKED
    assert tools.verified_check_passed()
    assert [mode for mode, _, _ in isaac_calls] == ["capture", "task"]
    assert isaac_calls[0][1] == isaac_calls[1][1]
    assert isaac_calls[0][2].capture_consumed
    assert not isaac_calls[0][2].place_consumed
    assert isaac_calls[1][2].place_consumed
    assert client.calls[0]["messages"][1]["content"] == GOAL
    corrected = CellConfig.model_validate_json((tools.root / "calibration/cell.json").read_text())
    actual = scene.board or placement_of(cell, Board())
    assert board_shift_mm(corrected, Board(), placement=actual) < 0.1
    assert corrected.feeder_directions == cell.feeder_directions
    parsed = Layout.model_validate_json((tools.root / "attempts/01/layout.json").read_text())
    task = Task.model_validate_json((tools.root / "execution/task.json").read_text())
    assert task == compile_task(parsed, corrected)
    saved_scene = SimScene.model_validate_json((tools.root / "evaluation/scene.json").read_text())
    assert saved_scene.board == actual
    assert (tools.root / "evaluation" / CAMERA_TRUTH_FILE).is_file()
    assert not (tools.root / "capture" / CAMERA_TRUTH_FILE).exists()
    # 배치는 데모 화면이 보여 주는 발표 영상을 쓴다.
    assert (tools.root / "execution/sim" / PRESENT_VIDEO_FILE).is_file()
    summaries = "\n".join(path.read_text() for path in (tools.root / "tool_results").iterdir())
    for excluded in (
        "camera_truth",
        "scene.json",
        "board_origin",
        str(tools.root),
        "sha256",
        "image_check",
    ):
        assert excluded not in summaries
    if offset:
        nominal = json.loads((tools.root / "capture/camera.json").read_text())
        truth = json.loads((tools.root / "evaluation/camera_truth.json").read_text())
        assert nominal != truth
    manifest = _manifest(tools)
    assert len(manifest.artifacts) == 5
    assert len({record.id for record in manifest.artifacts}) == 5
    # 추정 카메라와 보정한 보드로 읽은 최종 이미지에서 모든 부품이 계획한 칸에 있다.
    image_check = _image_check(tools)
    assert image_check.passed and image_check.match_count == 4, image_check


def test_repaired_layout_is_checked_after_repair(tmp_path: Path, overlap_layout: Layout) -> None:
    """수리 전 위반은 성공을 막지 않는다. 수리 후 계획과 파싱 결과를 검사한다."""
    text = make_planner_response_json(overlap_layout)
    tools, client = _tools(tmp_path / "run", overlap_layout, responses=[text, text])
    result = tools.prepare_layout(PrepareLayoutArguments(retry=False), tool_call_id="prepare")
    assert isinstance(result, LayoutResult)
    assert result.summary.plan_feasible and result.summary.parsed_feasible
    assert result.summary.repair_count > 0
    assert len(client.calls) == 2


@pytest.mark.parametrize("second_valid", [True, False])
def test_layout_retry_preserves_goal_and_stops_after_two_attempts(
    tmp_path: Path, valid_layout: Layout, second_valid: bool
) -> None:
    responses: list[str | None] = ["bad", "bad"]
    responses += [make_planner_response_json(valid_layout)] if second_valid else ["bad", "bad"]
    tools, client = _tools(tmp_path / "run", valid_layout, responses=responses)
    first = tools.prepare_layout(PrepareLayoutArguments(retry=False), tool_call_id="first")
    assert isinstance(first, ErrorResult)
    assert first.error_code == ErrorCode.LAYOUT_INVALID and first.recoverable
    assert tools.ledger.state == ToolState.LAYOUT_RETRYABLE
    second = tools.prepare_layout(PrepareLayoutArguments(retry=True), tool_call_id="retry")
    retry_prompt = client.calls[2]["messages"][1]["content"]
    assert retry_prompt.startswith(GOAL + "\n\n")
    assert "error_code=layout_invalid" in retry_prompt
    assert "Keep all required parts and conditions" in retry_prompt
    assert (tools.root / "attempts/01/plan_raw_response.txt").read_text().count("bad") == 2
    if second_valid:
        assert isinstance(second, LayoutResult)
        assert len(client.calls) == 3
    else:
        assert isinstance(second, ErrorResult) and not second.recoverable
        assert tools.ledger.state == ToolState.ABORTED
        assert len(client.calls) == 4
    third = tools.prepare_layout(PrepareLayoutArguments(retry=True), tool_call_id="third")
    assert isinstance(third, ErrorResult) and third.error_code == ErrorCode.PROTOCOL_ERROR
    assert tools.ledger.layout_attempts == 2
    assert len(client.calls) == (3 if second_valid else 4)


def test_truncated_layout_is_not_retried(tmp_path: Path, valid_layout: Layout) -> None:
    tools, client = _tools(tmp_path / "run", valid_layout, finish_reason="length")
    result = tools.prepare_layout(PrepareLayoutArguments(retry=False), tool_call_id="prepare")
    assert isinstance(result, ErrorResult)
    assert result.error_code == ErrorCode.REASONER_TRUNCATED and not result.recoverable
    assert tools.ledger.state == ToolState.ABORTED
    assert len(client.calls) == 1


def test_cosmos_http_failure_is_not_layout_retry(tmp_path: Path) -> None:
    def fail(request: httpx.Request) -> httpx.Response:
        return httpx.Response(503, json={"error": {"message": "private detail"}})

    with OpenAI(
        api_key="fake-key",
        base_url="https://cosmos.invalid/v1",
        max_retries=0,
        http_client=DefaultHttpxClient(transport=httpx.MockTransport(fail)),
    ) as client:
        tools = AssemblyTools(
            out_dir=tmp_path / "run",
            prompt=GOAL,
            cell=CellConfig(),
            scene=SimScene(),
            reasoner=CosmosReasoner(client=client, model="fake"),
            sim_settings=SimSettings(),  # pyright: ignore[reportCallIssue]
        )
        result = tools.prepare_layout(PrepareLayoutArguments(retry=False), tool_call_id="prepare")
    assert isinstance(result, ErrorResult)
    assert result.error_code == ErrorCode.REASONER_ERROR and not result.recoverable
    assert "private detail" not in result.model_dump_json()


@pytest.mark.parametrize("failure", ["markers", "compile", "runner"])
def test_calibration_failure_never_places_or_repeats_capture(
    tmp_path: Path,
    valid_layout: Layout,
    monkeypatch: pytest.MonkeyPatch,
    isaac_calls: list[tuple[str, str, ToolLedger]],
    failure: str,
) -> None:
    cell = CellConfig()
    expected = ErrorCode.RUNNER_ERROR
    if failure == "markers":
        cell = CellConfig(markers=cell.markers[:3])
        expected = ErrorCode.CALIBRATION_FAILED
    elif failure == "compile":
        cell = CellConfig(feeder_center=(0.6, 0.6))
        expected = ErrorCode.COMPILE_FAILED
    else:
        monkeypatch.setenv("FAKE_ISAAC_MODE", "crash")
    tools, _ = _tools(tmp_path / "run", valid_layout, cell=cell)
    layout_id = _prepare(tools)
    arguments = CalibrateBoardArguments(layout_id=layout_id)
    result = tools.calibrate_board(arguments, tool_call_id="calibrate")
    assert isinstance(result, ErrorResult) and result.error_code == expected
    assert not result.recoverable and tools.ledger.state == ToolState.ABORTED
    before = len(isaac_calls)
    assert before == (0 if failure == "compile" else 1)
    tools.calibrate_board(arguments, tool_call_id="repeat")
    tools.execute_cell(
        ExecuteCellArguments(layout_id=layout_id, calibration_id="0" * 32), tool_call_id="place"
    )
    assert len(isaac_calls) == before


@pytest.mark.parametrize(
    "mode",
    ["crash", "no_present_video", "wrong_version", "missing_part", "invalid_result", "miss"],
)
def test_execution_failure_keeps_evidence_and_never_repeats(
    tmp_path: Path,
    valid_layout: Layout,
    monkeypatch: pytest.MonkeyPatch,
    isaac_calls: list[tuple[str, str, ToolLedger]],
    mode: str,
) -> None:
    tools, _ = _tools(tmp_path / "run", valid_layout)
    layout_id = _prepare(tools)
    calibration_id = _calibrate(tools, layout_id)
    monkeypatch.setenv("FAKE_ISAAC_MODE", mode)
    arguments = ExecuteCellArguments(layout_id=layout_id, calibration_id=calibration_id)
    result = tools.execute_cell(arguments, tool_call_id="execute")
    if mode == "miss":
        assert isinstance(result, ExecutionResult) and result.status == "ok"
        assert not result.summary.passed and result.summary.failed_parts
        assert tools.ledger.state == ToolState.CHECKED
        assert tools.ledger.error_code == ErrorCode.CHECK_FAILED
        assert not tools.verified_check_passed()
    else:
        runner_failed = mode in ("crash", "no_present_video")
        expected = ErrorCode.RUNNER_ERROR if runner_failed else ErrorCode.RESULT_MISMATCH
        assert isinstance(result, ErrorResult) and result.error_code == expected
        assert tools.ledger.state == ToolState.ABORTED and not result.recoverable
    assert (tools.root / "execution/sim/place_task.log").is_file()
    assert [call[0] for call in isaac_calls] == ["capture", "task"]
    tools.execute_cell(arguments, tool_call_id="repeat")
    assert len(isaac_calls) == 2


def test_compile_failure_after_calibration_does_not_place(
    tmp_path: Path, valid_layout: Layout, isaac_calls: list[tuple[str, str, ToolLedger]]
) -> None:
    cell = CellConfig(part_chamfer=0.004)
    tools, _ = _tools(tmp_path / "run", valid_layout, cell=cell)
    layout_id = _prepare(tools)
    calibration_id = _calibrate(tools, layout_id)
    result = tools.execute_cell(
        ExecuteCellArguments(layout_id=layout_id, calibration_id=calibration_id),
        tool_call_id="execute",
    )
    assert isinstance(result, ErrorResult) and result.error_code == ErrorCode.COMPILE_FAILED
    assert len(isaac_calls) == 1 and not tools.ledger.place_consumed


@pytest.mark.parametrize("change", ["layout", "scene", "manifest", "unknown", "foreign", "symlink"])
def test_bad_artifact_does_not_start_capture(
    tmp_path: Path,
    valid_layout: Layout,
    isaac_calls: list[tuple[str, str, ToolLedger]],
    change: str,
) -> None:
    tools, _ = _tools(tmp_path / "run", valid_layout)
    layout_id = _prepare(tools)
    if change == "unknown":
        layout_id = "0" * 32
    elif change == "foreign":
        other, _ = _tools(tmp_path / "other", valid_layout)
        layout_id = _prepare(other)
    elif change == "symlink":
        path = tools.root / "attempts/01/layout.json"
        copy = tmp_path / "copy.json"
        copy.write_bytes(path.read_bytes())
        path.unlink()
        path.symlink_to(copy)
    else:
        relative = {
            "layout": "attempts/01/layout.json",
            "scene": "evaluation/scene.json",
            "manifest": "manifest.json",
        }[change]
        path = tools.root / relative
        path.write_text(path.read_text() + " ")
    result = tools.calibrate_board(
        CalibrateBoardArguments(layout_id=layout_id), tool_call_id="calibrate"
    )
    assert isinstance(result, ErrorResult) and result.error_code == ErrorCode.PROTOCOL_ERROR
    assert not isaac_calls


@pytest.mark.parametrize("change", ["calibration", "foreign-id", "duplicate-call"])
def test_bad_calibration_does_not_start_place(
    tmp_path: Path,
    valid_layout: Layout,
    isaac_calls: list[tuple[str, str, ToolLedger]],
    change: str,
) -> None:
    tools, _ = _tools(tmp_path / "run", valid_layout)
    layout_id = _prepare(tools)
    calibration_id = _calibrate(tools, layout_id)
    call_id = "execute"
    if change == "calibration":
        path = tools.root / "calibration/cell.json"
        path.write_text(path.read_text() + " ")
    elif change == "foreign-id":
        calibration_id = layout_id
    else:
        call_id = "calibrate"
    result = tools.execute_cell(
        ExecuteCellArguments(layout_id=layout_id, calibration_id=calibration_id),
        tool_call_id=call_id,
    )
    assert isinstance(result, ErrorResult) and result.error_code == ErrorCode.PROTOCOL_ERROR
    assert len(isaac_calls) == 1 and not tools.ledger.place_consumed


def test_forged_check_cannot_support_final_success(tmp_path: Path, valid_layout: Layout) -> None:
    tools, _ = _tools(tmp_path / "run", valid_layout)
    layout_id = _prepare(tools)
    calibration_id = _calibrate(tools, layout_id)
    _execute(tools, layout_id, calibration_id)
    path = tools.root / "execution/check.json"
    path.write_text(path.read_text() + " ")
    with pytest.raises(ArtifactError, match="changed"):
        tools.verified_check_passed()


class TestImageCheckRecord:
    """`execute_cell`이 최종 이미지의 카메라 판정을 따로 기록한다.

    카메라 판정은 도구 결과와 완료 조건을 바꾸지 않는다.
    """

    def test_report_is_separate_artifact(self, tmp_path: Path, valid_layout: Layout) -> None:
        """배치가 끝나면 보고서와 최종 이미지를 `image_check`로 등록한다. 도구 결과는 그대로다."""
        tools, result = _run_to_check(tmp_path / "run", valid_layout)
        records = _manifest(tools).artifacts
        assert [record.kind for record in records] == [*CHECKED_KINDS, "image_check"]
        assert [file.path for file in records[-1].files] == [IMAGE_CHECK_PATH, FINAL_IMAGE_PATH]
        assert records[-1].layout_id is None
        assert [reference.kind for reference in result.artifacts] == ["check"]
        assert result.artifacts[0].id == tools.ledger.check_id
        report = _image_check(tools)
        assert [part.status for part in report.parts] == [PartStatus.MATCH] * 4, report.parts
        assert not report.unexpected_cells

    def test_moved_part_is_mismatch(
        self, tmp_path: Path, valid_layout: Layout, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """실행기가 첫 부품을 10 mm 옮겨 놓으면 판독도 그 부품을 `mismatch`로 기록한다."""
        tools, _ = _tools(tmp_path / "run", valid_layout)
        layout_id = _prepare(tools)
        calibration_id = _calibrate(tools, layout_id)
        monkeypatch.setenv("FAKE_ISAAC_MODE", "miss")
        result = _execute(tools, layout_id, calibration_id)
        report = _image_check(tools)
        statuses = [part.status for part in report.parts]
        assert statuses == [PartStatus.MISMATCH, *[PartStatus.MATCH] * 3], report.parts
        assert not report.passed
        # 판독의 첫 부품은 참값 판정이 실패로 본 부품이다.
        assert [part.part_id for part in result.summary.failed_parts] == [report.parts[0].part_id]

    @pytest.mark.parametrize("change", ["edit", "delete"])
    def test_changed_report_does_not_change_final_check(
        self, tmp_path: Path, valid_layout: Layout, change: str
    ) -> None:
        """보고서를 바꾸거나 지워도 `verified_check_passed()`와 완료 기록은 같다."""
        tools, _ = _run_to_check(tmp_path / "run", valid_layout)
        path = tools.root / IMAGE_CHECK_PATH
        if change == "edit":
            path.write_text(path.read_text() + " ")
        else:
            path.unlink()
        assert tools.verified_check_passed()
        tools.finish(completed=True, error_code=None)
        assert tools.ledger.state == ToolState.COMPLETED

    def test_run_without_final_image_has_no_record(
        self,
        tmp_path: Path,
        valid_layout: Layout,
        monkeypatch: pytest.MonkeyPatch,
        caplog: pytest.LogCaptureFixture,
    ) -> None:
        """최종 이미지가 없으면 판독하지 않는다. 등록은 4개이고 경고 로그도 없다."""
        tools, _ = _tools(tmp_path / "run", valid_layout)
        layout_id = _prepare(tools)
        calibration_id = _calibrate(tools, layout_id)
        monkeypatch.setenv("FAKE_ISAAC_MODE", "no_final_image")
        with caplog.at_level(logging.WARNING, logger="pixel2place.agent.tools"):
            result = _execute(tools, layout_id, calibration_id)
        assert result.summary.passed
        assert not (tools.root / FINAL_IMAGE_PATH).exists()
        assert not (tools.root / IMAGE_CHECK_PATH).exists()
        assert [record.kind for record in _manifest(tools).artifacts] == CHECKED_KINDS
        assert not caplog.records, caplog.text
        assert tools.verified_check_passed()

    def test_reader_error_keeps_execution_result(
        self,
        tmp_path: Path,
        valid_layout: Layout,
        monkeypatch: pytest.MonkeyPatch,
        caplog: pytest.LogCaptureFixture,
    ) -> None:
        """판독이 예외를 내도 `ExecutionResult`는 `check` 참조 1개를 준다. 경고 로그를 쓴다."""
        tools, _ = _tools(tmp_path / "run", valid_layout)
        layout_id = _prepare(tools)
        calibration_id = _calibrate(tools, layout_id)
        monkeypatch.setenv("FAKE_ISAAC_MODE", "bad_final_image")
        with caplog.at_level(logging.WARNING, logger="pixel2place.agent.tools"):
            result = _execute(tools, layout_id, calibration_id)
        assert result.summary.passed
        assert [reference.kind for reference in result.artifacts] == ["check"]
        assert (tools.root / FINAL_IMAGE_PATH).is_file()
        assert not (tools.root / IMAGE_CHECK_PATH).exists()
        assert [record.kind for record in _manifest(tools).artifacts] == CHECKED_KINDS
        assert tools.ledger.state == ToolState.CHECKED and tools.ledger.error_code is None
        assert tools.verified_check_passed()
        (record,) = caplog.records
        assert record.levelno == logging.WARNING and record.exc_info is not None
        assert "Image check failed." in record.getMessage()


def test_wrong_order_and_retry_do_not_call_external_tools(
    tmp_path: Path, valid_layout: Layout, isaac_calls: list[tuple[str, str, ToolLedger]]
) -> None:
    tools, client = _tools(tmp_path / "run", valid_layout)
    result = tools.prepare_layout(PrepareLayoutArguments(retry=True), tool_call_id="retry")
    assert isinstance(result, ErrorResult) and result.error_code == ErrorCode.PROTOCOL_ERROR
    tools.calibrate_board(CalibrateBoardArguments(layout_id="0" * 32), tool_call_id="calibrate")
    assert not client.calls and not isaac_calls


def test_record_failure_stops_before_next_external_action(
    tmp_path: Path, valid_layout: Layout, isaac_calls: list[tuple[str, str, ToolLedger]]
) -> None:
    """도구 결과를 기록할 수 없으면 현재 객체도 다음 작업을 실행하지 않는다."""
    tools, client = _tools(tmp_path / "run", valid_layout)
    (tools.root / "tool_results/01.json").mkdir()
    with pytest.raises(OSError):
        _prepare(tools)
    with pytest.raises(OSError, match="record"):
        tools.calibrate_board(CalibrateBoardArguments(layout_id="0" * 32), tool_call_id="calibrate")
    assert len(client.calls) == 1 and not isaac_calls


@pytest.mark.parametrize("prompt", ["", "  ", "x" * 4001])
def test_invalid_goal_does_not_create_output(tmp_path: Path, prompt: str) -> None:
    client = FakeOpenAI([])
    with pytest.raises(ValidationError):
        AssemblyTools(
            out_dir=tmp_path / "run",
            prompt=prompt,
            cell=CellConfig(),
            scene=SimScene(),
            reasoner=CosmosReasoner(client=client, model="fake"),
            sim_settings=SimSettings(),  # pyright: ignore[reportCallIssue]
        )
    assert not (tmp_path / "run").exists() and not client.calls


def test_preflight_and_existing_output_do_not_call_cosmos(
    tmp_path: Path, valid_layout: Layout
) -> None:
    root = tmp_path / "existing"
    root.mkdir()
    with pytest.raises(FileExistsError):
        _tools(root, valid_layout)
    (tmp_path / "isaac/python.sh").unlink()
    with pytest.raises(SimSetupError):
        _tools(tmp_path / "new", valid_layout)
    assert not (tmp_path / "new").exists()


@pytest.mark.parametrize(
    ("model", "payload"),
    [
        (PrepareLayoutArguments, '{"retry": 1}'),
        (PrepareLayoutArguments, '{"retry": "false"}'),
        (PrepareLayoutArguments, '{"retry": false, "prompt": "new goal"}'),
        (PrepareLayoutArguments, "{}"),
        (CalibrateBoardArguments, '{"layout_id": "../layout.json"}'),
        (CalibrateBoardArguments, '{"layout_id": ""}'),
        (ExecuteCellArguments, '{"layout_id": null, "calibration_id": 5}'),
    ],
)
def test_arguments_reject_coercion_extra_fields_and_paths(
    model: type[PrepareLayoutArguments | CalibrateBoardArguments | ExecuteCellArguments],
    payload: str,
) -> None:
    with pytest.raises(ValidationError):
        model.model_validate_json(payload)


def test_artifact_store_rejects_changed_ownership_and_missing_files(tmp_path: Path) -> None:
    store = ArtifactStore(tmp_path / "store")
    path = store.root / "layout.json"
    path.write_text("{}")
    record = store.register("layout", (path,))
    with pytest.raises(ArtifactError):
        store.require(record.id, "calibration")
    with pytest.raises(ArtifactError):
        store.require(record.id, "layout", layout_id="0" * 32)
    outside = tmp_path / "outside.json"
    outside.write_text("{}")
    with pytest.raises(ArtifactError, match="outside"):
        store.register("layout", (outside,))
    path.unlink()
    with pytest.raises(ArtifactError, match="unavailable"):
        store.require(record.id, "layout")
    (store.root / "manifest.json").unlink()
    with pytest.raises(ArtifactError, match="manifest"):
        store.require(record.id, "layout")
