"""처분의 출하, 재작업과 불량 경로를 HTTP Fake와 실제 조립 도구로 검사한다.

결함은 가짜 실행기가 장면에서 읽는다. 공급 누락은 칸을 비운다. 다른 색 부품은 칸을 덮는다.
"""

import json
from collections.abc import Iterator
from pathlib import Path
from typing import Any

import pytest
from pydantic import ValidationError

from pixel2place.agent.artifacts import Manifest
from pixel2place.agent.contracts import (
    CalibrateBoardArguments,
    CalibrationResult,
    ErrorCode,
    ErrorResult,
    ExecuteCellArguments,
    ExecutionResult,
    ExecutionSummary,
    InspectedExecutionSummary,
    LayoutResult,
    PrepareLayoutArguments,
    ReworkCellArguments,
    ToolName,
    ToolState,
)
from pixel2place.agent.prompts import (
    DISPOSITION_GUIDANCE,
    REVIEW_GUIDANCE,
    SYSTEM_PROMPT,
    system_prompt,
)
from pixel2place.agent.runtime import allowed_tools, tool_schema
from pixel2place.agent.runtime_contracts import AgentBudget, AgentMetadata, AgentReport
from pixel2place.agent.tools import REWORK_SCENE_FILE, AssemblyTools, ToolLedger
from pixel2place.cell.image_check import ImageCheckReport, PartStatus
from pixel2place.cell.scene import SceneFaults, SimScene, WrongColor
from pixel2place.cell.task import Task
from pixel2place.config import DEFAULT_JUDGMENT_MODEL, DEFAULT_NEMOTRON_MODEL
from pixel2place.generation.synthetic import COLOR_RGB
from pixel2place.layout.schema import Color, Layout, PartType
from tests.factories import make_layout, make_part, make_planner_response_json
from tests.fakes.fake_agent_model import FakeAgentModel
from tests.unit.test_agent_runtime import make_runtime, read_events

pytestmark = [pytest.mark.unit, pytest.mark.usefixtures("fake_isaac")]
MISSING_SEQUENCE = 2
WRONG_COLOR_SEQUENCE = 1
ORANGE = WrongColor(
    sequence=WRONG_COLOR_SEQUENCE, color=Color.ORANGE, color_rgb=COLOR_RGB[Color.ORANGE]
)
MISSING = SimScene(faults=SceneFaults(missing=(MISSING_SEQUENCE,)))
WRONG_COLOR = SimScene(faults=SceneFaults(wrong_color=(ORANGE,)))
MISSING_AND_WRONG_COLOR = SimScene(
    faults=SceneFaults(missing=(MISSING_SEQUENCE,), wrong_color=(ORANGE,))
)
REWORK_RESULT_PATH = "execution/rework/sim/result.json"
RUN_ID = "0" * 32


@pytest.fixture
def api() -> Iterator[FakeAgentModel]:
    """테스트마다 HTTP Fake를 만든다. 종료 때 클라이언트를 닫는다."""
    fake = FakeAgentModel()
    yield fake
    for client in fake.clients:
        client.close()


def _body(api: FakeAgentModel, index: int) -> dict[str, Any]:
    return json.loads(api.model_posts[index].content)


def _state(body: dict[str, Any]) -> dict[str, Any]:
    return json.loads(body["messages"][0]["content"].split("Current runtime state: ")[1])


def _tool_names(body: dict[str, Any]) -> list[str]:
    return [tool["function"]["name"] for tool in body.get("tools", [])]


def _last_result(body: dict[str, Any]) -> dict[str, Any]:
    """요청의 마지막 도구 결과. 모델이 처분을 고를 때 읽는 값이다."""
    return json.loads(body["messages"][-1]["content"])


def _part_id(tools: AssemblyTools, sequence: int) -> str:
    """배치 순서가 `sequence`인 부품의 ID. 결함은 순서로 지정한다."""
    task = Task.model_validate_json((tools.root / "execution/task.json").read_text())
    return next(part.part_id for part in task.parts if part.sequence == sequence)


def _audit_reasons(tools: AssemblyTools) -> list[str]:
    lines = (tools.root / "model_audit.jsonl").read_text().splitlines()
    records = [json.loads(line) for line in lines]
    return [
        record["data"]["reason"]
        for record in records
        if record["kind"] == "validation" and not record["data"]["accepted"]
    ]


def _run_to_check(tools: AssemblyTools) -> ExecutionResult:
    """모델 없이 준비, 보정과 배치를 차례로 실행한다."""
    prepared = tools.prepare_layout(PrepareLayoutArguments(retry=False), tool_call_id="prepare")
    assert isinstance(prepared, LayoutResult), prepared
    layout_id = prepared.artifacts[0].id
    calibrated = tools.calibrate_board(
        CalibrateBoardArguments(layout_id=layout_id), tool_call_id="calibrate"
    )
    assert isinstance(calibrated, CalibrationResult), calibrated
    executed = tools.execute_cell(
        ExecuteCellArguments(layout_id=layout_id, calibration_id=calibrated.artifacts[0].id),
        tool_call_id="execute",
    )
    assert isinstance(executed, ExecutionResult), executed
    return executed


class TestDispositionOff:
    """처분을 끈 실행은 처분의 값과 도구를 모델에 주지 않는다."""

    def test_failed_run_has_no_inspection_and_no_rework(
        self, tmp_path: Path, valid_layout: Layout, api: FakeAgentModel
    ) -> None:
        """부품이 빠져도 모델 요청 4회로 끝난다. 결과에 카메라 판정이 없고 도구를 주지 않는다."""
        api.dispositions = [None]
        runtime, tools, _ = make_runtime(tmp_path / "run", valid_layout, api, scene=MISSING)
        report = runtime.run()
        assert report.error_code == ErrorCode.CHECK_FAILED and report.calls.nemotron == 4
        assert report.calls.place == 1 and report.reworked is False
        checked = _body(api, 3)
        assert checked["messages"][0]["content"].startswith(SYSTEM_PROMPT + "\nCurrent")
        assert "tools" not in checked and "rework_runs" not in _state(checked)
        assert "inspection" not in _last_result(checked)["summary"]
        execution = next(e.result for e in read_events(tools) if e.to_state == ToolState.CHECKED)
        assert isinstance(execution, ExecutionResult)
        assert type(execution.summary) is ExecutionSummary
        assert tools.ledger.image_check_id is None and tools.ledger.rework_targets == ()

    def test_rework_call_is_a_protocol_error(self, tmp_path: Path, valid_layout: Layout) -> None:
        """처분을 끈 실행의 `rework_cell`은 실행기를 띄우지 않고 중단한다."""
        _, tools, _ = make_runtime(tmp_path / "run", valid_layout, FakeAgentModel(), scene=MISSING)
        _run_to_check(tools)
        result = tools.rework_cell(
            ReworkCellArguments(part_ids=(_part_id(tools, MISSING_SEQUENCE),)),
            tool_call_id="rework",
        )
        assert isinstance(result, ErrorResult) and result.error_code == ErrorCode.PROTOCOL_ERROR
        assert not tools.ledger.rework_consumed and tools.ledger.state == ToolState.ABORTED
        assert not (tools.root / "execution/rework").exists()


class TestDisposition:
    """모델이 검사 결과로 출하, 재작업과 불량을 고른다. 코드가 선택을 검증한다."""

    def test_passed_inspection_is_released_in_four_inferences(
        self, tmp_path: Path, valid_layout: Layout, api: FakeAgentModel
    ) -> None:
        """두 판정이 통과하면 재작업을 주지 않는다. 모델의 완료를 받는다."""
        runtime, tools, _ = make_runtime(tmp_path / "run", valid_layout, api, disposition=True)
        report = runtime.run()
        assert report.state == ToolState.COMPLETED and report.error_code is None
        assert report.calls.nemotron == 4 and report.calls.place == 1
        assert report.reworked is False
        checked = _body(api, 3)
        prompt = system_prompt(review=False, disposition=True)
        assert checked["messages"][0]["content"].startswith(prompt + "\nCurrent runtime state")
        assert _state(checked)["rework_runs"] == 0 and _state(checked)["error_code"] is None
        assert "tools" not in checked
        inspection = _last_result(checked)["summary"]["inspection"]
        assert inspection == {
            "passed": True,
            "match_count": 4,
            "shadowed_count": 0,
            "mismatch_count": 0,
            "parts": [],
            "unexpected_cell_count": 0,
            "unexpected_cells": [],
        }
        ledger = tools.ledger
        assert ledger.image_check_passed is True and ledger.image_check_id is not None
        assert ledger.rework_targets == ()

    def test_missing_part_is_reworked_and_released(
        self, tmp_path: Path, valid_layout: Layout, api: FakeAgentModel
    ) -> None:
        """빈 칸의 부품을 재작업으로 다시 놓는다. 두 번째 판정이 통과하면 완료한다."""
        api.dispositions = [None]
        runtime, tools, _ = make_runtime(
            tmp_path / "run", valid_layout, api, scene=MISSING, disposition=True
        )
        report = runtime.run()
        assert report.state == ToolState.COMPLETED, report.error_code
        assert report.calls.nemotron == 5 and report.calls.place == 2
        assert report.reworked is True and report.check_passed is True
        saved = AgentReport.model_validate_json((tools.root / "agent_report.json").read_text())
        assert saved == report
        missing = _part_id(tools, MISSING_SEQUENCE)
        first, second = _body(api, 3), _body(api, 4)
        assert _tool_names(first) == ["rework_cell"]
        assert _state(first)["error_code"] == "check_failed" and _state(first)["rework_runs"] == 0
        summary = _last_result(first)["summary"]
        assert summary["passed"] is False and summary["passed_count"] == 3
        assert summary["inspection"]["parts"] == [
            {"part_id": missing, "status": "mismatch", "empty": True}
        ]
        assert summary["inspection"]["passed"] is False
        assert "tools" not in second and _state(second)["rework_runs"] == 1
        assert _state(second)["error_code"] is None
        assert _state(second)["check_id"] == report.check_id != _state(first)["check_id"]
        reworked = _last_result(second)["summary"]
        assert reworked["passed"] is True and reworked["inspection"]["passed"] is True
        calls = [e.call for e in read_events(tools) if e.kind == "tool_result" and e.call]
        assert [call.name for call in calls][-2:] == ["execute_cell", "rework_cell"]
        assert calls[-1].arguments == ReworkCellArguments(part_ids=(missing,))

    def test_rework_records_are_registered_and_truth_stays_local(
        self, tmp_path: Path, valid_layout: Layout, api: FakeAgentModel
    ) -> None:
        """재작업은 판정과 카메라 판정을 새로 등록한다. 재작업 장면은 모델에 가지 않는다."""
        api.dispositions = [None]
        runtime, tools, _ = make_runtime(
            tmp_path / "run", valid_layout, api, scene=MISSING, disposition=True
        )
        assert runtime.run().state == ToolState.COMPLETED
        manifest = Manifest.model_validate_json((tools.root / "manifest.json").read_text())
        assert [record.kind for record in manifest.artifacts] == [
            "inputs",
            "layout",
            "calibration",
            "check",
            "image_check",
            "check",
            "image_check",
        ]
        check, image_check = manifest.artifacts[-2:]
        assert check.id == tools.ledger.check_id and image_check.id == tools.ledger.image_check_id
        assert [file.path for file in check.files] == [
            "execution/rework/check.json",
            "execution/task.json",
            REWORK_RESULT_PATH,
            f"evaluation/{REWORK_SCENE_FILE}",
        ]
        assert [file.path for file in image_check.files] == [
            "execution/rework/image_check.json",
            "execution/rework/sim/topdown_final.png",
        ]
        rework = SimScene.model_validate_json(
            (tools.root / "evaluation" / REWORK_SCENE_FILE).read_text()
        )
        assert rework.faults is None and rework.preplaced is not None
        restored = sorted(part.sequence for part in rework.preplaced)
        assert restored == [1, 3, 4], f"재작업 대상이 아닌 부품만 복원한다: {restored}"
        report = ImageCheckReport.model_validate_json(
            (tools.root / "execution/rework/image_check.json").read_text()
        )
        assert [part.status for part in report.parts] == [PartStatus.MATCH] * 4
        model_text = "\n".join(request.content.decode() for request in api.model_posts)
        record_text = (tools.root / "events.jsonl").read_text()
        for excluded in ("rework_scene", "preplaced", "orientation", "rework_targets", "faults"):
            assert excluded not in model_text and excluded not in record_text, excluded

    def test_wrong_color_part_is_rejected(
        self, tmp_path: Path, valid_layout: Layout, api: FakeAgentModel
    ) -> None:
        """참값 판정만 통과한 실행은 출하하지 않는다. 칸이 덮여 있어서 재작업 대상이 없다."""
        runtime, tools, _ = make_runtime(
            tmp_path / "run", valid_layout, api, scene=WRONG_COLOR, disposition=True
        )
        report = runtime.run()
        assert report.state == ToolState.ABORTED
        assert report.error_code == ErrorCode.INSPECTION_REJECTED
        assert report.calls.nemotron == 4 and report.calls.place == 1
        assert report.check_passed is True and report.reworked is False
        checked = _body(api, 3)
        assert _tool_names(checked) == ["rework_cell"]
        assert _state(checked)["error_code"] == "inspection_rejected"
        summary = _last_result(checked)["summary"]
        assert summary["passed"] is True
        assert summary["inspection"]["parts"] == [
            {"part_id": _part_id(tools, WRONG_COLOR_SEQUENCE), "status": "mismatch", "empty": False}
        ]
        assert tools.ledger.rework_targets == () and not tools.ledger.rework_consumed

    def test_rework_that_does_not_pass_is_rejected(
        self, tmp_path: Path, valid_layout: Layout, api: FakeAgentModel
    ) -> None:
        """재작업 뒤에도 카메라 판정이 실패하면 불량이다. 재작업을 다시 주지 않는다."""
        api.dispositions = [None, None]
        runtime, tools, _ = make_runtime(
            tmp_path / "run", valid_layout, api, scene=MISSING_AND_WRONG_COLOR, disposition=True
        )
        report = runtime.run()
        assert report.state == ToolState.ABORTED
        assert report.error_code == ErrorCode.INSPECTION_REJECTED
        assert report.calls.nemotron == 5 and report.calls.place == 2
        assert report.reworked is True and report.check_passed is True
        first, second = _body(api, 3), _body(api, 4)
        assert _state(first)["error_code"] == "check_failed"
        assert "tools" not in second and _state(second)["rework_runs"] == 1
        assert _state(second)["error_code"] == "inspection_rejected"
        assert api.dispositions == [None], "두 번째 재작업은 제공하지 않는다."
        assert tools.ledger.rework_targets == ()

    def test_replan_and_rework_use_six_inferences(
        self, tmp_path: Path, valid_layout: Layout, api: FakeAgentModel
    ) -> None:
        """재계획과 재작업이 모두 있는 실행은 모델 요청 한도 6회 안에서 완료한다."""
        second_layout = make_layout(
            make_part("P001", PartType.SQUARE_TILE, origin=(1, 1)),
            make_part("P002", PartType.RECT_TILE, origin=(3, 2), color=Color.GRAY),
            make_part("P003", PartType.SQUARE_TILE, origin=(5, 5), color=Color.WHITE),
        )
        api.replans, api.dispositions = ["part_set_mismatch"], [None]
        runtime, _, _ = make_runtime(
            tmp_path / "run",
            valid_layout,
            api,
            responses=[
                make_planner_response_json(valid_layout),
                make_planner_response_json(second_layout),
            ],
            review=True,
            scene=MISSING,
            disposition=True,
        )
        report = runtime.run()
        assert report.state == ToolState.COMPLETED, report.error_code
        assert report.calls.nemotron == AgentBudget().max_inferences == 6
        assert report.calls.cosmos == 2 and report.calls.place == 2 and report.reworked

    @pytest.mark.parametrize("change", ["other_part", "extra_part", "repeated_part", "unknown"])
    def test_rework_parts_must_equal_the_targets(
        self, tmp_path: Path, valid_layout: Layout, api: FakeAgentModel, change: str
    ) -> None:
        """부품 목록이 재작업 대상과 다르면 실행기를 띄우지 않고 중단한다."""
        runtime, tools, _ = make_runtime(
            tmp_path / "run", valid_layout, api, scene=MISSING, disposition=True
        )

        def choose(request: object) -> None:
            if len(api.model_posts) != 4 or api.dispositions:
                return
            missing, other = _part_id(tools, MISSING_SEQUENCE), _part_id(tools, 1)
            api.dispositions = [
                {
                    "other_part": [other],
                    "extra_part": [missing, other],
                    "repeated_part": [missing, missing],
                    "unknown": ["P999"],
                }[change]
            ]

        api.on_request = choose
        report = runtime.run()
        assert report.error_code == ErrorCode.PROTOCOL_ERROR and report.calls.nemotron == 4
        assert report.calls.place == 1 and report.reworked is False
        assert _audit_reasons(tools) == ["invalid_rework_parts"]
        assert not (tools.root / "execution/rework").exists()

    @pytest.mark.parametrize("scene", [WRONG_COLOR, MISSING], ids=["wrong-color", "missing"])
    def test_completion_needs_both_checks(
        self, tmp_path: Path, valid_layout: Layout, api: FakeAgentModel, scene: SimScene
    ) -> None:
        """카메라 판정 또는 참값 판정이 실패한 실행의 완료 응답은 계약 위반이다."""
        api.variant, api.variant_stage = "early_complete", 4
        runtime, tools, _ = make_runtime(
            tmp_path / "run", valid_layout, api, scene=scene, disposition=True
        )
        report = runtime.run()
        assert report.state == ToolState.ABORTED
        assert report.error_code == ErrorCode.PROTOCOL_ERROR
        assert _audit_reasons(tools) == ["unverified_completion"]

    @pytest.mark.parametrize("variant", ["success", "early_complete"])
    def test_run_without_a_camera_report_is_not_released(
        self,
        tmp_path: Path,
        valid_layout: Layout,
        api: FakeAgentModel,
        monkeypatch: pytest.MonkeyPatch,
        variant: str,
    ) -> None:
        """최종 이미지가 없으면 검사 요약은 null이다. 참값 판정이 통과해도 출하하지 않는다."""
        monkeypatch.setenv("FAKE_ISAAC_MODE", "no_final_image")
        api.variant, api.variant_stage = variant, 4
        runtime, tools, _ = make_runtime(tmp_path / "run", valid_layout, api, disposition=True)
        report = runtime.run()
        assert report.state == ToolState.ABORTED and report.check_passed is True
        assert report.error_code == (
            ErrorCode.INSPECTION_REJECTED if variant == "success" else ErrorCode.PROTOCOL_ERROR
        )
        checked = _body(api, 3)
        assert _last_result(checked)["summary"]["inspection"] is None
        assert _state(checked)["error_code"] == "inspection_rejected"
        assert tools.ledger.image_check_id is None and tools.ledger.rework_targets == ()

    def test_changed_camera_report_cannot_support_release(
        self, tmp_path: Path, valid_layout: Layout, api: FakeAgentModel
    ) -> None:
        """등록한 카메라 판정의 보고서가 바뀌면 완료를 거부한다."""
        runtime, tools, _ = make_runtime(tmp_path / "run", valid_layout, api, disposition=True)

        def change(request: object) -> None:
            if len(api.model_posts) == 4:
                path = tools.root / "execution/image_check.json"
                path.write_text(path.read_text() + " ")

        api.on_request = change
        report = runtime.run()
        assert report.error_code == ErrorCode.PROTOCOL_ERROR and report.calls.nemotron == 4

    def test_changed_first_result_stops_the_rework(
        self, tmp_path: Path, valid_layout: Layout, api: FakeAgentModel
    ) -> None:
        """첫 배치의 결과가 바뀌면 그 자세로 재작업 장면을 만들지 않는다."""
        api.dispositions = [None]
        runtime, tools, _ = make_runtime(
            tmp_path / "run", valid_layout, api, scene=MISSING, disposition=True
        )

        def change(request: object) -> None:
            if len(api.model_posts) == 4:
                path = tools.root / "execution/sim/result.json"
                path.write_text(path.read_text() + " ")

        api.on_request = change
        report = runtime.run()
        assert report.error_code == ErrorCode.PROTOCOL_ERROR
        assert report.calls.place == 1 and not (tools.root / "execution/rework").exists()


class TestReworkTool:
    """`rework_cell`의 단계, 횟수와 실패를 모델 없이 검사한다."""

    def _tools(self, root: Path, layout: Layout, *, scene: SimScene = MISSING) -> AssemblyTools:
        _, tools, _ = make_runtime(root, layout, FakeAgentModel(), scene=scene, disposition=True)
        return tools

    def test_rework_needs_an_executed_cell(self, tmp_path: Path, valid_layout: Layout) -> None:
        """배치의 앞에서 부른 재작업은 `protocol_error`다."""
        tools = self._tools(tmp_path / "run", valid_layout)
        result = tools.rework_cell(ReworkCellArguments(part_ids=("P001",)), tool_call_id="rework")
        assert isinstance(result, ErrorResult) and result.error_code == ErrorCode.PROTOCOL_ERROR
        assert result.summary.tool == ToolName.REWORK_CELL and not tools.ledger.rework_consumed

    def test_other_parts_do_not_start_the_runner(
        self, tmp_path: Path, valid_layout: Layout
    ) -> None:
        """도구도 부품 목록을 재작업 대상과 비교한다. 다른 부품은 `protocol_error`다."""
        tools = self._tools(tmp_path / "run", valid_layout)
        _run_to_check(tools)
        assert tools.ledger.rework_targets == (_part_id(tools, MISSING_SEQUENCE),)
        result = tools.rework_cell(ReworkCellArguments(part_ids=("P999",)), tool_call_id="rework")
        assert isinstance(result, ErrorResult) and result.error_code == ErrorCode.PROTOCOL_ERROR
        assert not tools.ledger.rework_consumed
        assert not (tools.root / "execution/rework").exists()

    def test_second_rework_is_a_protocol_error(self, tmp_path: Path, valid_layout: Layout) -> None:
        """재작업은 실행마다 1회다. 두 번째 호출은 실행기를 띄우지 않는다."""
        tools = self._tools(tmp_path / "run", valid_layout, scene=MISSING_AND_WRONG_COLOR)
        _run_to_check(tools)
        arguments = ReworkCellArguments(part_ids=(_part_id(tools, MISSING_SEQUENCE),))
        first = tools.rework_cell(arguments, tool_call_id="rework-1")
        assert isinstance(first, ExecutionResult), first
        assert isinstance(first.summary, InspectedExecutionSummary)
        assert first.summary.passed and first.summary.inspection is not None
        assert not first.summary.inspection.passed
        assert tools.ledger.error_code == ErrorCode.INSPECTION_REJECTED
        result_path = tools.root / REWORK_RESULT_PATH
        written = result_path.stat().st_mtime_ns
        second = tools.rework_cell(arguments, tool_call_id="rework-2")
        assert isinstance(second, ErrorResult) and second.error_code == ErrorCode.PROTOCOL_ERROR
        assert result_path.stat().st_mtime_ns == written

    def test_runner_failure_in_rework_aborts(
        self, tmp_path: Path, valid_layout: Layout, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """재작업 실행이 결과를 쓰지 않으면 `runner_error`로 중단한다. 재작업은 소비한다."""
        tools = self._tools(tmp_path / "run", valid_layout)
        _run_to_check(tools)
        monkeypatch.setenv("FAKE_ISAAC_MODE", "crash")
        result = tools.rework_cell(
            ReworkCellArguments(part_ids=(_part_id(tools, MISSING_SEQUENCE),)),
            tool_call_id="rework",
        )
        assert isinstance(result, ErrorResult) and result.error_code == ErrorCode.RUNNER_ERROR
        assert tools.ledger.rework_consumed and tools.ledger.state == ToolState.ABORTED

    def test_first_result_without_orientation_cannot_be_reworked(
        self, tmp_path: Path, valid_layout: Layout, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """첫 배치의 결과에 quaternion이 없으면 놓인 부품을 복원하지 못한다."""
        tools = self._tools(tmp_path / "run", valid_layout)
        monkeypatch.setenv("FAKE_ISAAC_MODE", "no_orientation")
        _run_to_check(tools)
        result = tools.rework_cell(
            ReworkCellArguments(part_ids=(_part_id(tools, MISSING_SEQUENCE),)),
            tool_call_id="rework",
        )
        assert isinstance(result, ErrorResult) and result.error_code == ErrorCode.RESULT_MISMATCH
        assert not tools.ledger.rework_consumed
        assert not (tools.root / "execution/rework").exists()

    def test_rework_arguments_reject_empty_and_extra_values(self) -> None:
        """부품 목록은 1개 이상이다. 추가 필드와 문자열이 아닌 ID를 거부한다."""
        for payload in (
            '{"part_ids":[]}',
            '{"part_ids":["P001"],"force":true}',
            '{"part_ids":[1]}',
        ):
            with pytest.raises(ValidationError):
                ReworkCellArguments.model_validate_json(payload)
        schema = tool_schema(ToolName.REWORK_CELL)["function"]
        assert schema.get("strict") is True
        assert schema.get("parameters", {}).get("required") == ["part_ids"]


class TestDispositionModel:
    """`checked`의 요청만 처분 모델로 보낸다."""

    def test_only_checked_requests_use_the_disposition_model(
        self, tmp_path: Path, valid_layout: Layout, api: FakeAgentModel
    ) -> None:
        """재작업의 앞과 뒤의 `checked` 요청이 처분 모델로 간다. 다른 요청은 조율 모델로 간다."""
        api.dispositions = [None]
        runtime, tools, _ = make_runtime(
            tmp_path / "run",
            valid_layout,
            api,
            scene=MISSING,
            disposition=True,
            disposition_model=DEFAULT_JUDGMENT_MODEL,
        )
        assert runtime.run().state == ToolState.COMPLETED
        models = [json.loads(request.content)["model"] for request in api.model_posts]
        assert models == [DEFAULT_NEMOTRON_MODEL] * 3 + [DEFAULT_JUDGMENT_MODEL] * 2
        requested = [e.model for e in read_events(tools) if e.kind == "model_request"]
        assert requested == models

    def test_disposition_model_outside_the_catalog_stops_before_inference(
        self, tmp_path: Path, valid_layout: Layout, api: FakeAgentModel
    ) -> None:
        """계정의 모델 목록에 처분 모델이 없으면 모델 요청의 앞에서 중단한다."""
        api.models = [DEFAULT_NEMOTRON_MODEL]
        runtime, _, _ = make_runtime(
            tmp_path / "run",
            valid_layout,
            api,
            disposition=True,
            disposition_model=DEFAULT_JUDGMENT_MODEL,
        )
        report = runtime.run()
        assert report.error_code == ErrorCode.MODEL_ERROR and report.calls.nemotron == 0

    def test_disposition_model_is_not_used_when_disposition_is_off(
        self, tmp_path: Path, valid_layout: Layout, api: FakeAgentModel
    ) -> None:
        """처분을 끈 실행은 처분 모델을 쓰지 않는다. 모델 목록에 없어도 실행한다."""
        api.models = [DEFAULT_NEMOTRON_MODEL]
        runtime, _, _ = make_runtime(
            tmp_path / "run", valid_layout, api, disposition_model=DEFAULT_JUDGMENT_MODEL
        )
        assert runtime.run().state == ToolState.COMPLETED
        models = {json.loads(request.content)["model"] for request in api.model_posts}
        assert models == {DEFAULT_NEMOTRON_MODEL}


class TestAllowedTools:
    """`checked`에서 재작업을 제공하는 조건."""

    @pytest.mark.parametrize(
        ("disposition", "error_code", "rework_consumed", "expected"),
        [
            (True, ErrorCode.CHECK_FAILED, False, (ToolName.REWORK_CELL,)),
            (True, ErrorCode.INSPECTION_REJECTED, False, (ToolName.REWORK_CELL,)),
            (True, None, False, ()),
            (True, ErrorCode.CHECK_FAILED, True, ()),
            (False, ErrorCode.CHECK_FAILED, False, ()),
        ],
    )
    def test_rework_needs_disposition_a_failed_inspection_and_a_free_run(
        self,
        disposition: bool,
        error_code: ErrorCode | None,
        rework_consumed: bool,
        expected: tuple[ToolName, ...],
    ) -> None:
        """처분을 켠 실행에서 검사가 실패했고 재작업을 쓰지 않았을 때만 제공한다."""
        ledger = ToolLedger(
            run_id=RUN_ID,
            state=ToolState.CHECKED,
            error_code=error_code,
            rework_consumed=rework_consumed,
        )
        assert allowed_tools(ledger, review=True, disposition=disposition) == expected

    def test_rework_is_not_offered_before_the_check(self) -> None:
        """다른 상태에서는 처분을 켜도 재작업을 제공하지 않는다."""
        ledger = ToolLedger(
            run_id=RUN_ID, state=ToolState.CALIBRATED, error_code=ErrorCode.CHECK_FAILED
        )
        assert allowed_tools(ledger, review=False, disposition=True) == (ToolName.EXECUTE_CELL,)


class TestDispositionSettings:
    """처분 설정의 기본값과 지시문."""

    def test_disposition_needs_a_budget_of_two_place_runs(self) -> None:
        """배치 한도가 1회인 예산으로는 처분을 켤 수 없다."""
        with pytest.raises(ValidationError, match="two placement runs"):
            AgentMetadata(
                model=DEFAULT_NEMOTRON_MODEL,
                cosmos_model="fake",
                budget=AgentBudget(),
                system_prompt=SYSTEM_PROMPT,
                disposition=True,
            )

    def test_metadata_without_the_disposition_field_is_disposition_off(self) -> None:
        """처분 필드가 없는 기록은 처분을 끈 실행으로 읽는다."""
        metadata = AgentMetadata.model_validate(
            {
                "model": DEFAULT_NEMOTRON_MODEL,
                "cosmos_model": "fake",
                "budget": AgentBudget().model_dump(),
                "system_prompt": SYSTEM_PROMPT,
            }
        )
        assert metadata.disposition is False and metadata.disposition_model is None

    @pytest.mark.parametrize("review", [False, True])
    def test_disposition_prompt_allows_the_rework_tool(self, review: bool) -> None:
        """처분을 켠 지시문은 셀 실행의 반복을 막는 문장을 바꾸고 처분 규칙을 더한다."""
        prompt = system_prompt(review=review, disposition=True)
        assert "Do not retry calibration or cell execution." not in prompt
        assert "Do not call calibrate_board or execute_cell again." in prompt
        assert "After a passed check and a passed inspection, return only" in prompt
        assert "5. In checked, select the disposition." in prompt
        assert prompt.endswith(DISPOSITION_GUIDANCE)
        assert (REVIEW_GUIDANCE in prompt) is review
        off = system_prompt(review=review)
        assert "rework" not in off and "inspection" not in off and "disposition" not in off
