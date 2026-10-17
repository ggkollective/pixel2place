"""계획 검토의 부품 목록, 재계획과 거부 경계를 HTTP Fake와 실제 조립 도구로 검사한다."""

import json
from collections.abc import Iterator
from pathlib import Path
from typing import Any

import pytest

from pixel2place.agent.contracts import (
    ErrorCode,
    ErrorResult,
    LayoutResult,
    LayoutSummary,
    PrepareLayoutArguments,
    ReplanLayoutArguments,
    ReviewedLayoutSummary,
    ReviewReason,
    ToolState,
)
from pixel2place.agent.prompts import REVIEW_SYSTEM_PROMPT, SYSTEM_PROMPT
from pixel2place.agent.runtime import allowed_tools, tool_schema
from pixel2place.agent.runtime_contracts import AgentMetadata
from pixel2place.agent.tools import REVIEW_FEEDBACK, AssemblyTools, ToolLedger, ToolRun
from pixel2place.cell.task import Task
from pixel2place.config import DEFAULT_JUDGMENT_MODEL, DEFAULT_NEMOTRON_MODEL
from pixel2place.layout.schema import Color, Layout, PartType
from tests.factories import make_layout, make_part, make_planner_response_json
from tests.fakes.fake_agent_model import FakeAgentModel
from tests.unit.test_agent_runtime import GOAL, make_runtime, read_events

pytestmark = [pytest.mark.unit, pytest.mark.usefixtures("fake_isaac")]


@pytest.fixture
def api() -> Iterator[FakeAgentModel]:
    """테스트마다 HTTP Fake를 만든다. 종료 때 클라이언트를 닫는다."""
    fake = FakeAgentModel()
    yield fake
    for client in fake.clients:
        client.close()


@pytest.fixture
def second_layout() -> Layout:
    """재계획이 돌려주는 레이아웃. 첫 계획과 부품 수가 다르다."""
    return make_layout(
        make_part("P001", PartType.SQUARE_TILE, origin=(1, 1)),
        make_part("P002", PartType.RECT_TILE, origin=(3, 2), color=Color.GRAY),
    )


def _body(api: FakeAgentModel, index: int) -> dict[str, Any]:
    return json.loads(api.model_posts[index].content)


def _state(body: dict[str, Any]) -> dict[str, Any]:
    return json.loads(body["messages"][0]["content"].split("Current runtime state: ")[1])


def _tool_names(body: dict[str, Any]) -> list[str]:
    return [tool["function"]["name"] for tool in body.get("tools", [])]


class TestReviewOff:
    """계획 검토를 끈 실행의 요청과 기록이 바뀌지 않는지 검사한다."""

    def test_requests_and_results_have_no_review_fields(
        self, tmp_path: Path, valid_layout: Layout, api: FakeAgentModel
    ) -> None:
        """상태에 준비 시도 수가 없다. 도구는 1개다. 준비 결과에 부품 목록이 없다."""
        runtime, tools, _ = make_runtime(tmp_path / "run", valid_layout, api)
        assert runtime.run().state == ToolState.COMPLETED
        body = _body(api, 1)
        assert body["messages"][0]["content"].startswith(SYSTEM_PROMPT + "\nCurrent runtime state")
        assert list(_state(body)) == [
            "state",
            "error_code",
            "layout_id",
            "calibration_id",
            "check_id",
        ]
        assert _tool_names(body) == ["calibrate_board"]
        summary = json.loads(body["messages"][-1]["content"])["summary"]
        assert "parts" not in summary and "part_counts" not in summary
        prepared = read_events(tools)[4].result
        assert isinstance(prepared, LayoutResult) and type(prepared.summary) is LayoutSummary
        inputs = ToolRun.model_validate_json((tools.root / "run.json").read_text())
        assert inputs.agent is not None and inputs.agent.review is False

    @pytest.mark.parametrize("variant", ["replan_unoffered", "plan_rejected"])
    def test_review_actions_are_protocol_errors(
        self, tmp_path: Path, valid_layout: Layout, api: FakeAgentModel, variant: str
    ) -> None:
        """재계획 호출과 `plan_rejected` 중단을 거부한다. 캡처를 실행하지 않는다."""
        api.variant, api.variant_stage = variant, 2
        runtime, _, cosmos = make_runtime(tmp_path / "run", valid_layout, api)
        report = runtime.run()
        assert report.error_code == ErrorCode.PROTOCOL_ERROR
        assert report.calls.nemotron == 2 and report.calls.capture == 0
        assert len(cosmos.calls) == 1


class TestPlanReview:
    """검토를 켠 실행의 요청, 재계획과 마지막 시도를 검사한다."""

    def test_layout_ready_request_offers_replan_and_the_part_list(
        self, tmp_path: Path, valid_layout: Layout, api: FakeAgentModel
    ) -> None:
        """모델은 부품 목록과 도구 2개를 받는다. 보정을 고르면 모델 요청 4회로 완료한다."""
        runtime, tools, _ = make_runtime(tmp_path / "run", valid_layout, api, review=True)
        report = runtime.run()
        assert report.state == ToolState.COMPLETED and report.calls.nemotron == 4
        body = _body(api, 1)
        assert body["messages"][0]["content"].startswith(
            REVIEW_SYSTEM_PROMPT + "\nCurrent runtime state"
        )
        assert _state(body)["layout_attempts"] == 1
        assert _tool_names(body) == ["calibrate_board", "replan_layout"]
        reason = body["tools"][1]["function"]["parameters"]["properties"]["reason"]
        assert reason["enum"] == [item.value for item in ReviewReason]
        assert "$ref" not in json.dumps(body["tools"]) and body["tools"][1]["function"]["strict"]
        summary = json.loads(body["messages"][-1]["content"])["summary"]
        assert [part["part_id"] for part in summary["parts"]] == ["P001", "P002", "P003", "P004"]
        # 90도 돌린 직사각은 셀 (4, 4)와 (5, 4)를 쓴다. 위치는 왼쪽 위 셀이다.
        assert summary["parts"][2] == {
            "part_id": "P003",
            "type": "rect_tile",
            "color": "blue",
            "rotation_deg": 90,
            "row": 4,
            "col": 4,
        }
        assert {"type": "T_tile", "color": "gray", "count": 1} in summary["part_counts"]
        assert _tool_names(_body(api, 0)) == ["prepare_layout"]
        assert _tool_names(_body(api, 2)) == ["execute_cell"]
        prepared = read_events(tools)[4].result
        assert isinstance(prepared, LayoutResult)
        assert isinstance(prepared.summary, ReviewedLayoutSummary)
        inputs = ToolRun.model_validate_json((tools.root / "run.json").read_text())
        assert inputs.agent is not None and inputs.agent.review is True
        assert inputs.agent.system_prompt == REVIEW_SYSTEM_PROMPT

    @pytest.mark.parametrize("reason", list(ReviewReason))
    def test_replan_sends_code_feedback_and_executes_the_new_layout(
        self,
        tmp_path: Path,
        valid_layout: Layout,
        second_layout: Layout,
        api: FakeAgentModel,
        reason: ReviewReason,
    ) -> None:
        """재계획은 원문과 사유별 고정 문장을 Cosmos에 보낸다. 셀은 두 번째 계획을 실행한다."""
        api.replans = [reason.value]
        runtime, tools, cosmos = make_runtime(
            tmp_path / "run",
            valid_layout,
            api,
            review=True,
            responses=[
                make_planner_response_json(valid_layout),
                make_planner_response_json(second_layout),
            ],
        )
        report = runtime.run()
        assert report.state == ToolState.COMPLETED and report.calls.nemotron == 5
        assert report.calls.cosmos == 2 and tools.ledger.layout_attempts == 2
        assert cosmos.calls[0]["messages"][1]["content"] == GOAL
        assert cosmos.calls[1]["messages"][1]["content"] == f"{GOAL}\n\n{REVIEW_FEEDBACK[reason]}"
        results = [event for event in read_events(tools) if event.kind == "tool_result"]
        assert [event.call.name for event in results if event.call] == [
            "prepare_layout",
            "replan_layout",
            "calibrate_board",
            "execute_cell",
        ]
        first, second = results[0].result, results[1].result
        assert isinstance(first, LayoutResult) and isinstance(second, LayoutResult)
        assert first.artifacts[0].id != second.artifacts[0].id
        assert tools.ledger.layout_id == second.artifacts[0].id
        assert results[1].from_state == results[1].to_state == ToolState.LAYOUT_READY
        third = _body(api, 2)
        assert _tool_names(third) == ["calibrate_board"] and _state(third)["layout_attempts"] == 2
        assert _state(third)["layout_id"] == second.artifacts[0].id
        task = Task.model_validate_json((tools.root / "execution" / "task.json").read_text())
        assert len(task.parts) == len(second_layout.parts)
        audit = [
            json.loads(line) for line in (tools.root / "model_audit.jsonl").read_text().splitlines()
        ]
        call = next(
            record["data"]["response"]["choices"][0]["tool_calls"][0]
            for record in audit
            if record["inference"] == 2 and record["kind"] == "sdk_response"
        )
        assert call["name"] == "replan_layout" and call["arguments"]["reason"] == reason.value

    def test_failed_replan_aborts_without_capture(
        self, tmp_path: Path, valid_layout: Layout, api: FakeAgentModel
    ) -> None:
        """두 번째 계획이 무효이면 중단한다. 첫 계획으로 돌아가지 않는다."""
        api.replans = ["part_set_mismatch"]
        runtime, tools, _ = make_runtime(
            tmp_path / "run",
            valid_layout,
            api,
            review=True,
            responses=[make_planner_response_json(valid_layout), "bad", "bad"],
        )
        report = runtime.run()
        assert report.state == ToolState.ABORTED and report.error_code == ErrorCode.LAYOUT_INVALID
        assert report.calls.nemotron == 2 and report.calls.cosmos == 3
        assert report.calls.capture == report.calls.place == 0
        failed = read_events(tools)[-2].result
        assert isinstance(failed, ErrorResult) and not failed.recoverable
        assert failed.summary.tool == "replan_layout" and failed.summary.layout_attempt == 2

    def test_review_after_a_format_retry_offers_only_calibration(
        self, tmp_path: Path, valid_layout: Layout, api: FakeAgentModel
    ) -> None:
        """형식 오류 재시도가 준비 시도 2회를 쓰면 재계획을 제공하지 않는다."""
        api.replans = ["position_mismatch"]
        runtime, tools, _ = make_runtime(
            tmp_path / "run",
            valid_layout,
            api,
            review=True,
            responses=["bad", "bad", make_planner_response_json(valid_layout)],
        )
        report = runtime.run()
        assert report.state == ToolState.COMPLETED and report.calls.nemotron == 5
        third = _body(api, 2)
        assert _tool_names(third) == ["calibrate_board"] and _state(third)["layout_attempts"] == 2
        assert api.replans == ["position_mismatch"] and tools.ledger.layout_attempts == 2

    @pytest.mark.parametrize(("stage", "accepted"), [(1, False), (2, True), (3, False)])
    def test_plan_rejected_abort_is_valid_only_in_layout_ready(
        self,
        tmp_path: Path,
        valid_layout: Layout,
        api: FakeAgentModel,
        stage: int,
        accepted: bool,
    ) -> None:
        """모델은 `layout_ready`에서만 계획을 거부할 수 있다. 거부하면 캡처를 실행하지 않는다."""
        api.variant, api.variant_stage = "plan_rejected", stage
        runtime, _, _ = make_runtime(tmp_path / "run", valid_layout, api, review=True)
        report = runtime.run()
        assert report.state == ToolState.ABORTED and report.calls.nemotron == stage
        expected = ErrorCode.PLAN_REJECTED if accepted else ErrorCode.PROTOCOL_ERROR
        assert report.error_code == expected
        assert report.calls.capture == (stage > 2) and report.calls.place == 0

    def test_replan_with_another_layout_id_is_rejected(
        self, tmp_path: Path, valid_layout: Layout, api: FakeAgentModel
    ) -> None:
        """현재 레이아웃이 아닌 ID의 재계획은 Cosmos를 다시 호출하지 않는다."""
        api.variant, api.variant_stage = "replan_wrong_id", 2
        runtime, _, cosmos = make_runtime(tmp_path / "run", valid_layout, api, review=True)
        report = runtime.run()
        assert report.error_code == ErrorCode.PROTOCOL_ERROR and len(cosmos.calls) == 1

    def test_second_replan_is_rejected(
        self, tmp_path: Path, valid_layout: Layout, second_layout: Layout, api: FakeAgentModel
    ) -> None:
        """재계획 뒤에는 재계획을 다시 받지 않는다. 준비 시도는 2회다."""
        api.replans = ["arrangement_mismatch"]
        api.variant, api.variant_stage = "replan_unoffered", 3
        runtime, tools, cosmos = make_runtime(
            tmp_path / "run",
            valid_layout,
            api,
            review=True,
            responses=[
                make_planner_response_json(valid_layout),
                make_planner_response_json(second_layout),
            ],
        )
        report = runtime.run()
        assert report.error_code == ErrorCode.PROTOCOL_ERROR and report.calls.nemotron == 3
        assert len(cosmos.calls) == 2 and tools.ledger.layout_attempts == 2
        assert report.calls.capture == 0


class TestReviewModel:
    """계획 검토의 요청만 판단 모델로 보내는지 검사한다."""

    def test_only_layout_ready_requests_use_the_review_model(
        self, tmp_path: Path, valid_layout: Layout, second_layout: Layout, api: FakeAgentModel
    ) -> None:
        """준비, 실행과 완료는 조율 모델이다. 재계획 뒤의 검토도 판단 모델이다."""
        api.replans = ["part_set_mismatch"]
        runtime, tools, _ = make_runtime(
            tmp_path / "run",
            valid_layout,
            api,
            review=True,
            review_model=DEFAULT_JUDGMENT_MODEL,
            responses=[
                make_planner_response_json(valid_layout),
                make_planner_response_json(second_layout),
            ],
        )
        report = runtime.run()
        assert report.state == ToolState.COMPLETED and report.calls.catalog == 1
        models = [_body(api, index)["model"] for index in range(report.calls.nemotron)]
        assert models == [
            DEFAULT_NEMOTRON_MODEL,
            DEFAULT_JUDGMENT_MODEL,
            DEFAULT_JUDGMENT_MODEL,
            DEFAULT_NEMOTRON_MODEL,
            DEFAULT_NEMOTRON_MODEL,
        ]
        events = read_events(tools)
        results = [event for event in events if event.kind == "model_result"]
        assert [event.model for event in results] == models
        assert all(
            event.evidence is not None and event.evidence.model == event.model for event in results
        )
        requests = [event for event in events if event.kind == "model_request"]
        assert [event.model for event in requests] == models
        assert all(
            event.model == DEFAULT_NEMOTRON_MODEL
            for event in events
            if event.kind not in ("model_request", "model_result")
        )
        inputs = ToolRun.model_validate_json((tools.root / "run.json").read_text())
        assert inputs.agent is not None
        assert inputs.agent.model == DEFAULT_NEMOTRON_MODEL
        assert inputs.agent.review_model == DEFAULT_JUDGMENT_MODEL

    def test_review_model_is_not_used_when_review_is_off(
        self, tmp_path: Path, valid_layout: Layout, api: FakeAgentModel
    ) -> None:
        """검토를 끈 실행은 모든 요청을 조율 모델로 보낸다."""
        runtime, _, _ = make_runtime(
            tmp_path / "run", valid_layout, api, review_model=DEFAULT_JUDGMENT_MODEL
        )
        report = runtime.run()
        assert report.state == ToolState.COMPLETED
        assert {_body(api, index)["model"] for index in range(4)} == {DEFAULT_NEMOTRON_MODEL}

    def test_review_model_outside_the_catalog_stops_before_inference(
        self, tmp_path: Path, valid_layout: Layout, api: FakeAgentModel
    ) -> None:
        """계정 목록에 판단 모델이 없으면 모델 요청과 조립 도구를 시작하지 않는다."""
        api.models = [DEFAULT_NEMOTRON_MODEL]
        runtime, _, cosmos = make_runtime(
            tmp_path / "run", valid_layout, api, review=True, review_model=DEFAULT_JUDGMENT_MODEL
        )
        report = runtime.run()
        assert report.state == ToolState.ABORTED and report.error_code == ErrorCode.MODEL_ERROR
        assert report.calls.catalog == 1 and report.calls.nemotron == 0
        assert not api.model_posts and not cosmos.calls

    def test_response_from_another_model_in_the_review_is_a_model_error(
        self, tmp_path: Path, valid_layout: Layout, api: FakeAgentModel
    ) -> None:
        """검토 응답의 모델 ID가 판단 모델과 다르면 중단한다. 캡처를 실행하지 않는다."""
        api.variant, api.variant_stage = "different_model", 2
        runtime, _, _ = make_runtime(
            tmp_path / "run", valid_layout, api, review=True, review_model=DEFAULT_JUDGMENT_MODEL
        )
        report = runtime.run()
        assert report.error_code == ErrorCode.MODEL_ERROR
        assert report.calls.nemotron == 2 and report.calls.capture == 0

    def test_review_without_a_review_model_uses_the_coordination_model(
        self, tmp_path: Path, valid_layout: Layout, api: FakeAgentModel
    ) -> None:
        """판단 모델을 주지 않으면 검토도 조율 모델로 보낸다."""
        runtime, _, _ = make_runtime(tmp_path / "run", valid_layout, api, review=True)
        assert runtime.run().state == ToolState.COMPLETED
        assert {_body(api, index)["model"] for index in range(4)} == {DEFAULT_NEMOTRON_MODEL}


def test_arrangement_feedback_gives_the_board_numbers() -> None:
    """배치 방식의 재계획 문장은 직사각이 쓰는 col의 수와 row의 길이를 쓴다."""
    feedback = REVIEW_FEEDBACK[ReviewReason.ARRANGEMENT_MISMATCH]
    assert feedback.endswith(
        "Parts of the same type must have the same row and different cols. "
        "A rectangular tile uses 2 cols. A row has 8 cols."
    ), feedback
    assert all(text.count("\n") == 1 for text in REVIEW_FEEDBACK.values())


class TestReplanTool:
    """모델 없이 도구의 상태와 설정 검사를 확인한다."""

    def _tools(self, root: Path, layout: Layout, *, review: bool) -> AssemblyTools:
        runtime_parts = make_runtime(root, layout, FakeAgentModel(), review=review)
        return runtime_parts[1]

    def test_replan_needs_the_review_setting(self, tmp_path: Path, valid_layout: Layout) -> None:
        """검토를 끈 실행에서 재계획 도구는 계약 위반이다."""
        tools = self._tools(tmp_path / "run", valid_layout, review=False)
        prepared = tools.prepare_layout(PrepareLayoutArguments(retry=False), tool_call_id="a")
        assert isinstance(prepared, LayoutResult)
        result = tools.replan_layout(
            ReplanLayoutArguments(
                layout_id=prepared.artifacts[0].id, reason=ReviewReason.PART_SET_MISMATCH
            ),
            tool_call_id="b",
        )
        assert isinstance(result, ErrorResult) and result.error_code == ErrorCode.PROTOCOL_ERROR
        assert tools.ledger.state == ToolState.ABORTED and tools.ledger.layout_attempts == 1

    def test_replan_needs_a_prepared_layout(self, tmp_path: Path, valid_layout: Layout) -> None:
        """준비 전에는 재계획할 레이아웃이 없다."""
        tools = self._tools(tmp_path / "run", valid_layout, review=True)
        result = tools.replan_layout(
            ReplanLayoutArguments(layout_id="0" * 32, reason=ReviewReason.POSITION_MISMATCH),
            tool_call_id="a",
        )
        assert isinstance(result, ErrorResult) and result.error_code == ErrorCode.PROTOCOL_ERROR
        assert tools.ledger.layout_attempts == 0


class TestAllowedTools:
    """상태와 준비 시도 수에서 제공 도구를 정한다."""

    @pytest.mark.parametrize(
        ("state", "attempts", "review", "expected"),
        [
            (ToolState.LAYOUT_READY, 1, True, ["calibrate_board", "replan_layout"]),
            (ToolState.LAYOUT_READY, 2, True, ["calibrate_board"]),
            (ToolState.LAYOUT_READY, 1, False, ["calibrate_board"]),
            (ToolState.CREATED, 0, True, ["prepare_layout"]),
            (ToolState.LAYOUT_RETRYABLE, 1, True, ["prepare_layout"]),
            (ToolState.CALIBRATED, 1, True, ["execute_cell"]),
            (ToolState.CHECKED, 1, True, []),
        ],
    )
    def test_replan_is_offered_only_with_review_and_a_free_attempt(
        self, state: ToolState, attempts: int, review: bool, expected: list[str]
    ) -> None:
        ledger = ToolLedger(run_id="a" * 32, state=state, layout_attempts=attempts)
        assert [tool.value for tool in allowed_tools(ledger, review=review)] == expected
        for tool in allowed_tools(ledger, review=review):
            assert tool_schema(tool)["function"]["name"] == tool.value


def test_metadata_without_the_review_field_is_review_off() -> None:
    """현재 계약의 `run.json`에는 `review`가 없다. 없으면 검토를 끈 실행이다."""
    fields = AgentMetadata.model_fields
    assert fields["review"].default is False
