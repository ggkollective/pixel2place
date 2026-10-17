"""HTTP Fake와 실제 조립 도구로 조율 프로그램 상태와 실패 경계를 검사한다."""

import json
from collections.abc import Iterator, Sequence
from datetime import timedelta
from pathlib import Path

import pytest
from openai import OpenAI

from pixel2place.agent.artifacts import Manifest
from pixel2place.agent.contracts import ErrorCode, ToolState
from pixel2place.agent.prompts import system_prompt
from pixel2place.agent.runtime import TIMING_FILE, AgentRuntime
from pixel2place.agent.runtime_contracts import (
    MAX_REQUEST_RETRIES,
    AgentBudget,
    AgentEvent,
    AgentMetadata,
    AgentReport,
    EventTiming,
)
from pixel2place.agent.tools import AssemblyTools, ToolLedger, ToolRun
from pixel2place.cell.config import CellConfig
from pixel2place.cell.scene import SimScene
from pixel2place.config import DEFAULT_NEMOTRON_MODEL, SimSettings
from pixel2place.layout.schema import Layout
from pixel2place.reasoner.client import CosmosReasoner
from tests.factories import make_planner_response_json
from tests.fakes.fake_agent_model import FAKE_AGENT_KEY, FAKE_REASONING, FakeAgentModel
from tests.fakes.fake_openai import FakeOpenAI

pytestmark = [pytest.mark.unit, pytest.mark.usefixtures("fake_isaac")]
GOAL = "Place all four parts on the board. Keep their colors."


@pytest.fixture
def api() -> Iterator[FakeAgentModel]:
    """테스트마다 HTTP Fake를 만든다. 종료 때 클라이언트를 닫는다."""
    fake = FakeAgentModel()
    yield fake
    for client in fake.clients:
        client.close()


def make_runtime(
    root: Path,
    layout: Layout,
    api: FakeAgentModel,
    *,
    budget: AgentBudget | None = None,
    responses: Sequence[str | None] | None = None,
    cell: CellConfig | None = None,
    review: bool = False,
    review_model: str | None = None,
    scene: SimScene | None = None,
    disposition: bool = False,
    disposition_model: str | None = None,
) -> tuple[AgentRuntime, AssemblyTools, FakeOpenAI]:
    """도메인 코드는 유지한다. 모델 HTTP와 Isaac 프로세스만 Fake로 바꾼다.

    처분을 켠 실행의 기본 예산은 배치 2회다.
    """
    budget = budget or AgentBudget(max_place_runs=2 if disposition else 1)
    cosmos = FakeOpenAI(
        list(responses) if responses is not None else [make_planner_response_json(layout)]
    )
    tools = AssemblyTools(
        out_dir=root,
        prompt=GOAL,
        cell=cell or CellConfig(),
        scene=scene or SimScene(),
        reasoner=CosmosReasoner(client=cosmos, model="fake"),
        sim_settings=SimSettings(),  # pyright: ignore[reportCallIssue]
        agent=AgentMetadata(
            model=DEFAULT_NEMOTRON_MODEL,
            cosmos_model="fake",
            budget=budget,
            system_prompt=system_prompt(review=review, disposition=disposition),
            review=review,
            review_model=review_model,
            disposition=disposition,
            disposition_model=disposition_model,
        ),
    )
    client = OpenAI(
        api_key=FAKE_AGENT_KEY,
        base_url="https://api.tokenfactory.nebius.com/v1/",
        max_retries=0,
        http_client=api.make_client(),
    )
    return (
        AgentRuntime(
            tools=tools,
            client=client,
            model=DEFAULT_NEMOTRON_MODEL,
            budget=budget,
            prompt=GOAL,
            review_model=review_model,
            disposition_model=disposition_model,
        ),
        tools,
        cosmos,
    )


def first_request_bytes(root: Path, layout: Layout) -> int:
    """첫 모델 요청의 JSON 크기. 모델 요청 1회로 끝나는 실행의 요청 자료다."""
    fake = FakeAgentModel()
    runtime, _, _ = make_runtime(root, layout, fake, budget=AgentBudget(max_inferences=1))
    report = runtime.run()
    for client in fake.clients:
        client.close()
    return report.calls.request_bytes


def read_events(tools: AssemblyTools) -> list[AgentEvent]:
    """저장한 모든 이벤트를 계약 모델로 다시 읽는다."""
    return [
        AgentEvent.model_validate_json(line)
        for line in (tools.root / "events.jsonl").read_text().splitlines()
    ]


class TestAgentRuntime:
    """정상 완료, 재계획과 모델 중단을 검사한다."""

    def test_complete_requires_model_decision_and_verified_check(
        self,
        tmp_path: Path,
        valid_layout: Layout,
        api: FakeAgentModel,
    ) -> None:
        """모델이 세 도구와 완료를 선택하면 판정 근거를 검사하고 완료한다."""
        runtime, tools, cosmos = make_runtime(tmp_path / "run", valid_layout, api)
        report = runtime.run()
        assert report.state == ToolState.COMPLETED and report.error_code is None
        assert report.calls.nemotron == 4 and report.calls.catalog == 1
        assert report.calls.cosmos == 1 and report.calls.capture == report.calls.place == 1
        assert report.check_passed is True and report.check_id is not None
        saved = AgentReport.model_validate_json((tools.root / "agent_report.json").read_text())
        assert saved == report
        inputs = ToolRun.model_validate_json((tools.root / "run.json").read_text())
        assert inputs.agent is not None and inputs.agent.budget == AgentBudget()
        events = read_events(tools)
        assert [e.sequence for e in events] == list(range(1, len(events) + 1))
        assert [e.call.name for e in events if e.kind == "tool_result" and e.call] == [
            "prepare_layout",
            "calibrate_board",
            "execute_cell",
        ]
        assert events[-1].to_state == ToolState.COMPLETED
        assert cosmos.calls[0]["messages"][1]["content"] == GOAL
        for index, request in enumerate(api.model_posts):
            body = json.loads(request.content)
            assert body["messages"][1]["content"] == GOAL
            assert body["max_tokens"] == 2048
            assert request.extensions["timeout"]["read"] == 120
            if index < 3:
                assert body["tool_choice"] == "auto" and len(body["tools"]) == 1
            else:
                assert "tools" not in body and "tool_choice" not in body
            if index:
                assert body["messages"][-1]["role"] == "tool"
                assert (
                    body["messages"][-2]["tool_calls"][0]["id"]
                    == body["messages"][-1]["tool_call_id"]
                )
        model_text = "\n".join(r.content.decode() for r in api.model_posts)
        record_text = (tools.root / "events.jsonl").read_text()
        for excluded in (
            FAKE_AGENT_KEY,
            FAKE_REASONING,
            "camera_truth",
            "scene.json",
            "sha256",
            str(tools.root),
        ):
            assert excluded not in model_text and excluded not in record_text
        assert all(e.evidence.reasoning_tokens == 20 for e in events if e.evidence)
        with pytest.raises(ValueError, match="twice"):
            runtime.run()
        assert len(api.model_posts) == 4

    def test_recoverable_preparation_allows_one_model_selected_retry(
        self,
        tmp_path: Path,
        valid_layout: Layout,
        api: FakeAgentModel,
    ) -> None:
        """형식 오류 뒤 모델이 재계획을 선택하면 원문을 유지하고 완료한다."""
        runtime, tools, cosmos = make_runtime(
            tmp_path / "run",
            valid_layout,
            api,
            responses=["bad", "bad", make_planner_response_json(valid_layout)],
        )
        report = runtime.run()
        assert report.state == ToolState.COMPLETED and report.calls.nemotron == 5
        assert report.calls.cosmos == 3 and tools.ledger.layout_attempts == 2
        assert cosmos.calls[2]["messages"][1]["content"].startswith(GOAL + "\n\n")

    @pytest.mark.parametrize("stage", [1, 2, 3, 4])
    def test_voluntary_abort_stops_at_current_stage(
        self,
        tmp_path: Path,
        valid_layout: Layout,
        api: FakeAgentModel,
        stage: int,
    ) -> None:
        """모델이 중단을 선택하면 다음 도구를 실행하지 않는다."""
        api.variant, api.variant_stage = "abort", stage
        runtime, _, cosmos = make_runtime(tmp_path / "run", valid_layout, api)
        report = runtime.run()
        assert report.state == ToolState.ABORTED and report.error_code == ErrorCode.AGENT_ABORTED
        assert report.calls.nemotron == stage
        assert len(cosmos.calls) == (stage > 1)
        assert report.calls.capture == (stage > 2) and report.calls.place == (stage > 3)

    def test_check_failure_is_read_by_model_without_replacement(
        self,
        tmp_path: Path,
        valid_layout: Layout,
        api: FakeAgentModel,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """배치 판정이 실패하면 모델에 전달하고 재배치 없이 중단한다."""
        monkeypatch.setenv("FAKE_ISAAC_MODE", "miss")
        runtime, _, _ = make_runtime(tmp_path / "run", valid_layout, api)
        report = runtime.run()
        assert report.error_code == ErrorCode.CHECK_FAILED and report.calls.nemotron == 4
        assert report.calls.place == 1 and report.check_passed is False
        result = json.loads(json.loads(api.model_posts[-1].content)["messages"][-1]["content"])
        assert result["status"] == "ok" and result["summary"]["passed"] is False

    def test_failed_second_preparation_stops_without_more_inference(
        self,
        tmp_path: Path,
        valid_layout: Layout,
        api: FakeAgentModel,
    ) -> None:
        """재계획도 실패하면 세 번째 모델 요청과 Isaac 실행을 하지 않는다."""
        runtime, _, _ = make_runtime(tmp_path / "run", valid_layout, api, responses=["bad"] * 4)
        report = runtime.run()
        assert report.error_code == ErrorCode.LAYOUT_INVALID and report.calls.nemotron == 2
        assert report.calls.cosmos == 4 and report.calls.capture == report.calls.place == 0

    def test_calibration_failure_stops_without_more_inference(
        self,
        tmp_path: Path,
        valid_layout: Layout,
        api: FakeAgentModel,
    ) -> None:
        """캡처 보정이 실패하면 공칭 셀로 배치하지 않는다."""
        cell = CellConfig()
        cell = cell.model_copy(update={"markers": cell.markers[:3]})
        runtime, _, _ = make_runtime(tmp_path / "run", valid_layout, api, cell=cell)
        report = runtime.run()
        assert report.error_code == ErrorCode.CALIBRATION_FAILED and report.calls.nemotron == 2
        assert report.calls.capture == 1 and report.calls.place == 0


class TestAgentProtocol:
    """잘못된 모델 응답과 도구 호출을 실행 전에 거부한다."""

    @pytest.mark.parametrize(
        "variant",
        [
            "wrong_tool",
            "multiple_tools",
            "extra_arguments",
            "coercion",
            "wrong_retry",
            "bad_json",
            "blank_id",
            "free_text",
            "early_complete",
            "bad_abort",
            "plain_final",
            "extra_final",
            "duplicate_final",
            "empty_choices",
            "multiple_choices",
            "missing_id",
            "refusal",
            "legacy_call",
            "empty_final",
        ],
    )
    def test_invalid_first_response_never_executes_tools(
        self,
        tmp_path: Path,
        valid_layout: Layout,
        api: FakeAgentModel,
        variant: str,
    ) -> None:
        """응답 계약이 틀리면 Cosmos와 Isaac을 호출하지 않는다."""
        api.variant = variant
        runtime, tools, cosmos = make_runtime(tmp_path / "run", valid_layout, api)
        report = runtime.run()
        assert report.error_code == ErrorCode.PROTOCOL_ERROR
        assert report.calls.nemotron == 1 and not cosmos.calls
        assert report.calls.capture == report.calls.place == 0
        assert read_events(tools)[-1].error_code == ErrorCode.PROTOCOL_ERROR

    @pytest.mark.parametrize(
        ("variant", "stage"),
        [
            ("duplicate_id", 2),
            ("wrong_id", 2),
            ("wrong_calibration", 3),
        ],
    )
    def test_bad_call_does_not_start_next_isaac_action(
        self,
        tmp_path: Path,
        valid_layout: Layout,
        api: FakeAgentModel,
        variant: str,
        stage: int,
    ) -> None:
        """중복 호출이나 잘못된 ID를 받으면 다음 외부 작업을 하지 않는다."""
        api.variant, api.variant_stage = variant, stage
        runtime, _, _ = make_runtime(tmp_path / "run", valid_layout, api)
        report = runtime.run()
        assert report.error_code == ErrorCode.PROTOCOL_ERROR and report.calls.nemotron == stage
        assert report.calls.capture == (stage == 3) and report.calls.place == 0

    @pytest.mark.parametrize(
        "variant", ["truncated", "different_model", "http_timeout", "http_connection", "http_503"]
    )
    def test_model_failure_has_no_retry_or_tool_call(
        self,
        tmp_path: Path,
        valid_layout: Layout,
        api: FakeAgentModel,
        variant: str,
    ) -> None:
        """모델 또는 HTTP가 실패하면 한 번만 요청하고 도구를 호출하지 않는다."""
        api.variant = variant
        runtime, tools, cosmos = make_runtime(tmp_path / "run", valid_layout, api)
        report = runtime.run()
        assert report.error_code == ErrorCode.MODEL_ERROR and report.calls.nemotron == 1
        assert not cosmos.calls and len(api.model_posts) == 1
        if variant.startswith("http_"):
            result = next(e for e in read_events(tools) if e.kind == "model_result")
            assert result.evidence is None
        assert FAKE_AGENT_KEY not in (tools.root / "events.jsonl").read_text()

    @pytest.mark.parametrize("failure", ["unavailable", "http"])
    def test_catalog_failure_stops_before_inference(
        self,
        tmp_path: Path,
        valid_layout: Layout,
        api: FakeAgentModel,
        failure: str,
    ) -> None:
        """모델 목록 검사가 실패하면 모델 요청을 하지 않는다."""
        if failure == "unavailable":
            api.models = []
        else:
            api.variant, api.variant_stage = "http_503", 0
        runtime, _, cosmos = make_runtime(tmp_path / "run", valid_layout, api)
        report = runtime.run()
        assert report.error_code == ErrorCode.MODEL_ERROR and report.calls.catalog == 1
        assert report.calls.nemotron == 0 and not cosmos.calls

    def test_missing_usage_is_valid_without_invented_counts(
        self,
        tmp_path: Path,
        valid_layout: Layout,
        api: FakeAgentModel,
    ) -> None:
        """사용량이 없으면 null로 기록하고 판정 성공을 검증한다."""
        api.variant = "no_usage"
        runtime, tools, _ = make_runtime(tmp_path / "run", valid_layout, api)
        assert runtime.run().state == ToolState.COMPLETED
        assert all(e.evidence.prompt_tokens is None for e in read_events(tools) if e.evidence)

    @pytest.mark.parametrize("variant", ["http_503", "truncated", "early_complete"])
    def test_model_failure_after_placement_cannot_report_success(
        self,
        tmp_path: Path,
        valid_layout: Layout,
        api: FakeAgentModel,
        monkeypatch: pytest.MonkeyPatch,
        variant: str,
    ) -> None:
        """실행 후 모델 오류나 거짓 완료 요청이 있으면 배치를 반복하지 않는다."""
        api.variant, api.variant_stage = variant, 4
        if variant == "early_complete":
            monkeypatch.setenv("FAKE_ISAAC_MODE", "miss")
        runtime, _, _ = make_runtime(tmp_path / "run", valid_layout, api)
        report = runtime.run()
        assert report.state == ToolState.ABORTED and report.calls.nemotron == 4
        assert report.calls.capture == report.calls.place == 1
        assert report.error_code == (
            ErrorCode.PROTOCOL_ERROR if variant == "early_complete" else ErrorCode.MODEL_ERROR
        )


class TestRequestRetry:
    """재전송을 켠 예산에서 일시 오류의 재전송, 집계와 기록을 검사한다."""

    RETRY = AgentBudget(request_retries=MAX_REQUEST_RETRIES)

    @pytest.mark.parametrize("stage", [0, 1, 4])
    def test_one_503_is_sent_again_and_the_run_completes(
        self,
        tmp_path: Path,
        valid_layout: Layout,
        api: FakeAgentModel,
        retry_waits: list[float],
        stage: int,
    ) -> None:
        """모델 목록 또는 모델 요청의 503이 1회이면 다시 보내고 완료한다.

        모델 요청 수는 늘지 않는다.
        """
        api.variant, api.variant_stage, api.http_failures = "http_503", stage, 1
        runtime, tools, _ = make_runtime(tmp_path / "run", valid_layout, api, budget=self.RETRY)
        report = runtime.run()
        assert report.state == ToolState.COMPLETED, report.error_code
        assert report.calls.retries == 1 and retry_waits == [3.0], retry_waits
        assert report.calls.catalog == 1 and report.calls.nemotron == 4
        gets = [request for request in api.requests if request.method == "GET"]
        assert len(gets) == 1 + (stage == 0) and len(api.model_posts) == 4 + (stage > 0)
        events = read_events(tools)
        assert sum(event.kind == "model_request" for event in events) == 4
        assert sum(event.kind == "catalog_request" for event in events) == 1
        saved = AgentReport.model_validate_json((tools.root / "agent_report.json").read_text())
        assert saved.calls.retries == 1
        assert FAKE_AGENT_KEY not in (tools.root / "model_audit.jsonl").read_text()

    @pytest.mark.parametrize("stage", [0, 1])
    def test_four_failed_attempts_stop_with_model_error(
        self,
        tmp_path: Path,
        valid_layout: Layout,
        api: FakeAgentModel,
        retry_waits: list[float],
        stage: int,
    ) -> None:
        """같은 요청이 4회 연속으로 실패하면 `model_error`로 중단한다. 도구는 실행하지 않는다."""
        api.variant, api.variant_stage, api.http_failures = "http_503", stage, 4
        runtime, tools, cosmos = make_runtime(
            tmp_path / "run", valid_layout, api, budget=self.RETRY
        )
        report = runtime.run()
        assert report.state == ToolState.ABORTED and report.error_code == ErrorCode.MODEL_ERROR
        assert report.calls.retries == 3 and retry_waits == [3.0, 9.0, 20.0], retry_waits
        assert report.calls.catalog == 1 and report.calls.nemotron == stage
        assert len(api.requests) == 4 + stage and not cosmos.calls
        records = [
            json.loads(line) for line in (tools.root / "model_audit.jsonl").read_text().splitlines()
        ]
        attempts = [record for record in records if record["kind"] == "attempt"]
        failed = [record for record in attempts if record["inference"] == stage]
        assert [record["data"]["attempt"] for record in failed] == [1, 2, 3, 4]
        assert {record["data"]["target"] for record in failed} == {
            "completion" if stage else "catalog"
        }
        assert failed[-1]["data"] == {
            "target": "completion" if stage else "catalog",
            "attempt": 4,
            "outcome": "http_error",
            "status": 503,
            "wait_s": None,
        }

    @pytest.mark.parametrize(
        ("variant", "error"),
        [
            ("http_400", ErrorCode.MODEL_ERROR),
            ("truncated", ErrorCode.MODEL_ERROR),
            ("different_model", ErrorCode.MODEL_ERROR),
            ("wrong_tool", ErrorCode.PROTOCOL_ERROR),
            ("empty_choices", ErrorCode.PROTOCOL_ERROR),
        ],
    )
    def test_other_failures_are_not_sent_again(
        self,
        tmp_path: Path,
        valid_layout: Layout,
        api: FakeAgentModel,
        retry_waits: list[float],
        variant: str,
        error: ErrorCode,
    ) -> None:
        """다른 4xx, 잘린 응답, 다른 모델과 응답 형식의 오류는 다시 보내지 않는다."""
        api.variant = variant
        runtime, _, cosmos = make_runtime(tmp_path / "run", valid_layout, api, budget=self.RETRY)
        report = runtime.run()
        assert report.error_code == error and report.calls.nemotron == 1
        assert report.calls.retries == 0 and not retry_waits
        assert len(api.model_posts) == 1 and not cosmos.calls

    def test_unavailable_model_is_not_requested_again(
        self,
        tmp_path: Path,
        valid_layout: Layout,
        api: FakeAgentModel,
        retry_waits: list[float],
    ) -> None:
        """모델 목록에 모델이 없으면 목록 요청을 다시 보내지 않는다."""
        api.models = []
        runtime, tools, _ = make_runtime(tmp_path / "run", valid_layout, api, budget=self.RETRY)
        report = runtime.run()
        assert report.error_code == ErrorCode.MODEL_ERROR and report.calls.retries == 0
        assert len(api.requests) == 1 and not retry_waits
        record = json.loads((tools.root / "model_audit.jsonl").read_text().splitlines()[0])
        assert record["kind"] == "attempt" and record["inference"] == 0
        assert record["data"]["outcome"] == "invalid_response"

    def test_each_attempt_is_recorded_and_counts_request_bytes(
        self,
        tmp_path: Path,
        valid_layout: Layout,
        api: FakeAgentModel,
        retry_waits: list[float],
    ) -> None:
        """시도마다 기록 1줄을 쓴다. 다시 보낸 요청의 크기도 요청 자료에 센다."""
        size = first_request_bytes(tmp_path / "single", valid_layout)
        api.variant, api.http_failures = "http_503", 2
        runtime, tools, _ = make_runtime(
            tmp_path / "run",
            valid_layout,
            api,
            budget=AgentBudget(max_inferences=1, request_retries=MAX_REQUEST_RETRIES),
        )
        report = runtime.run()
        assert report.calls.nemotron == 1 and report.calls.retries == 2
        assert report.calls.request_bytes == 3 * size, (report.calls.request_bytes, size)
        assert retry_waits == [3.0, 9.0]
        records = [
            json.loads(line) for line in (tools.root / "model_audit.jsonl").read_text().splitlines()
        ]
        assert [record["kind"] for record in records if record["inference"] == 0] == ["attempt"]
        assert [record["kind"] for record in records if record["inference"] == 1] == [
            "request",
            "http_response",
            "attempt",
            "http_response",
            "attempt",
            "http_response",
            "sdk_response",
            "attempt",
            "validation",
        ]

    def test_retry_stops_at_the_request_byte_limit(
        self,
        tmp_path: Path,
        valid_layout: Layout,
        api: FakeAgentModel,
        retry_waits: list[float],
    ) -> None:
        """요청 자료의 한도가 다음 시도를 받지 못하면 다시 보내지 않는다."""
        size = first_request_bytes(tmp_path / "single", valid_layout)
        api.variant, api.http_failures = "http_503", 4
        budget = AgentBudget(request_retries=MAX_REQUEST_RETRIES, max_request_bytes=2 * size)
        runtime, _, _ = make_runtime(tmp_path / "run", valid_layout, api, budget=budget)
        report = runtime.run()
        assert report.error_code == ErrorCode.MODEL_ERROR
        assert report.calls.retries == 1 and report.calls.request_bytes == 2 * size
        assert len(api.model_posts) == 2 and retry_waits == [3.0]


class TestAgentLimitsAndRecords:
    """예산과 파일 기록 실패 뒤 외부 작업을 막는다."""

    @pytest.mark.parametrize(
        ("budget", "requests", "captures"),
        [
            (AgentBudget(max_inferences=1), 1, 0),
            (AgentBudget(max_request_bytes=1), 0, 0),
            (AgentBudget(max_tool_bytes=1), 1, 0),
        ],
    )
    def test_exhausted_budget_stops_without_next_external_action(
        self,
        tmp_path: Path,
        valid_layout: Layout,
        api: FakeAgentModel,
        budget: AgentBudget,
        requests: int,
        captures: int,
    ) -> None:
        """호출 또는 자료 한도를 소진하면 다음 요청을 하지 않는다."""
        runtime, _, _ = make_runtime(tmp_path / "run", valid_layout, api, budget=budget)
        report = runtime.run()
        assert report.error_code == ErrorCode.BUDGET_EXHAUSTED
        assert report.calls.nemotron == requests and report.calls.capture == captures
        assert report.calls.place == 0

    @pytest.mark.parametrize("file", ["attempts/01/layout.json", "execution/check.json"])
    def test_changed_artifact_cannot_support_next_action_or_completion(
        self,
        tmp_path: Path,
        valid_layout: Layout,
        api: FakeAgentModel,
        file: str,
    ) -> None:
        """등록한 파일이 바뀌면 다음 작업 또는 최종 완료를 거부한다."""
        runtime, tools, _ = make_runtime(tmp_path / "run", valid_layout, api)
        stage = 2 if file.startswith("attempts") else 4

        def change(request: object) -> None:
            if len(api.model_posts) == stage:
                path = tools.root / file
                path.write_text(path.read_text() + " ")

        api.on_request = change
        report = runtime.run()
        assert report.error_code == ErrorCode.PROTOCOL_ERROR and report.calls.nemotron == stage
        assert report.calls.capture == report.calls.place == (stage == 4)

    def test_event_write_failure_stops_before_tool_execution(
        self,
        tmp_path: Path,
        valid_layout: Layout,
        api: FakeAgentModel,
    ) -> None:
        """모델 결과 기록이 실패하면 Cosmos를 호출하지 않고 중단한다."""
        runtime, tools, cosmos = make_runtime(tmp_path / "run", valid_layout, api)

        def obstruct(request: object) -> None:
            if api.model_posts:
                path = tools.root / "events.jsonl"
                path.unlink()
                path.mkdir()

        api.on_request = obstruct
        report = runtime.run()
        assert report.error_code == ErrorCode.RECORDING_ERROR and not cosmos.calls
        assert report.calls.nemotron == 1 and report.calls.capture == report.calls.place == 0

    def test_final_report_failure_cannot_return_success(
        self,
        tmp_path: Path,
        valid_layout: Layout,
        api: FakeAgentModel,
    ) -> None:
        """최종 보고를 쓸 수 없으면 성공을 반환하지 않는다."""
        runtime, tools, _ = make_runtime(tmp_path / "run", valid_layout, api)
        (tools.root / "agent_report.json").mkdir()
        report = runtime.run()
        assert report.state == ToolState.ABORTED and report.error_code == ErrorCode.RECORDING_ERROR
        ledger = ToolLedger.model_validate_json((tools.root / "tool_state.json").read_text())
        assert ledger.state == ToolState.ABORTED
        assert read_events(tools)[-1].error_code == ErrorCode.RECORDING_ERROR

    @pytest.mark.parametrize("file", ["tool_results/01.json", "tool_state.json"])
    def test_tool_record_failure_stops_before_capture(
        self,
        tmp_path: Path,
        valid_layout: Layout,
        api: FakeAgentModel,
        file: str,
    ) -> None:
        """도구 결과나 상태 기록이 실패하면 다음 캡처를 실행하지 않는다."""
        runtime, tools, cosmos = make_runtime(tmp_path / "run", valid_layout, api)
        stage = 1 if file.startswith("tool_results") else 2

        def obstruct(request: object) -> None:
            if len(api.model_posts) == stage:
                path = tools.root / file
                path.unlink(missing_ok=True)
                path.mkdir()

        api.on_request = obstruct
        report = runtime.run()
        assert report.error_code == ErrorCode.RECORDING_ERROR
        assert len(cosmos.calls) == 1 and report.calls.nemotron == stage
        assert report.calls.capture == report.calls.place == 0


class TestEventTiming:
    """이벤트의 시각 기록과 그 기록의 실패를 검사한다."""

    def test_each_event_has_one_timing_line(
        self, tmp_path: Path, valid_layout: Layout, api: FakeAgentModel
    ) -> None:
        """이벤트마다 순번, 종류, UTC 시각과 경과 초를 쓴다. manifest에는 등록하지 않는다."""
        runtime, tools, _ = make_runtime(tmp_path / "run", valid_layout, api)
        assert runtime.run().state == ToolState.COMPLETED
        events = read_events(tools)
        timings = [
            EventTiming.model_validate_json(line)
            for line in (tools.root / TIMING_FILE).read_text().splitlines()
        ]
        assert [(t.sequence, t.kind) for t in timings] == [(e.sequence, e.kind) for e in events]
        elapsed = [t.elapsed_s for t in timings]
        assert elapsed == sorted(elapsed), f"경과 초가 줄었다: {elapsed}"
        # 벽시계는 뒤로 갈 수 있다. 순서는 경과 초로만 검사한다.
        assert all(t.utc.utcoffset() == timedelta(0) for t in timings)
        manifest = Manifest.model_validate_json((tools.root / "manifest.json").read_text())
        registered = {file.path for record in manifest.artifacts for file in record.files}
        assert TIMING_FILE not in registered and "events.jsonl" not in registered

    def test_timing_write_failure_stops_before_inference(
        self, tmp_path: Path, valid_layout: Layout, api: FakeAgentModel
    ) -> None:
        """시각 기록을 쓸 수 없으면 기록 오류로 중단한다. 모델을 호출하지 않는다."""
        runtime, tools, cosmos = make_runtime(tmp_path / "run", valid_layout, api)
        (tools.root / TIMING_FILE).mkdir()
        report = runtime.run()
        assert report.state == ToolState.ABORTED
        assert report.error_code == ErrorCode.RECORDING_ERROR
        assert report.calls.nemotron == 0 and not api.requests and not cosmos.calls


class TestAgentResponseAudit:
    """전송 자료, SDK 응답과 거부 판정을 인증값 없이 연결한다."""

    def test_rejected_second_response_preserves_wire_and_sdk_arguments(
        self, tmp_path: Path, valid_layout: Layout, api: FakeAgentModel
    ) -> None:
        """준비 성공 뒤 같은 도구와 문자열 불리언을 받으면 근거를 남기고 차단한다."""
        api.variant, api.variant_stage = "repeat_prepare", 2
        runtime, tools, cosmos = make_runtime(tmp_path / "run", valid_layout, api)
        report = runtime.run()
        assert report.error_code == ErrorCode.PROTOCOL_ERROR and report.calls.nemotron == 2
        assert len(cosmos.calls) == 1 and report.calls.capture == report.calls.place == 0
        records = [
            json.loads(line) for line in (tools.root / "model_audit.jsonl").read_text().splitlines()
        ]
        second = [record for record in records if record["inference"] == 2]
        assert [record["kind"] for record in second] == [
            "request",
            "http_response",
            "sdk_response",
            "validation",
        ]
        wire, sdk = second[1]["data"]["response"], second[2]["data"]["response"]
        assert wire == sdk
        call = wire["choices"][0]["tool_calls"][0]
        assert call["name"] == "prepare_layout"
        assert call["arguments"] == {"retry": "True"}
        assert call["argument_types"] == {"retry": "str"}
        assert second[-1]["data"]["accepted"] is False
        assert second[-1]["data"]["reason"] == "tool_not_allowed"
        request = second[0]["data"]["request"]
        assert request["tools"][0]["function"]["name"] == "calibrate_board"
        assert request["messages"][-1]["role"] == "tool"
        text = (tools.root / "model_audit.jsonl").read_text()
        assert FAKE_AGENT_KEY not in text and FAKE_REASONING not in text

    def test_string_boolean_has_a_separate_validation_reason(
        self, tmp_path: Path, valid_layout: Layout, api: FakeAgentModel
    ) -> None:
        """허용된 도구의 문자열 불리언은 도구 이름 오류와 구분한다."""
        api.variant = "coercion"
        runtime, tools, cosmos = make_runtime(tmp_path / "run", valid_layout, api)
        report = runtime.run()
        assert report.error_code == ErrorCode.PROTOCOL_ERROR and not cosmos.calls
        records = [
            json.loads(line) for line in (tools.root / "model_audit.jsonl").read_text().splitlines()
        ]
        assert records[-1]["data"]["reason"] == "invalid_arguments"
        assert records[-1]["data"]["issues"] == [{"location": ["retry"], "type": "bool_type"}]

    def test_audit_write_failure_stops_before_request(
        self, tmp_path: Path, valid_layout: Layout, api: FakeAgentModel
    ) -> None:
        """모델 요청 전 관찰 기록이 실패하면 모델과 도구를 호출하지 않는다."""
        runtime, tools, cosmos = make_runtime(tmp_path / "run", valid_layout, api)
        (tools.root / "model_audit.jsonl").mkdir()
        report = runtime.run()
        assert report.error_code == ErrorCode.RECORDING_ERROR
        assert report.calls.nemotron == 0 and not api.model_posts and not cosmos.calls

    def test_http_audit_write_failure_stops_before_tool(
        self, tmp_path: Path, valid_layout: Layout, api: FakeAgentModel
    ) -> None:
        """응답 관찰 기록이 실패하면 이미 받은 도구 호출도 실행하지 않는다."""
        runtime, tools, cosmos = make_runtime(tmp_path / "run", valid_layout, api)

        def obstruct(request: object) -> None:
            if api.model_posts:
                path = tools.root / "model_audit.jsonl"
                path.unlink()
                path.mkdir()

        api.on_request = obstruct
        report = runtime.run()
        assert report.error_code == ErrorCode.RECORDING_ERROR
        assert report.calls.nemotron == 1 and not cosmos.calls

    def test_requests_preserve_auto_abort_with_strict_tools_and_state_rules(
        self, tmp_path: Path, valid_layout: Layout, api: FakeAgentModel
    ) -> None:
        """상태 규칙과 서버 생성 제약을 전달한다. 모델의 중단 선택은 유지한다."""
        runtime, _, _ = make_runtime(
            tmp_path / "run", valid_layout, api, budget=AgentBudget(max_inferences=1)
        )
        runtime.run()
        request = json.loads(api.model_posts[0].content)
        assert request["tools"][0]["function"]["strict"] is True
        assert request["tool_choice"] == "auto"
        assert "In layout_ready, call calibrate_board" in request["messages"][0]["content"]
        assert "Do not use strings for booleans." in request["messages"][0]["content"]
        assert request["messages"][1]["content"] == GOAL
