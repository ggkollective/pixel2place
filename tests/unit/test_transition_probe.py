"""실제 모델 비교 CLI의 입력 검사와 비실행 계약을 HTTP Fake로 검사한다."""

import json
from pathlib import Path

import pytest
from typer.testing import CliRunner

from pixel2place.agent.runtime_contracts import AgentBudget
from pixel2place.agent.transition_probe import PROBE_CASES, ProbeReport, load_probe_context
from pixel2place.cli import app
from pixel2place.layout.schema import Layout
from pixel2place.reasoner import nemotron
from tests.fakes.fake_agent_model import FAKE_AGENT_KEY, FakeAgentModel
from tests.unit.test_agent_runtime import make_runtime

pytestmark = [pytest.mark.unit, pytest.mark.usefixtures("fake_isaac")]


@pytest.fixture
def prepared_run(tmp_path: Path, valid_layout: Layout) -> Path:
    """준비만 성공한 실행을 만든다. 캡처와 배치는 호출하지 않는다."""
    fake = FakeAgentModel()
    fake.call_prefix = "source-call"
    runtime, tools, _ = make_runtime(
        tmp_path / "source", valid_layout, fake, budget=AgentBudget(max_inferences=1)
    )
    report = runtime.run()
    assert report.calls.capture == report.calls.place == 0
    for client in fake.clients:
        client.close()
    return tools.root


@pytest.fixture
def probe_api(monkeypatch: pytest.MonkeyPatch) -> FakeAgentModel:
    """모델 경계만 대체한다. 실제 CLI와 응답 검사는 유지한다."""
    fake = FakeAgentModel()
    monkeypatch.setenv("NEBIUS_API_KEY", FAKE_AGENT_KEY)
    monkeypatch.setattr(nemotron, "DefaultHttpxClient", fake.make_client)
    return fake


def test_probe_runs_four_independent_requests_without_assembly_tools(
    tmp_path: Path, prepared_run: Path, probe_api: FakeAgentModel
) -> None:
    """준비 결과를 그대로 쓴다. 각 조건에서 한 번 모델 요청하고 SDK 값을 비교한다."""
    before = (prepared_run / "events.jsonl").read_bytes()
    output = tmp_path / "probe"
    result = CliRunner().invoke(
        app, ["verify-agent-transition", "--run", str(prepared_run), "-o", str(output)]
    )
    assert result.exit_code == 0, result.output
    report = ProbeReport.model_validate_json((output / "probe_report.json").read_text())
    assert report.request_count == 4 and tuple(item.case for item in report.results) == PROBE_CASES
    assert all(item.accepted and item.selected_tool == "calibrate_board" for item in report.results)
    assert all(item.wire_matches_sdk for item in report.results)
    requests = [json.loads(request.content) for request in probe_api.model_posts]
    assert len(requests) == 4
    assert requests[0]["messages"] == requests[2]["messages"] == requests[3]["messages"]
    assert requests[0]["tools"] == requests[1]["tools"] == requests[3]["tools"]
    assert requests[2]["tools"][0]["function"]["strict"] is True
    assert "strict" not in requests[0]["tools"][0]["function"]
    assert requests[3]["tool_choice"] == {
        "type": "function",
        "function": {"name": "calibrate_board"},
    }
    assert all(
        request["messages"][-1]["content"] == requests[0]["messages"][-1]["content"]
        for request in requests
    )
    assert (prepared_run / "events.jsonl").read_bytes() == before
    assert all(client.is_closed for client in probe_api.clients)
    assert not (output / "attempts").exists() and not (output / "calibration").exists()


def test_failed_comparison_is_preserved_and_does_not_retry(
    tmp_path: Path, prepared_run: Path, probe_api: FakeAgentModel
) -> None:
    """원래 조건의 실패 뒤 다음 독립 조건을 실행한다. 실패 요청은 다시 하지 않는다."""
    probe_api.variant = "repeat_prepare"
    output = tmp_path / "probe"
    result = CliRunner().invoke(
        app, ["verify-agent-transition", "--run", str(prepared_run), "-o", str(output)]
    )
    assert result.exit_code == 1
    report = ProbeReport.model_validate_json((output / "probe_report.json").read_text())
    assert report.results[0].reason == "tool_not_allowed"
    assert report.request_count == len(probe_api.model_posts) == 4


@pytest.mark.parametrize(
    "failure", ["existing_output", "bad_source", "changed_result", "invalid_first_retry"]
)
def test_probe_preflight_failures_never_send_requests(
    tmp_path: Path, prepared_run: Path, probe_api: FakeAgentModel, failure: str
) -> None:
    """출력 충돌, 잘못된 기록과 결과 불일치는 외부 요청 전에 거부한다."""
    output = tmp_path / "probe"
    source = prepared_run
    if failure == "existing_output":
        output.mkdir()
    elif failure == "bad_source":
        (source / "run.json").write_text("{}")
    elif failure == "invalid_first_retry":
        path = source / "events.jsonl"
        events = [json.loads(line) for line in path.read_text().splitlines()]
        prepared = next(event for event in events if event["kind"] == "tool_result")
        prepared["call"]["arguments"]["retry"] = True
        path.write_text("".join(json.dumps(event) + "\n" for event in events))
    else:
        path = source / "tool_results" / "01.json"
        saved = json.loads(path.read_text())
        saved["summary"]["part_count"] += 1
        path.write_text(json.dumps(saved))
    result = CliRunner().invoke(
        app, ["verify-agent-transition", "--run", str(source), "-o", str(output)]
    )
    assert result.exit_code == 2
    assert not probe_api.requests


def test_source_without_preparation_is_rejected(tmp_path: Path, prepared_run: Path) -> None:
    """완료되지 않은 첫 준비를 비교 입력으로 사용하지 않는다."""
    (prepared_run / "events.jsonl").write_text("")
    with pytest.raises(ValueError, match="preparation result"):
        load_probe_context(prepared_run)


def test_confirmation_preserves_model_selected_abort(
    tmp_path: Path, prepared_run: Path, probe_api: FakeAgentModel
) -> None:
    """수정 요청은 보정과 중단을 모두 허용한다. 특정 함수를 강제하지 않는다."""
    probe_api.variant, probe_api.variant_stage = "abort", 2
    output = tmp_path / "confirm"
    result = CliRunner().invoke(
        app,
        ["verify-agent-transition", "--run", str(prepared_run), "-o", str(output), "--confirm"],
    )
    assert result.exit_code == 0, result.output
    report = ProbeReport.model_validate_json((output / "probe_report.json").read_text())
    assert report.request_count == 2 and report.results[0].selected_tool == "calibrate_board"
    assert report.results[1].final is not None and report.results[1].final.decision == "abort"
    for request in probe_api.model_posts:
        body = json.loads(request.content)
        assert body["tool_choice"] == "auto" and body["tools"][0]["function"]["strict"] is True
