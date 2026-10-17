"""실제 SDK의 HTTP 경계에서 Nemotron 도구 계약과 호출 한도를 검사한다."""

import json
from pathlib import Path

import httpx2 as httpx
import pytest
from typer.testing import CliRunner

from pixel2place.cli import app
from pixel2place.config import DEFAULT_NEMOTRON_MODEL
from pixel2place.reasoner import nemotron
from pixel2place.reasoner.nemotron_tools import SYSTEM_PROMPT, NemotronToolsVerification

API_KEY = "test-tool-secret-not-for-output"
REASONING = "reasoning-must-not-be-stored"
runner = CliRunner()


class MockTokenFactory:
    """HTTP 요청을 보존하고 시나리오별 도구 응답을 반환한다."""

    def __init__(self) -> None:
        self.requests: list[httpx.Request] = []
        self.clients: list[httpx.Client] = []
        self.variant = "success"
        self.failure_stage = 1
        self.models = [DEFAULT_NEMOTRON_MODEL]

    def make_client(self, *, follow_redirects: bool) -> httpx.Client:
        """실제 SDK에 mock HTTP transport만 제공한다."""
        client = httpx.Client(
            transport=httpx.MockTransport(self.handle), follow_redirects=follow_redirects
        )
        self.clients.append(client)
        return client

    def handle(self, request: httpx.Request) -> httpx.Response:
        """모델 목록, 도구 호출 또는 도구 결과를 읽은 최종 응답을 반환한다."""
        self.requests.append(request)
        assert request.headers["authorization"] == f"Bearer {API_KEY}"
        stage = sum(item.method == "POST" for item in self.requests)
        if self.variant.startswith("http_") and stage == self.failure_stage:
            if self.variant == "http_timeout":
                raise httpx.ReadTimeout(API_KEY, request=request)
            if self.variant == "http_connection":
                raise httpx.ConnectError(API_KEY, request=request)
            return httpx.Response(
                int(self.variant.removeprefix("http_")), json={"error": {"message": API_KEY}}
            )
        if request.method == "GET":
            return httpx.Response(
                200,
                json={
                    "object": "list",
                    "data": [
                        {"id": model, "object": "model", "created": 0, "owned_by": "nvidia"}
                        for model in self.models
                    ],
                },
            )
        body = json.loads(request.content)
        model = body["model"]
        if stage == 1:
            request_id = body["messages"][1]["content"].split()[-1].removesuffix(".")
            args = {"request_id": request_id}
            if self.variant == "wrong_request":
                args["request_id"] = "0" * 32
            if self.variant == "extra_argument":
                args["command"] = "never execute this"
            call = {
                "id": "call-1",
                "type": "function",
                "function": {
                    "name": "wrong_tool" if self.variant == "wrong_tool" else "verify_contract",
                    "arguments": "{" if self.variant == "bad_arguments" else json.dumps(args),
                },
            }
            if self.variant == "empty_call_id":
                call["id"] = " "
            calls = [call, call] if self.variant == "multiple_tools" else [call]
            message = {
                "role": "assistant",
                "content": None,
                "tool_calls": calls,
                "reasoning_content": REASONING,
            }
            finish = "tool_calls"
            if self.variant == "no_tool":
                message = {"role": "assistant", "content": "Done."}
                finish = "stop"
        else:
            result = json.loads(body["messages"][-1]["content"])
            if self.variant == "wrong_receipt":
                result["receipt"] = "0" * 32
            if self.variant == "false_verified":
                result["verified"] = False
            if self.variant == "numeric_verified":
                result["verified"] = 1
            if self.variant == "extra_final":
                result["extra"] = "no"
            if self.variant == "missing_receipt":
                del result["receipt"]
            message = {
                "role": "assistant",
                "content": "Done." if self.variant == "plain_final" else json.dumps(result),
            }
            finish = "stop"
        if self.variant == "truncated" and stage == self.failure_stage:
            finish = "length"
        if self.variant == "different_model" and stage == self.failure_stage:
            model = "nvidia/Nemotron-other"
        payload = {
            "id": f"completion-{stage}",
            "object": "chat.completion",
            "created": 0,
            "model": model,
            "choices": [{"index": 0, "message": message, "finish_reason": finish}],
            "usage": {
                "prompt_tokens": 100,
                "completion_tokens": 50,
                "total_tokens": 150,
                "completion_tokens_details": {"reasoning_tokens": 20},
            },
        }
        if self.variant == "empty_choices":
            payload["choices"] = []
        if self.variant == "multiple_choices":
            payload["choices"] *= 2
        if self.variant == "no_usage":
            del payload["usage"]
        if self.variant == "invalid_response":
            del payload["id"]
        return httpx.Response(200, json=payload)


@pytest.fixture
def api(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> MockTokenFactory:
    """실제 .env와 키를 쓰지 않고 SDK의 HTTP 경계만 대체한다."""
    monkeypatch.chdir(tmp_path)
    for name in (
        "NEBIUS_API_KEY",
        "NEBIUS_BASE_URL",
        "NEMOTRON_MODEL",
        "NEMOTRON_TIMEOUT_S",
        "NEMOTRON_MAX_TOKENS",
    ):
        monkeypatch.delenv(name, raising=False)
    monkeypatch.setenv("NEBIUS_API_KEY", API_KEY)
    mock = MockTokenFactory()
    monkeypatch.setattr(nemotron, "DefaultHttpxClient", mock.make_client)
    return mock


@pytest.mark.unit
class TestNemotronTools:
    """도구 결과를 읽은 증거, 호출 수와 실패 시 무재시도를 검사한다."""

    def test_success_checks_receipt_and_safe_evidence(
        self, api: MockTokenFactory, tmp_path: Path
    ) -> None:
        """도구를 고르고 무작위 결과를 읽으면 2회 모델 요청 근거를 저장한다."""
        output = tmp_path / "evidence" / "verify.json"
        result = runner.invoke(app, ["verify-nemotron-tools", "-o", str(output)])
        assert result.exit_code == 0, result.output
        report = NemotronToolsVerification.model_validate_json(output.read_text())
        assert report.passed and report.error_code is None
        assert report.request_count == report.max_completions == 2
        assert report.final_response is not None
        assert report.tool_receipt is not None
        assert report.final_response.receipt == report.tool_receipt
        assert report.max_tokens == 2048 and report.timeout_s == 120 and report.max_retries == 0
        assert [request.method for request in api.requests] == ["GET", "POST", "POST"]
        first = json.loads(api.requests[1].content)
        second = json.loads(api.requests[2].content)
        assert first["tool_choice"] == "auto" and first["max_tokens"] == 2048
        assert first["messages"][0]["content"] == report.system_prompt == SYSTEM_PROMPT
        assert first["messages"][1]["content"] == report.user_prompt
        assert "tools" not in second and "tool_choice" not in second
        assert second["messages"][-1]["tool_call_id"] == report.tool_call_id
        assert report.tool_receipt not in json.dumps(first)
        assert REASONING not in json.dumps(second)
        assert API_KEY not in output.read_text() + result.output
        assert REASONING not in output.read_text()
        assert all(client.is_closed and not client.follow_redirects for client in api.clients)
        assert all(request.extensions["timeout"]["read"] == 120 for request in api.requests)
        assert [
            (item.prompt_tokens, item.completion_tokens, item.reasoning_tokens)
            for item in report.completions
        ] == [(100, 50, 20), (100, 50, 20)]

    @pytest.mark.parametrize(
        ("variant", "error"),
        [
            ("wrong_tool", "tool_call_invalid"),
            ("empty_call_id", "tool_call_invalid"),
            ("multiple_tools", "tool_call_invalid"),
            ("no_tool", "tool_call_invalid"),
            ("bad_arguments", "tool_arguments_invalid"),
            ("wrong_request", "tool_arguments_invalid"),
            ("extra_argument", "tool_arguments_invalid"),
            ("empty_choices", "invalid_response"),
            ("multiple_choices", "invalid_response"),
            ("invalid_response", "invalid_response"),
            ("different_model", "different_model"),
            ("truncated", "token_limit"),
        ],
    )
    def test_invalid_call_does_not_request_final(
        self, api: MockTokenFactory, tmp_path: Path, variant: str, error: str
    ) -> None:
        """첫 계약 위반에서는 도구 결과와 추가 모델 요청을 만들지 않는다."""
        api.variant = variant
        output = tmp_path / "failure.json"
        result = runner.invoke(app, ["verify-nemotron-tools", "-o", str(output)])
        report = NemotronToolsVerification.model_validate_json(output.read_text())
        assert result.exit_code == 1 and not report.passed
        assert report.error_code == error
        assert report.request_count == 1 and len(api.requests) == 2
        assert report.tool_receipt is None and report.final_response is None
        assert all(client.is_closed for client in api.clients)

    @pytest.mark.parametrize(
        "variant",
        [
            "wrong_receipt",
            "false_verified",
            "numeric_verified",
            "extra_final",
            "missing_receipt",
            "plain_final",
        ],
    )
    def test_invalid_final_is_not_success(
        self, api: MockTokenFactory, tmp_path: Path, variant: str
    ) -> None:
        """결과를 잘못 읽거나 형식을 어기면 성공으로 저장하지 않는다."""
        api.variant = variant
        output = tmp_path / "failure.json"
        result = runner.invoke(app, ["verify-nemotron-tools", "-o", str(output)])
        report = NemotronToolsVerification.model_validate_json(output.read_text())
        assert result.exit_code == 1
        assert not report.passed and report.error_code == "final_response_invalid"
        assert report.request_count == 2 and len(api.requests) == 3
        assert report.final_response is None

    @pytest.mark.parametrize(
        ("variant", "error"), [("truncated", "token_limit"), ("different_model", "different_model")]
    )
    def test_final_response_limits(
        self, api: MockTokenFactory, tmp_path: Path, variant: str, error: str
    ) -> None:
        """최종 응답의 모델 변경과 토큰 한도 종료도 실패로 처리한다."""
        api.variant, api.failure_stage = variant, 2
        output = tmp_path / "failure.json"
        result = runner.invoke(app, ["verify-nemotron-tools", "-o", str(output)])
        report = NemotronToolsVerification.model_validate_json(output.read_text())
        assert result.exit_code == 1 and report.error_code == error
        assert report.request_count == 2 and len(api.requests) == 3

    @pytest.mark.parametrize("stage", [0, 1, 2])
    @pytest.mark.parametrize(
        ("variant", "error"),
        [
            ("http_timeout", "http_timeout"),
            ("http_connection", "http_connection"),
            ("http_401", "http_status"),
            ("http_429", "http_status"),
            ("http_503", "http_status"),
        ],
    )
    def test_http_failure_is_safe_and_not_retried(
        self, api: MockTokenFactory, tmp_path: Path, stage: int, variant: str, error: str
    ) -> None:
        """목록 또는 모델 요청의 HTTP 실패는 재시도하지 않고 본문과 키를 저장하지 않는다."""
        api.variant, api.failure_stage = variant, stage
        output = tmp_path / "failure.json"
        result = runner.invoke(app, ["verify-nemotron-tools", "-o", str(output)])
        report = NemotronToolsVerification.model_validate_json(output.read_text())
        assert result.exit_code == 1 and report.error_code == error
        assert report.request_count == stage and len(api.requests) == stage + 1
        assert API_KEY not in output.read_text() + result.output
        assert all(client.is_closed for client in api.clients)

    def test_no_usage_stays_unknown(self, api: MockTokenFactory, tmp_path: Path) -> None:
        """사용량이 없으면 토큰 수를 추정하지 않는다."""
        api.variant = "no_usage"
        output = tmp_path / "verify.json"
        result = runner.invoke(app, ["verify-nemotron-tools", "-o", str(output)])
        report = NemotronToolsVerification.model_validate_json(output.read_text())
        assert result.exit_code == 0
        assert all(
            item.prompt_tokens is None
            and item.completion_tokens is None
            and item.reasoning_tokens is None
            for item in report.completions
        )

    def test_model_override_is_per_command(self, api: MockTokenFactory, tmp_path: Path) -> None:
        """지정한 모델을 2회 쓰고 .env 설정과 기존 검증 한도를 보존한다."""
        model = "nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B"
        api.models.append(model)
        result = runner.invoke(
            app, ["verify-nemotron-tools", "--model", model, "--max-tokens", "512"]
        )
        assert result.exit_code == 0, result.output
        assert all(
            json.loads(request.content)["model"] == model
            and json.loads(request.content)["max_tokens"] == 512
            for request in api.requests
            if request.method == "POST"
        )
        assert not (tmp_path / ".env").exists()

    def test_existing_output_makes_no_request(self, api: MockTokenFactory, tmp_path: Path) -> None:
        """기존 근거 파일이 있으면 API와 덮어쓰기 없이 종료한다."""
        output = tmp_path / "verify.json"
        output.write_text("preserve")
        result = runner.invoke(app, ["verify-nemotron-tools", "-o", str(output)])
        assert result.exit_code == 2 and output.read_text() == "preserve"
        assert not api.requests and not api.clients

    @pytest.mark.parametrize(
        "args",
        [
            ["--model", " "],
            ["--model", "other/model"],
            ["--max-tokens", "0"],
            ["--max-tokens", "4097"],
        ],
    )
    def test_invalid_settings_make_no_request(self, api: MockTokenFactory, args: list[str]) -> None:
        """잘못된 모델과 호출 한도는 외부 요청 전에 거부한다."""
        result = runner.invoke(app, ["verify-nemotron-tools", *args])
        assert result.exit_code == 2 and not api.requests and not api.clients

    def test_unavailable_model_stops_before_inference(self, api: MockTokenFactory) -> None:
        """계정에 없는 Nemotron은 다른 모델로 바꾸거나 모델 요청하지 않는다."""
        result = runner.invoke(app, ["verify-nemotron-tools", "--model", "nvidia/Nemotron-missing"])
        assert result.exit_code == 1 and len(api.requests) == 1
        assert "catalog_error" in result.output
