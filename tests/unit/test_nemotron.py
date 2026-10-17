"""Token Factory 설정과 접속 검증 CLI를 HTTP 경계에서 검사한다."""

import json
from collections.abc import Callable
from pathlib import Path

import httpx2 as httpx
import pytest
from pydantic import ValidationError
from typer.testing import CliRunner

from pixel2place.cli import EXIT_INFEASIBLE, EXIT_USAGE, app
from pixel2place.config import DEFAULT_NEMOTRON_MODEL, TOKEN_FACTORY_BASE_URL, TokenFactorySettings
from pixel2place.reasoner import nemotron
from pixel2place.reasoner.nemotron import NemotronVerification, verify_nemotron

FAKE_API_KEY = "test-nebius-secret-not-for-output"
ENV_NAMES = (
    "NEBIUS_API_KEY",
    "NEBIUS_BASE_URL",
    "NEMOTRON_MODEL",
    "NEMOTRON_TIMEOUT_S",
    "NEMOTRON_MAX_TOKENS",
)
runner = CliRunner()


@pytest.fixture(autouse=True)
def isolated_env(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    """실제 키와 프로젝트 .env 대신 테스트 환경만 쓴다."""
    monkeypatch.chdir(tmp_path)
    for name in ENV_NAMES:
        monkeypatch.delenv(name, raising=False)
    monkeypatch.setenv("NEBIUS_API_KEY", FAKE_API_KEY)


@pytest.fixture
def settings() -> TokenFactorySettings:
    """기본 설정과 테스트 키를 쓴다."""
    return TokenFactorySettings()  # pyright: ignore[reportCallIssue]


def _mock_http(
    monkeypatch: pytest.MonkeyPatch, handler: Callable[[httpx.Request], httpx.Response]
) -> list[httpx.Client]:
    """SDK의 HTTP 경계만 대체하고 클라이언트 종료를 관찰한다."""
    clients: list[httpx.Client] = []

    def factory(*, follow_redirects: bool) -> httpx.Client:
        client = httpx.Client(
            transport=httpx.MockTransport(handler), follow_redirects=follow_redirects
        )
        clients.append(client)
        return client

    monkeypatch.setattr(nemotron, "DefaultHttpxClient", factory)
    return clients


def _models_response(*models: str) -> httpx.Response:
    """계정 모델 목록 응답을 만든다."""
    return httpx.Response(
        200,
        json={
            "object": "list",
            "data": [
                {"id": model, "object": "model", "created": 0, "owned_by": "nvidia"}
                for model in models
            ],
        },
    )


def _completion_response(
    *,
    content: str | None = "READY",
    finish_reason: str = "stop",
    model: str = DEFAULT_NEMOTRON_MODEL,
    empty_choices: bool = False,
    usage: bool = True,
    reasoning: bool = True,
) -> httpx.Response:
    """사용량과 종료 상태를 지정한 SDK 응답을 만든다."""
    payload: dict[str, object] = {
        "id": "chatcmpl-test",
        "object": "chat.completion",
        "created": 0,
        "model": model,
        "choices": []
        if empty_choices
        else [
            {
                "index": 0,
                "message": {"role": "assistant", "content": content},
                "finish_reason": finish_reason,
            }
        ],
    }
    if usage:
        token_usage: dict[str, object] = {
            "prompt_tokens": 22,
            "completion_tokens": 180,
            "total_tokens": 202,
        }
        if reasoning:
            token_usage["completion_tokens_details"] = {"reasoning_tokens": 179}
        payload["usage"] = token_usage
    return httpx.Response(200, json=payload)


@pytest.mark.unit
class TestTokenFactorySettings:
    """인증, 모델과 API 주소 설정의 안전 경계."""

    def test_defaults_keep_key_secret(self, settings: TokenFactorySettings) -> None:
        """기본 모델과 공식 주소를 쓰고 키를 문자열 표현에서 숨긴다."""
        assert settings.nemotron_model == DEFAULT_NEMOTRON_MODEL
        assert str(settings.nebius_base_url) == TOKEN_FACTORY_BASE_URL
        assert settings.nemotron_timeout_s == 120
        assert settings.nemotron_max_tokens == 1024
        assert FAKE_API_KEY not in str(settings)

    @pytest.mark.parametrize(
        ("name", "value"),
        [
            ("NEBIUS_API_KEY", ""),
            ("NEBIUS_API_KEY", " "),
            ("NEBIUS_API_KEY", f" {FAKE_API_KEY}"),
            ("NEBIUS_BASE_URL", "http://api.tokenfactory.nebius.com/v1/"),
            ("NEBIUS_BASE_URL", "https://other.example/v1/"),
            ("NEBIUS_BASE_URL", "https://api.tokenfactory.nebius.com:8443/v1/"),
            ("NEBIUS_BASE_URL", f"https://{FAKE_API_KEY}@api.tokenfactory.nebius.com/v1/"),
            ("NEBIUS_BASE_URL", f"https://api.tokenfactory.nebius.com/v1/?key={FAKE_API_KEY}"),
            ("NEBIUS_BASE_URL", "https://api.tokenfactory.nebius.com/v1/#fragment"),
            ("NEBIUS_BASE_URL", "https://api.tokenfactory.nebius.com/v2/"),
            ("NEMOTRON_MODEL", " "),
            ("NEMOTRON_MODEL", "other/Nemotron"),
            ("NEMOTRON_MODEL", "nvidia/Cosmos3-Nano"),
            ("NEMOTRON_TIMEOUT_S", "0"),
            ("NEMOTRON_TIMEOUT_S", "inf"),
            ("NEMOTRON_MAX_TOKENS", "0"),
            ("NEMOTRON_MAX_TOKENS", "4097"),
        ],
    )
    def test_invalid_settings_make_no_request(
        self, monkeypatch: pytest.MonkeyPatch, name: str, value: str
    ) -> None:
        """잘못된 설정은 API 호출 전에 실패하고 입력값을 출력하지 않는다."""
        monkeypatch.setenv(name, value)
        requests: list[httpx.Request] = []

        def handler(request: httpx.Request) -> httpx.Response:
            requests.append(request)
            return _models_response(DEFAULT_NEMOTRON_MODEL)

        clients = _mock_http(monkeypatch, handler)
        result = runner.invoke(app, ["verify-nemotron"])
        assert result.exit_code == EXIT_USAGE, result.output
        assert name in result.output
        assert FAKE_API_KEY not in result.output
        assert not requests
        assert not clients

    def test_missing_key(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """키가 없으면 필수 변수 이름을 안내한다."""
        monkeypatch.delenv("NEBIUS_API_KEY")
        result = runner.invoke(app, ["verify-nemotron"])
        assert result.exit_code == EXIT_USAGE
        assert "NEBIUS_API_KEY" in result.output

    def test_validation_error_hides_input(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """설정 예외의 문자열도 잘못된 주소에 포함된 비밀값을 숨긴다."""
        monkeypatch.setenv("NEBIUS_BASE_URL", f"https://example.com/?key={FAKE_API_KEY}")
        with pytest.raises(ValidationError) as caught:
            TokenFactorySettings()  # pyright: ignore[reportCallIssue]
        assert FAKE_API_KEY not in str(caught.value)


@pytest.mark.unit
class TestNemotronVerification:
    """실제 SDK를 통해 모델 목록, 모델 요청과 오류 처리를 검사한다."""

    def test_success_saves_safe_evidence(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        """목록 조회 뒤 모델 요청을 1회 호출하고 인증값 없이 측정 근거를 쓴다."""
        requests: list[httpx.Request] = []

        def handler(request: httpx.Request) -> httpx.Response:
            requests.append(request)
            assert request.headers["authorization"] == f"Bearer {FAKE_API_KEY}"
            assert request.extensions["timeout"]["read"] == 120
            if request.method == "GET":
                assert request.url.path == "/v1/models"
                return _models_response(DEFAULT_NEMOTRON_MODEL, "nvidia/Cosmos3-Nano")
            assert request.url.path == "/v1/chat/completions"
            body = json.loads(request.content)
            assert body == {
                "model": DEFAULT_NEMOTRON_MODEL,
                "messages": [{"role": "user", "content": nemotron.VERIFICATION_PROMPT}],
                "max_tokens": 1024,
            }
            return _completion_response()

        clients = _mock_http(monkeypatch, handler)
        output = tmp_path / "nested" / "verify.json"
        result = runner.invoke(app, ["verify-nemotron", "-o", str(output)])
        assert result.exit_code == 0, result.output
        assert "Verified" in result.output
        assert [request.method for request in requests] == ["GET", "POST"]
        report = NemotronVerification.model_validate_json(output.read_text())
        assert report.available_models == (DEFAULT_NEMOTRON_MODEL,)
        assert report.requested_model == report.served_model == DEFAULT_NEMOTRON_MODEL
        assert report.response_text == "READY"
        assert report.finish_reason == "stop"
        assert report.max_retries == 0
        assert report.completion_tokens == 180
        assert report.prompt_tokens == 22
        assert report.reasoning_tokens == 179
        assert report.verified_at.utcoffset() is not None
        assert report.latency_s >= 0
        assert FAKE_API_KEY not in output.read_text()
        assert FAKE_API_KEY not in result.output
        assert all(client.is_closed for client in clients)
        assert all(not client.follow_redirects for client in clients)

    @pytest.mark.parametrize(("usage", "reasoning"), [(False, False), (True, False)])
    def test_optional_usage(
        self,
        monkeypatch: pytest.MonkeyPatch,
        settings: TokenFactorySettings,
        usage: bool,
        reasoning: bool,
    ) -> None:
        """선택 사용량 필드가 없으면 누락 값을 추정하지 않는다."""

        def handler(request: httpx.Request) -> httpx.Response:
            if request.method == "GET":
                return _models_response(DEFAULT_NEMOTRON_MODEL)
            return _completion_response(usage=usage, reasoning=reasoning)

        clients = _mock_http(monkeypatch, handler)
        report = verify_nemotron(settings)
        assert report.reasoning_tokens is None
        assert report.prompt_tokens == (22 if usage else None)
        assert all(client.is_closed for client in clients)

    def test_model_not_available_stops_before_inference(
        self, monkeypatch: pytest.MonkeyPatch, settings: TokenFactorySettings
    ) -> None:
        """설정한 모델이 없으면 유료 모델 요청이나 다른 모델을 호출하지 않는다."""
        requests: list[httpx.Request] = []

        def handler(request: httpx.Request) -> httpx.Response:
            requests.append(request)
            return _models_response("nvidia/nemotron-3-super-120b-a12b")

        clients = _mock_http(monkeypatch, handler)
        with pytest.raises(ValueError, match="not available"):
            verify_nemotron(settings)
        assert len(requests) == 1
        assert requests[0].method == "GET"
        assert all(client.is_closed for client in clients)

    @pytest.mark.parametrize("status", [302, 401, 403, 429, 500])
    @pytest.mark.parametrize("stage", ["models", "completion"])
    def test_http_errors_hide_body_and_do_not_retry(
        self,
        monkeypatch: pytest.MonkeyPatch,
        tmp_path: Path,
        status: int,
        stage: str,
    ) -> None:
        """조회와 모델 요청의 HTTP 실패는 본문을 숨기고 재시도나 리디렉션을 하지 않는다."""
        requests: list[httpx.Request] = []

        def handler(request: httpx.Request) -> httpx.Response:
            requests.append(request)
            if request.method == "GET" and stage == "completion":
                return _models_response(DEFAULT_NEMOTRON_MODEL)
            return httpx.Response(
                status,
                json={"error": {"message": FAKE_API_KEY}},
                headers={"Location": "https://other.example/"},
            )

        clients = _mock_http(monkeypatch, handler)
        output = tmp_path / "failed.json"
        result = runner.invoke(app, ["verify-nemotron", "-o", str(output)])
        assert result.exit_code == EXIT_INFEASIBLE, result.output
        assert f"HTTP {status}" in result.output
        assert FAKE_API_KEY not in result.output
        assert len(requests) == (1 if stage == "models" else 2)
        assert not output.exists()
        assert all(client.is_closed for client in clients)

    @pytest.mark.parametrize("error", [httpx.ReadTimeout, httpx.ConnectError])
    def test_transport_errors(
        self, monkeypatch: pytest.MonkeyPatch, error: type[httpx.TransportError]
    ) -> None:
        """timeout과 연결 오류는 인증값을 숨기고 클라이언트를 닫는다."""
        requests: list[httpx.Request] = []

        def handler(request: httpx.Request) -> httpx.Response:
            requests.append(request)
            raise error(FAKE_API_KEY, request=request)

        clients = _mock_http(monkeypatch, handler)
        result = runner.invoke(app, ["verify-nemotron"])
        assert result.exit_code == EXIT_INFEASIBLE, result.output
        assert "No retry" in result.output
        assert FAKE_API_KEY not in result.output
        assert len(requests) == 1
        assert all(client.is_closed for client in clients)

    @pytest.mark.parametrize(
        ("content", "finish_reason", "model", "empty_choices"),
        [
            (None, "stop", DEFAULT_NEMOTRON_MODEL, False),
            (" ", "stop", DEFAULT_NEMOTRON_MODEL, False),
            ("READY", "length", DEFAULT_NEMOTRON_MODEL, False),
            ("READY", "stop", "nvidia/other-Nemotron", False),
            ("READY", "stop", DEFAULT_NEMOTRON_MODEL, True),
        ],
    )
    def test_incomplete_response_is_not_success(
        self,
        monkeypatch: pytest.MonkeyPatch,
        tmp_path: Path,
        content: str | None,
        finish_reason: str,
        model: str,
        empty_choices: bool,
    ) -> None:
        """빈 응답, 토큰 한도와 모델 불일치를 성공 근거로 저장하지 않는다."""

        def handler(request: httpx.Request) -> httpx.Response:
            if request.method == "GET":
                return _models_response(DEFAULT_NEMOTRON_MODEL)
            return _completion_response(
                content=content,
                finish_reason=finish_reason,
                model=model,
                empty_choices=empty_choices,
            )

        clients = _mock_http(monkeypatch, handler)
        output = tmp_path / "failed.json"
        result = runner.invoke(app, ["verify-nemotron", "-o", str(output)])
        assert result.exit_code == EXIT_INFEASIBLE, result.output
        assert "verification failed" in result.output
        assert not output.exists()
        assert all(client.is_closed for client in clients)

    def test_success_without_output(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """출력 경로가 없으면 접속 결과만 표시한다."""

        def handler(request: httpx.Request) -> httpx.Response:
            if request.method == "GET":
                return _models_response(DEFAULT_NEMOTRON_MODEL)
            return _completion_response()

        _mock_http(monkeypatch, handler)
        result = runner.invoke(app, ["verify-nemotron"])
        assert result.exit_code == 0, result.output
        assert "finish=stop" in result.output

    @pytest.mark.parametrize("stage", ["models", "completion"])
    @pytest.mark.parametrize("payload", [{"invalid": FAKE_API_KEY}, FAKE_API_KEY])
    def test_invalid_schema_hides_input(
        self,
        monkeypatch: pytest.MonkeyPatch,
        stage: str,
        payload: dict[str, str] | str,
    ) -> None:
        """깨진 외부 응답은 성공으로 처리하지 않고 원문을 숨긴다."""

        def handler(request: httpx.Request) -> httpx.Response:
            if request.method == "GET" and stage == "completion":
                return _models_response(DEFAULT_NEMOTRON_MODEL)
            return httpx.Response(200, json=payload)

        clients = _mock_http(monkeypatch, handler)
        result = runner.invoke(app, ["verify-nemotron"])
        assert result.exit_code == EXIT_INFEASIBLE, result.output
        assert FAKE_API_KEY not in result.output
        assert all(client.is_closed for client in clients)

    def test_environment_overrides(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """환경 변수의 모델, timeout과 토큰 한도를 실제 SDK 요청에 적용한다."""
        model = "nvidia/nemotron-3-super-120b-a12b"
        monkeypatch.setenv("NEMOTRON_MODEL", model)
        monkeypatch.setenv("NEMOTRON_TIMEOUT_S", "30")
        monkeypatch.setenv("NEMOTRON_MAX_TOKENS", "512")

        def handler(request: httpx.Request) -> httpx.Response:
            assert request.extensions["timeout"]["read"] == 30
            if request.method == "GET":
                return _models_response(model)
            body = json.loads(request.content)
            assert body["model"] == model
            assert body["max_tokens"] == 512
            return _completion_response(model=model)

        _mock_http(monkeypatch, handler)
        result = runner.invoke(app, ["verify-nemotron"])
        assert result.exit_code == 0, result.output
        assert model in result.output

    def test_existing_evidence_stops_before_request(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        """기존 측정 파일이 있으면 덮어쓰거나 유료 요청을 하지 않는다."""
        output = tmp_path / "verify.json"
        output.write_text("existing evidence")
        requests: list[httpx.Request] = []

        def handler(request: httpx.Request) -> httpx.Response:
            requests.append(request)
            return _models_response(DEFAULT_NEMOTRON_MODEL)

        _mock_http(monkeypatch, handler)
        result = runner.invoke(app, ["verify-nemotron", "-o", str(output)])
        assert result.exit_code == EXIT_USAGE
        assert "already exists" in result.output
        assert output.read_text() == "existing evidence"
        assert not requests

    def test_evidence_created_during_request_is_preserved(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        """검증 중 다른 실행이 쓴 근거도 exclusive 쓰기로 보존한다."""
        output = tmp_path / "verify.json"

        def handler(request: httpx.Request) -> httpx.Response:
            if request.method == "GET":
                return _models_response(DEFAULT_NEMOTRON_MODEL)
            output.write_text("concurrent evidence")
            return _completion_response()

        clients = _mock_http(monkeypatch, handler)
        result = runner.invoke(app, ["verify-nemotron", "-o", str(output)])
        assert result.exit_code == EXIT_USAGE
        assert output.read_text() == "concurrent evidence"
        assert all(client.is_closed for client in clients)
