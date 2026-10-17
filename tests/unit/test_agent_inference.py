"""모델 요청 1회의 전송, 재전송, 관찰 기록과 HTTP 오류 경계를 HTTP Fake로 검사한다."""

import json
from collections.abc import Callable, Iterator
from pathlib import Path
from typing import Any

import httpx2 as httpx
import pytest
from openai import APIConnectionError, APIStatusError, APITimeoutError, OpenAI, OpenAIError
from openai.types.chat import ChatCompletion, ChatCompletionMessageParam

from pixel2place.agent.audit import AUDIT_FILE, ModelAudit
from pixel2place.agent.contracts import ToolName
from pixel2place.agent.inference import (
    NO_RETRY,
    REQUEST_RETRY,
    RETRY_DELAYS_S,
    ObservedResponse,
    RetryPolicy,
    completion_evidence,
    request_completion,
    with_retry,
)
from pixel2place.agent.runtime import tool_schema
from pixel2place.agent.runtime_contracts import MAX_REQUEST_RETRIES
from pixel2place.config import DEFAULT_NEMOTRON_MODEL
from tests.fakes.fake_agent_model import FAKE_AGENT_KEY, FakeAgentModel

pytestmark = pytest.mark.unit
LAYOUT_ID = "a" * 32


@pytest.fixture
def api() -> Iterator[FakeAgentModel]:
    """테스트마다 HTTP Fake를 만든다. 종료 때 클라이언트를 닫는다."""
    fake = FakeAgentModel()
    yield fake
    for client in fake.clients:
        client.close()


def _client(api: FakeAgentModel) -> OpenAI:
    return OpenAI(
        api_key=FAKE_AGENT_KEY,
        base_url="https://api.tokenfactory.nebius.com/v1/",
        max_retries=0,
        http_client=api.make_client(),
    )


def _messages(state: str) -> list[ChatCompletionMessageParam]:
    context = json.dumps(
        {"state": state, "error_code": None, "layout_id": LAYOUT_ID, "calibration_id": LAYOUT_ID}
    )
    return [
        {"role": "system", "content": f"Coordinate.\nCurrent runtime state: {context}"},
        {"role": "user", "content": "Place the parts."},
    ]


def _records(root: Path) -> list[dict[str, object]]:
    return [json.loads(line) for line in (root / AUDIT_FILE).read_text().splitlines()]


class TestRequestCompletion:
    """요청 본문, 관찰 기록의 순서와 오류 전달을 검사한다."""

    def test_tool_request_records_wire_and_sdk_responses(
        self, tmp_path: Path, api: FakeAgentModel
    ) -> None:
        """도구가 있으면 `tools`와 `tool_choice`를 보낸다. 두 응답 관찰이 같다."""
        audit = ModelAudit(root=tmp_path, run_id="b" * 32, secret=FAKE_AGENT_KEY)
        observed = request_completion(
            client=_client(api),
            audit=audit,
            inference=3,
            model=DEFAULT_NEMOTRON_MODEL,
            messages=_messages("layout_ready"),
            tools=[tool_schema(ToolName.CALIBRATE_BOARD)],
            max_tokens=2048,
            timeout_s=120.0,
        )
        body = json.loads(api.model_posts[0].content)
        assert body["tool_choice"] == "auto" and len(body["tools"]) == 1
        assert api.model_posts[0].extensions["timeout"]["read"] == 120
        assert observed.wire_matches_sdk and observed.latency_s >= 0
        assert isinstance(observed.response, ChatCompletion)
        evidence = completion_evidence(observed.response, latency_s=observed.latency_s)
        assert evidence.finish_reason == "tool_calls" and evidence.reasoning_tokens == 20
        records = _records(tmp_path)
        assert [(r["kind"], r["inference"]) for r in records] == [
            ("http_response", 3),
            ("sdk_response", 3),
        ]

    def test_request_without_tools_omits_tool_fields(
        self, tmp_path: Path, api: FakeAgentModel
    ) -> None:
        """도구가 없으면 요청에 `tools`와 `tool_choice`가 없다."""
        audit = ModelAudit(root=tmp_path, run_id="b" * 32, secret=FAKE_AGENT_KEY)
        observed = request_completion(
            client=_client(api),
            audit=audit,
            inference=1,
            model=DEFAULT_NEMOTRON_MODEL,
            messages=_messages("checked"),
            tools=[],
            max_tokens=2048,
            timeout_s=120.0,
        )
        body = json.loads(api.model_posts[0].content)
        assert "tools" not in body and "tool_choice" not in body
        assert isinstance(observed.response, ChatCompletion)
        assert observed.response.choices[0].finish_reason == "stop"

    def test_http_status_error_records_only_the_status(
        self, tmp_path: Path, api: FakeAgentModel
    ) -> None:
        """HTTP 오류는 상태 코드만 기록하고 전달한다. 오류 본문은 기록하지 않는다."""
        api.variant, api.variant_stage = "http_503", 1
        audit = ModelAudit(root=tmp_path, run_id="b" * 32, secret=FAKE_AGENT_KEY)
        with pytest.raises(APIStatusError):
            request_completion(
                client=_client(api),
                audit=audit,
                inference=1,
                model=DEFAULT_NEMOTRON_MODEL,
                messages=_messages("checked"),
                tools=[],
                max_tokens=2048,
                timeout_s=120.0,
            )
        records = _records(tmp_path)
        assert [r["kind"] for r in records] == ["http_response"]
        assert records[0]["data"] == {"status": 503}
        assert FAKE_AGENT_KEY not in (tmp_path / AUDIT_FILE).read_text()


def _send(api: FakeAgentModel, audit: ModelAudit) -> Callable[[], ObservedResponse]:
    """도구 없는 모델 요청 1회. 시도마다 같은 요청을 보낸다."""
    client = _client(api)
    return lambda: request_completion(
        client=client,
        audit=audit,
        inference=1,
        model=DEFAULT_NEMOTRON_MODEL,
        messages=_messages("checked"),
        tools=[],
        max_tokens=2048,
        timeout_s=120.0,
    )


def _attempts(root: Path) -> list[dict[str, Any]]:
    records = [json.loads(line) for line in (root / AUDIT_FILE).read_text().splitlines()]
    return [record["data"] for record in records if record["kind"] == "attempt"]


class TestWithRetry:
    """일시 오류의 재전송, 대기와 시도 기록을 검사한다."""

    @pytest.mark.parametrize(
        ("variant", "outcome", "status"),
        [
            ("http_503", "http_error", 503),
            ("http_500", "http_error", 500),
            ("http_429", "http_error", 429),
            ("http_connection", "connection_error", None),
        ],
    )
    def test_transient_error_is_sent_again_after_the_first_delay(
        self,
        tmp_path: Path,
        api: FakeAgentModel,
        retry_waits: list[float],
        variant: str,
        outcome: str,
        status: int | None,
    ) -> None:
        """연결 오류, 429와 5xx는 3초 뒤에 다시 보낸다. 시도마다 기록 1줄을 쓴다."""
        api.variant, api.http_failures = variant, 1
        audit = ModelAudit(root=tmp_path, run_id="b" * 32, secret=FAKE_AGENT_KEY)
        observed = with_retry(
            _send(api, audit), policy=REQUEST_RETRY, audit=audit, inference=1, target="completion"
        )
        assert isinstance(observed.response, ChatCompletion)
        assert len(api.model_posts) == 2 and retry_waits == [3.0], retry_waits
        assert _attempts(tmp_path) == [
            {
                "target": "completion",
                "attempt": 1,
                "outcome": outcome,
                "status": status,
                "wait_s": 3.0,
            },
            {"target": "completion", "attempt": 2, "outcome": "ok", "status": None, "wait_s": None},
        ]
        assert FAKE_AGENT_KEY not in (tmp_path / AUDIT_FILE).read_text()

    def test_timeout_is_not_sent_again(
        self, tmp_path: Path, api: FakeAgentModel, retry_waits: list[float]
    ) -> None:
        """timeout은 다시 보내지 않는다. 요청은 1개이고 대기는 없다."""
        api.variant, api.http_failures = "http_timeout", 1
        audit = ModelAudit(root=tmp_path, run_id="b" * 32, secret=FAKE_AGENT_KEY)
        with pytest.raises(APITimeoutError):
            with_retry(
                _send(api, audit),
                policy=REQUEST_RETRY,
                audit=audit,
                inference=1,
                target="completion",
            )
        assert len(api.model_posts) == 1 and retry_waits == []
        assert _attempts(tmp_path) == [
            {
                "target": "completion",
                "attempt": 1,
                "outcome": "timeout",
                "status": None,
                "wait_s": None,
            }
        ]

    def test_four_failures_raise_the_last_error(
        self, tmp_path: Path, api: FakeAgentModel, retry_waits: list[float]
    ) -> None:
        """4회 연속으로 실패하면 3초, 9초와 20초를 기다린 뒤에 마지막 오류를 전달한다."""
        api.variant, api.http_failures = "http_503", 4
        audit = ModelAudit(root=tmp_path, run_id="b" * 32, secret=FAKE_AGENT_KEY)
        with pytest.raises(APIStatusError):
            with_retry(
                _send(api, audit),
                policy=REQUEST_RETRY,
                audit=audit,
                inference=1,
                target="completion",
            )
        assert len(api.model_posts) == 4, f"요청 4회여야 한다: {len(api.model_posts)}"
        assert retry_waits == [3.0, 9.0, 20.0] == list(RETRY_DELAYS_S)
        attempts = _attempts(tmp_path)
        assert [a["attempt"] for a in attempts] == [1, 2, 3, 4]
        assert [a["wait_s"] for a in attempts] == [3.0, 9.0, 20.0, None]

    @pytest.mark.parametrize("variant", ["http_400", "http_401", "http_404", "http_422"])
    def test_other_http_errors_are_not_sent_again(
        self, tmp_path: Path, api: FakeAgentModel, retry_waits: list[float], variant: str
    ) -> None:
        """429가 아닌 4xx는 다시 보내지 않는다. 기록은 상태 코드와 대기 없음을 쓴다."""
        api.variant, api.http_failures = variant, 4
        audit = ModelAudit(root=tmp_path, run_id="b" * 32, secret=FAKE_AGENT_KEY)
        with pytest.raises(APIStatusError):
            with_retry(
                _send(api, audit),
                policy=REQUEST_RETRY,
                audit=audit,
                inference=1,
                target="completion",
            )
        assert len(api.model_posts) == 1 and not retry_waits
        assert _attempts(tmp_path) == [
            {
                "target": "completion",
                "attempt": 1,
                "outcome": "http_error",
                "status": int(variant.removeprefix("http_")),
                "wait_s": None,
            }
        ]

    def test_default_policy_sends_once_and_writes_no_attempt_record(
        self, tmp_path: Path, api: FakeAgentModel, retry_waits: list[float]
    ) -> None:
        """재전송을 끈 방침은 요청 1회와 기존 관찰 기록만 남긴다."""
        api.variant, api.http_failures = "http_503", 4
        audit = ModelAudit(root=tmp_path, run_id="b" * 32, secret=FAKE_AGENT_KEY)
        with pytest.raises(APIStatusError):
            with_retry(
                _send(api, audit), policy=NO_RETRY, audit=audit, inference=1, target="completion"
            )
        assert len(api.model_posts) == 1 and not retry_waits
        assert [record["kind"] for record in _records(tmp_path)] == ["http_response"]

    def test_refused_reservation_stops_the_retry(
        self, tmp_path: Path, api: FakeAgentModel, retry_waits: list[float]
    ) -> None:
        """호출자가 다음 시도를 받지 않으면 기다리지 않고 오류를 전달한다."""
        api.variant, api.http_failures = "http_503", 4
        audit = ModelAudit(root=tmp_path, run_id="b" * 32, secret=FAKE_AGENT_KEY)
        reservations = iter([True, False])
        with pytest.raises(APIStatusError):
            with_retry(
                _send(api, audit),
                policy=REQUEST_RETRY,
                audit=audit,
                inference=1,
                target="completion",
                reserve_retry=lambda: next(reservations),
            )
        assert len(api.model_posts) == 2 and retry_waits == [3.0]
        assert [a["wait_s"] for a in _attempts(tmp_path)] == [3.0, None]

    @pytest.mark.parametrize(
        ("error", "outcome"),
        [
            (ValueError("The catalog is not valid."), "invalid_response"),
            (OpenAIError("The response is not valid."), "sdk_error"),
        ],
    )
    def test_content_and_parse_errors_are_not_sent_again(
        self, tmp_path: Path, retry_waits: list[float], error: Exception, outcome: str
    ) -> None:
        """응답 내용의 거부와 SDK의 다른 오류는 1회 기록하고 그대로 전달한다."""
        audit = ModelAudit(root=tmp_path, run_id="b" * 32, secret=FAKE_AGENT_KEY)
        calls: list[int] = []

        def send() -> None:
            calls.append(1)
            raise error

        with pytest.raises(type(error)):
            with_retry(send, policy=REQUEST_RETRY, audit=audit, inference=0, target="catalog")
        assert len(calls) == 1 and not retry_waits
        assert _attempts(tmp_path) == [
            {"target": "catalog", "attempt": 1, "outcome": outcome, "status": None, "wait_s": None}
        ]

    def test_policy_limit_matches_the_delay_schedule(self) -> None:
        """재전송 횟수의 상한은 대기 표의 길이와 같다. 범위 밖의 횟수는 거부한다."""
        assert len(RETRY_DELAYS_S) == MAX_REQUEST_RETRIES == REQUEST_RETRY.retries
        assert NO_RETRY.retries == 0
        for retries in (-1, MAX_REQUEST_RETRIES + 1):
            with pytest.raises(ValueError, match="retry count"):
                RetryPolicy(retries=retries)

    def test_wait_follows_the_attempt_and_the_error_kind(self) -> None:
        """대기는 시도 번호와 오류의 종류가 정한다. 마지막 시도 뒤에는 대기가 없다.

        timeout은 연결 오류의 하위 클래스다. 그래도 대기가 없다.
        """
        policy = RetryPolicy(retries=2)
        request = httpx.Request("GET", "https://api.tokenfactory.nebius.com/v1/models")
        refused = APIConnectionError(request=request)
        assert policy.wait_after(1, refused) == 3.0 and policy.wait_after(2, refused) == 9.0
        assert policy.wait_after(3, refused) is None
        assert policy.wait_after(1, APITimeoutError(request)) is None
        assert policy.wait_after(1, OpenAIError("other")) is None
