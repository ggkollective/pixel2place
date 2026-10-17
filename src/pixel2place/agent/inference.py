"""Nemotron 요청 1회의 전송, 일시 오류의 재전송, 응답 관찰과 응답 형태 검사."""

import json
from collections.abc import Callable
from dataclasses import dataclass
from time import perf_counter, sleep
from typing import Literal, NamedTuple

from openai import APIConnectionError, APIStatusError, APITimeoutError, OpenAI, OpenAIError
from openai.types.chat import (
    ChatCompletion,
    ChatCompletionMessageFunctionToolCall,
    ChatCompletionMessageParam,
    ChatCompletionNamedToolChoiceParam,
    ChatCompletionToolParam,
)

from pixel2place.agent.audit import ModelAudit
from pixel2place.agent.runtime_contracts import MAX_REQUEST_RETRIES, RequestEvidence

type ToolChoice = Literal["auto"] | ChatCompletionNamedToolChoiceParam
type RetryTarget = Literal["catalog", "completion"]
"""다시 보내는 요청의 종류. 모델 목록 요청 또는 모델 요청이다."""

RETRY_DELAYS_S = (3.0, 9.0, 20.0)
"""재전송 앞의 대기. 합은 관찰한 Token Factory 장애(약 20초)보다 길다."""
CATALOG_INFERENCE = 0
"""모델 목록 요청의 시도 기록이 쓰는 회차. 모델 요청의 회차는 1부터다."""
_RATE_LIMIT_STATUS = 429
_SERVER_ERROR_STATUS = 500


def is_transient(error: OpenAIError) -> bool:
    """같은 요청을 다시 보내면 곧 성공할 수 있는 오류인지.

    대상은 연결 오류, HTTP 429와 5xx다. 이 오류는 빨리 온다.
    timeout은 대상이 아니다. timeout은 연결 오류의 하위 클래스라서 먼저 뺀다.
    서버는 timeout이 난 요청의 생성을 계속할 수 있다. 다시 보내면 대기와 비용이 시도 수만큼 는다.
    다른 HTTP 오류와 응답 파싱 오류는 다시 보내도 결과가 같다.
    """
    if isinstance(error, APITimeoutError):
        return False
    if isinstance(error, APIStatusError):
        return error.status_code == _RATE_LIMIT_STATUS or error.status_code >= _SERVER_ERROR_STATUS
    return isinstance(error, APIConnectionError)


@dataclass(frozen=True)
class RetryPolicy:
    """일시 오류가 난 요청 1개를 다시 보내는 횟수. 기본은 재전송 없음이다.

    대기는 `RETRY_DELAYS_S`가 정한다. 오류의 분류는 `is_transient`가 정한다.
    요청 자료의 한도와 재전송 수의 집계는 호출자가 소유한다.
    """

    retries: int = 0

    def __post_init__(self) -> None:
        if not 0 <= self.retries <= MAX_REQUEST_RETRIES:
            raise ValueError("The retry count is outside the allowed range.")

    def wait_after(self, attempt: int, error: OpenAIError) -> float | None:
        """`attempt`번째 시도가 `error`로 실패한 뒤의 대기 초. 다시 보내지 않으면 None이다."""
        if attempt > self.retries or not is_transient(error):
            return None
        return RETRY_DELAYS_S[attempt - 1]


NO_RETRY = RetryPolicy()
REQUEST_RETRY = RetryPolicy(retries=MAX_REQUEST_RETRIES)
"""CLI와 데모의 경로가 켜는 방침이다."""


def _failure(error: OpenAIError) -> tuple[str, int | None]:
    """시도 기록의 결과 이름과 HTTP 상태 코드. 오류 본문은 쓰지 않는다."""
    if isinstance(error, APIStatusError):
        return "http_error", error.status_code
    if isinstance(error, APITimeoutError):
        return "timeout", None
    if isinstance(error, APIConnectionError):
        return "connection_error", None
    return "sdk_error", None


def with_retry[T](
    send: Callable[[], T],
    *,
    policy: RetryPolicy,
    audit: ModelAudit,
    inference: int,
    target: RetryTarget,
    reserve_retry: Callable[[], bool] | None = None,
) -> T:
    """`send`를 실행한다. 일시 오류이면 대기한 뒤 다시 실행한다.

    재전송을 켠 방침은 시도마다 `attempt` 기록 1줄을 쓴다.
    줄은 대상, 시도 번호, 결과, HTTP 상태 코드와 다음 시도 앞의 대기 초다.
    재전송을 끈 방침은 `send`를 1회 실행한다. 기록을 쓰지 않는다.

    Args:
        send: 요청 1회. 전송 오류는 `OpenAIError`로, 응답 내용의 거부는 `ValueError`로 낸다.
        policy: 재전송 횟수.
        audit: 시도 기록을 쓰는 자리.
        inference: 시도 기록의 회차. 모델 목록 요청은 `CATALOG_INFERENCE`다.
        target: 요청의 종류.
        reserve_retry: 다음 시도를 예약한다. 대기의 앞에서 부른다. False이면 다시 보내지 않는다.
            호출자는 이 함수에서 재전송 수와 요청 자료를 센다.

    Raises:
        openai.OpenAIError: 다시 보내지 않는 오류일 때 또는 시도를 모두 썼을 때. 마지막 오류다.
        ValueError: `send`가 응답의 내용을 거부했을 때. 다시 보내지 않는다.
        OSError: 시도 기록을 쓸 수 없을 때.
    """

    def record(outcome: str, *, status: int | None = None, wait_s: float | None = None) -> None:
        if policy.retries:
            audit.write(
                inference=inference,
                kind="attempt",
                data={
                    "target": target,
                    "attempt": attempt,
                    "outcome": outcome,
                    "status": status,
                    "wait_s": wait_s,
                },
            )

    attempt = 1
    while True:
        try:
            result = send()
        except OpenAIError as exc:
            wait_s = policy.wait_after(attempt, exc)
            if wait_s is not None and reserve_retry is not None and not reserve_retry():
                wait_s = None
            outcome, status = _failure(exc)
            record(outcome, status=status, wait_s=wait_s)
            if wait_s is None:
                raise
        except ValueError:
            record("invalid_response")
            raise
        else:
            record("ok")
            return result
        sleep(wait_s)
        attempt += 1


class ObservedResponse(NamedTuple):
    """SDK가 파싱한 응답과 관찰 값. 응답 형식은 호출자가 검사한다."""

    response: object
    latency_s: float
    wire_matches_sdk: bool
    """HTTP 응답과 SDK 응답의 도구 필드가 같은지."""


def request_completion(
    *,
    client: OpenAI,
    audit: ModelAudit,
    inference: int,
    model: str,
    messages: list[ChatCompletionMessageParam],
    tools: list[ChatCompletionToolParam],
    tool_choice: ToolChoice = "auto",
    max_tokens: int,
    timeout_s: float,
) -> ObservedResponse:
    """요청 1회를 보낸다. HTTP 응답과 SDK 응답의 도구 필드를 기록한다.

    도구가 없으면 요청에 `tools`와 `tool_choice`를 넣지 않는다.
    HTTP 상태 오류는 상태 코드를 기록한 뒤 전달한다.

    Raises:
        openai.OpenAIError: 요청 또는 응답 파싱이 실패할 때.
        OSError: 관찰 기록을 쓸 수 없을 때.
    """
    started = perf_counter()
    try:
        if tools:
            raw = client.chat.completions.with_raw_response.create(
                model=model,
                messages=messages,
                max_tokens=max_tokens,
                tools=tools,
                tool_choice=tool_choice,
                timeout=timeout_s,
            )
        else:
            raw = client.chat.completions.with_raw_response.create(
                model=model, messages=messages, max_tokens=max_tokens, timeout=timeout_s
            )
    except APIStatusError as exc:
        audit.write(inference=inference, kind="http_response", data={"status": exc.status_code})
        raise
    latency_s = perf_counter() - started
    try:
        payload = raw.http_response.json()
    except ValueError:
        payload = None
    wire = audit.response(payload)
    audit.write(
        inference=inference,
        kind="http_response",
        data=audit.http_response(
            payload=payload,
            request=raw.http_response.request.content,
            status=raw.http_response.status_code,
        ),
    )
    parse_started = perf_counter()
    response = raw.parse()
    latency_s += perf_counter() - parse_started
    sdk = audit.response(
        response.model_dump(warnings=False) if isinstance(response, ChatCompletion) else None
    )
    audit.write(inference=inference, kind="sdk_response", data={"response": sdk})
    return ObservedResponse(response=response, latency_s=latency_s, wire_matches_sdk=wire == sdk)


def single_tool_call(completion: ChatCompletion) -> ChatCompletionMessageFunctionToolCall | None:
    """응답의 함수 호출 1개를 돌려준다. 도구 호출이 없으면 None이다.

    Raises:
        ValueError: 거절, 이전 형식의 호출, 여러 호출 또는 자유 텍스트가 있을 때.
    """
    choice = completion.choices[0]
    message = choice.message
    if message.refusal:
        raise ValueError("Model refused the request.")
    if message.function_call is not None:
        raise ValueError("Legacy function calls are not allowed.")
    calls = message.tool_calls
    if not calls:
        return None
    if choice.finish_reason != "tool_calls" or len(calls) != 1:
        raise ValueError("Expected one tool call.")
    call = calls[0]
    if call.type != "function" or (message.content and message.content.strip()):
        raise ValueError("Expected a function call without free text.")
    return call


def require_unique_keys(text: str) -> None:
    """JSON 문법과 키 중복을 검사한다. pydantic은 중복 키의 마지막 값을 쓴다.

    Raises:
        ValueError: JSON이 아니거나 같은 객체에 같은 키가 2번 있을 때.
    """

    def reject(pairs: list[tuple[str, object]]) -> dict[str, object]:
        result: dict[str, object] = {}
        for key, value in pairs:
            if key in result:
                raise ValueError("JSON contains a duplicate key.")
            result[key] = value
        return result

    json.loads(text, object_pairs_hook=reject)


def completion_evidence(completion: ChatCompletion, *, latency_s: float) -> RequestEvidence:
    """검증한 응답에서 ID, 종료 이유와 사용량만 남긴다."""
    usage = completion.usage
    details = usage.completion_tokens_details if usage is not None else None
    return RequestEvidence(
        completion_id=completion.id,
        model=completion.model,
        finish_reason=completion.choices[0].finish_reason,
        latency_s=latency_s,
        prompt_tokens=usage.prompt_tokens if usage is not None else None,
        completion_tokens=usage.completion_tokens if usage is not None else None,
        reasoning_tokens=details.reasoning_tokens if details is not None else None,
    )
