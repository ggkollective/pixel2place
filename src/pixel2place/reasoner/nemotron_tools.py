"""Nemotron의 도구 선택과 결과 읽기 계약을 검증한다.

두 번의 모델 요청과 응답 검증, 키를 제외한 근거를 소유한다.
Cosmos 계획, agent 상태와 Isaac 실행은 소유하지 않는다.
"""

from datetime import UTC, datetime
from enum import StrEnum
from time import perf_counter
from typing import Annotated, Literal
from uuid import uuid4

from openai import APIConnectionError, APIError, APIStatusError, APITimeoutError, OpenAI
from openai.types.chat import ChatCompletion, ChatCompletionMessageParam, ChatCompletionToolParam
from pydantic import BaseModel, ConfigDict, Field, ValidationError

from pixel2place.config import TokenFactorySettings
from pixel2place.reasoner.nemotron import create_nemotron_client, require_nemotron_model

DEFAULT_TOOL_MAX_TOKENS = 2048
VERIFICATION_TOOL = "verify_contract"
TOOL_RESULT = "contract_ok"
SYSTEM_PROMPT = """You coordinate a tool contract check.
Call verify_contract once with the request_id from the user.
Use one function call in each response.
After the tool result, return only one JSON object.
Use the keys verified, tool_result, and receipt.
Set verified to true.
Copy tool_result and receipt from the tool result.
Do not create a receipt."""
USER_PROMPT_PREFIX = "Verify tool calling for request_id"
HexId = Annotated[str, Field(pattern=r"^[a-f0-9]{32}$")]


class _Contract(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True, frozen=True)


class VerificationArguments(_Contract):
    """도구가 받을 실행 ID. 경로와 실행 명령은 받지 않는다."""

    request_id: HexId


class VerificationReceipt(_Contract):
    """모델이 도구 결과에서 읽은 완료 값과 무작위 receipt."""

    verified: bool
    tool_result: Literal["contract_ok"]
    receipt: HexId


class ToolVerificationError(StrEnum):
    """원문 응답을 노출하지 않는 검증 실패 코드."""

    CATALOG_ERROR = "catalog_error"
    INVALID_RESPONSE = "invalid_response"
    DIFFERENT_MODEL = "different_model"
    TOKEN_LIMIT = "token_limit"
    TOOL_CALL_INVALID = "tool_call_invalid"
    TOOL_ARGUMENTS_INVALID = "tool_arguments_invalid"
    FINAL_RESPONSE_INVALID = "final_response_invalid"
    HTTP_TIMEOUT = "http_timeout"
    HTTP_CONNECTION = "http_connection"
    HTTP_STATUS = "http_status"
    API_RESPONSE = "api_response"


class ToolProtocolError(ValueError):
    """호출 또는 응답 계약 위반. 검증된 오류 코드만 보관한다."""

    def __init__(self, code: ToolVerificationError) -> None:
        self.code = code
        super().__init__(f"Nemotron tool contract failed: {code}.")


class CompletionEvidence(BaseModel):
    """응답의 식별자, 종료, 지연과 사용량. reasoning 원문은 제외한다."""

    model_config = ConfigDict(frozen=True)

    stage: Literal["select_tool", "read_result"]
    completion_id: str
    served_model: str
    finish_reason: str
    latency_s: float
    prompt_tokens: int | None
    completion_tokens: int | None
    reasoning_tokens: int | None


class NemotronToolsVerification(BaseModel):
    """성공 또는 실패한 검증과 실제로 소비한 호출의 안전한 근거."""

    model_config = ConfigDict(frozen=True)

    schema_version: Literal[1] = 1
    verified_at: datetime
    base_url: str
    requested_model: str
    available_models: tuple[str, ...]
    system_prompt: str
    user_prompt: str
    tool_name: str
    request_id: str
    tool_call_id: str | None
    tool_receipt: str | None
    final_response: VerificationReceipt | None
    max_tokens: int
    timeout_s: float
    max_retries: Literal[0] = 0
    max_completions: Literal[2] = 2
    request_count: int
    completions: tuple[CompletionEvidence, ...]
    passed: bool
    error_code: ToolVerificationError | None
    http_status: int | None


def validate_completion(response: object, *, model: str) -> ChatCompletion:
    """SDK 응답의 필수 구조, 모델 ID와 단일 choice를 검증한다."""
    if not isinstance(response, ChatCompletion):
        raise ToolProtocolError(ToolVerificationError.INVALID_RESPONSE)
    completion = ChatCompletion.model_validate(response.model_dump(warnings=False), strict=True)
    if len(completion.choices) != 1 or completion.choices[0].index != 0:
        raise ToolProtocolError(ToolVerificationError.INVALID_RESPONSE)
    if completion.model != model:
        raise ToolProtocolError(ToolVerificationError.DIFFERENT_MODEL)
    if not completion.id.strip() or completion.choices[0].message.role != "assistant":
        raise ToolProtocolError(ToolVerificationError.INVALID_RESPONSE)
    return completion


def _evidence(
    completion: ChatCompletion, *, stage: Literal["select_tool", "read_result"], latency_s: float
) -> CompletionEvidence:
    usage = completion.usage
    details = usage.completion_tokens_details if usage is not None else None
    return CompletionEvidence(
        stage=stage,
        completion_id=completion.id,
        served_model=completion.model,
        finish_reason=completion.choices[0].finish_reason,
        latency_s=latency_s,
        prompt_tokens=usage.prompt_tokens if usage is not None else None,
        completion_tokens=usage.completion_tokens if usage is not None else None,
        reasoning_tokens=details.reasoning_tokens if details is not None else None,
    )


def _verify_tool_call(completion: ChatCompletion, *, request_id: str) -> str:
    choice = completion.choices[0]
    if choice.finish_reason == "length":
        raise ToolProtocolError(ToolVerificationError.TOKEN_LIMIT)
    calls = choice.message.tool_calls
    if choice.finish_reason != "tool_calls" or not calls or len(calls) != 1:
        raise ToolProtocolError(ToolVerificationError.TOOL_CALL_INVALID)
    call = calls[0]
    if call.type != "function" or call.function.name != VERIFICATION_TOOL or not call.id.strip():
        raise ToolProtocolError(ToolVerificationError.TOOL_CALL_INVALID)
    try:
        args = VerificationArguments.model_validate_json(call.function.arguments)
    except ValidationError as exc:
        raise ToolProtocolError(ToolVerificationError.TOOL_ARGUMENTS_INVALID) from exc
    if args.request_id != request_id:
        raise ToolProtocolError(ToolVerificationError.TOOL_ARGUMENTS_INVALID)
    return call.id


def _verify_final(completion: ChatCompletion, *, receipt: str) -> VerificationReceipt:
    choice = completion.choices[0]
    if choice.finish_reason == "length":
        raise ToolProtocolError(ToolVerificationError.TOKEN_LIMIT)
    if choice.finish_reason != "stop" or choice.message.tool_calls or not choice.message.content:
        raise ToolProtocolError(ToolVerificationError.FINAL_RESPONSE_INVALID)
    try:
        final = VerificationReceipt.model_validate_json(choice.message.content)
    except ValidationError as exc:
        raise ToolProtocolError(ToolVerificationError.FINAL_RESPONSE_INVALID) from exc
    if not final.verified or final.receipt != receipt:
        raise ToolProtocolError(ToolVerificationError.FINAL_RESPONSE_INVALID)
    return final


def _complete(
    client: OpenAI,
    *,
    model: str,
    messages: list[ChatCompletionMessageParam],
    max_tokens: int,
    stage: Literal["select_tool", "read_result"],
    tool: ChatCompletionToolParam | None = None,
) -> tuple[ChatCompletion, CompletionEvidence]:
    """요청의 지연을 재고 응답 구조와 모델을 검증한다."""
    started = perf_counter()
    if tool is None:
        response = client.chat.completions.create(
            model=model,
            messages=messages,
            max_tokens=max_tokens,
        )
    else:
        response = client.chat.completions.create(
            model=model,
            messages=messages,
            max_tokens=max_tokens,
            tools=[tool],
            tool_choice="auto",
        )
    latency_s = perf_counter() - started
    completion = validate_completion(response, model=model)
    return completion, _evidence(completion, stage=stage, latency_s=latency_s)


def _receipt_messages(
    *, call_id: str, request_id: str, receipt: str
) -> list[ChatCompletionMessageParam]:
    """검증된 도구 호출과 결과만 연결한다. reasoning 원문은 전달하지 않는다."""
    return [
        {
            "role": "assistant",
            "tool_calls": [
                {
                    "id": call_id,
                    "type": "function",
                    "function": {
                        "name": VERIFICATION_TOOL,
                        "arguments": VerificationArguments(request_id=request_id).model_dump_json(),
                    },
                }
            ],
        },
        {
            "role": "tool",
            "tool_call_id": call_id,
            "content": VerificationReceipt(
                verified=True, tool_result=TOOL_RESULT, receipt=receipt
            ).model_dump_json(),
        },
    ]


def verify_nemotron_tools(
    settings: TokenFactorySettings, *, max_tokens: int = DEFAULT_TOOL_MAX_TOKENS
) -> NemotronToolsVerification:
    """모델 목록을 확인하고 최대 2회의 모델 요청으로 도구 계약을 검증한다."""
    if type(max_tokens) is not int or not 1 <= max_tokens <= 4096:
        raise ValueError("Tool verification max_tokens must be between 1 and 4096.")
    request_id = uuid4().hex
    user_prompt = f"{USER_PROMPT_PREFIX} {request_id}."
    messages: list[ChatCompletionMessageParam] = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": user_prompt},
    ]
    tool: ChatCompletionToolParam = {
        "type": "function",
        "function": {
            "name": VERIFICATION_TOOL,
            "description": "Verify the request ID and return a receipt.",
            "parameters": VerificationArguments.model_json_schema(),
        },
    }
    available: tuple[str, ...] = ()
    records: list[CompletionEvidence] = []
    request_count = 0
    call_id: str | None = None
    receipt: str | None = None
    final: VerificationReceipt | None = None
    error: ToolVerificationError | None = None
    http_status: int | None = None
    try:
        with create_nemotron_client(settings) as client:
            available = require_nemotron_model(client, model=settings.nemotron_model)
            request_count += 1
            completion, evidence = _complete(
                client,
                model=settings.nemotron_model,
                messages=messages,
                max_tokens=max_tokens,
                stage="select_tool",
                tool=tool,
            )
            records.append(evidence)
            call_id = _verify_tool_call(completion, request_id=request_id)
            # 첫 요청에는 없는 값을 반환해야 결과를 읽었는지 검사할 수 있다.
            receipt = uuid4().hex
            messages.extend(
                _receipt_messages(call_id=call_id, request_id=request_id, receipt=receipt)
            )
            request_count += 1
            completion, evidence = _complete(
                client,
                model=settings.nemotron_model,
                messages=messages,
                max_tokens=max_tokens,
                stage="read_result",
            )
            records.append(evidence)
            final = _verify_final(completion, receipt=receipt)
    except ToolProtocolError as exc:
        error = exc.code
    except ValidationError:
        error = ToolVerificationError.INVALID_RESPONSE
    except APITimeoutError:
        error = ToolVerificationError.HTTP_TIMEOUT
    except APIConnectionError:
        error = ToolVerificationError.HTTP_CONNECTION
    except APIStatusError as exc:
        error = ToolVerificationError.HTTP_STATUS
        http_status = exc.status_code
    except APIError:
        error = ToolVerificationError.API_RESPONSE
    except ValueError:
        error = (
            ToolVerificationError.CATALOG_ERROR
            if request_count == 0
            else ToolVerificationError.INVALID_RESPONSE
        )
    return NemotronToolsVerification(
        verified_at=datetime.now(UTC),
        base_url=str(settings.nebius_base_url),
        requested_model=settings.nemotron_model,
        available_models=available,
        system_prompt=SYSTEM_PROMPT,
        user_prompt=user_prompt,
        tool_name=VERIFICATION_TOOL,
        request_id=request_id,
        tool_call_id=call_id,
        tool_receipt=receipt,
        final_response=final,
        max_tokens=max_tokens,
        timeout_s=settings.nemotron_timeout_s,
        request_count=request_count,
        completions=tuple(records),
        passed=error is None,
        error_code=error,
        http_status=http_status,
    )
