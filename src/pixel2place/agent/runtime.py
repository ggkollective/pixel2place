"""Nemotron 응답 검증, 호출 예산과 조립 도구 조율."""

import json
from collections.abc import Callable
from contextlib import suppress
from datetime import UTC, datetime
from time import perf_counter

from openai import OpenAI, OpenAIError
from openai.types.chat import ChatCompletion, ChatCompletionMessageParam, ChatCompletionToolParam
from pydantic import TypeAdapter, ValidationError

from pixel2place.agent.artifacts import ArtifactError, write_record
from pixel2place.agent.audit import ModelAudit
from pixel2place.agent.contracts import (
    CalibrateBoardArguments,
    ErrorCode,
    ErrorResult,
    ExecuteCellArguments,
    PrepareLayoutArguments,
    ReplanLayoutArguments,
    ReworkCellArguments,
    ToolName,
    ToolResult,
    ToolState,
)
from pixel2place.agent.inference import (
    CATALOG_INFERENCE,
    RetryPolicy,
    completion_evidence,
    request_completion,
    require_unique_keys,
    single_tool_call,
    with_retry,
)
from pixel2place.agent.prompts import system_prompt
from pixel2place.agent.runtime_contracts import (
    AgentBudget,
    AgentEvent,
    AgentReport,
    CallCounts,
    CompleteDecision,
    EventTiming,
    FinalDecision,
    RequestEvidence,
    ValidatedCall,
)
from pixel2place.agent.schema import strict_parameters
from pixel2place.agent.tools import MAX_LAYOUT_ATTEMPTS, AssemblyTools, ToolLedger
from pixel2place.reasoner.nemotron import require_nemotron_model
from pixel2place.reasoner.nemotron_tools import (
    ToolProtocolError,
    ToolVerificationError,
    validate_completion,
)

EVENTS_FILE = "events.jsonl"
TIMING_FILE = "timing.jsonl"
"""이벤트마다 시각 1줄을 쓰는 파일. manifest에 등록하지 않는다."""

_ARGUMENT_MODELS = {
    ToolName.PREPARE_LAYOUT: PrepareLayoutArguments,
    ToolName.CALIBRATE_BOARD: CalibrateBoardArguments,
    ToolName.EXECUTE_CELL: ExecuteCellArguments,
    ToolName.REPLAN_LAYOUT: ReplanLayoutArguments,
    ToolName.REWORK_CELL: ReworkCellArguments,
}
_DESCRIPTIONS = {
    ToolName.PREPARE_LAYOUT: "Prepare and check a layout for the original user goal.",
    ToolName.CALIBRATE_BOARD: "Capture the board and calibrate it from nominal observations.",
    ToolName.EXECUTE_CELL: "Compile, execute once, and check the calibrated assembly task.",
    ToolName.REPLAN_LAYOUT: "Reject the current layout and prepare one new layout for the goal.",
    ToolName.REWORK_CELL: "Place the parts of the empty cells again and check the full task.",
}
_ALLOWED = {
    ToolState.CREATED: (ToolName.PREPARE_LAYOUT,),
    ToolState.LAYOUT_RETRYABLE: (ToolName.PREPARE_LAYOUT,),
    ToolState.LAYOUT_READY: (ToolName.CALIBRATE_BOARD,),
    ToolState.CALIBRATED: (ToolName.EXECUTE_CELL,),
    ToolState.CHECKED: (),
}


def allowed_tools(
    ledger: ToolLedger, *, review: bool, disposition: bool = False
) -> tuple[ToolName, ...]:
    """현재 상태에서 모델에 제공하는 도구.

    계획 검토를 켠 실행은 준비 시도가 남아 있을 때 `layout_ready`에서 재계획도 제공한다.
    처분을 켠 실행은 `checked`에서 재작업을 제공한다. 조건은 실패한 검사와 쓰지 않은 재작업이다.
    재작업 대상이 없어도 제공한다. 대상의 판단은 모델이 한다.
    """
    allowed = _ALLOWED[ledger.state]
    if (
        review
        and ledger.state == ToolState.LAYOUT_READY
        and ledger.layout_attempts < MAX_LAYOUT_ATTEMPTS
    ):
        return (*allowed, ToolName.REPLAN_LAYOUT)
    if (
        disposition
        and ledger.state == ToolState.CHECKED
        and ledger.error_code is not None
        and not ledger.rework_consumed
    ):
        return (*allowed, ToolName.REWORK_CELL)
    return allowed


def tool_schema(name: ToolName, *, strict: bool = True) -> ChatCompletionToolParam:
    """서버 생성 제약을 요청한다. 비교 도구는 기존 정의도 만들 수 있다."""
    tool: ChatCompletionToolParam = {
        "type": "function",
        "function": {
            "name": name.value,
            "description": _DESCRIPTIONS[name],
            "parameters": strict_parameters(_ARGUMENT_MODELS[name]),
        },
    }
    if strict:
        tool["function"]["strict"] = True
    return tool


def request_messages(
    *,
    ledger: ToolLedger,
    messages: list[ChatCompletionMessageParam],
    system_prompt: str,
    review: bool = False,
    disposition: bool = False,
) -> list[ChatCompletionMessageParam]:
    """현재 상태와 검증된 대화를 같은 요청 규약으로 결합한다.

    계획 검토를 켠 실행은 준비 시도 수를 상태에 넣는다. 모델이 재계획 가능 여부를 읽는다.
    처분을 켠 실행은 재작업 실행의 수를 상태에 넣는다. 모델이 재작업 가능 여부를 읽는다.
    """
    context: dict[str, object] = {
        "state": ledger.state,
        "error_code": ledger.error_code,
        "layout_id": ledger.layout_id,
        "calibration_id": ledger.calibration_id,
        "check_id": ledger.check_id,
    }
    if review:
        context["layout_attempts"] = ledger.layout_attempts
    if disposition:
        context["rework_runs"] = int(ledger.rework_consumed)
    state = json.dumps(context, separators=(",", ":"))
    return [
        {"role": "system", "content": f"{system_prompt}\nCurrent runtime state: {state}"},
        *messages,
    ]


def validate_action(
    completion: ChatCompletion,
    *,
    allowed: tuple[ToolName, ...],
    ledger: ToolLedger,
    release_allowed: Callable[[], bool],
    review: bool = False,
) -> ValidatedCall | FinalDecision:
    """도구·타입·상태·ID 계약을 검사한다. 완료에는 출하 규칙의 허용을 요구한다.

    계획 검토를 켠 실행은 `layout_ready`에서 `plan_rejected` 중단을 받는다.
    재작업 호출의 부품 목록은 장부의 재작업 대상과 같아야 한다.
    """
    choice = completion.choices[0]
    message = choice.message
    call = single_tool_call(completion)
    if call is not None:
        name = ToolName(call.function.name)
        if name not in allowed or call.id in ledger.call_ids:
            raise ValueError("Tool call is not allowed.")
        require_unique_keys(call.function.arguments)
        arguments = _ARGUMENT_MODELS[name].model_validate_json(call.function.arguments)
        if isinstance(arguments, PrepareLayoutArguments):
            if arguments.retry != (ledger.state == ToolState.LAYOUT_RETRYABLE):
                raise ValueError("Layout retry does not match the current state.")
        elif isinstance(arguments, ReworkCellArguments):
            parts = set(arguments.part_ids)
            if len(parts) != len(arguments.part_ids) or parts != set(ledger.rework_targets):
                raise ValueError("Rework parts are not the rework targets.")
        else:
            if arguments.layout_id != ledger.layout_id:
                raise ValueError("Layout is not current.")
            if (
                isinstance(arguments, ExecuteCellArguments)
                and arguments.calibration_id != ledger.calibration_id
            ):
                raise ValueError("Calibration is not current.")
        return ValidatedCall(id=call.id, name=name, arguments=arguments)
    if choice.finish_reason != "stop" or not message.content:
        raise ValueError("Expected a final JSON decision.")
    require_unique_keys(message.content)
    final = TypeAdapter(FinalDecision).validate_json(message.content)
    if isinstance(final, CompleteDecision):
        if not release_allowed():
            raise ValueError("Completion requires a passed check.")
    elif final.reason_code == ErrorCode.PLAN_REJECTED:
        if not review or ledger.state != ToolState.LAYOUT_READY:
            raise ValueError("Abort reason does not match the current error.")
    elif final.reason_code not in (ErrorCode.AGENT_ABORTED, ledger.error_code):
        raise ValueError("Abort reason does not match the current error.")
    return final


class AgentRuntime:
    """도구를 한 번 조율한다. 호출자가 외부 클라이언트를 닫는다."""

    def __init__(
        self,
        *,
        tools: AssemblyTools,
        client: OpenAI,
        model: str,
        budget: AgentBudget,
        prompt: str,
        review_model: str | None = None,
        disposition_model: str | None = None,
    ) -> None:
        """`review_model`은 계획 검토를 켠 실행의 `layout_ready` 요청을 보내는 모델이다.

        `disposition_model`은 처분을 켠 실행의 `checked` 요청을 보내는 모델이다.
        None인 모델의 요청은 `model`로 보낸다.
        """
        self._tools = tools
        self._client = client.with_options(max_retries=0, timeout=budget.nemotron_timeout_s)
        self._model = model
        self._review_model = review_model if tools.review and review_model else model
        self._disposition_model = (
            disposition_model if tools.disposition and disposition_model else model
        )
        self._budget = budget
        self._retry = RetryPolicy(retries=budget.request_retries)
        self._messages: list[ChatCompletionMessageParam] = [{"role": "user", "content": prompt}]
        self._inferences = 0
        self._catalog = 0
        self._retries = 0
        self._request_bytes = 0
        self._sequence = 0
        self._started = False
        self._started_at = 0.0
        self._error: ErrorCode | None = None
        self._final: FinalDecision | None = None
        self._review = tools.review
        self._disposition = tools.disposition
        self._system_prompt = system_prompt(review=tools.review, disposition=tools.disposition)
        self._audit = ModelAudit(root=tools.root, run_id=tools.run_id, secret=client.api_key)

    def run(self) -> AgentReport:
        """모델 목록을 1회 읽는다. 모델 요청 한도를 지킨다. 기록 실패 뒤 외부 작업을 막는다.

        일시 오류가 난 요청은 예산의 `request_retries`만큼 다시 보낸다.
        재전송은 모델 목록 요청 수와 모델 요청 수에 넣지 않는다.
        """
        if self._started:
            raise ValueError("The run controller cannot resume or run twice.")
        self._started = True
        self._started_at = perf_counter()
        try:
            self._event("catalog_request")
            self._catalog += 1
            try:
                available = with_retry(
                    lambda: require_nemotron_model(self._client, model=self._model),
                    policy=self._retry,
                    audit=self._audit,
                    inference=CATALOG_INFERENCE,
                    target="catalog",
                    reserve_retry=lambda: self._reserve_retry(0),
                )
                if not {self._review_model, self._disposition_model} <= set(available):
                    raise ValueError("A judgment model is not available in this account.")
            except (OpenAIError, ValueError):
                self._error = ErrorCode.MODEL_ERROR
            self._event("catalog_result", error=self._error)
            while self._error is None and self._final is None:
                if self._inferences >= self._budget.max_inferences:
                    self._error = ErrorCode.BUDGET_EXHAUSTED
                    break
                self._step()
        except OSError:
            self._error = ErrorCode.RECORDING_ERROR
        except ArtifactError:
            self._error = ErrorCode.PROTOCOL_ERROR
        except KeyboardInterrupt:
            self._error = ErrorCode.AGENT_ABORTED
        return self._finish()

    def _step(self) -> None:
        before = self._tools.ledger.state
        # 계획 검토와 처분은 그 역할의 모델이 맡는다. 다른 상태는 조율 모델이 맡는다.
        model = {
            ToolState.LAYOUT_READY: self._review_model,
            ToolState.CHECKED: self._disposition_model,
        }.get(before, self._model)
        allowed = allowed_tools(
            self._tools.ledger, review=self._review, disposition=self._disposition
        )
        messages = request_messages(
            ledger=self._tools.ledger,
            messages=self._messages,
            system_prompt=self._system_prompt,
            review=self._review,
            disposition=self._disposition,
        )
        schemas = [tool_schema(name) for name in allowed]
        body: dict[str, object] = {
            "model": model,
            "messages": messages,
            "max_tokens": self._budget.max_tokens,
        }
        if schemas:
            body.update(tools=schemas, tool_choice="auto")
        # ASCII 이스케이프와 공백을 포함한 크기로 요청 자료를 보수적으로 센다.
        size = len(json.dumps(body).encode("utf-8"))
        if self._request_bytes + size > self._budget.max_request_bytes:
            self._error = ErrorCode.BUDGET_EXHAUSTED
            return
        self._audit.write(inference=self._inferences + 1, kind="request", data={"request": body})
        self._event("model_request", allowed=allowed, model=model)
        self._inferences += 1
        self._request_bytes += size
        evidence = None
        action: ValidatedCall | FinalDecision | None = None
        try:
            observed = with_retry(
                lambda: request_completion(
                    client=self._client,
                    audit=self._audit,
                    inference=self._inferences,
                    model=model,
                    messages=messages,
                    tools=schemas,
                    max_tokens=self._budget.max_tokens,
                    timeout_s=self._budget.nemotron_timeout_s,
                ),
                policy=self._retry,
                audit=self._audit,
                inference=self._inferences,
                target="completion",
                reserve_retry=lambda: self._reserve_retry(size),
            )
            completion = validate_completion(observed.response, model=model)
            evidence = completion_evidence(completion, latency_s=observed.latency_s)
            if completion.choices[0].finish_reason == "length":
                self._error = ErrorCode.MODEL_ERROR
                self._audit.write(
                    inference=self._inferences,
                    kind="validation",
                    data={"accepted": False, "reason": "token_limit"},
                )
            else:
                action = validate_action(
                    completion,
                    allowed=allowed,
                    ledger=self._tools.ledger,
                    release_allowed=self._tools.release_allowed,
                    review=self._review,
                )
                self._audit.write(
                    inference=self._inferences, kind="validation", data={"accepted": True}
                )
        except ToolProtocolError as exc:
            self._error = (
                ErrorCode.MODEL_ERROR
                if exc.code == ToolVerificationError.DIFFERENT_MODEL
                else ErrorCode.PROTOCOL_ERROR
            )
            self._audit.write(
                inference=self._inferences,
                kind="validation",
                data={"accepted": False, "reason": exc.code.value},
            )
        except OpenAIError:
            self._error = ErrorCode.MODEL_ERROR
            self._audit.write(
                inference=self._inferences,
                kind="validation",
                data={"accepted": False, "reason": "model_error"},
            )
        except (ValidationError, ValueError) as exc:
            self._error = ErrorCode.PROTOCOL_ERROR
            self._audit.write(
                inference=self._inferences, kind="validation", data=self._audit.rejection(exc)
            )
        self._event(
            "model_result",
            allowed=allowed,
            model=model,
            evidence=evidence,
            call=action if isinstance(action, ValidatedCall) else None,
            final=action if action is not None and not isinstance(action, ValidatedCall) else None,
            error=self._error,
        )
        if self._error is not None:
            return
        if not isinstance(action, ValidatedCall):
            self._final = action
            if not isinstance(action, CompleteDecision):
                assert action is not None
                self._error = action.reason_code
            return
        result = self._dispatch(action)
        self._event("tool_result", from_state=before, call=action, result=result)
        if isinstance(result, ErrorResult) and not result.recoverable:
            self._error = result.error_code
            return
        summary = result.model_dump_json()
        if len(summary.encode("utf-8")) > self._budget.max_tool_bytes:
            self._error = ErrorCode.BUDGET_EXHAUSTED
            return
        self._messages.extend(
            [
                {
                    "role": "assistant",
                    "tool_calls": [
                        {
                            "id": action.id,
                            "type": "function",
                            "function": {
                                "name": action.name.value,
                                "arguments": action.arguments.model_dump_json(),
                            },
                        }
                    ],
                },
                {"role": "tool", "tool_call_id": action.id, "content": summary},
            ]
        )

    def _reserve_retry(self, size: int) -> bool:
        """재전송 1회를 센다. `size`는 다시 보내는 요청 JSON의 크기다.

        요청 자료의 한도가 이 시도를 받지 못하면 세지 않는다. False를 돌려준다.
        """
        if self._request_bytes + size > self._budget.max_request_bytes:
            return False
        self._retries += 1
        self._request_bytes += size
        return True

    def _dispatch(self, call: ValidatedCall) -> ToolResult:
        if isinstance(call.arguments, PrepareLayoutArguments):
            return self._tools.prepare_layout(call.arguments, tool_call_id=call.id)
        if isinstance(call.arguments, CalibrateBoardArguments):
            return self._tools.calibrate_board(call.arguments, tool_call_id=call.id)
        if isinstance(call.arguments, ReplanLayoutArguments):
            return self._tools.replan_layout(call.arguments, tool_call_id=call.id)
        if isinstance(call.arguments, ReworkCellArguments):
            return self._tools.rework_cell(call.arguments, tool_call_id=call.id)
        return self._tools.execute_cell(call.arguments, tool_call_id=call.id)

    def _event(
        self,
        kind: str,
        *,
        allowed: tuple[ToolName, ...] = (),
        from_state: ToolState | None = None,
        evidence: RequestEvidence | None = None,
        call: ValidatedCall | None = None,
        result: ToolResult | None = None,
        final: FinalDecision | None = None,
        error: ErrorCode | None = None,
        model: str | None = None,
    ) -> None:
        """이벤트 1줄과 그 시각 1줄을 쓴다.

        `model`은 요청을 보낸 모델이다. 요청이 아닌 이벤트는 조율 모델을 쓴다.
        시각은 `timing.jsonl`에 쓴다. 이벤트의 형식에는 시각이 없다.
        """
        self._sequence += 1
        event = AgentEvent(
            run_id=self._tools.run_id,
            sequence=self._sequence,
            kind=kind,  # pyright: ignore[reportArgumentType]
            inference=self._inferences + (kind == "model_request"),
            from_state=from_state or self._tools.ledger.state,
            to_state=self._tools.ledger.state,
            allowed_tools=allowed,
            model=model or self._model,
            timeout_s=self._budget.nemotron_timeout_s,
            evidence=evidence,
            call=call,
            result=result,
            final=final,
            error_code=error,
        )
        timing = EventTiming(
            sequence=event.sequence,
            kind=event.kind,
            utc=datetime.now(UTC),
            elapsed_s=perf_counter() - self._started_at,
        )
        # 시각 줄을 먼저 쓴다. 시각 기록이 실패한 이벤트는 `events.jsonl`에 남지 않는다.
        # 그래서 완료를 적은 이벤트와 기록 오류의 보고가 함께 남지 않는다.
        with (self._tools.root / TIMING_FILE).open("a", encoding="utf-8") as stream:
            stream.write(timing.model_dump_json() + "\n")
        with (self._tools.root / EVENTS_FILE).open("a", encoding="utf-8") as stream:
            stream.write(event.model_dump_json() + "\n")

    def _finish(self) -> AgentReport:
        completed = self._error is None and isinstance(self._final, CompleteDecision)
        before = self._tools.ledger.state
        try:
            self._tools.finish(completed=completed, error_code=self._error)
            self._event("final", from_state=before, final=self._final, error=self._error)
        except OSError:
            self._error = ErrorCode.RECORDING_ERROR
            completed = False
        except ArtifactError:
            self._error = ErrorCode.PROTOCOL_ERROR
            completed = False
        if not completed:
            try:
                self._tools.finish(completed=False, error_code=self._error)
            except OSError:
                self._error = ErrorCode.RECORDING_ERROR
        ledger = self._tools.ledger
        report = AgentReport(
            run_id=self._tools.run_id,
            state=ToolState.COMPLETED if completed else ToolState.ABORTED,
            error_code=self._error,
            calls=CallCounts(
                catalog=self._catalog,
                nemotron=self._inferences,
                cosmos=self._tools.cosmos_requests,
                capture=int(ledger.capture_consumed),
                place=int(ledger.place_consumed) + int(ledger.rework_consumed),
                request_bytes=self._request_bytes,
                retries=self._retries,
            ),
            final_decision=self._final,
            check_id=ledger.check_id,
            check_passed=ledger.check_passed,
            reworked=ledger.rework_consumed,
        )
        try:
            write_record(self._tools.root / "agent_report.json", report)
        except OSError:
            report = report.model_copy(
                update={"state": ToolState.ABORTED, "error_code": ErrorCode.RECORDING_ERROR}
            )
            with suppress(OSError):
                self._tools.finish(completed=False, error_code=ErrorCode.RECORDING_ERROR)
                self._event("final", final=self._final, error=ErrorCode.RECORDING_ERROR)
        return report
