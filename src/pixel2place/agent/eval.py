"""작업 지시 접수, 계획 검토, 조율과 처분의 오프라인 평가 실행과 채점.

평가는 실행 프로그램과 같은 요청 생성 코드와 검증 코드를 쓴다. 조립 도구는 실행하지 않는다.
사례와 정답은 `pixel2place.agent.eval_cases`가 만든다.
이 모듈은 요청을 순서대로 보낸다. 답을 정답과 비교하고 기준 판정을 쓴다.
"""

import hashlib
import json
import logging
from collections.abc import Iterable, Sequence
from enum import StrEnum
from pathlib import Path
from typing import Annotated, Literal, NamedTuple, Self
from uuid import uuid4

from openai import OpenAI, OpenAIError
from openai.types.chat import ChatCompletionMessageParam, ChatCompletionToolParam
from pydantic import Field, model_validator

from pixel2place.agent.artifacts import write_record
from pixel2place.agent.audit import ModelAudit
from pixel2place.agent.contracts import (
    ArtifactRef,
    CalibrateBoardArguments,
    CalibrationResult,
    CalibrationSummary,
    Contract,
    Count,
    ErrorCode,
    ErrorResult,
    ErrorSummary,
    ExecuteCellArguments,
    ExecutionResult,
    LayoutResult,
    OpaqueId,
    PrepareLayoutArguments,
    ReplanLayoutArguments,
    ReviewReason,
    ReworkCellArguments,
    ToolName,
    ToolState,
)
from pixel2place.agent.eval_cases import (
    CoordinationCase,
    CoordinationState,
    DispositionAction,
    DispositionCase,
    DispositionGroup,
    EvalCase,
    Feature,
    IntakeCase,
    IntakeGroup,
    PlaceRun,
    ReviewAction,
    ReviewCase,
    ReviewGroup,
    SavedPair,
    build_cases,
    cases_jsonl,
    order_key,
    place_reports,
    select_cases,
)
from pixel2place.agent.inference import completion_evidence, request_completion
from pixel2place.agent.inspection import (
    checked_error,
    inspect_layout,
    inspected_summary,
    review_summary,
    rework_targets,
    summarize_execution,
)
from pixel2place.agent.intake import (
    ARGUMENT_MODELS,
    INTAKE_SYSTEM_PROMPT,
    IntakeOutcome,
    IntakeTool,
    RejectReason,
    intake_messages,
    intake_tools,
    read_order,
)
from pixel2place.agent.prompts import system_prompt
from pixel2place.agent.runtime import allowed_tools, request_messages, tool_schema, validate_action
from pixel2place.agent.runtime_contracts import (
    AbortDecision,
    AgentBudget,
    CompleteDecision,
    FinalDecision,
    RequestEvidence,
    ValidatedCall,
)
from pixel2place.agent.tools import ToolLedger
from pixel2place.config import (
    DEFAULT_JUDGMENT_MODEL,
    DEFAULT_NEMOTRON_MODEL,
    TokenFactorySettings,
)
from pixel2place.layout.schema import Layout
from pixel2place.order import Order
from pixel2place.reasoner.nemotron import create_nemotron_client, require_nemotron_model
from pixel2place.reasoner.nemotron_tools import (
    ToolProtocolError,
    ToolVerificationError,
    validate_completion,
)

logger = logging.getLogger(__name__)

CASES_FILE = "cases.jsonl"
EVAL_FILE = "eval.json"
RESULTS_FILE = "results.jsonl"
REPORT_FILE = "report.json"
# 반복 실행하는 사례의 수다. 같은 사례의 답이 실행마다 같은지 본다.
CONSISTENCY_CASES = 20
MAX_REPEATS = 5
# 응답이 없는 요청이 이 수만큼 이어지면 멈춘다. 남은 요청에 비용을 쓰지 않는다.
MAX_CONSECUTIVE_FAILURES = 5
# 요청 JSON의 바이트 수를 입력 토큰 수로 바꾸는 값이다.
# 실제 실행의 요청은 토큰 1개에 2.7~3.2바이트다.
BYTES_PER_TOKEN = 3
# 비용 예상에 쓰는 요청 1회의 출력 토큰 수다. 실제 실행의 평균은 약 90개다.
EXPECTED_OUTPUT_TOKENS = 300
TOKENS_PER_PRICE_UNIT = 1_000_000
# Token Factory 모델 목록의 가격이다. 100만 토큰의 입력 달러와 출력 달러다.
PRICES_USD: dict[str, tuple[float, float]] = {
    DEFAULT_NEMOTRON_MODEL: (0.06, 0.24),
    "nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B": (0.06, 0.24),
    "nvidia/nemotron-3-super-120b-a12b": (0.30, 0.90),
    DEFAULT_JUDGMENT_MODEL: (1.00, 3.00),
}

_PROCEED_ANSWER = ToolName.CALIBRATE_BOARD.value
_REWORK_ANSWER = ToolName.REWORK_CELL.value
# 기능의 요청이 모델에 제공하는 모든 조립 도구다.
_STATE_TOOLS = {
    Feature.REVIEW: (ToolName.CALIBRATE_BOARD, ToolName.REPLAN_LAYOUT),
    Feature.COORDINATION: (ToolName.PREPARE_LAYOUT, ToolName.EXECUTE_CELL),
    Feature.DISPOSITION: (ToolName.REWORK_CELL,),
}
# 실행 프로그램이 답의 내용을 거부한 이유의 코드다. 그 응답의 형식은 계약과 같다.
_FALSE_RELEASE_FAILURE = "unverified_completion"
_WRONG_PARTS_FAILURE = "invalid_rework_parts"
# 요청의 대화에 넣는 보정 결과다. Fake 실행의 공칭 장면에서 얻은 대표값이다.
_CALIBRATION = CalibrationSummary(
    markers_found=6,
    markers_total=6,
    pins_used=64,
    pins_total=64,
    marker_max_residual_px=0.018,
    pin_rms_mm=0.101,
    pin_max_mm=0.193,
    board_shift_mm=0.031,
    board_dx_mm=-0.001,
    board_dy_mm=-0.03,
    board_yaw_deg=-0.001,
)
_VALID_OUTCOMES = (
    IntakeOutcome.ACCEPTED,
    IntakeOutcome.REJECTED,
    IntakeOutcome.ORDER_NOT_FEASIBLE,
)


class ResponseStatus(StrEnum):
    """요청 1회의 응답 상태."""

    VALID = "valid"
    """응답이 실행 프로그램의 검증을 통과했다. 답이 정답인지는 따로 본다."""
    INVALID = "invalid"
    """응답이 계약과 다르다. 잘린 응답도 포함한다."""
    NO_RESPONSE = "no_response"
    """HTTP 오류 또는 다른 응답 모델이다. 판정할 답이 없다."""


class EvalConfig(Contract):
    """평가 1회의 입력. 같은 값과 같은 저장 계획은 같은 사례를 만든다."""

    feature: Feature
    model: Annotated[str, Field(min_length=1, pattern=r"^\S+$")]
    seed: int = 0
    limit: Annotated[int, Field(ge=1)] | None = None
    repeats: Annotated[int, Field(ge=1, le=MAX_REPEATS)] = 1
    """반복 사례 1개의 실행 횟수. 1은 반복하지 않는다."""
    disposition: bool = False
    """처분을 켠 실행의 지시문과 상태로 요청을 만든다.

    계획 검토와 조율은 두 값을 모두 받는다. 처분은 True만 받는다. 접수는 False만 받는다.
    """

    @model_validator(mode="after")
    def _disposition_fits_the_feature(self) -> Self:
        required = {Feature.DISPOSITION: True, Feature.INTAKE: False}.get(self.feature)
        if required is not None and self.disposition is not required:
            raise ValueError("The disposition setting does not fit the feature.")
        return self


class CostEstimate(Contract):
    """요청을 보내기 전의 어림값. 가격을 모르는 모델은 비용이 None이다."""

    input_tokens: Count
    expected_usd: float | None
    max_usd: float | None
    """모든 응답이 출력 한도를 채울 때의 비용."""


class EvalManifest(Contract):
    """평가 조건. 인증값과 로컬 경로는 없다."""

    schema_version: Literal[1] = 1
    eval_id: OpaqueId
    config: EvalConfig
    dry_run: bool
    saved_pairs: Count
    system_prompt: str
    tools: tuple[dict[str, object], ...]
    max_tokens: int
    timeout_s: float
    sdk_retries: int
    case_count: Count
    request_count: Count
    cases_sha256: str
    estimate: CostEstimate


class PreparedEval(NamedTuple):
    """기록한 평가 조건과 그 사례."""

    manifest: EvalManifest
    cases: Sequence[EvalCase]


class EvalResult(Contract):
    """요청 1회의 답과 채점. 자유 텍스트는 없다."""

    case_id: str
    run: Count
    inference: Count
    status: ResponseStatus
    answer: str | None = None
    """검증을 통과한 답. 예: `accepted`, `rejected:too_many_parts`, `calibrate_board`."""
    order: Order | None = None
    """접수에서 주문 계약을 통과한 주문."""
    failure: str | None = None
    """검증을 통과하지 못한 이유의 코드."""
    correct: bool
    evidence: RequestEvidence | None = None


class Metric(Contract):
    """사례 묶음 1개의 수. 기준이 있으면 통과 여부가 있다. 사례가 없으면 None이다."""

    name: str
    count: Count
    total: Count
    criterion: str | None = None
    passed: bool | None = None


class Consistency(Contract):
    """반복 사례에서 모든 실행의 답이 같은 사례의 수. 응답 없는 실행이 있는 사례는 뺀다."""

    cases: Count
    runs: Count
    agreed: Count


class Usage(Contract):
    """응답이 보고한 토큰 수와 그 비용. 가격을 모르는 모델은 비용이 None이다."""

    prompt_tokens: Count
    completion_tokens: Count
    reasoning_tokens: Count
    cost_usd: float | None


class EvalReport(Contract):
    """평가 1회의 지표와 기준 판정."""

    schema_version: Literal[1] = 1
    eval_id: OpaqueId
    feature: Feature
    model: str
    case_count: Count
    request_count: Count
    completed_requests: Count
    stopped: bool
    """응답 없는 요청이 이어져서 남은 요청을 보내지 않았다."""
    metrics: tuple[Metric, ...]
    consistency: Consistency | None
    usage: Usage
    passed: bool
    """멈추지 않았고 사례가 있는 모든 기준을 통과했다."""


class _Bound(NamedTuple):
    """`count / total`의 하한 또는 상한. 정수로 비교한다."""

    at_least: bool
    numerator: int
    denominator: int

    def holds(self, count: int, total: int) -> bool:
        left, right = count * self.denominator, self.numerator * total
        return left >= right if self.at_least else left <= right

    def __str__(self) -> str:
        return f"{'>=' if self.at_least else '<='} {self.numerator}/{self.denominator}"


_CONTRACT_VALID = _Bound(True, 99, 100)
_KIT_MATCH = _Bound(True, 38, 40)
_BOARD_MATCH = _Bound(True, 27, 30)
_FALSE_ACCEPT = _Bound(False, 2, 48)
_FALSE_DETECTION = _Bound(False, 1, 96)
_NEVER = _Bound(False, 0, 1)
_TOOL_MATCH = _Bound(True, 29, 30)
_RELEASE_MATCH = _Bound(True, 95, 100)
_REWORK_MATCH = _Bound(True, 90, 100)
_REJECT_MATCH = _Bound(True, 90, 100)
_FALSE_RELEASE = _Bound(False, 2, 100)
_DETECTION = {
    ReviewReason.PART_SET_MISMATCH: _Bound(True, 90, 100),
    ReviewReason.POSITION_MISMATCH: _Bound(True, 85, 100),
    ReviewReason.ARRANGEMENT_MISMATCH: _Bound(True, 80, 100),
}


def prepare_eval(
    config: EvalConfig, *, out_dir: Path, saved: Sequence[SavedPair] = (), dry_run: bool
) -> PreparedEval:
    """사례를 만든다. 새 디렉터리에 사례와 평가 조건을 쓴다. 네트워크를 쓰지 않는다.

    모든 요청 본문을 만들어서 입력 토큰과 비용을 어림한다.

    Raises:
        FileExistsError: 출력 디렉터리가 이미 있을 때.
        ValueError: 저장한 계획이 준비 검사를 통과하지 못할 때.
        OSError: 기록을 쓸 수 없을 때.
    """
    cases = select_cases(
        build_cases(config.feature, seed=config.seed, saved=saved), limit=config.limit
    )
    requests = _requests(cases, repeats=config.repeats)
    request_bytes = sum(
        len(
            json.dumps(
                _request_body(case, model=config.model, disposition=config.disposition)
            ).encode("utf-8")
        )
        for case, _ in requests
    )
    budget = AgentBudget()
    text = cases_jsonl(cases)
    manifest = EvalManifest(
        eval_id=uuid4().hex,
        config=config,
        dry_run=dry_run,
        saved_pairs=len(saved),
        system_prompt=(
            INTAKE_SYSTEM_PROMPT
            if config.feature == Feature.INTAKE
            else system_prompt(review=True, disposition=config.disposition)
        ),
        tools=tuple(dict(tool) for tool in _offered_tools(config.feature)),
        max_tokens=budget.max_tokens,
        timeout_s=budget.nemotron_timeout_s,
        sdk_retries=budget.sdk_retries,
        case_count=len(cases),
        request_count=len(requests),
        cases_sha256=hashlib.sha256(text.encode("utf-8")).hexdigest(),
        estimate=_estimate(
            config.model,
            input_tokens=request_bytes // BYTES_PER_TOKEN,
            requests=len(requests),
            max_tokens=budget.max_tokens,
        ),
    )
    # 출력 경로를 먼저 확보한다. 충돌 뒤에 유료 호출을 하지 않는다.
    out_dir.mkdir(parents=True, exist_ok=False)
    (out_dir / CASES_FILE).write_text(text, encoding="utf-8")
    write_record(out_dir / EVAL_FILE, manifest)
    return PreparedEval(manifest=manifest, cases=cases)


def run_eval(
    prepared: PreparedEval, *, out_dir: Path, settings: TokenFactorySettings
) -> EvalReport:
    """모델 목록을 1회 읽는다. 요청을 순서대로 보내고 보고를 쓴다. 재시도는 없다.

    응답 없는 요청이 `MAX_CONSECUTIVE_FAILURES`번 이어지면 멈춘다.

    Raises:
        openai.OpenAIError: 모델 목록을 읽을 수 없을 때.
        ValueError: 계정의 모델 목록에 모델이 없을 때.
        OSError: 기록을 쓸 수 없을 때.
    """
    manifest, cases = prepared
    model = manifest.config.model
    configured = settings.model_copy(
        update={"nemotron_model": model, "nemotron_timeout_s": manifest.timeout_s}
    )
    audit = ModelAudit(
        root=out_dir,
        run_id=manifest.eval_id,
        secret=configured.nebius_api_key.get_secret_value(),
        argument_models=(
            {tool.value: ARGUMENT_MODELS[tool] for tool in IntakeTool}
            if manifest.config.feature == Feature.INTAKE
            else None
        ),
    )
    requests = _requests(cases, repeats=manifest.config.repeats)
    results: list[EvalResult] = []
    failures = 0
    with create_nemotron_client(configured) as client:
        require_nemotron_model(client, model=model)
        for inference, (case, run) in enumerate(requests, start=1):
            result = _ask(
                case,
                run=run,
                inference=inference,
                client=client,
                model=model,
                audit=audit,
                disposition=manifest.config.disposition,
            )
            with (out_dir / RESULTS_FILE).open("a", encoding="utf-8") as stream:
                stream.write(f"{result.model_dump_json()}\n")
            results.append(result)
            logger.info(
                "Evaluation request finished.",
                extra={
                    "inference": inference,
                    "requests": len(requests),
                    "case_id": case.case_id,
                    "status": result.status.value,
                    "correct": result.correct,
                },
            )
            failures = failures + 1 if result.status == ResponseStatus.NO_RESPONSE else 0
            if failures >= MAX_CONSECUTIVE_FAILURES:
                break
    report = _report(manifest, cases, results)
    write_record(out_dir / REPORT_FILE, report)
    return report


def _requests(cases: Sequence[EvalCase], *, repeats: int) -> list[tuple[EvalCase, int]]:
    """모든 사례를 1번 실행한다. 그 뒤에 반복 사례를 남은 횟수만큼 실행한다."""
    repeated = select_cases(cases, limit=CONSISTENCY_CASES)
    return [(case, 1) for case in cases] + [
        (case, run) for run in range(2, repeats + 1) for case in repeated
    ]


def _offered_tools(feature: Feature) -> list[ChatCompletionToolParam]:
    """기능의 요청이 모델에 제공하는 모든 도구."""
    if feature == Feature.INTAKE:
        return intake_tools()
    return [tool_schema(name) for name in _STATE_TOOLS[feature]]


def _estimate(model: str, *, input_tokens: int, requests: int, max_tokens: int) -> CostEstimate:
    return CostEstimate(
        input_tokens=input_tokens,
        expected_usd=_cost(model, input_tokens, requests * EXPECTED_OUTPUT_TOKENS),
        max_usd=_cost(model, input_tokens, requests * max_tokens),
    )


def _cost(model: str, input_tokens: int, output_tokens: int) -> float | None:
    if model not in PRICES_USD:
        return None
    input_price, output_price = PRICES_USD[model]
    return (input_tokens * input_price + output_tokens * output_price) / TOKENS_PER_PRICE_UNIT


type _StateCase = ReviewCase | CoordinationCase | DispositionCase


class _StateRequest(NamedTuple):
    """실행 프로그램 상태 1개의 요청. 실행 프로그램이 같은 상태에서 만드는 값이다."""

    ledger: ToolLedger
    allowed: tuple[ToolName, ...]
    messages: list[ChatCompletionMessageParam]
    tools: list[ChatCompletionToolParam]


def _request_body(case: EvalCase, *, model: str, disposition: bool) -> dict[str, object]:
    """모델에 보내는 요청 본문. 입력 토큰의 어림에 쓴다."""
    if isinstance(case, IntakeCase):
        return _body(model, messages=intake_messages(case.text), tools=intake_tools())
    request = _state_request(case, disposition=disposition)
    return _body(model, messages=request.messages, tools=request.tools)


def _body(
    model: str,
    *,
    messages: list[ChatCompletionMessageParam],
    tools: list[ChatCompletionToolParam],
) -> dict[str, object]:
    """실행 프로그램의 요청 본문과 같다. 제공하는 도구가 없으면 도구의 키도 없다."""
    body: dict[str, object] = {
        "model": model,
        "messages": messages,
        "max_tokens": AgentBudget().max_tokens,
    }
    if tools:
        body.update(tools=tools, tool_choice="auto")
    return body


def _state_request(case: _StateCase, *, disposition: bool) -> _StateRequest:
    """사례의 상태에 있는 대화와 장부를 만든다. 조립 도구는 실행하지 않는다.

    ID는 사례 ID에서 만든다. 같은 사례는 같은 요청이 된다.
    `disposition`은 처분을 켠 실행의 지시문과 상태를 쓴다. 처분 사례는 항상 그 값이다.
    """
    if isinstance(case, ReviewCase):
        ledger, messages = _review_state(case)
    elif isinstance(case, CoordinationCase):
        ledger, messages = _coordination_state(case)
    else:
        ledger, messages = _disposition_state(case)
        disposition = True
    allowed = allowed_tools(ledger, review=True, disposition=disposition)
    return _StateRequest(
        ledger=ledger,
        allowed=allowed,
        messages=request_messages(
            ledger=ledger,
            messages=messages,
            system_prompt=system_prompt(review=True, disposition=disposition),
            review=True,
            disposition=disposition,
        ),
        tools=[tool_schema(name) for name in allowed],
    )


def _prepared(run_id: str, layout: Layout, *, call_id: str, artifact_id: str) -> str:
    """부품 목록이 있는 준비 결과. 계획 검토를 켠 실행의 `prepare_layout` 결과다."""
    inspection = inspect_layout(layout, layout, board=layout.board, repair_count=0)
    return LayoutResult(
        run_id=run_id,
        tool_call_id=call_id,
        artifacts=(ArtifactRef(id=artifact_id, kind="layout"),),
        summary=review_summary(inspection.summary, layout=layout),
    ).model_dump_json()


def _review_state(case: ReviewCase) -> tuple[ToolLedger, list[ChatCompletionMessageParam]]:
    """`layout_ready`의 장부와 대화.

    준비 시도가 2회인 사례는 형식 오류 재시도 또는 재계획의 대화를 둔다.
    """
    run_id = _hex_id(case.case_id, "run")
    first_call, second_call = _call_id(case.case_id, 1), _call_id(case.case_id, 2)
    layout_id = _hex_id(case.case_id, "layout")
    messages: list[ChatCompletionMessageParam] = [{"role": "user", "content": case.goal}]
    first_arguments = PrepareLayoutArguments(retry=False).model_dump_json()
    if case.layout_attempts == 1:
        messages += _exchange(
            first_call,
            ToolName.PREPARE_LAYOUT,
            first_arguments,
            _prepared(run_id, case.layout, call_id=first_call, artifact_id=layout_id),
        )
    elif case.rejected is None:
        messages += _exchange(
            first_call,
            ToolName.PREPARE_LAYOUT,
            first_arguments,
            _layout_error(run_id, call_id=first_call),
        )
        messages += _exchange(
            second_call,
            ToolName.PREPARE_LAYOUT,
            PrepareLayoutArguments(retry=True).model_dump_json(),
            _prepared(run_id, case.layout, call_id=second_call, artifact_id=layout_id),
        )
    else:
        rejected_id = _hex_id(case.case_id, "rejected")
        messages += _exchange(
            first_call,
            ToolName.PREPARE_LAYOUT,
            first_arguments,
            _prepared(run_id, case.rejected.layout, call_id=first_call, artifact_id=rejected_id),
        )
        messages += _exchange(
            second_call,
            ToolName.REPLAN_LAYOUT,
            ReplanLayoutArguments(
                layout_id=rejected_id, reason=case.rejected.reason
            ).model_dump_json(),
            _prepared(run_id, case.layout, call_id=second_call, artifact_id=layout_id),
        )
    ledger = ToolLedger(
        run_id=run_id,
        state=ToolState.LAYOUT_READY,
        layout_attempts=case.layout_attempts,
        result_count=case.layout_attempts,
        layout_id=layout_id,
        call_ids=(first_call, second_call)[: case.layout_attempts],
    )
    return ledger, messages


def _layout_error(run_id: str, *, call_id: str) -> str:
    """첫 준비의 복구 가능한 형식 오류."""
    return ErrorResult(
        run_id=run_id,
        tool_call_id=call_id,
        error_code=ErrorCode.LAYOUT_INVALID,
        recoverable=True,
        summary=ErrorSummary(tool=ToolName.PREPARE_LAYOUT, layout_attempt=1),
    ).model_dump_json()


def _coordination_state(
    case: CoordinationCase,
) -> tuple[ToolLedger, list[ChatCompletionMessageParam]]:
    """`created`, `layout_retryable` 또는 `calibrated`의 장부와 대화."""
    run_id = _hex_id(case.case_id, "run")
    if case.group == CoordinationState.CALIBRATED:
        assert case.layout is not None
        return _calibrated_state(case.case_id, goal=case.goal, layout=case.layout)
    messages: list[ChatCompletionMessageParam] = [{"role": "user", "content": case.goal}]
    if case.group == CoordinationState.CREATED:
        return ToolLedger(run_id=run_id), messages
    call_id = _call_id(case.case_id, 1)
    messages += _exchange(
        call_id,
        ToolName.PREPARE_LAYOUT,
        PrepareLayoutArguments(retry=False).model_dump_json(),
        _layout_error(run_id, call_id=call_id),
    )
    ledger = ToolLedger(
        run_id=run_id,
        state=ToolState.LAYOUT_RETRYABLE,
        layout_attempts=1,
        result_count=1,
        call_ids=(call_id,),
        error_code=ErrorCode.LAYOUT_INVALID,
    )
    return ledger, messages


def _calibrated_state(
    case_id: str, *, goal: str, layout: Layout
) -> tuple[ToolLedger, list[ChatCompletionMessageParam]]:
    """준비 1회와 보정이 끝난 `calibrated`의 장부와 대화."""
    run_id = _hex_id(case_id, "run")
    prepare_call, calibrate_call = _call_id(case_id, 1), _call_id(case_id, 2)
    layout_id, calibration_id = _hex_id(case_id, "layout"), _hex_id(case_id, "calibration")
    calibrated = CalibrationResult(
        run_id=run_id,
        tool_call_id=calibrate_call,
        artifacts=(ArtifactRef(id=calibration_id, kind="calibration"),),
        summary=_CALIBRATION,
    )
    messages: list[ChatCompletionMessageParam] = [
        {"role": "user", "content": goal},
        *_exchange(
            prepare_call,
            ToolName.PREPARE_LAYOUT,
            PrepareLayoutArguments(retry=False).model_dump_json(),
            _prepared(run_id, layout, call_id=prepare_call, artifact_id=layout_id),
        ),
        *_exchange(
            calibrate_call,
            ToolName.CALIBRATE_BOARD,
            CalibrateBoardArguments(layout_id=layout_id).model_dump_json(),
            calibrated.model_dump_json(),
        ),
    ]
    ledger = ToolLedger(
        run_id=run_id,
        state=ToolState.CALIBRATED,
        layout_attempts=1,
        capture_consumed=True,
        result_count=2,
        call_ids=(prepare_call, calibrate_call),
        layout_id=layout_id,
        calibration_id=calibration_id,
    )
    return ledger, messages


def _disposition_state(
    case: DispositionCase,
) -> tuple[ToolLedger, list[ChatCompletionMessageParam]]:
    """배치가 끝난 `checked`의 장부와 대화. 재작업 뒤의 사례는 재작업의 대화도 둔다.

    도구 결과의 요약, 오류 코드와 재작업 대상은 실행 프로그램과 같은 함수가 정한다.
    """
    ledger, messages = _calibrated_state(case.case_id, goal=case.goal, layout=case.layout)
    assert ledger.layout_id is not None and ledger.calibration_id is not None
    execute = ExecuteCellArguments(layout_id=ledger.layout_id, calibration_id=ledger.calibration_id)
    runs: list[tuple[ToolName, str, PlaceRun]] = [
        (ToolName.EXECUTE_CELL, execute.model_dump_json(), case.first or case.run)
    ]
    if case.first is not None:
        _, first_report = place_reports(case.layout, case.first)
        assert first_report is not None
        rework = ReworkCellArguments(part_ids=rework_targets(first_report))
        runs.append((ToolName.REWORK_CELL, rework.model_dump_json(), case.run))
    for index, (tool, arguments, run) in enumerate(runs, start=3):
        call_id = _call_id(case.case_id, index)
        check, report = place_reports(case.layout, run)
        check_id = _hex_id(case.case_id, f"check{index}")
        reworked = tool == ToolName.REWORK_CELL
        result = ExecutionResult(
            run_id=ledger.run_id,
            tool_call_id=call_id,
            artifacts=(ArtifactRef(id=check_id, kind="check"),),
            summary=inspected_summary(summarize_execution(check), report=report),
        )
        messages += _exchange(call_id, tool, arguments, result.model_dump_json())
        ledger = ledger.model_copy(
            update={
                "state": ToolState.CHECKED,
                "place_consumed": True,
                "rework_consumed": reworked,
                "result_count": index,
                "call_ids": (*ledger.call_ids, call_id),
                "check_id": check_id,
                "check_passed": check.passed,
                "error_code": checked_error(check_passed=check.passed, report=report),
                "image_check_id": None
                if report is None
                else _hex_id(case.case_id, f"image{index}"),
                "image_check_passed": None if report is None else report.passed,
                "rework_targets": () if report is None or reworked else rework_targets(report),
            }
        )
    return ledger, messages


def _exchange(
    call_id: str, tool: ToolName, arguments: str, result: str
) -> list[ChatCompletionMessageParam]:
    """모델의 도구 호출 1개와 그 결과. 실행 프로그램이 대화에 더하는 형식이다."""
    return [
        {
            "role": "assistant",
            "tool_calls": [
                {
                    "id": call_id,
                    "type": "function",
                    "function": {"name": tool.value, "arguments": arguments},
                }
            ],
        },
        {"role": "tool", "tool_call_id": call_id, "content": result},
    ]


def _hex_id(case_id: str, label: str) -> str:
    return hashlib.sha256(f"{case_id}:{label}".encode()).hexdigest()[:32]


def _call_id(case_id: str, index: int) -> str:
    return f"call_{_hex_id(case_id, f'call{index}')[:24]}"


def _ask(
    case: EvalCase,
    *,
    run: int,
    inference: int,
    client: OpenAI,
    model: str,
    audit: ModelAudit,
    disposition: bool,
) -> EvalResult:
    if isinstance(case, IntakeCase):
        return _ask_intake(
            case, run=run, inference=inference, client=client, model=model, audit=audit
        )
    return _ask_state(
        case,
        run=run,
        inference=inference,
        client=client,
        model=model,
        audit=audit,
        disposition=disposition,
    )


def _ask_intake(
    case: IntakeCase, *, run: int, inference: int, client: OpenAI, model: str, audit: ModelAudit
) -> EvalResult:
    """실행 프로그램의 접수 함수로 문장 1개를 읽는다. 읽은 주문을 정답과 비교한다."""
    reading = read_order(case.text, client=client, model=model, audit=audit, inference=inference)
    outcome = reading.outcome
    if outcome in _VALID_OUTCOMES:
        status = ResponseStatus.VALID
    elif outcome == IntakeOutcome.MODEL_ERROR and reading.evidence is None:
        status = ResponseStatus.NO_RESPONSE
    else:
        status = ResponseStatus.INVALID
    if case.order is not None:
        correct = (
            outcome == IntakeOutcome.ACCEPTED
            and reading.order is not None
            and order_key(reading.order) == order_key(case.order)
        )
    else:
        correct = outcome == IntakeOutcome.REJECTED and reading.reject_reason == case.reject_reason
    answer = (
        outcome.value
        if reading.reject_reason is None
        else f"{outcome.value}:{reading.reject_reason.value}"
    )
    is_valid = status == ResponseStatus.VALID
    return EvalResult(
        case_id=case.case_id,
        run=run,
        inference=inference,
        status=status,
        answer=answer if is_valid else None,
        order=reading.order,
        failure=None if is_valid else outcome.value,
        correct=correct,
        evidence=reading.evidence,
    )


def _ask_state(
    case: _StateCase,
    *,
    run: int,
    inference: int,
    client: OpenAI,
    model: str,
    audit: ModelAudit,
    disposition: bool,
) -> EvalResult:
    """실행 프로그램의 요청과 검증으로 상태 1개의 답을 받는다. 도구는 실행하지 않는다."""
    budget = AgentBudget()
    request = _state_request(case, disposition=disposition)
    body = _body(model, messages=request.messages, tools=request.tools)
    audit.write(
        inference=inference, kind="request", data={"case_id": case.case_id, "request": body}
    )
    # 출하할 수 있는 실행은 정답이 출하인 처분 사례뿐이다. 다른 완료 응답은 계약 위반이다.
    releasable = isinstance(case, DispositionCase) and case.expected == DispositionAction.RELEASE
    evidence = None
    action: ValidatedCall | FinalDecision | None = None
    status, failure = ResponseStatus.INVALID, None
    rejection: dict[str, object] | None = None
    try:
        observed = request_completion(
            client=client,
            audit=audit,
            inference=inference,
            model=model,
            messages=request.messages,
            tools=request.tools,
            max_tokens=budget.max_tokens,
            timeout_s=budget.nemotron_timeout_s,
        )
        completion = validate_completion(observed.response, model=model)
        evidence = completion_evidence(completion, latency_s=observed.latency_s)
        if completion.choices[0].finish_reason == "length":
            failure = "token_limit"
        else:
            action = validate_action(
                completion,
                allowed=request.allowed,
                ledger=request.ledger,
                release_allowed=lambda: releasable,
                review=True,
            )
            status = ResponseStatus.VALID
    except ToolProtocolError as exc:
        failure = exc.code.value
        if exc.code == ToolVerificationError.DIFFERENT_MODEL:
            status = ResponseStatus.NO_RESPONSE
    except OpenAIError:
        status, failure = ResponseStatus.NO_RESPONSE, "model_error"
    except ValueError as exc:
        rejection = audit.rejection(exc)
        failure = str(rejection["reason"])
    if rejection is None:
        rejection = {"accepted": True} if action else {"accepted": False, "reason": failure}
    audit.write(inference=inference, kind="validation", data=rejection)
    answer, correct = (None, False) if action is None else _score(case, action, request.ledger)
    return EvalResult(
        case_id=case.case_id,
        run=run,
        inference=inference,
        status=status,
        answer=answer,
        failure=failure,
        correct=correct,
        evidence=evidence,
    )


def _score(
    case: _StateCase, action: ValidatedCall | FinalDecision, ledger: ToolLedger
) -> tuple[str, bool]:
    """검증한 응답의 표기와 정답 여부."""
    if isinstance(case, ReviewCase):
        answer, chosen, reason = _review_answer(action)
        correct = chosen == case.expected and (
            chosen != ReviewAction.REPLAN or reason == case.mismatch
        )
        return answer, correct
    if isinstance(case, CoordinationCase):
        if isinstance(action, ValidatedCall):
            return action.name.value, action.name == case.expected
        return _final_answer(action), False
    answer, disposition = _disposition_answer(action, ledger)
    return answer, disposition == case.expected


def _review_answer(
    action: ValidatedCall | FinalDecision,
) -> tuple[str, ReviewAction | None, ReviewReason | None]:
    """검증한 응답의 표기, 행동과 재계획 사유. 다른 중단은 행동이 None이다."""
    if isinstance(action, ValidatedCall):
        if isinstance(action.arguments, ReplanLayoutArguments):
            reason = action.arguments.reason
            return f"{action.name.value}:{reason.value}", ReviewAction.REPLAN, reason
        assert isinstance(action.arguments, CalibrateBoardArguments)
        return action.name.value, ReviewAction.PROCEED, None
    assert isinstance(action, AbortDecision)
    rejected = action.reason_code == ErrorCode.PLAN_REJECTED
    return _final_answer(action), ReviewAction.REJECT if rejected else None, None


def _disposition_answer(
    action: ValidatedCall | FinalDecision, ledger: ToolLedger
) -> tuple[str, DispositionAction | None]:
    """검증한 응답의 표기와 처분. 현재 오류 코드가 아닌 중단은 처분이 None이다.

    검증을 통과한 재작업 호출의 부품은 장부의 재작업 대상과 같다.
    """
    if isinstance(action, ValidatedCall):
        assert isinstance(action.arguments, ReworkCellArguments)
        parts = ",".join(sorted(action.arguments.part_ids))
        return f"{_REWORK_ANSWER}:{parts}", DispositionAction.REWORK
    if isinstance(action, CompleteDecision):
        return _final_answer(action), DispositionAction.RELEASE
    rejected = action.reason_code == ledger.error_code
    return _final_answer(action), DispositionAction.REJECT if rejected else None


def _final_answer(final: FinalDecision) -> str:
    """최종 판단의 표기. 중단은 사유 코드를 붙인다."""
    if isinstance(final, CompleteDecision):
        return final.decision
    return f"{final.decision}:{final.reason_code.value}"


def _report(
    manifest: EvalManifest, cases: Sequence[EvalCase], results: Sequence[EvalResult]
) -> EvalReport:
    first = {result.case_id: result for result in results if result.run == 1}
    scored = [(case, first[case.case_id]) for case in cases if case.case_id in first]
    feature = manifest.config.feature
    if feature == Feature.DISPOSITION:
        # 실행 프로그램이 답의 내용을 거부한 응답은 형식이 맞다. 그 응답은 판단의 지표가 센다.
        metrics = [
            _metric(
                "format_valid",
                (
                    result.status == ResponseStatus.VALID
                    or result.failure in (_FALSE_RELEASE_FAILURE, _WRONG_PARTS_FAILURE)
                    for result in results
                ),
                _CONTRACT_VALID,
            )
        ]
        metrics += _disposition_metrics(
            [(c, r) for c, r in scored if isinstance(c, DispositionCase)]
        )
    else:
        metrics = [
            _metric(
                "contract_valid",
                (result.status == ResponseStatus.VALID for result in results),
                _CONTRACT_VALID,
            )
        ]
    if feature == Feature.INTAKE:
        metrics += _intake_metrics([(c, r) for c, r in scored if isinstance(c, IntakeCase)])
    elif feature == Feature.REVIEW:
        metrics += _review_metrics([(c, r) for c, r in scored if isinstance(c, ReviewCase)])
    elif feature == Feature.COORDINATION:
        metrics += _coordination_metrics(
            [(c, r) for c, r in scored if isinstance(c, CoordinationCase)]
        )
    stopped = len(results) < manifest.request_count
    model = manifest.config.model
    evidence = [result.evidence for result in results if result.evidence is not None]
    prompt_tokens = sum(item.prompt_tokens or 0 for item in evidence)
    completion_tokens = sum(item.completion_tokens or 0 for item in evidence)
    return EvalReport(
        eval_id=manifest.eval_id,
        feature=manifest.config.feature,
        model=model,
        case_count=manifest.case_count,
        request_count=manifest.request_count,
        completed_requests=len(results),
        stopped=stopped,
        metrics=tuple(metrics),
        consistency=_consistency(results, repeats=manifest.config.repeats),
        usage=Usage(
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            reasoning_tokens=sum(item.reasoning_tokens or 0 for item in evidence),
            cost_usd=_cost(model, prompt_tokens, completion_tokens),
        ),
        passed=not stopped and all(metric.passed is not False for metric in metrics),
    )


def _metric(name: str, flags: Iterable[bool], bound: _Bound | None = None) -> Metric:
    values = list(flags)
    count, total = sum(values), len(values)
    return Metric(
        name=name,
        count=count,
        total=total,
        criterion=None if bound is None else str(bound),
        passed=None if bound is None or total == 0 else bound.holds(count, total),
    )


def _intake_metrics(scored: Sequence[tuple[IntakeCase, EvalResult]]) -> list[Metric]:
    """기준은 영어 문장에만 건다. 한국어 문장은 수만 보고한다."""

    def group(name: IntakeGroup) -> list[tuple[IntakeCase, EvalResult]]:
        return [(case, result) for case, result in scored if case.group == name]

    accepted = IntakeOutcome.ACCEPTED.value
    out_of_scope = group(IntakeGroup.OUT_OF_SCOPE)
    return [
        _metric("kit_order_match", (r.correct for _, r in group(IntakeGroup.KIT)), _KIT_MATCH),
        _metric(
            "board_order_match", (r.correct for _, r in group(IntakeGroup.BOARD)), _BOARD_MATCH
        ),
        _metric("false_accept", (r.answer == accepted for _, r in out_of_scope), _FALSE_ACCEPT),
        _metric(
            "non_assembly_accept",
            (
                result.answer == accepted
                for case, result in out_of_scope
                if case.reject_reason == RejectReason.NOT_ASSEMBLY_REQUEST
            ),
            _NEVER,
        ),
        _metric("reject_reason_match", (r.correct for _, r in out_of_scope)),
        _metric("korean_match", (r.correct for _, r in group(IntakeGroup.KOREAN))),
    ]


def _review_metrics(scored: Sequence[tuple[ReviewCase, EvalResult]]) -> list[Metric]:
    """오검출은 올바른 계획에서 보정이 아닌 답이다. 계약과 다른 응답은 세지 않는다."""

    def group(name: ReviewGroup) -> list[tuple[ReviewCase, EvalResult]]:
        return [(case, result) for case, result in scored if case.group == name]

    def stops(result: EvalResult) -> bool:
        return result.status == ResponseStatus.VALID and result.answer != _PROCEED_ANSWER

    correct, wrong = group(ReviewGroup.CORRECT), group(ReviewGroup.WRONG)
    metrics = [
        _metric("false_detection", (stops(result) for _, result in correct), _FALSE_DETECTION),
        _metric(
            "saved_false_detection",
            (stops(result) for case, result in correct if case.saved_name is not None),
            _NEVER,
        ),
    ]
    for reason, bound in _DETECTION.items():
        metrics.append(
            _metric(
                f"detected_{reason.value}",
                (result.correct for case, result in wrong if case.mismatch == reason),
                bound,
            )
        )
    metrics += [
        _metric("wrong_plan_not_accepted", (stops(result) for _, result in wrong)),
        _metric("last_attempt_match", (r.correct for _, r in group(ReviewGroup.LAST_ATTEMPT))),
    ]
    return metrics


def _coordination_metrics(scored: Sequence[tuple[CoordinationCase, EvalResult]]) -> list[Metric]:
    """정답은 그 상태의 도구 호출이다. 중단은 계약과 같아도 정답이 아니다."""
    metrics = [_metric("tool_match", (result.correct for _, result in scored), _TOOL_MATCH)]
    for state in CoordinationState:
        metrics.append(
            _metric(
                f"tool_match_{state.value}",
                (result.correct for case, result in scored if case.group == state),
            )
        )
    return metrics


def _disposition_metrics(scored: Sequence[tuple[DispositionCase, EvalResult]]) -> list[Metric]:
    """정답 처분마다 일치한 답을 센다. 실행 프로그램이 거부한 답은 따로 센다."""

    def expecting(action: DispositionAction) -> list[EvalResult]:
        return [result for case, result in scored if case.expected == action]

    failing = [result for case, result in scored if case.expected != DispositionAction.RELEASE]
    return [
        _metric(
            "release_match",
            (result.correct for result in expecting(DispositionAction.RELEASE)),
            _RELEASE_MATCH,
        ),
        _metric(
            "rework_match",
            (result.correct for result in expecting(DispositionAction.REWORK)),
            _REWORK_MATCH,
        ),
        _metric(
            "reject_match",
            (result.correct for result in expecting(DispositionAction.REJECT)),
            _REJECT_MATCH,
        ),
        _metric(
            "false_release",
            (result.failure == _FALSE_RELEASE_FAILURE for result in failing),
            _FALSE_RELEASE,
        ),
        _metric("wrong_rework_parts", (r.failure == _WRONG_PARTS_FAILURE for r in failing)),
        _metric(
            "needless_rework",
            (
                result.answer is not None and result.answer.startswith(_REWORK_ANSWER)
                for result in expecting(DispositionAction.REJECT)
            ),
        ),
        _metric(
            "after_rework_match",
            (
                result.correct
                for case, result in scored
                if case.group == DispositionGroup.AFTER_REWORK
            ),
        ),
    ]


def _consistency(results: Sequence[EvalResult], *, repeats: int) -> Consistency | None:
    """모든 실행에 응답이 있는 반복 사례만 센다. 응답 없는 실행은 모델의 답이 아니다."""
    if repeats < 2:
        return None
    runs: dict[str, list[EvalResult]] = {}
    for result in results:
        runs.setdefault(result.case_id, []).append(result)
    repeated = [
        items
        for items in runs.values()
        if len(items) == repeats
        and all(item.status != ResponseStatus.NO_RESPONSE for item in items)
    ]
    return Consistency(
        cases=len(repeated),
        runs=repeats,
        agreed=sum(1 for items in repeated if len({_answer_key(item) for item in items}) == 1),
    )


def _answer_key(result: EvalResult) -> tuple[object, ...]:
    order = None if result.order is None else order_key(result.order)
    return (result.status, result.answer, result.failure, order)
