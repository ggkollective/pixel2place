"""작업 실행 예산, 이벤트와 최종 판단 계약."""

from datetime import datetime
from enum import StrEnum
from typing import Annotated, Literal, Self

from pydantic import Field, model_validator

from pixel2place.agent.contracts import (
    CalibrateBoardArguments,
    CallId,
    Contract,
    Count,
    ErrorCode,
    ExecuteCellArguments,
    Measure,
    OpaqueId,
    PrepareLayoutArguments,
    ReplanLayoutArguments,
    ReworkCellArguments,
    ToolName,
    ToolResult,
    ToolState,
)

MAX_REQUEST_RETRIES = 3
"""Nemotron 요청 1개를 다시 보내는 횟수의 상한."""


class AgentBudget(Contract):
    """외부 호출 한도. 테스트는 호출 수와 자료 한도를 낮출 수 있다."""

    max_inferences: Annotated[int, Field(ge=1, le=6)] = 6
    max_tokens: Literal[2048] = 2048
    nemotron_timeout_s: Annotated[float, Field(ge=120, le=120)] = 120.0
    cosmos_timeout_s: Annotated[float, Field(ge=900, le=900)] = 900.0
    cosmos_max_tokens: Literal[4096] = 4096
    sdk_retries: Literal[0] = 0
    """SDK의 자동 재시도 수. Nemotron과 Cosmos의 클라이언트가 함께 쓴다."""
    request_retries: Annotated[int, Field(ge=0, le=MAX_REQUEST_RETRIES)] = 0
    """일시 오류가 난 Nemotron 요청을 다시 보내는 횟수. 0이면 다시 보내지 않는다."""
    max_layout_attempts: Literal[2] = 2
    cosmos_requests_per_attempt: Literal[2] = 2
    max_capture_runs: Literal[1] = 1
    max_place_runs: Literal[1, 2] = 1
    """배치 실행의 한도. 2는 첫 배치와 재작업 1회다."""
    sim_timeout_s: Annotated[float, Field(ge=1800, le=1800)] = 1800.0
    max_tool_bytes: Annotated[int, Field(ge=1, le=8192)] = 8192
    max_request_bytes: Annotated[int, Field(ge=1, le=131072)] = 131072


class AgentMetadata(Contract):
    """입력 파일을 manifest에 등록하기 전에 고정하는 모델과 호출 조건."""

    model: str
    cosmos_model: str
    budget: AgentBudget
    system_prompt: str
    review: bool = False
    """계획 검토를 켠 실행이다. 준비 결과에 부품 목록이 있고 `replan_layout`을 쓸 수 있다."""
    review_model: str | None = None
    """`layout_ready`의 요청을 보내는 모델. None이면 `model`을 쓴다."""
    disposition: bool = False
    """처분을 켠 실행이다. 실행 결과에 카메라 판정이 있고 `rework_cell`을 쓸 수 있다.

    출하는 참값 판정과 카메라 판정의 통과를 모두 요구한다.
    """
    disposition_model: str | None = None
    """처분을 켠 실행에서 `checked`의 요청을 보내는 모델. None이면 `model`을 쓴다."""

    @model_validator(mode="after")
    def _disposition_has_a_rework_run(self) -> Self:
        if self.disposition and self.budget.max_place_runs != 2:
            raise ValueError("Disposition requires a limit of two placement runs.")
        return self


class CompleteDecision(Contract):
    """실제 판정 통과를 요구하는 모델 완료 요청."""

    decision: Literal["complete"]
    reason_code: Literal["check_passed"]


class AbortDecision(Contract):
    """현재 오류 또는 자발적 중단의 모델 판단."""

    decision: Literal["abort"]
    reason_code: ErrorCode


type FinalDecision = Annotated[CompleteDecision | AbortDecision, Field(discriminator="decision")]
type ToolArguments = (
    PrepareLayoutArguments
    | CalibrateBoardArguments
    | ExecuteCellArguments
    | ReplanLayoutArguments
    | ReworkCellArguments
)


class ValidatedCall(Contract):
    """실행 전에 검증한 단일 도구 호출."""

    id: CallId
    name: ToolName
    arguments: ToolArguments


class RequestEvidence(Contract):
    """reasoning 원문과 응답 본문을 제외한 요청 결과."""

    completion_id: str
    model: str
    finish_reason: str
    latency_s: Annotated[float, Field(ge=0)]
    prompt_tokens: Count | None
    completion_tokens: Count | None
    reasoning_tokens: Count | None


type EventKind = Literal[
    "catalog_request", "catalog_result", "model_request", "model_result", "tool_result", "final"
]


class AgentEvent(Contract):
    """외부 요청과 상태 전이의 순서 기록."""

    schema_version: Literal[1] = 1
    run_id: OpaqueId
    sequence: Count
    kind: EventKind
    inference: Count
    from_state: ToolState
    to_state: ToolState
    allowed_tools: tuple[ToolName, ...] = ()
    model: str
    timeout_s: float
    evidence: RequestEvidence | None = None
    call: ValidatedCall | None = None
    result: ToolResult | None = None
    final: FinalDecision | None = None
    error_code: ErrorCode | None = None


class EventTiming(Contract):
    """이벤트 1개를 쓴 시각. `sequence`가 같은 `AgentEvent`와 짝이다.

    이벤트의 형식은 시각을 갖지 않는다. 시각은 따로 쓴다.
    """

    sequence: Count
    kind: EventKind
    utc: datetime
    elapsed_s: Measure
    """`AgentRuntime.run`의 시작부터 이 이벤트까지의 초."""


class CallCounts(Contract):
    """시작한 요청 수와 소비한 외부 작업 수."""

    catalog: Count
    nemotron: Count
    cosmos: Count
    capture: Count
    place: Count
    """배치 실행의 수. 재작업 실행도 센다."""
    request_bytes: Count
    """보낸 Nemotron 요청의 JSON 크기의 합. 다시 보낸 요청도 센다."""
    retries: Count = 0
    """다시 보낸 Nemotron 요청의 수. 모델 목록 요청과 모델 요청을 함께 센다.

    `catalog`와 `nemotron`은 이 수를 포함하지 않는다.
    """


class Disposition(StrEnum):
    """끝난 실행 1건의 처분."""

    RELEASED = "released"
    """재작업 없이 출하했다."""
    REWORKED_RELEASED = "reworked_released"
    """재작업 뒤에 출하했다."""
    REJECTED = "rejected"
    """불량이다. 참값 판정 또는 카메라 판정이 실패한 채로 끝났다."""
    ABORTED = "aborted"
    """중단이다. 실행이 판정의 앞에서 끝났거나 다른 오류로 끝났다."""


_REJECTED_CODES = (ErrorCode.CHECK_FAILED, ErrorCode.INSPECTION_REJECTED)


class AgentReport(Contract):
    """완료 또는 중단과 연결된 판정 ID. HTTP 오류 원문은 없다."""

    schema_version: Literal[1] = 1
    run_id: OpaqueId
    state: Literal[ToolState.COMPLETED, ToolState.ABORTED]
    error_code: ErrorCode | None
    calls: CallCounts
    final_decision: FinalDecision | None
    check_id: OpaqueId | None
    check_passed: bool | None
    """마지막 배치 실행의 판정이다. 재작업을 한 실행은 재작업 뒤의 판정이다."""
    reworked: bool = False
    """재작업 실행을 했다. 이 값이 없는 보고는 False로 읽는다."""

    @property
    def disposition(self) -> Disposition:
        """종료 상태, 오류 코드와 재작업 여부에서 구한 처분. 저장하지 않는다."""
        if self.state == ToolState.COMPLETED:
            return Disposition.REWORKED_RELEASED if self.reworked else Disposition.RELEASED
        if self.error_code in _REJECTED_CODES:
            return Disposition.REJECTED
        return Disposition.ABORTED
