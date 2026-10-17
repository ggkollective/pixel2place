"""보존한 준비 성공 결과에서 두 번째 모델 요청만 비교한다."""

import hashlib
import json
from pathlib import Path
from typing import Annotated, Literal
from uuid import uuid4

from openai import OpenAI, OpenAIError
from openai.types.chat import ChatCompletionMessageParam
from pydantic import Field

from pixel2place.agent.artifacts import write_record
from pixel2place.agent.audit import ModelAudit
from pixel2place.agent.contracts import (
    Contract,
    LayoutResult,
    PrepareLayoutArguments,
    ToolName,
    ToolState,
)
from pixel2place.agent.inference import ToolChoice, completion_evidence, request_completion
from pixel2place.agent.prompts import STATE_GUIDANCE, SYSTEM_PROMPT
from pixel2place.agent.runtime import request_messages, tool_schema, validate_action
from pixel2place.agent.runtime_contracts import (
    AgentBudget,
    AgentEvent,
    FinalDecision,
    RequestEvidence,
    ValidatedCall,
)
from pixel2place.agent.tools import ToolLedger, ToolRun
from pixel2place.config import DEFAULT_NEMOTRON_MODEL, TokenFactorySettings
from pixel2place.reasoner.nemotron import create_nemotron_client, require_nemotron_model
from pixel2place.reasoner.nemotron_tools import ToolProtocolError, validate_completion

PROBE_CASES = ("baseline", "state_guidance", "strict", "named")
CONFIRM_CASES = ("fixed", "fixed_abort")


class ProbeResult(Contract):
    """독립 요청의 계약 판정. 도구는 실행하지 않는다."""

    case: str
    accepted: bool
    selected_tool: ToolName | None = None
    final: FinalDecision | None = None
    evidence: RequestEvidence | None = None
    reason: str | None = None
    wire_matches_sdk: bool | None = None


class ProbeReport(Contract):
    """모델 요청만 실행한 비교다. M3 완료나 로봇 성능을 뜻하지 않는다."""

    schema_version: Literal[1] = 1
    run_id: str
    source_run_id: str
    model: str = DEFAULT_NEMOTRON_MODEL
    max_tokens: Literal[2048] = 2048
    timeout_s: Annotated[float, Field(ge=120, le=120)] = 120.0
    sdk_retries: Literal[0] = 0
    request_count: int
    results: tuple[ProbeResult, ...]


def load_probe_context(
    source: Path,
) -> tuple[ToolRun, ToolLedger, list[ChatCompletionMessageParam]]:
    """공칭·오차 기록의 첫 준비 성공만 읽는다. 장면과 Isaac 파일은 읽지 않는다."""
    run = ToolRun.model_validate_json((source / "run.json").read_text(encoding="utf-8"))
    if run.agent is None or run.agent.model != DEFAULT_NEMOTRON_MODEL:
        raise ValueError("The source run must use Lightning.")
    events = [
        AgentEvent.model_validate_json(line)
        for line in (source / "events.jsonl").read_text(encoding="utf-8").splitlines()
    ]
    prepared = next((event for event in events if event.kind == "tool_result"), None)
    if (
        prepared is None
        or prepared.run_id != run.run_id
        or prepared.from_state != ToolState.CREATED
        or prepared.to_state != ToolState.LAYOUT_READY
        or prepared.call is None
        or prepared.call.name != ToolName.PREPARE_LAYOUT
        or not isinstance(prepared.call.arguments, PrepareLayoutArguments)
        or prepared.call.arguments.retry
        or not isinstance(prepared.result, LayoutResult)
        or prepared.result.run_id != run.run_id
        or prepared.result.tool_call_id != prepared.call.id
        or len(prepared.result.artifacts) != 1
        or prepared.result.artifacts[0].kind != "layout"
    ):
        raise ValueError("The source must contain a valid first preparation result.")
    saved = LayoutResult.model_validate_json(
        (source / "tool_results" / "01.json").read_text(encoding="utf-8")
    )
    if saved != prepared.result:
        raise ValueError("The preparation result does not match the event.")
    ledger = ToolLedger(
        run_id=run.run_id,
        state=ToolState.LAYOUT_READY,
        layout_attempts=1,
        result_count=1,
        layout_id=saved.artifacts[0].id,
        call_ids=(prepared.call.id,),
    )
    messages: list[ChatCompletionMessageParam] = [
        {"role": "user", "content": run.prompt},
        {
            "role": "assistant",
            "tool_calls": [
                {
                    "id": prepared.call.id,
                    "type": "function",
                    "function": {
                        "name": prepared.call.name.value,
                        "arguments": prepared.call.arguments.model_dump_json(),
                    },
                }
            ],
        },
        {"role": "tool", "tool_call_id": prepared.call.id, "content": saved.model_dump_json()},
    ]
    return run, ledger, messages


def probe_case(
    *,
    client: OpenAI,
    audit: ModelAudit,
    inference: int,
    case: str,
    ledger: ToolLedger,
    messages: list[ChatCompletionMessageParam],
    system_prompt: str,
) -> ProbeResult:
    """조건 하나만 바꾼다. 운영 validator로 응답을 검사하고 dispatch하지 않는다."""
    budget = AgentBudget()
    schema = tool_schema(ToolName.CALIBRATE_BOARD, strict=case in ("strict", *CONFIRM_CASES))
    prompt = system_prompt + STATE_GUIDANCE if case == "state_guidance" else system_prompt
    request = request_messages(ledger=ledger, messages=messages, system_prompt=prompt)
    if case == "fixed_abort":
        request.append({"role": "user", "content": "Abort this run now. Do not call a tool."})
    choice: ToolChoice = (
        {"type": "function", "function": {"name": "calibrate_board"}} if case == "named" else "auto"
    )
    body: dict[str, object] = {
        "model": DEFAULT_NEMOTRON_MODEL,
        "messages": request,
        "tools": [schema],
        "tool_choice": choice,
        "max_tokens": budget.max_tokens,
    }
    audit.write(inference=inference, kind="request", data={"case": case, "request": body})
    evidence = None
    matches = None
    reason = None
    accepted = False
    action: ValidatedCall | FinalDecision | None = None
    try:
        observed = request_completion(
            client=client,
            audit=audit,
            inference=inference,
            model=DEFAULT_NEMOTRON_MODEL,
            messages=request,
            tools=[schema],
            tool_choice=choice,
            max_tokens=budget.max_tokens,
            timeout_s=budget.nemotron_timeout_s,
        )
        matches = observed.wire_matches_sdk
        completion = validate_completion(observed.response, model=DEFAULT_NEMOTRON_MODEL)
        evidence = completion_evidence(completion, latency_s=observed.latency_s)
        if completion.choices[0].finish_reason == "length":
            reason = "token_limit"
        else:
            action = validate_action(
                completion,
                allowed=(ToolName.CALIBRATE_BOARD,),
                ledger=ledger,
                release_allowed=lambda: False,
            )
            accepted = True
    except ToolProtocolError as exc:
        reason = exc.code.value
    except OpenAIError:
        reason = "model_error"
    except ValueError as exc:
        rejection = audit.rejection(exc)
        audit.write(inference=inference, kind="validation", data=rejection)
        return ProbeResult(
            case=case,
            accepted=False,
            evidence=evidence,
            reason=str(rejection["reason"]),
            wire_matches_sdk=matches,
        )
    audit.write(
        inference=inference, kind="validation", data={"accepted": accepted, "reason": reason}
    )
    return ProbeResult(
        case=case,
        accepted=accepted,
        selected_tool=action.name if isinstance(action, ValidatedCall) else None,
        final=action if action is not None and not isinstance(action, ValidatedCall) else None,
        evidence=evidence,
        reason=reason,
        wire_matches_sdk=matches,
    )


def run_transition_probe(
    *, source: Path, out_dir: Path, settings: TokenFactorySettings, confirm: bool = False
) -> ProbeReport:
    """네 조건을 비교하거나 현재 수정의 진행·중단을 두 번 확인한다."""
    run, ledger, messages = load_probe_context(source)
    assert run.agent is not None
    cases = CONFIRM_CASES if confirm else PROBE_CASES
    system_prompt = SYSTEM_PROMPT if confirm else run.agent.system_prompt
    out_dir.mkdir(parents=True, exist_ok=False)
    budget = AgentBudget()
    configured = settings.model_copy(
        update={
            "nemotron_model": DEFAULT_NEMOTRON_MODEL,
            "nemotron_timeout_s": budget.nemotron_timeout_s,
        }
    )
    probe_id = uuid4().hex
    audit = ModelAudit(
        root=out_dir, run_id=probe_id, secret=configured.nebius_api_key.get_secret_value()
    )
    metadata = {
        "source_run_id": run.run_id,
        "model": DEFAULT_NEMOTRON_MODEL,
        "system_prompt": system_prompt,
        "state_guidance": STATE_GUIDANCE,
        "cases": cases,
        "budget": budget.model_dump(mode="json"),
        "source_sha256": {
            name: hashlib.sha256((source / name).read_bytes()).hexdigest()
            for name in ("run.json", "events.jsonl", "tool_results/01.json")
        },
    }
    (out_dir / "probe.json").write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")
    results: list[ProbeResult] = []
    with create_nemotron_client(configured) as client:
        require_nemotron_model(client, model=DEFAULT_NEMOTRON_MODEL)
        for inference, case in enumerate(cases, 1):
            result = probe_case(
                client=client,
                audit=audit,
                inference=inference,
                case=case,
                ledger=ledger,
                messages=messages,
                system_prompt=system_prompt,
            )
            results.append(result)
            write_record(out_dir / f"{case}.json", result)
    report = ProbeReport(
        run_id=probe_id,
        source_run_id=run.run_id,
        request_count=len(results),
        results=tuple(results),
    )
    write_record(out_dir / "probe_report.json", report)
    return report
