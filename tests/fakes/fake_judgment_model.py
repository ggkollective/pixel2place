"""대기열 또는 응답기의 응답을 반환하는 판단 모델 HTTP Fake."""

import json
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from typing import Any

import httpx2 as httpx

from pixel2place.agent.eval_cases import (
    DispositionAction,
    DispositionCase,
    EvalCase,
    IntakeCase,
    ReviewAction,
    ReviewCase,
    place_reports,
)
from pixel2place.agent.inspection import inspected_summary, summarize_execution
from pixel2place.agent.intake import IntakeTool
from pixel2place.config import DEFAULT_JUDGMENT_MODEL, DEFAULT_NEMOTRON_MODEL
from pixel2place.order import KitOrder

FAKE_JUDGMENT_KEY = "judgment-test-key-never-store"
FAKE_JUDGMENT_REASONING = "judgment-private-reasoning-never-store"


@dataclass(frozen=True)
class Reply:
    """모델 요청 1회의 응답. `tool`이 없으면 `content`만 있는 메시지다."""

    tool: str | None = None
    arguments: dict[str, Any] | str | None = None
    """dict는 JSON으로 쓴다. 문자열은 그대로 쓴다. 손상된 JSON을 만들 때 쓴다."""
    content: str | None = None
    status: int = 200
    finish: str | None = None
    model: str | None = None
    call_count: int = 1


type Responder = Callable[[dict[str, Any]], Reply]
"""모델 요청의 본문을 받아서 응답을 정한다."""


def runtime_state(body: dict[str, Any]) -> dict[str, Any]:
    """계획 검토 요청의 지시문 끝에 있는 실행 프로그램 상태."""
    return json.loads(body["messages"][0]["content"].split("Current runtime state: ")[1])


def correct_replies(cases: Sequence[EvalCase]) -> Responder:
    """요청에서 사례를 찾아서 정답대로 답하는 응답기를 만든다.

    - 접수 요청은 사용자 문장으로 찾는다.
    - 검토 요청은 목표, 준비 시도 수와 부품 목록으로 찾는다.
    - 처분 요청은 목표, 재작업 실행의 수와 마지막 도구 결과의 요약으로 찾는다.
    - 조율 요청의 정답은 요청의 상태가 정한다. 사례를 찾지 않는다.
    """
    texts: dict[str, IntakeCase] = {}
    plans: dict[tuple[object, ...], ReviewCase] = {}
    dispositions: dict[tuple[object, ...], DispositionCase] = {}
    for case in cases:
        if isinstance(case, IntakeCase):
            texts[case.text] = case
        elif isinstance(case, ReviewCase):
            parts = tuple(
                (
                    part.type.value,
                    part.color.value,
                    int(part.rotation_deg),
                    min(row for row, _ in part.cells),
                    min(col for _, col in part.cells),
                )
                for part in case.layout.parts
            )
            plans[(case.goal, case.layout_attempts, parts)] = case
        elif isinstance(case, DispositionCase):
            check, report = place_reports(case.layout, case.run)
            summary = inspected_summary(summarize_execution(check), report=report)
            key = (case.goal, int(case.first is not None), _summary_key(summary.model_dump()))
            dispositions[key] = case

    def respond(body: dict[str, Any]) -> Reply:
        messages = body["messages"]
        if "Current runtime state: " not in messages[0]["content"]:
            return intake_reply(texts[messages[-1]["content"]])
        state = runtime_state(body)
        if state["state"] == "checked":
            summary = json.loads(messages[-1]["content"])["summary"]
            key = (messages[1]["content"], state["rework_runs"], _summary_key(summary))
            return disposition_reply(dispositions[key], state)
        if state["state"] != "layout_ready":
            return coordination_reply(state)
        parts = tuple(
            (part["type"], part["color"], part["rotation_deg"], part["row"], part["col"])
            for part in json.loads(messages[-1]["content"])["summary"]["parts"]
        )
        case = plans[(messages[1]["content"], state["layout_attempts"], parts)]
        return review_reply(case.expected, state, reason=case.mismatch)

    return respond


def _summary_key(summary: dict[str, Any]) -> str:
    """도구 결과의 요약을 비교하는 값. JSON으로 쓴 값과 모델의 값이 같은 문자열이 된다."""
    return json.dumps(summary, sort_keys=True, default=str)


def intake_reply(case: IntakeCase) -> Reply:
    """접수 사례의 정답 주문 또는 정답 거절."""
    if case.order is None:
        assert case.reject_reason is not None
        return Reply(
            tool=IntakeTool.REJECT_REQUEST.value, arguments={"reason": case.reject_reason.value}
        )
    tool = (
        IntakeTool.SUBMIT_KIT_ORDER
        if isinstance(case.order, KitOrder)
        else IntakeTool.SUBMIT_BOARD_ORDER
    )
    return Reply(tool=tool.value, arguments=case.order.model_dump(mode="json"))


def review_reply(
    action: ReviewAction, state: dict[str, Any], *, reason: str | None = None
) -> Reply:
    """현재 레이아웃 ID로 보정, 재계획 또는 계획 거부를 답한다."""
    if action == ReviewAction.PROCEED:
        return Reply(tool="calibrate_board", arguments={"layout_id": state["layout_id"]})
    if action == ReviewAction.REPLAN:
        return Reply(
            tool="replan_layout", arguments={"layout_id": state["layout_id"], "reason": reason}
        )
    return Reply(content='{"decision":"abort","reason_code":"plan_rejected"}')


def coordination_reply(state: dict[str, Any]) -> Reply:
    """요청의 상태에 맞는 준비 또는 실행 호출."""
    if state["state"] == "calibrated":
        arguments = {"layout_id": state["layout_id"], "calibration_id": state["calibration_id"]}
        return Reply(tool="execute_cell", arguments=arguments)
    return Reply(tool="prepare_layout", arguments={"retry": state["state"] == "layout_retryable"})


def disposition_reply(case: DispositionCase, state: dict[str, Any]) -> Reply:
    """처분 사례의 정답. 출하, 정답 부품의 재작업 또는 현재 오류 코드의 중단이다."""
    if case.expected == DispositionAction.RELEASE:
        return Reply(content='{"decision":"complete","reason_code":"check_passed"}')
    if case.expected == DispositionAction.REWORK:
        return Reply(tool="rework_cell", arguments={"part_ids": list(case.expected_parts)})
    return Reply(content=json.dumps({"decision": "abort", "reason_code": state["error_code"]}))


class FakeJudgmentModel:
    """모델 목록과 모델 요청에 답한다.

    응답기가 있으면 요청 본문으로 응답을 정한다. 없으면 대기열의 다음 응답을 쓴다.
    """

    def __init__(
        self, replies: Sequence[Reply] = (), *, responder: Responder | None = None
    ) -> None:
        self.replies = list(replies)
        self.responder = responder
        self.requests: list[httpx.Request] = []
        self.clients: list[httpx.Client] = []
        self.models = [DEFAULT_NEMOTRON_MODEL, DEFAULT_JUDGMENT_MODEL]
        self.catalog_statuses: list[int] = []
        """모델 목록 요청의 HTTP 상태의 대기열. 비면 목록을 반환한다."""

    def make_client(self, *, follow_redirects: bool = False) -> httpx.Client:
        """mock transport를 가진 클라이언트를 만들고 종료 여부를 보존한다."""
        client = httpx.Client(
            transport=httpx.MockTransport(self.handle), follow_redirects=follow_redirects
        )
        self.clients.append(client)
        return client

    @property
    def model_posts(self) -> list[httpx.Request]:
        """모델 목록을 제외한 모델 요청."""
        return [request for request in self.requests if request.method == "POST"]

    def handle(self, request: httpx.Request) -> httpx.Response:
        """인증 헤더를 검사한다. 응답기가 없고 대기열이 비면 테스트가 실패한다."""
        self.requests.append(request)
        assert request.headers["authorization"] == f"Bearer {FAKE_JUDGMENT_KEY}"
        if request.method == "GET":
            if self.catalog_statuses:
                status = self.catalog_statuses.pop(0)
                return httpx.Response(status, json={"error": {"message": FAKE_JUDGMENT_KEY}})
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
        reply = (
            self.responder(json.loads(request.content))
            if self.responder is not None
            else self.replies.pop(0)
        )
        if reply.status != 200:
            return httpx.Response(reply.status, json={"error": {"message": FAKE_JUDGMENT_KEY}})
        index = len(self.model_posts)
        message: dict[str, Any] = {"role": "assistant", "content": reply.content}
        if reply.tool is not None:
            arguments = reply.arguments
            call = {
                "id": f"call-{index}",
                "type": "function",
                "function": {
                    "name": reply.tool,
                    "arguments": arguments if isinstance(arguments, str) else json.dumps(arguments),
                },
            }
            message["tool_calls"] = [call] * reply.call_count
            message["reasoning_content"] = FAKE_JUDGMENT_REASONING
        return httpx.Response(
            200,
            json={
                "id": f"completion-{index}",
                "object": "chat.completion",
                "created": 0,
                "model": reply.model or json.loads(request.content)["model"],
                "choices": [
                    {
                        "index": 0,
                        "message": message,
                        "finish_reason": reply.finish
                        or ("tool_calls" if reply.tool is not None else "stop"),
                    }
                ],
                "usage": {
                    "prompt_tokens": 100,
                    "completion_tokens": 50,
                    "total_tokens": 150,
                    "completion_tokens_details": {"reasoning_tokens": 20},
                },
            },
        )
