"""현재 상태와 도구 결과를 읽는 조율 프로그램 HTTP Fake."""

import json
from collections.abc import Callable
from typing import Any

import httpx2 as httpx

from pixel2place.config import DEFAULT_JUDGMENT_MODEL, DEFAULT_NEMOTRON_MODEL

FAKE_AGENT_KEY = "agent-test-key-never-store"
FAKE_REASONING = "agent-private-reasoning-never-store"


def _empty_parts(body: dict[str, Any]) -> list[str]:
    """마지막 도구 결과의 검사 요약에서 불일치이고 빈 칸인 부품의 ID."""
    inspection = json.loads(body["messages"][-1]["content"])["summary"]["inspection"]
    return [
        part["part_id"]
        for part in inspection["parts"]
        if part["status"] == "mismatch" and part["empty"]
    ]


class FakeAgentModel:
    """SDK 요청을 읽는다. 도구 호출 1개 또는 최종 JSON을 반환한다."""

    def __init__(self) -> None:
        self.requests: list[httpx.Request] = []
        self.clients: list[httpx.Client] = []
        self.variant = "success"
        self.variant_stage = 1
        self.http_failures: int | None = None
        """HTTP 결함을 넣는 연속 요청의 수. None이면 `variant_stage`의 요청에만 넣는다.

        재전송은 모델 요청의 수를 늘린다. 같은 모델 요청의 여러 시도를 실패시킬 때 이 값을 쓴다.
        """
        self._http_failed = 0
        self.call_prefix = "call"
        self.models = [DEFAULT_NEMOTRON_MODEL, DEFAULT_JUDGMENT_MODEL]
        self.cosmos_response = ""
        self.replans: list[str] = []
        """재계획 사유의 대기열. `replan_layout`을 제공받은 요청에서 앞의 사유를 쓴다."""
        self.dispositions: list[list[str] | None] = []
        """재작업 호출의 대기열. `rework_cell`을 제공받은 요청에서 앞의 값을 쓴다.

        값은 호출에 넣는 부품 ID의 목록이다.
        None이면 마지막 도구 결과의 검사 요약에서 불일치이고 빈 칸인 부품을 고른다.
        대기열이 비면 `checked`에서 최종 판단을 낸다.
        """
        self.on_request: Callable[[httpx.Request], None] | None = None

    def make_client(self, *, follow_redirects: bool = False) -> httpx.Client:
        """mock transport를 가진 클라이언트를 만들고 종료 여부를 보존한다."""
        client = httpx.Client(
            transport=httpx.MockTransport(self.handle), follow_redirects=follow_redirects
        )
        self.clients.append(client)
        return client

    @property
    def model_posts(self) -> list[httpx.Request]:
        """Cosmos와 모델 목록을 제외한 Nemotron 요청."""
        return [r for r in self.requests if r.method == "POST" and r.url.host != "cosmos.invalid"]

    def handle(self, request: httpx.Request) -> httpx.Response:
        """요청 내용을 읽고 응답한다. 필요하면 HTTP 오류와 계약 오류를 만든다."""
        self.requests.append(request)
        if self.on_request is not None:
            self.on_request(request)
        if request.url.host == "cosmos.invalid":
            body = json.loads(request.content)
            return self._response(
                body["model"], {"role": "assistant", "content": self.cosmos_response}, "stop", 0
            )
        assert request.headers["authorization"] == f"Bearer {FAKE_AGENT_KEY}"
        stage = len(self.model_posts)
        active = stage == self.variant_stage
        if self.variant.startswith("http_") and self._http_fault_due(stage):
            if self.variant == "http_timeout":
                raise httpx.ReadTimeout(FAKE_AGENT_KEY, request=request)
            if self.variant == "http_connection":
                raise httpx.ConnectError(FAKE_AGENT_KEY, request=request)
            status = int(self.variant.removeprefix("http_"))
            return httpx.Response(status, json={"error": {"message": FAKE_AGENT_KEY}})
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
        state = json.loads(body["messages"][0]["content"].split("Current runtime state: ")[1])
        error = state["error_code"]
        message: dict[str, Any]
        arguments: dict[str, object]
        offered = [tool["function"]["name"] for tool in body.get("tools", [])]
        rework = bool(self.dispositions) and "rework_cell" in offered
        if state["state"] in ("created", "layout_retryable"):
            name = "prepare_layout"
            arguments = {"retry": state["state"] == "layout_retryable"}
        elif state["state"] == "layout_ready":
            name = "calibrate_board"
            arguments = {"layout_id": state["layout_id"]}
            if self.replans and "replan_layout" in offered:
                name = "replan_layout"
                arguments["reason"] = self.replans.pop(0)
        elif rework:
            name = "rework_cell"
            chosen = self.dispositions.pop(0)
            arguments = {"part_ids": _empty_parts(body) if chosen is None else chosen}
        else:
            name = "execute_cell"
            arguments = {"layout_id": state["layout_id"], "calibration_id": state["calibration_id"]}
        if (state["state"] == "checked" and not rework) or (
            active
            and self.variant
            in (
                "abort",
                "plan_rejected",
                "early_complete",
                "bad_abort",
                "plain_final",
                "extra_final",
                "duplicate_final",
            )
        ):
            final = (
                {"decision": "abort", "reason_code": error or "agent_aborted"}
                if error
                else {"decision": "complete", "reason_code": "check_passed"}
            )
            if active and self.variant == "abort":
                final = {"decision": "abort", "reason_code": "agent_aborted"}
            if active and self.variant == "bad_abort":
                final = {"decision": "abort", "reason_code": "runner_error"}
            if active and self.variant == "plan_rejected":
                final = {"decision": "abort", "reason_code": "plan_rejected"}
            if active and self.variant == "early_complete":
                final = {"decision": "complete", "reason_code": "check_passed"}
            if active and self.variant == "extra_final":
                final["extra"] = "invalid"
            content = "All done." if active and self.variant == "plain_final" else json.dumps(final)
            if active and self.variant == "duplicate_final":
                content = '{"decision":"abort","decision":"complete","reason_code":"check_passed"}'
            message = {"role": "assistant", "content": content}
            finish = "stop"
        else:
            if active and self.variant == "repeat_prepare":
                name = "prepare_layout"
                arguments = {"retry": "True"}
            if active and self.variant == "wrong_tool":
                name = "execute_cell"
            if active and self.variant in ("replan_unoffered", "replan_wrong_id"):
                name = "replan_layout"
                arguments = {"layout_id": state["layout_id"], "reason": "part_set_mismatch"}
            if active and self.variant == "replan_wrong_id":
                arguments["layout_id"] = "0" * 32
            if active and self.variant == "extra_arguments":
                arguments["command"] = "never execute"
            if active and self.variant == "coercion":
                arguments["retry"] = "false"
            if active and self.variant == "wrong_id":
                arguments["layout_id"] = "0" * 32
            if active and self.variant == "wrong_retry":
                arguments["retry"] = True
            if active and self.variant == "wrong_calibration":
                arguments["calibration_id"] = "0" * 32
            call = {
                "id": f"{self.call_prefix}-{stage}",
                "type": "function",
                "function": {
                    "name": name,
                    "arguments": "{"
                    if active and self.variant == "bad_json"
                    else json.dumps(arguments),
                },
            }
            if active and self.variant == "duplicate_id":
                call["id"] = "call-1"
            if active and self.variant == "blank_id":
                call["id"] = " "
            calls = [call, call] if active and self.variant == "multiple_tools" else [call]
            message = {
                "role": "assistant",
                "content": None,
                "tool_calls": calls,
                "reasoning_content": FAKE_REASONING,
            }
            if active and self.variant == "free_text":
                message["content"] = "Execute now."
            finish = "tool_calls"
        if active and self.variant == "truncated":
            finish = "length"
        if active and self.variant == "refusal":
            message["refusal"] = "Request refused."
        if active and self.variant == "legacy_call":
            message["function_call"] = {"name": "execute_cell", "arguments": "{}"}
        if active and self.variant == "empty_final":
            message = {"role": "assistant", "content": None}
            finish = "stop"
        model = (
            "nvidia/Nemotron-other"
            if active and self.variant == "different_model"
            else body["model"]
        )
        return self._response(model, message, finish, stage)

    def _http_fault_due(self, stage: int) -> bool:
        """이 요청에 HTTP 결함을 넣을지 정한다.

        `http_failures`가 None이면 대상은 `variant_stage`번째 모델 요청이다.
        `variant_stage`가 0이면 모든 요청이다.
        값이 있으면 대상은 그 첫 요청부터 연속한 `http_failures`개의 요청이다.
        """
        starts = self.variant_stage in (0, stage)
        if self.http_failures is None:
            return starts
        if self._http_failed >= self.http_failures or not (starts or self._http_failed):
            return False
        self._http_failed += 1
        return True

    def _response(
        self, model: str, message: dict[str, Any], finish: str, stage: int
    ) -> httpx.Response:
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
        if stage == self.variant_stage and self.variant == "empty_choices":
            payload["choices"] = []
        if stage == self.variant_stage and self.variant == "multiple_choices":
            payload["choices"] *= 2
        if stage == self.variant_stage and self.variant == "missing_id":
            del payload["id"]
        if self.variant == "no_usage":
            del payload["usage"]
        return httpx.Response(200, json=payload)
