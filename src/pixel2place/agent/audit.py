"""모델 요청, HTTP·SDK 도구 응답과 판정의 안전한 관찰."""

import hashlib
import json
import re
from collections.abc import Iterator, Mapping
from pathlib import Path

from pydantic import BaseModel, ValidationError

from pixel2place.agent.schema import strict_parameters

AUDIT_FILE = "model_audit.jsonl"
REDACTED = "[redacted]"
MAX_OBSERVED_ITEMS = 16
MAX_ARGUMENT_BYTES = 8192
_ARGUMENT_NAMES = frozenset(("retry", "layout_id", "calibration_id", "reason", "part_ids"))
_SAFE_ID = re.compile(r"^[a-f0-9]{32}$")
_SAFE_CODE = re.compile(r"^[a-z_]{1,64}$")
_SAFE_PART_ID = re.compile(r"^[A-Za-z0-9_-]{1,64}$")
_SAFE_LABEL = re.compile(r"^[A-Za-z0-9_./-]{1,256}$")
_REASONS = {
    "Tool call is not allowed.": "tool_not_allowed",
    "Expected one tool call.": "invalid_call_count",
    "Expected a function call without free text.": "invalid_call_message",
    "Model refused the request.": "refusal",
    "Legacy function calls are not allowed.": "legacy_call",
    "Layout retry does not match the current state.": "invalid_retry_state",
    "Layout is not current.": "invalid_layout_id",
    "Calibration is not current.": "invalid_calibration_id",
    "Rework parts are not the rework targets.": "invalid_rework_parts",
    "JSON contains a duplicate key.": "duplicate_json_key",
    "Expected a final JSON decision.": "invalid_final_message",
    "Completion requires a passed check.": "unverified_completion",
    "Abort reason does not match the current error.": "invalid_abort_reason",
}


def _mapping(value: object) -> dict[str, object]:
    return value if isinstance(value, dict) else {}


def _property_names(schema: object) -> Iterator[str]:
    """스키마의 모든 깊이에서 속성 이름을 찾는다. 이름은 코드가 정한 값이다."""
    if isinstance(schema, dict):
        yield from _mapping(schema.get("properties"))
        for value in schema.values():
            yield from _property_names(value)
    elif isinstance(schema, list):
        for item in schema:
            yield from _property_names(item)


class ModelAudit:
    """인증값과 임의 응답 텍스트를 제외한다. 쓰기 오류는 호출자에게 전달한다.

    `argument_models`는 도구 이름과 strict 인자 모델의 표다.
    그 모델을 통과한 인자는 전체를 기록한다. 통과하지 못한 인자는 이름 목록으로만 투영한다.
    """

    def __init__(
        self,
        *,
        root: Path,
        run_id: str,
        secret: str,
        argument_models: Mapping[str, type[BaseModel]] | None = None,
    ) -> None:
        self._path = root / AUDIT_FILE
        self._run_id = run_id
        self._secret = secret
        self._argument_models = dict(argument_models or {})
        self._names = _ARGUMENT_NAMES.union(
            *(_property_names(strict_parameters(model)) for model in self._argument_models.values())
        )

    def write(self, *, inference: int, kind: str, data: dict[str, object]) -> None:
        """각 관찰을 회차에 연결한다. 실패한 기록을 성공으로 처리하지 않는다."""
        record = {
            "schema_version": 1,
            "run_id": self._run_id,
            "inference": inference,
            "kind": kind,
            "data": self._mask(data),
        }
        with self._path.open("a", encoding="utf-8") as stream:
            stream.write(json.dumps(record, ensure_ascii=False, allow_nan=False) + "\n")

    def _mask(self, value: object) -> object:
        if isinstance(value, str):
            return value.replace(self._secret, REDACTED) if self._secret else value
        if isinstance(value, dict):
            return {str(self._mask(key)): self._mask(item) for key, item in value.items()}
        if isinstance(value, (list, tuple)):
            return [self._mask(item) for item in value]
        return value

    def _label(self, value: object) -> str | None:
        if not isinstance(value, str):
            return None
        return value if _SAFE_LABEL.fullmatch(value) else REDACTED

    def _arguments(self, raw: object, *, tool: object) -> dict[str, object]:
        duplicates: list[str] = []

        def pairs(items: list[tuple[str, object]]) -> dict[str, object]:
            result: dict[str, object] = {}
            for key, value in items:
                if key in result:
                    duplicates.append(key if key in _ARGUMENT_NAMES else REDACTED)
                result[key] = value
            return result

        arguments: dict[str, object] = {}
        types: dict[str, str] = {}
        parsed: object = None
        valid = False
        if isinstance(raw, str) and len(raw.encode("utf-8")) <= MAX_ARGUMENT_BYTES:
            try:
                parsed = json.loads(raw, object_pairs_hook=pairs)
                valid = True
            except ValueError:
                pass
        values = _mapping(parsed)
        model = self._argument_models.get(tool) if isinstance(tool, str) else None
        if model is not None and isinstance(raw, str) and valid and not duplicates:
            try:
                model.model_validate_json(raw)
            except ValidationError:
                pass
            else:
                return {
                    "arguments": values,
                    "arguments_json": raw,
                    "argument_types": {key: type(value).__name__ for key, value in values.items()},
                    "json_valid": True,
                    "json_type": type(parsed).__name__,
                    "duplicate_keys": [],
                    "extra_argument_count": 0,
                }
        for key, value in values.items():
            if key not in _ARGUMENT_NAMES:
                continue
            types[key] = type(value).__name__
            if key == "retry":
                safe = isinstance(value, bool) or (
                    isinstance(value, str) and value in ("true", "false", "True", "False")
                )
            elif key == "reason":
                safe = isinstance(value, str) and _SAFE_CODE.fullmatch(value) is not None
            elif key == "part_ids":
                safe = (
                    isinstance(value, list)
                    and len(value) <= MAX_OBSERVED_ITEMS
                    and all(
                        isinstance(item, str) and _SAFE_PART_ID.fullmatch(item) is not None
                        for item in value
                    )
                )
            else:
                safe = isinstance(value, str) and _SAFE_ID.fullmatch(value) is not None
            arguments[key] = value if safe else REDACTED
        extra_count = len(values.keys() - _ARGUMENT_NAMES)
        raw_safe = valid and isinstance(parsed, dict) and not extra_count and arguments == parsed
        return {
            "arguments": arguments,
            "arguments_json": raw if raw_safe else None,
            "argument_types": types,
            "json_valid": valid,
            "json_type": type(parsed).__name__ if valid else None,
            "duplicate_keys": duplicates[:MAX_OBSERVED_ITEMS],
            "extra_argument_count": extra_count,
        }

    def response(self, payload: object) -> dict[str, object]:
        """도구 응답의 허용 필드만 투영한다. 자유 텍스트는 존재 여부만 쓴다."""
        body = _mapping(payload)
        raw_choices = body.get("choices")
        choices: list[dict[str, object]] = []
        for raw_choice in raw_choices[:MAX_OBSERVED_ITEMS] if isinstance(raw_choices, list) else []:
            choice = _mapping(raw_choice)
            message = _mapping(choice.get("message"))
            raw_calls = message.get("tool_calls")
            calls: list[dict[str, object]] = []
            for raw_call in raw_calls[:MAX_OBSERVED_ITEMS] if isinstance(raw_calls, list) else []:
                call = _mapping(raw_call)
                function = _mapping(call.get("function"))
                calls.append(
                    {
                        "id": self._label(call.get("id")),
                        "type": self._label(call.get("type")),
                        "name": self._label(function.get("name")),
                        **self._arguments(function.get("arguments"), tool=function.get("name")),
                    }
                )
            content = message.get("content")
            choices.append(
                {
                    "finish_reason": self._label(choice.get("finish_reason")),
                    "tool_calls": calls,
                    "tool_call_count": len(raw_calls) if isinstance(raw_calls, list) else 0,
                    "free_text_present": bool(content.strip())
                    if isinstance(content, str)
                    else content is not None,
                    "refusal_present": bool(message.get("refusal")),
                    "legacy_function_call_present": message.get("function_call") is not None,
                }
            )
        return {
            "completion_id": self._label(body.get("id")),
            "model": self._label(body.get("model")),
            "choice_count": len(raw_choices) if isinstance(raw_choices, list) else 0,
            "choices": choices,
        }

    def http_response(self, *, payload: object, request: bytes, status: int) -> dict[str, object]:
        """실제 전송 JSON과 응답 도구를 저장한다. HTTP 헤더는 읽지 않는다."""
        return {
            "status": status,
            "request_bytes": len(request),
            "request_sha256": hashlib.sha256(request).hexdigest(),
            "request": json.loads(request),
            "response": self.response(payload),
        }

    def rejection(self, error: ValueError) -> dict[str, object]:
        """고정 거부 분류와 검증 위치만 보존한다. 입력값은 제외한다."""
        if isinstance(error, ValidationError):
            issues = [
                {
                    "location": [
                        item if isinstance(item, int) or item in self._names else REDACTED
                        for item in issue["loc"]
                    ],
                    "type": issue["type"],
                }
                for issue in error.errors(include_input=False, include_context=False)
            ]
            return {"accepted": False, "reason": "invalid_arguments", "issues": issues}
        return {"accepted": False, "reason": _REASONS.get(str(error), "invalid_action")}
