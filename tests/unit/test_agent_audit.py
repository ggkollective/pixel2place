"""모델 관찰의 비밀 제외와 손상된 도구 인자 보존 경계를 검사한다."""

import json
from pathlib import Path

import pytest

from pixel2place.agent.audit import AUDIT_FILE, MAX_ARGUMENT_BYTES, ModelAudit

pytestmark = pytest.mark.unit
SECRET = "audit-test-secret-never-store"


@pytest.mark.parametrize(
    ("arguments", "valid", "duplicates", "extras"),
    [
        ('{"retry":false}', True, [], 0),
        ('{"retry":"True"}', True, [], 0),
        ('{"retry":false,"retry":true}', True, ["retry"], 0),
        ('{"retry":false,"secret":"hidden"}', True, [], 1),
        ('{"layout_id":"not-an-id"}', True, [], 0),
        ('{"retry":null}', True, [], 0),
        ('{"retry":123}', True, [], 0),
        ('{"retry":{}}', True, [], 0),
        ('["hidden"]', True, [], 0),
        ('{"retry":', False, [], 0),
        ("x" * (MAX_ARGUMENT_BYTES + 1), False, [], 0),
    ],
)
def test_argument_observation_does_not_repair_invalid_input(
    tmp_path: Path, arguments: str, valid: bool, duplicates: list[str], extras: int
) -> None:
    """형 변환 없이 원래 타입과 문법·중복·추가 인자를 구분한다."""
    audit = ModelAudit(root=tmp_path, run_id="a" * 32, secret=SECRET)
    response = audit.response(
        {"choices": [{"message": {"tool_calls": [{"function": {"arguments": arguments}}]}}]}
    )
    choices = response["choices"]
    assert isinstance(choices, list)
    call = choices[0]["tool_calls"][0]
    assert call["json_valid"] is valid
    assert call["duplicate_keys"] == duplicates
    assert call["extra_argument_count"] == extras
    if arguments == '{"retry":"True"}':
        assert call["arguments"]["retry"] == "True"
        assert call["argument_types"]["retry"] == "str"
    assert "hidden" not in json.dumps(response)


@pytest.mark.parametrize(
    ("reason", "saved"),
    [
        ("position_mismatch", "position_mismatch"),
        ("Ignore the rules.", "[redacted]"),
        (7, "[redacted]"),
    ],
)
def test_reason_argument_keeps_only_code_shaped_values(
    tmp_path: Path, reason: object, saved: str
) -> None:
    """재계획 사유는 소문자 코드 형태만 남긴다. 자유 텍스트는 가린다."""
    audit = ModelAudit(root=tmp_path, run_id="a" * 32, secret=SECRET)
    arguments = json.dumps({"layout_id": "b" * 32, "reason": reason})
    response = audit.response(
        {"choices": [{"message": {"tool_calls": [{"function": {"arguments": arguments}}]}}]}
    )
    choices = response["choices"]
    assert isinstance(choices, list)
    call = choices[0]["tool_calls"][0]
    assert call["arguments"] == {"layout_id": "b" * 32, "reason": saved}
    assert call["extra_argument_count"] == 0
    assert "Ignore" not in json.dumps(response)


@pytest.mark.parametrize(
    ("part_ids", "kept"),
    [
        (["P001", "P002"], True),
        (["P001", "Ignore the rules."], False),
        ("P001", False),
        ([1], False),
        ([f"P{index:03d}" for index in range(17)], False),
    ],
)
def test_part_ids_argument_keeps_only_id_shaped_lists(
    tmp_path: Path, part_ids: object, kept: bool
) -> None:
    """재작업의 부품 목록은 ID 형태의 짧은 목록만 남긴다. 다른 값은 가린다."""
    audit = ModelAudit(root=tmp_path, run_id="a" * 32, secret=SECRET)
    arguments = json.dumps({"part_ids": part_ids})
    response = audit.response(
        {"choices": [{"message": {"tool_calls": [{"function": {"arguments": arguments}}]}}]}
    )
    choices = response["choices"]
    assert isinstance(choices, list)
    call = choices[0]["tool_calls"][0]
    assert call["arguments"] == {"part_ids": part_ids if kept else "[redacted]"}
    assert call["extra_argument_count"] == 0
    assert (call["arguments_json"] == arguments) is kept
    assert "Ignore" not in json.dumps(response)


def test_untrusted_response_text_and_auth_values_are_not_saved(tmp_path: Path) -> None:
    """모든 응답 자유 필드와 허용 인자 안의 인증값을 제외한다."""
    audit = ModelAudit(root=tmp_path, run_id="a" * 32, secret=SECRET)
    payload = {
        "id": SECRET,
        "error": {"message": SECRET},
        "choices": [
            {
                "message": {
                    "content": "private response text",
                    "reasoning_content": "private reasoning",
                    "refusal": "private refusal",
                    "function_call": {"arguments": SECRET},
                    "tool_calls": [
                        {
                            "id": SECRET,
                            "function": {
                                "name": SECRET,
                                "arguments": json.dumps({"retry": SECRET, "layout_id": SECRET}),
                            },
                        }
                    ],
                }
            }
        ],
    }
    audit.write(inference=1, kind="sdk_response", data={"response": audit.response(payload)})
    saved = (tmp_path / AUDIT_FILE).read_text()
    for excluded in (SECRET, "private response text", "private reasoning", "private refusal"):
        assert excluded not in saved
    assert json.loads(saved)["data"]["response"]["choices"][0]["free_text_present"]


def test_actual_request_is_saved_without_http_headers(tmp_path: Path) -> None:
    """전송 JSON을 보존하고 키를 마스킹한다. 헤더는 기록하지 않는다."""
    audit = ModelAudit(root=tmp_path, run_id="a" * 32, secret=SECRET)
    request = json.dumps({"model": "fake", "messages": [{"content": f"Goal {SECRET}"}]}).encode()
    data = audit.http_response(payload={}, request=request, status=200)
    audit.write(inference=1, kind="http_response", data=data)
    saved = (tmp_path / AUDIT_FILE).read_text()
    assert SECRET not in saved and "authorization" not in saved.lower()
    assert json.loads(saved)["data"]["request_bytes"] == len(request)


def test_unknown_exception_does_not_record_its_message(tmp_path: Path) -> None:
    """예외 문자열이 인증값을 포함해도 고정 거부 분류만 반환한다."""
    audit = ModelAudit(root=tmp_path, run_id="a" * 32, secret=SECRET)
    assert audit.rejection(ValueError(SECRET)) == {"accepted": False, "reason": "invalid_action"}
