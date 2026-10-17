"""작업 지시 접수의 주문 추출, 거절과 실패 경계를 HTTP Fake로 검사한다."""

import json
from collections.abc import Iterator
from pathlib import Path
from typing import Any

import pytest
from openai import OpenAI
from typer.testing import CliRunner

from pixel2place.agent.audit import AUDIT_FILE, ModelAudit
from pixel2place.agent.inference import NO_RETRY, REQUEST_RETRY, RetryPolicy
from pixel2place.agent.intake import (
    ARGUMENT_MODELS,
    INTAKE_FILE,
    INTAKE_SYSTEM_PROMPT,
    MAX_TEXT_CHARS,
    IntakeOutcome,
    IntakeRecord,
    IntakeTool,
    RejectReason,
    intake_tools,
    read_order,
)
from pixel2place.cli import app
from pixel2place.config import DEFAULT_JUDGMENT_MODEL, DEFAULT_NEMOTRON_MODEL
from pixel2place.feasibility.checker import IssueCode
from pixel2place.order import KIT_COLORS, KitOrder, order_instruction, parse_order
from pixel2place.reasoner import nemotron
from tests.fakes.fake_judgment_model import (
    FAKE_JUDGMENT_KEY,
    FAKE_JUDGMENT_REASONING,
    FakeJudgmentModel,
    Reply,
)

pytestmark = pytest.mark.unit
runner = CliRunner()
TEXT = "I need four square tiles and two rectangular tiles. Keep each type in one row."
KIT = {
    "kind": "kit",
    "items": [{"type": "square_tile", "count": 4}, {"type": "rect_tile", "count": 2}],
    "arrangement": "type_rows",
}
BOARD = {
    "kind": "board",
    "parts": [
        {"type": "L_tile", "color": "orange", "rotation_deg": 90, "row": 0, "col": 0},
        {"type": "square_tile", "color": "blue", "rotation_deg": 0, "row": 5, "col": 6},
    ],
}
OVERLAP = {
    "kind": "board",
    "parts": [
        {"type": "square_tile", "color": "blue", "rotation_deg": 0, "row": 1, "col": 1},
        {"type": "rect_tile", "color": "gray", "rotation_deg": 0, "row": 1, "col": 1},
    ],
}
TOO_MANY = {
    "kind": "kit",
    "items": [{"type": "square_tile", "count": 4}, {"type": "rect_tile", "count": 3}],
    "arrangement": "type_rows",
}
TOO_MANY_RECTS = {
    "kind": "kit",
    "items": [{"type": "rect_tile", "count": 5}],
    "arrangement": "type_rows",
}


@pytest.fixture
def api() -> Iterator[FakeJudgmentModel]:
    """테스트마다 HTTP Fake를 만든다. 종료 때 클라이언트를 닫는다."""
    fake = FakeJudgmentModel()
    yield fake
    for client in fake.clients:
        client.close()


def _read(
    api: FakeJudgmentModel,
    root: Path,
    *replies: Reply,
    text: str = TEXT,
    retry: RetryPolicy = NO_RETRY,
) -> Any:
    api.replies = list(replies)
    client = OpenAI(
        api_key=FAKE_JUDGMENT_KEY,
        base_url="https://api.tokenfactory.nebius.com/v1/",
        max_retries=0,
        http_client=api.make_client(),
    )
    audit = ModelAudit(
        root=root,
        run_id="c" * 32,
        secret=FAKE_JUDGMENT_KEY,
        argument_models={tool.value: ARGUMENT_MODELS[tool] for tool in IntakeTool},
    )
    return read_order(text, client=client, model=DEFAULT_NEMOTRON_MODEL, audit=audit, retry=retry)


def _audit(root: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in (root / AUDIT_FILE).read_text().splitlines()]


def _schema_nodes(node: object) -> Iterator[dict[str, Any]]:
    if isinstance(node, dict):
        yield node
        for key, value in node.items():
            # 속성 이름의 표는 스키마 노드가 아니다. 그 값만 본다.
            children = value.values() if key == "properties" else [value]
            for child in children:
                yield from _schema_nodes(child)
    elif isinstance(node, list):
        for item in node:
            yield from _schema_nodes(item)


class TestIntakeRequest:
    """모델에 보내는 지시문과 도구 정의를 검사한다."""

    def test_request_has_three_strict_tools_and_the_request_text(
        self, tmp_path: Path, api: FakeJudgmentModel
    ) -> None:
        """요청은 지시문, 원문과 strict 도구 3개를 보낸다. 재시도는 없다."""
        _read(api, tmp_path, Reply(tool="submit_kit_order", arguments=KIT))
        assert len(api.model_posts) == 1
        body = json.loads(api.model_posts[0].content)
        assert body["messages"] == [
            {"role": "system", "content": INTAKE_SYSTEM_PROMPT},
            {"role": "user", "content": TEXT},
        ]
        assert body["tool_choice"] == "auto" and body["max_tokens"] == 2048
        assert [tool["function"]["name"] for tool in body["tools"]] == [t.value for t in IntakeTool]
        assert all(tool["function"]["strict"] is True for tool in body["tools"])
        assert api.model_posts[0].extensions["timeout"]["read"] == 120

    def test_tool_schemas_are_flat_and_every_property_is_required(self) -> None:
        """스키마에 `$ref`, `$defs`, 설명과 기본값이 없다. 모든 속성은 필수다."""
        for tool in intake_tools():
            parameters = tool["function"].get("parameters")
            assert parameters is not None
            for node in _schema_nodes(parameters):
                assert not {"$ref", "$defs", "description", "default"} & node.keys(), node
                if "properties" in node:
                    assert node["required"] == list(node["properties"]), node
                    assert node["additionalProperties"] is False, node

    def test_kit_tool_offers_only_the_kit_part_types(self) -> None:
        """키트 도구의 부품 종류는 키트 계약이 받는 종류와 같다."""
        kit = intake_tools()[0]["function"].get("parameters")
        assert kit is not None
        item: Any = kit["properties"]
        assert item["items"]["items"]["properties"]["type"]["enum"] == [p.value for p in KIT_COLORS]

    def test_prompt_states_the_order_limits(self) -> None:
        """지시문은 주문 계약의 한도와 모든 거절 사유를 쓴다."""
        assert "maximum of 6 parts" in INTAKE_SYSTEM_PROMPT
        assert "maximum of 4 rect_tile parts" in INTAKE_SYSTEM_PROMPT
        assert "square_tile parts" not in INTAKE_SYSTEM_PROMPT
        assert "maximum of 16 parts" in INTAKE_SYSTEM_PROMPT
        assert "blue square_tile and gray rect_tile" in INTAKE_SYSTEM_PROMPT
        assert all(f"{reason.value}:" in INTAKE_SYSTEM_PROMPT for reason in RejectReason)

    def test_prompt_separates_kit_requests_from_ambiguous_requests(self) -> None:
        """키트 요청은 색과 위치가 없어도 된다. 모델이 값을 고르는 요청은 거절하게 한다."""
        assert "A kit request does not have to give colors or positions." in INTAKE_SYSTEM_PROMPT
        ambiguous = next(
            line for line in INTAKE_SYSTEM_PROMPT.splitlines() if "ambiguous_request:" in line
        )
        # 위치가 없는 요청은 키트 주문이다. 일부 부품에만 위치가 있을 때가 모호한 요청이다.
        assert "only some of the parts have a position" in ambiguous
        assert "a position is missing" not in ambiguous
        assert "gives the part types and the quantities" in INTAKE_SYSTEM_PROMPT
        assert "tells you to select the values, reject it." in INTAKE_SYSTEM_PROMPT


class TestReadOrder:
    """수락, 모델의 거절과 코드의 거절을 검사한다."""

    def test_kit_order_is_accepted_with_the_code_instruction(
        self, tmp_path: Path, api: FakeJudgmentModel
    ) -> None:
        """키트 주문을 수락한다. 지시문은 주문 계약의 코드가 만든다."""
        reading = _read(api, tmp_path, Reply(tool="submit_kit_order", arguments=KIT))
        order = parse_order(json.dumps(KIT))
        assert reading.outcome == IntakeOutcome.ACCEPTED and reading.order == order
        assert isinstance(reading.order, KitOrder)
        assert reading.instruction == order_instruction(order)
        assert reading.evidence.finish_reason == "tool_calls"
        records = _audit(tmp_path)
        assert [record["kind"] for record in records] == [
            "request",
            "http_response",
            "sdk_response",
            "validation",
        ]
        call = records[2]["data"]["response"]["choices"][0]["tool_calls"][0]
        assert call["name"] == "submit_kit_order" and call["arguments"] == KIT
        assert records[3]["data"] == {"accepted": True}
        text = (tmp_path / AUDIT_FILE).read_text()
        assert FAKE_JUDGMENT_KEY not in text and FAKE_JUDGMENT_REASONING not in text

    def test_board_order_is_accepted_with_positions(
        self, tmp_path: Path, api: FakeJudgmentModel
    ) -> None:
        """보드 주문을 수락한다. 지시문은 각 부품의 row와 col을 쓴다."""
        reading = _read(api, tmp_path, Reply(tool="submit_board_order", arguments=BOARD))
        assert reading.outcome == IntakeOutcome.ACCEPTED
        assert reading.order == parse_order(json.dumps(BOARD))
        assert "row 0 col 0" in reading.instruction and "row 5 col 6" in reading.instruction

    @pytest.mark.parametrize("reason", list(RejectReason))
    def test_model_reject_keeps_the_reason_and_no_order(
        self, tmp_path: Path, api: FakeJudgmentModel, reason: RejectReason
    ) -> None:
        """모델이 거절하면 사유만 남긴다. 주문과 지시문은 없다."""
        reading = _read(
            api, tmp_path, Reply(tool="reject_request", arguments={"reason": reason.value})
        )
        assert reading.outcome == IntakeOutcome.REJECTED and reading.reject_reason == reason
        assert reading.order is None and reading.instruction is None

    @pytest.mark.parametrize("arguments", [TOO_MANY, TOO_MANY_RECTS])
    def test_order_outside_the_contract_is_order_invalid(
        self, tmp_path: Path, api: FakeJudgmentModel, arguments: dict[str, object]
    ) -> None:
        """합계 또는 종류별 수량이 키트 한도를 넘으면 코드가 거절한다. 모델의 거절과 구분한다."""
        reading = _read(api, tmp_path, Reply(tool="submit_kit_order", arguments=arguments))
        assert reading.outcome == IntakeOutcome.ORDER_INVALID
        assert reading.order is None and reading.reject_reason is None
        validation = _audit(tmp_path)[-1]["data"]
        assert validation["accepted"] is False and validation["reason"] == "order_invalid"
        assert validation["issues"] and "input" not in json.dumps(validation)

    def test_board_order_with_overlap_is_not_feasible(
        self, tmp_path: Path, api: FakeJudgmentModel
    ) -> None:
        """배치 규칙을 어긴 보드 주문은 위반 코드를 남긴다. 지시문은 없다."""
        reading = _read(api, tmp_path, Reply(tool="submit_board_order", arguments=OVERLAP))
        assert reading.outcome == IntakeOutcome.ORDER_NOT_FEASIBLE
        assert reading.issues == (IssueCode.OVERLAP,) and reading.instruction is None
        assert reading.order == parse_order(json.dumps(OVERLAP))

    @pytest.mark.parametrize(
        ("reply", "reason"),
        [
            (Reply(content="Place four squares."), "invalid_call_count"),
            (Reply(tool="execute_cell", arguments={}), "tool_not_allowed"),
            (Reply(tool="submit_kit_order", arguments=KIT, call_count=2), "invalid_call_count"),
            (
                Reply(tool="submit_kit_order", arguments=KIT, content="Done."),
                "invalid_call_message",
            ),
            (
                Reply(tool="reject_request", arguments='{"reason":"a","reason":"too_many_parts"}'),
                "duplicate_json_key",
            ),
            (Reply(tool="reject_request", arguments={"reason": "other"}), "invalid_arguments"),
            (Reply(tool="reject_request", arguments="{"), "invalid_action"),
        ],
    )
    def test_response_outside_the_protocol_is_protocol_error(
        self, tmp_path: Path, api: FakeJudgmentModel, reply: Reply, reason: str
    ) -> None:
        """함수 호출 1개가 아닌 응답과 계약 밖의 거절 인자는 주문이 되지 않는다."""
        reading = _read(api, tmp_path, reply)
        assert reading.outcome == IntakeOutcome.PROTOCOL_ERROR and reading.order is None
        assert _audit(tmp_path)[-1]["data"]["reason"] == reason
        assert len(api.model_posts) == 1

    @pytest.mark.parametrize(
        ("reply", "reason"),
        [
            (Reply(status=503), "model_error"),
            (Reply(tool="submit_kit_order", arguments=KIT, finish="length"), "token_limit"),
            (
                Reply(tool="submit_kit_order", arguments=KIT, model="nvidia/Nemotron-other"),
                "different_model",
            ),
        ],
    )
    def test_model_failure_is_model_error_without_retry(
        self, tmp_path: Path, api: FakeJudgmentModel, reply: Reply, reason: str
    ) -> None:
        """HTTP 오류, 잘린 응답과 다른 모델은 모델 오류다. 다시 요청하지 않는다."""
        reading = _read(api, tmp_path, reply)
        assert reading.outcome == IntakeOutcome.MODEL_ERROR and reading.order is None
        assert _audit(tmp_path)[-1]["data"] == {"accepted": False, "reason": reason}
        assert len(api.model_posts) == 1
        assert FAKE_JUDGMENT_KEY not in (tmp_path / AUDIT_FILE).read_text()


class TestReadOrderRetry:
    """재전송 방침을 준 접수에서 일시 오류의 재전송을 검사한다."""

    def test_one_503_is_sent_again_and_the_order_is_accepted(
        self, tmp_path: Path, api: FakeJudgmentModel, retry_waits: list[float]
    ) -> None:
        """503이 1회이면 3초 뒤에 다시 보내고 주문을 수락한다."""
        reading = _read(
            api,
            tmp_path,
            Reply(status=503),
            Reply(tool="submit_kit_order", arguments=KIT),
            retry=REQUEST_RETRY,
        )
        assert reading.outcome == IntakeOutcome.ACCEPTED, reading
        assert len(api.model_posts) == 2 and retry_waits == [3.0]
        assert [record["kind"] for record in _audit(tmp_path)] == [
            "request",
            "http_response",
            "attempt",
            "http_response",
            "sdk_response",
            "attempt",
            "validation",
        ]

    def test_four_failed_attempts_are_model_error(
        self, tmp_path: Path, api: FakeJudgmentModel, retry_waits: list[float]
    ) -> None:
        """503이 4회 연속이면 모델 오류다. 대기는 3초, 9초와 20초다."""
        reading = _read(api, tmp_path, *[Reply(status=503)] * 4, retry=REQUEST_RETRY)
        assert reading.outcome == IntakeOutcome.MODEL_ERROR and reading.order is None
        assert len(api.model_posts) == 4 and retry_waits == [3.0, 9.0, 20.0]
        assert _audit(tmp_path)[-1]["data"] == {"accepted": False, "reason": "model_error"}
        assert FAKE_JUDGMENT_KEY not in (tmp_path / AUDIT_FILE).read_text()

    @pytest.mark.parametrize(
        ("reply", "outcome"),
        [
            (Reply(status=400), IntakeOutcome.MODEL_ERROR),
            (
                Reply(tool="submit_kit_order", arguments=KIT, finish="length"),
                IntakeOutcome.MODEL_ERROR,
            ),
            (
                Reply(tool="submit_kit_order", arguments=KIT, model="nvidia/Nemotron-other"),
                IntakeOutcome.MODEL_ERROR,
            ),
            (Reply(content="Place four squares."), IntakeOutcome.PROTOCOL_ERROR),
            (Reply(tool="submit_kit_order", arguments=TOO_MANY), IntakeOutcome.ORDER_INVALID),
        ],
    )
    def test_other_failures_are_not_sent_again(
        self,
        tmp_path: Path,
        api: FakeJudgmentModel,
        retry_waits: list[float],
        reply: Reply,
        outcome: IntakeOutcome,
    ) -> None:
        """다른 4xx, 잘린 응답, 다른 모델과 응답 형식의 오류는 다시 보내지 않는다."""
        reading = _read(api, tmp_path, reply, retry=REQUEST_RETRY)
        assert reading.outcome == outcome
        assert len(api.model_posts) == 1 and not retry_waits


class TestIntakeOrderCLI:
    """`intake-order`의 종료 코드, 표준 출력과 기록 파일을 검사한다."""

    @pytest.fixture
    def cli_api(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch, api: FakeJudgmentModel
    ) -> FakeJudgmentModel:
        """환경은 테스트 값으로 고정한다. SDK의 HTTP 클라이언트만 대체한다."""
        monkeypatch.chdir(tmp_path)
        monkeypatch.setenv("NEBIUS_API_KEY", FAKE_JUDGMENT_KEY)
        monkeypatch.setenv("NEBIUS_BASE_URL", "https://api.tokenfactory.nebius.com/v1/")
        monkeypatch.setenv("NEMOTRON_MODEL", "nvidia/Nemotron-other")
        monkeypatch.setattr(nemotron, "DefaultHttpxClient", api.make_client)
        return api

    def test_accepted_order_writes_the_instruction_and_exit_0(
        self, tmp_path: Path, cli_api: FakeJudgmentModel
    ) -> None:
        """수락하면 표준 출력에 지시문만 쓴다. 모델은 환경 값이 아니라 판단 모델이다."""
        cli_api.replies = [Reply(tool="submit_kit_order", arguments=KIT)]
        out_dir = tmp_path / "intake"
        result = runner.invoke(app, ["intake-order", TEXT, "-o", str(out_dir)])
        assert result.exit_code == 0, result.output
        assert result.stdout.strip() == order_instruction(parse_order(json.dumps(KIT)))
        record = IntakeRecord.model_validate_json((out_dir / INTAKE_FILE).read_text())
        assert record.text == TEXT and record.model == DEFAULT_JUDGMENT_MODEL
        assert json.loads(cli_api.model_posts[0].content)["model"] == DEFAULT_JUDGMENT_MODEL
        assert record.reading.outcome == IntakeOutcome.ACCEPTED
        assert record.intake_id == _audit(out_dir)[0]["run_id"]
        assert [request.method for request in cli_api.requests] == ["GET", "POST"]
        assert all(client.is_closed for client in cli_api.clients)
        assert FAKE_JUDGMENT_KEY not in (out_dir / INTAKE_FILE).read_text()

    def test_failed_requests_are_sent_again_and_the_order_is_accepted(
        self, tmp_path: Path, cli_api: FakeJudgmentModel, retry_waits: list[float]
    ) -> None:
        """모델 목록과 모델 요청의 503이 각 1회이면 CLI는 두 요청을 다시 보내고 exit 0이다."""
        cli_api.catalog_statuses = [503]
        cli_api.replies = [Reply(status=503), Reply(tool="submit_kit_order", arguments=KIT)]
        out_dir = tmp_path / "intake"
        result = runner.invoke(app, ["intake-order", TEXT, "-o", str(out_dir)])
        assert result.exit_code == 0, result.output
        assert [request.method for request in cli_api.requests] == ["GET", "GET", "POST", "POST"]
        assert retry_waits == [3.0, 3.0]
        attempts = [record for record in _audit(out_dir) if record["kind"] == "attempt"]
        assert [(record["inference"], record["data"]["outcome"]) for record in attempts] == [
            (0, "http_error"),
            (0, "ok"),
            (1, "http_error"),
            (1, "ok"),
        ]

    def test_four_failed_catalog_requests_are_exit_1_without_inference(
        self, tmp_path: Path, cli_api: FakeJudgmentModel, retry_waits: list[float]
    ) -> None:
        """모델 목록 요청이 4회 연속으로 실패하면 모델 요청 없이 모델 오류로 끝난다."""
        cli_api.catalog_statuses = [503] * 4
        out_dir = tmp_path / "intake"
        result = runner.invoke(app, ["intake-order", TEXT, "-o", str(out_dir)])
        assert result.exit_code == 1 and not cli_api.model_posts, result.output
        assert len(cli_api.requests) == 4 and retry_waits == [3.0, 9.0, 20.0]
        record = IntakeRecord.model_validate_json((out_dir / INTAKE_FILE).read_text())
        assert record.reading.outcome == IntakeOutcome.MODEL_ERROR

    def test_model_option_selects_another_listed_model(
        self, tmp_path: Path, cli_api: FakeJudgmentModel
    ) -> None:
        """`--model`은 계정 목록에 있는 다른 Nemotron을 쓴다."""
        other = "nvidia/nemotron-3-super-120b-a12b"
        cli_api.models = [DEFAULT_NEMOTRON_MODEL, other]
        cli_api.replies = [Reply(tool="submit_kit_order", arguments=KIT)]
        out_dir = tmp_path / "intake"
        result = runner.invoke(app, ["intake-order", TEXT, "-o", str(out_dir), "--model", other])
        assert result.exit_code == 0, result.output
        assert json.loads(cli_api.model_posts[0].content)["model"] == other

    def test_rejected_request_is_exit_1_with_the_reason(
        self, tmp_path: Path, cli_api: FakeJudgmentModel
    ) -> None:
        """거절하면 표준 출력이 비어 있다. 사유는 표준 오류와 기록에 있다."""
        cli_api.replies = [Reply(tool="reject_request", arguments={"reason": "too_many_parts"})]
        out_dir = tmp_path / "intake"
        result = runner.invoke(app, ["intake-order", "Place 40 squares.", "-o", str(out_dir)])
        assert result.exit_code == 1 and result.stdout == ""
        assert "outcome=rejected detail=too_many_parts" in result.stderr
        record = IntakeRecord.model_validate_json((out_dir / INTAKE_FILE).read_text())
        assert record.reading.reject_reason == RejectReason.TOO_MANY_PARTS

    def test_unlisted_model_is_exit_1_without_inference(
        self, tmp_path: Path, cli_api: FakeJudgmentModel
    ) -> None:
        """계정 목록에 모델이 없으면 모델 요청하지 않는다."""
        cli_api.models = []
        out_dir = tmp_path / "intake"
        result = runner.invoke(app, ["intake-order", TEXT, "-o", str(out_dir)])
        assert result.exit_code == 1 and not cli_api.model_posts
        record = IntakeRecord.model_validate_json((out_dir / INTAKE_FILE).read_text())
        assert record.reading.outcome == IntakeOutcome.MODEL_ERROR

    @pytest.mark.parametrize("text", [" ", "x" * (MAX_TEXT_CHARS + 1)])
    def test_invalid_text_is_exit_2_without_request(
        self, tmp_path: Path, cli_api: FakeJudgmentModel, text: str
    ) -> None:
        """빈 문장과 한도를 넘는 문장은 외부 요청 전에 거부한다."""
        out_dir = tmp_path / "intake"
        result = runner.invoke(app, ["intake-order", text, "-o", str(out_dir)])
        assert result.exit_code == 2 and not cli_api.requests
        assert not out_dir.exists()

    def test_existing_output_is_exit_2_without_request(
        self, tmp_path: Path, cli_api: FakeJudgmentModel
    ) -> None:
        """기존 출력 디렉터리를 덮어쓰지 않는다."""
        out_dir = tmp_path / "intake"
        out_dir.mkdir()
        result = runner.invoke(app, ["intake-order", TEXT, "-o", str(out_dir)])
        assert result.exit_code == 2 and not cli_api.requests

    def test_missing_key_is_exit_2(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        """인증 키가 없으면 출력 디렉터리를 만들지 않는다."""
        monkeypatch.chdir(tmp_path)
        monkeypatch.delenv("NEBIUS_API_KEY", raising=False)
        out_dir = tmp_path / "intake"
        result = runner.invoke(app, ["intake-order", TEXT, "-o", str(out_dir)])
        assert result.exit_code == 2 and not out_dir.exists()
