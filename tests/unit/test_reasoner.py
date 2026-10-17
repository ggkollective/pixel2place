import base64
import io
import json
import logging

import pytest
from openai import OpenAI
from PIL import Image

from pixel2place.config import Settings
from pixel2place.layout.schema import Board, Color
from pixel2place.reasoner.client import (
    SAMPLING_PARAMS,
    SAMPLING_SEED,
    UNUSED_API_KEY,
    VLLM_SAMPLING_PARAMS,
    CosmosReasoner,
    ReasonerOutputError,
)
from pixel2place.reasoner.prompts import load_system_prompt
from tests.factories import make_parser_response_json, make_planner_response_json
from tests.fakes.fake_openai import FakeOpenAI

MODEL = "fake-cosmos-reasoner"
BASE_URL = "http://reasoner.test:8000/v1/"


def _reasoner(fake: FakeOpenAI) -> CosmosReasoner:
    return CosmosReasoner(client=fake, model=MODEL)


@pytest.mark.unit
class TestLoadSystemPrompt:
    """시스템 지시문 로더 테스트."""

    @pytest.mark.parametrize("name", ["planner", "parser"])
    def test_fills_all_placeholders(self, name):
        """모든 자리 표시자를 보드·부품·팔레트 값으로 채운다."""
        prompt = load_system_prompt(name, board=Board(rows=6, cols=7))
        placeholders = ("{rows}", "{cols}", "{colors}", "{shape_table}", "{schema_example}")
        for placeholder in (*placeholders, "{parser_example}"):
            assert placeholder not in prompt, f"{placeholder}가 남아 있다"
        assert "6 rows and 7 columns" in prompt
        assert "L_tile -> 0: [[0, 0], [1, 0], [2, 0], [2, 1]]" in prompt
        assert ", ".join(color.value for color in Color) in prompt


@pytest.mark.unit
class TestCosmosReasoner:
    """Cosmos Reasoner 클라이언트 테스트."""

    def test_plan_returns_layout(self, valid_layout):
        """배치 계획 모델 응답의 JSON을 Layout으로 읽는다."""
        fake = FakeOpenAI([f"<think>ok</think>{make_planner_response_json(valid_layout)}"])
        assert _reasoner(fake).plan("symmetric tray") == valid_layout
        call = fake.calls[0]
        assert call["model"] == MODEL
        assert call["messages"][1] == {"role": "user", "content": "symmetric tray"}

    def test_parse_sends_png_data_url(self, valid_layout):
        """판독기는 이미지를 PNG data URL로 보낸다."""
        fake = FakeOpenAI([make_parser_response_json(valid_layout)])
        assert _reasoner(fake).parse(Image.new("RGB", (8, 8))) == valid_layout
        url = fake.calls[0]["messages"][1]["content"][1]["image_url"]["url"]
        assert url.startswith("data:image/png;base64,")

    def test_parse_sends_original_image_and_parser_schema(self, valid_layout):
        """판독기는 이미지를 바꾸지 않고 보낸다. bbox 응답 스키마를 요청한다."""
        fake = FakeOpenAI([make_parser_response_json(valid_layout)])
        _reasoner(fake).parse(Image.new("RGB", (80, 80)))
        call = fake.calls[0]
        url = call["messages"][1]["content"][1]["image_url"]["url"]
        sent = Image.open(io.BytesIO(base64.b64decode(url.split(",", 1)[1])))
        assert sent.size == (80, 80)
        schema = call["response_format"]["json_schema"]["schema"]
        assert "bbox_2d" in schema["$defs"]["ParserPart"]["properties"]

    def test_retries_once_with_feedback(self, valid_layout, caplog):
        """첫 응답이 틀리면 오류를 알려 주고 다시 요청한다."""
        fake = FakeOpenAI(["not json", make_planner_response_json(valid_layout)])
        with caplog.at_level(logging.WARNING):
            assert _reasoner(fake).plan("tray") == valid_layout
        retry_messages = fake.calls[1]["messages"]
        assert retry_messages[-2] == {"role": "assistant", "content": "not json"}
        assert "not a valid layout" in retry_messages[-1]["content"]
        assert "not a valid layout" in caplog.text

    def test_planner_retries_once_with_placement_feedback(self, valid_layout, overlap_layout):
        """배치 계획 모델 레이아웃이 배치 규칙을 어기면 위반 내용을 알려 주고 다시 요청한다."""
        fake = FakeOpenAI(
            [make_planner_response_json(overlap_layout), make_planner_response_json(valid_layout)]
        )
        assert _reasoner(fake).plan("tray") == valid_layout
        feedback = fake.calls[1]["messages"][-1]["content"]
        assert "breaks the placement rules" in feedback
        assert "is also in part P001" in feedback

    def test_planner_returns_last_layout_when_still_infeasible(self, overlap_layout):
        """두 번 모두 배치 규칙을 어기면 두 번째 레이아웃을 반환한다. 판정은 호출자가 한다."""
        fake = FakeOpenAI([make_planner_response_json(overlap_layout)] * 2)
        assert _reasoner(fake).plan("tray") == overlap_layout
        assert len(fake.calls) == 2

    def test_planner_keeps_infeasible_layout_when_retry_format_fails(self, overlap_layout):
        """재요청 응답의 형식이 틀리면 배치 규칙만 어긴 첫 레이아웃을 반환한다."""
        fake = FakeOpenAI([make_planner_response_json(overlap_layout), "not json"])
        assert _reasoner(fake).plan("tray") == overlap_layout

    def test_parser_does_not_retry_placement(self, overlap_layout):
        """판독기는 배치 규칙 위반을 고치게 하지 않는다."""
        fake = FakeOpenAI([make_parser_response_json(overlap_layout)])
        assert _reasoner(fake).parse(Image.new("RGB", (8, 8))) == overlap_layout
        assert len(fake.calls) == 1

    def test_fail_after_invalid_schema_twice(self):
        """두 번 모두 스키마가 틀리면 ReasonerOutputError를 낸다."""
        bad = json.dumps({"palette": ["purple"], "parts": []})
        fake = FakeOpenAI([bad, bad])
        with pytest.raises(ReasonerOutputError, match="after 2 attempts"):
            _reasoner(fake).plan("tray")

    def test_sends_sampling_params_and_token_limit(self, valid_layout):
        """권장 샘플링 값과 max_tokens를 요청에 넣는다."""
        fake = FakeOpenAI([make_planner_response_json(valid_layout)])
        CosmosReasoner(client=fake, model=MODEL, max_tokens=123).plan("tray")
        call = fake.calls[0]
        assert call["max_tokens"] == 123
        assert {key: call[key] for key in SAMPLING_PARAMS} == SAMPLING_PARAMS
        assert call["extra_body"] == VLLM_SAMPLING_PARAMS
        assert call["seed"] == SAMPLING_SEED

    def test_sends_response_schema_for_board(self, valid_layout):
        """planner는 보드 크기에 맞는 JSON Schema를 structured output으로 요청한다."""
        # 8x8 레이아웃은 5x5 보드 밖에 있다. 배치 재요청이 일어나므로 응답을 2개 둔다.
        fake = FakeOpenAI([make_planner_response_json(valid_layout)] * 2)
        reasoner = CosmosReasoner(client=fake, model=MODEL, board=Board(rows=5, cols=5))
        reasoner.plan("tray")
        response_format = fake.calls[0]["response_format"]
        assert response_format["type"] == "json_schema"
        schema = response_format["json_schema"]["schema"]
        assert schema["$defs"]["PlannerPart"]["properties"]["row"]["maximum"] == 4

    def test_error_keeps_raw_responses(self):
        """실패하면 받은 원시 응답을 순서대로 예외에 담는다."""
        fake = FakeOpenAI(["first", "second"])
        with pytest.raises(ReasonerOutputError) as exc_info:
            _reasoner(fake).plan("tray")
        assert exc_info.value.responses == ["first", "second"]

    def test_fail_without_retry_when_truncated(self, valid_layout):
        """응답이 토큰 상한에서 잘리면 다시 요청하지 않고 ReasonerOutputError를 낸다."""
        fake = FakeOpenAI([make_planner_response_json(valid_layout)] * 2, finish_reason="length")
        with pytest.raises(ReasonerOutputError, match="limit of") as exc_info:
            _reasoner(fake).plan("tray")
        assert len(fake.calls) == 1, f"재요청 횟수: {len(fake.calls) - 1}"
        assert exc_info.value.responses == [make_planner_response_json(valid_layout)]
        assert exc_info.value.truncated

    def test_fail_with_empty_response(self):
        """빈 응답이면 ReasonerOutputError를 낸다."""
        with pytest.raises(ReasonerOutputError, match="empty response"):
            _reasoner(FakeOpenAI([None])).plan("tray")

    @pytest.mark.parametrize(("api_key", "expected_key"), [("test-key", "test-key"), ("", None)])
    def test_from_settings_builds_client(self, monkeypatch, api_key, expected_key):
        """설정의 base URL, 모델 ID, API 키로 클라이언트를 만든다. 키가 없으면 대체 값을 쓴다."""
        monkeypatch.setenv("REASONER_BASE_URL", BASE_URL)
        monkeypatch.setenv("COSMOS_REASONER_MODEL", MODEL)
        monkeypatch.setenv("REASONER_API_KEY", api_key)
        settings = Settings(_env_file=None)  # pyright: ignore[reportCallIssue]
        reasoner = CosmosReasoner.from_settings(settings)
        client = reasoner._client
        assert isinstance(client, OpenAI)
        assert str(client.base_url) == BASE_URL
        assert client.api_key == (expected_key or UNUSED_API_KEY)
        assert reasoner._model == MODEL
        assert reasoner._max_tokens == settings.reasoner_max_tokens
