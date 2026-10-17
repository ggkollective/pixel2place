import json
import re

import pytest
from PIL import Image

from pixel2place.layout.schema import Board
from pixel2place.reasoner.client import CosmosReasoner, ReasonerOutputError
from pixel2place.reasoner.expansion import (
    EXPANSION_SCHEMA_NAME,
    ImagePrompt,
    VideoPrompt,
    expansion_json_schema,
)
from pixel2place.reasoner.prompts import load_system_prompt
from tests.fakes.fake_openai import FakeOpenAI

SUBJECT = {
    "description": "A gray plastic kitting tray.",
    "location": "The center of the image.",
    "appearance_details": "Matte plastic with square slots.",
}
IMAGE_PROMPT = {
    "subjects": [SUBJECT],
    "background_setting": "A steel workbench.",
    "lighting": "Soft light from above.",
    "aesthetics": "Neutral colors.",
    "cinematography": "Top-down view.",
    "style_medium": "A realistic industrial photograph.",
}
VIDEO_PROMPT = {
    **IMAGE_PROMPT,
    "temporal_caption": "A gripper places an orange tile in the empty slot.",
    "actions": [{"time": "0-2s", "description": "The gripper moves down."}],
}


def _reasoner(fake: FakeOpenAI) -> CosmosReasoner:
    return CosmosReasoner(client=fake, model="m")


@pytest.mark.unit
class TestExpansionSchema:
    """확장 응답 스키마 테스트."""

    def test_video_schema_adds_temporal_keys(self):
        """영상 스키마는 이미지 키에 시간 순서 설명과 동작 목록을 더한다."""
        image_keys = set(expansion_json_schema("image")["properties"])
        video_keys = set(expansion_json_schema("video")["properties"])
        assert video_keys - image_keys == {"temporal_caption", "actions"}

    @pytest.mark.parametrize("target", ["image", "video"])
    def test_schema_has_no_korean_description(self, target):
        """서버에 보내는 스키마에는 한국어 docstring이 없다."""
        schema_text = json.dumps(expansion_json_schema(target), ensure_ascii=False)
        assert re.search("[가-힣]", schema_text) is None

    @pytest.mark.parametrize("name", ["expand_image", "expand_video"])
    def test_loads_system_prompt(self, name):
        """확장 시스템 지시문을 읽는다."""
        assert "Write one JSON object" in load_system_prompt(name, board=Board())


@pytest.mark.unit
class TestExpandPrompt:
    """CosmosReasoner.expand_prompt 테스트."""

    def test_image_returns_validated_json(self):
        """응답을 검증하고 ImagePrompt JSON 문자열을 반환한다."""
        fake = FakeOpenAI([f"<think>ok</think>{json.dumps(IMAGE_PROMPT)}"])
        expanded = _reasoner(fake).expand_prompt("a kitting tray", target="image")
        assert ImagePrompt.model_validate_json(expanded).model_dump() == IMAGE_PROMPT
        call = fake.calls[0]
        assert call["messages"][1] == {"role": "user", "content": "a kitting tray"}
        json_schema = call["response_format"]["json_schema"]
        assert json_schema["name"] == EXPANSION_SCHEMA_NAME
        assert json_schema["schema"] == expansion_json_schema("image")

    def test_video_sends_first_frame(self):
        """첫 프레임을 PNG data URL로 함께 보내고 영상 스키마를 쓴다."""
        fake = FakeOpenAI([json.dumps(VIDEO_PROMPT)])
        frame = Image.new("RGB", (8, 8), "white")
        expanded = _reasoner(fake).expand_prompt("place a tile", target="video", image=frame)
        assert VideoPrompt.model_validate_json(expanded).actions[0].time == "0-2s"
        text_part, image_part = fake.calls[0]["messages"][1]["content"]
        assert text_part == {"type": "text", "text": "place a tile"}
        assert image_part["image_url"]["url"].startswith("data:image/png;base64,")
        schema = fake.calls[0]["response_format"]["json_schema"]["schema"]
        assert schema == expansion_json_schema("video")

    def test_fail_without_retry_when_invalid(self):
        """형식이 틀리면 재요청하지 않고 원시 응답과 함께 실패한다."""
        fake = FakeOpenAI([json.dumps({"subjects": []})])
        with pytest.raises(ReasonerOutputError, match="not a valid image prompt") as exc_info:
            _reasoner(fake).expand_prompt("a kitting tray", target="image")
        assert exc_info.value.responses == [json.dumps({"subjects": []})]
        assert len(fake.calls) == 1
