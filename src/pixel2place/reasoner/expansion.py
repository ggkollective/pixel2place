"""지시문 확장 응답 스키마.

Cosmos3 생성 모델은 구조화 JSON 지시문에서 품질이 높다.
VLM이 짧은 지시문을 이 스키마의 JSON으로 확장한다. 키 구성은 Cosmos3 구조화 지시문의 핵심 키다.
"""

from typing import Annotated, Any, Literal

from pydantic import BaseModel, ConfigDict, Field

ExpansionTarget = Literal["image", "video"]
"""확장한 지시문을 쓰는 생성 기능. image는 text-to-image, video는 image-to-video다."""

EXPANSION_SCHEMA_NAME = "generation_prompt"
# 문자열과 목록에 상한이 없으면 모델이 출력을 끝없이 늘릴 수 있다.
MAX_TEXT_CHARS = 600
MAX_SUBJECTS = 8
MAX_ACTIONS = 8

Text = Annotated[str, Field(min_length=1, max_length=MAX_TEXT_CHARS)]


class _PromptModel(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)


class Subject(_PromptModel):
    """장면 안의 대상 1개."""

    description: Text
    location: Text
    appearance_details: Text


class Action(_PromptModel):
    """영상 안의 동작 1개. `time`은 `0-2s`처럼 동작 구간을 쓴다."""

    time: Text
    description: Text


class ImagePrompt(_PromptModel):
    """text-to-image 구조화 지시문."""

    subjects: list[Subject] = Field(min_length=1, max_length=MAX_SUBJECTS)
    background_setting: Text
    lighting: Text
    aesthetics: Text
    cinematography: Text
    style_medium: Text


class VideoPrompt(ImagePrompt):
    """image-to-video 구조화 지시문. 이미지 키에 시간 순서 설명과 동작 목록을 더한다."""

    temporal_caption: Text
    actions: list[Action] = Field(min_length=1, max_length=MAX_ACTIONS)


PROMPT_MODELS: dict[ExpansionTarget, type[ImagePrompt]] = {
    "image": ImagePrompt,
    "video": VideoPrompt,
}


def expansion_json_schema(target: ExpansionTarget) -> dict[str, Any]:
    """확장 요청의 vLLM structured output JSON Schema를 만든다."""
    schema = PROMPT_MODELS[target].model_json_schema()
    # description에는 한국어 docstring이 들어간다. 서버에 보내는 값은 영어만 쓴다.
    for definition in [schema, *schema.get("$defs", {}).values()]:
        definition.pop("description", None)
    return schema


def parse_expansion_response(data: dict[str, Any], target: ExpansionTarget) -> str:
    """확장 응답 JSON을 검증하고 생성 서버에 보낼 JSON 문자열로 바꾼다."""
    return PROMPT_MODELS[target].model_validate(data).model_dump_json()
