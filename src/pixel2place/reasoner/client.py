"""Cosmos Reasoner 클라이언트.

Reasoner 서버(vLLM)는 OpenAI 호환 API를 제공한다. 그래서 `openai.OpenAI` 클라이언트를 쓴다.
"""

import base64
import io
import logging
import time
from typing import Any, Protocol

from openai import OpenAI
from PIL import Image
from pydantic import ValidationError

from pixel2place.config import DEFAULT_REASONER_MAX_TOKENS, Settings
from pixel2place.feasibility.checker import Issue, check_layout
from pixel2place.layout.schema import Board, Layout
from pixel2place.reasoner.expansion import (
    EXPANSION_SCHEMA_NAME,
    ExpansionTarget,
    expansion_json_schema,
    parse_expansion_response,
)
from pixel2place.reasoner.extract import JsonExtractionError, extract_json_object
from pixel2place.reasoner.prompts import PromptName, load_system_prompt
from pixel2place.reasoner.response import (
    RESPONSE_SCHEMA_NAME,
    LayoutConverter,
    parse_parser_response,
    parse_planner_response,
    parser_json_schema,
    planner_json_schema,
)

logger = logging.getLogger(__name__)

# 모델 출력의 형식이나 배치가 틀리면 문제를 알려 주고 1회 다시 요청한다.
MAX_ATTEMPTS = 2
ERROR_EXCERPT_CHARS = 500
PARSER_USER_TEXT = "Detect every part on this board."
# openai 클라이언트는 빈 API 키를 거부한다. 키를 검사하지 않는 서버에는 이 값을 보낸다.
UNUSED_API_KEY = "unused"
FINISH_REASON_LENGTH = "length"

# Cosmos3 README의 권장 샘플링 값. 기본 샘플링에서는 모델이 같은 출력을 반복하고 멈추지 않는다.
SAMPLING_PARAMS: dict[str, float] = {"temperature": 0.7, "top_p": 0.8, "presence_penalty": 1.5}
# OpenAI 표준 인자가 아닌 vLLM 전용 샘플링 값.
VLLM_SAMPLING_PARAMS: dict[str, float] = {"top_k": 20, "repetition_penalty": 1.0}
# Cosmos3 README의 권장 seed. 같은 서버에서 같은 입력은 같은 출력이 된다.
SAMPLING_SEED = 3407

Message = dict[str, Any]


class ChatClient(Protocol):
    """테스트에서 Fake로 바꿀 수 있는 OpenAI 클라이언트의 최소 형태."""

    @property
    def chat(self) -> Any: ...


class ReasonerOutputError(RuntimeError):
    """모델 응답이 유효한 레이아웃이 아니다.

    `responses`에는 실패까지 받은 원시 응답 텍스트가 순서대로 있다.
    `truncated`는 출력 토큰 한도로 응답이 잘렸는지 나타낸다.
    """

    def __init__(self, message: str, *, responses: list[str], truncated: bool = False) -> None:
        super().__init__(message)
        self.responses = responses
        self.truncated = truncated


class CosmosReasoner:
    """지시문이나 보드 이미지에서 `Layout`을 만든다."""

    def __init__(
        self,
        *,
        client: ChatClient,
        model: str,
        max_tokens: int = DEFAULT_REASONER_MAX_TOKENS,
        board: Board | None = None,
    ) -> None:
        self._client = client
        self._model = model
        self._max_tokens = max_tokens
        self._board = board or Board()
        self._request_count = 0
        # 지시문마다 structured output 형식과 응답 변환 함수가 다르다.
        self._formats: dict[PromptName, tuple[dict[str, Any], LayoutConverter]] = {
            "planner": (
                _response_format(RESPONSE_SCHEMA_NAME, planner_json_schema(self._board)),
                parse_planner_response,
            ),
            "parser": (
                _response_format(RESPONSE_SCHEMA_NAME, parser_json_schema(self._board)),
                parse_parser_response,
            ),
        }

    @property
    def request_count(self) -> int:
        """HTTP 실패도 포함한 논리 요청 수. SDK 내부 재시도는 제외한다."""
        return self._request_count

    @classmethod
    def from_settings(cls, settings: Settings) -> "CosmosReasoner":
        api_key = settings.reasoner_api_key
        client = OpenAI(
            base_url=settings.reasoner_base_url,
            api_key=api_key.get_secret_value() if api_key else UNUSED_API_KEY,
            timeout=settings.reasoner_timeout_s,
            max_retries=settings.reasoner_max_retries,
        )
        return cls(
            client=client,
            model=settings.cosmos_reasoner_model,
            max_tokens=settings.reasoner_max_tokens,
        )

    def plan(self, prompt: str) -> Layout:
        """텍스트 지시문에서 목표 레이아웃을 만든다.

        레이아웃이 배치 규칙을 어기면 위반 내용을 알려 주고 1회 다시 요청한다.
        """
        return self._request_layout("planner", user_content=prompt, check_placement=True)

    def parse(self, image: Image.Image) -> Layout:
        """top-down 보드 이미지에서 레이아웃을 읽는다.

        모델은 부품마다 bbox를 쓴다. 코드가 bbox를 격자에 맞춘다.
        """
        user_content = [
            {"type": "text", "text": PARSER_USER_TEXT},
            {"type": "image_url", "image_url": {"url": _png_data_url(image)}},
        ]
        # 판독기는 이미지를 그대로 읽어야 한다. 그래서 배치 규칙 위반을 고치게 하지 않는다.
        return self._request_layout("parser", user_content=user_content, check_placement=False)

    def expand_prompt(
        self, text: str, *, target: ExpansionTarget, image: Image.Image | None = None
    ) -> str:
        """짧은 지시문을 생성 서버용 구조화 JSON 문자열로 확장한다.

        `image`가 있으면 VLM이 첫 프레임을 보고 장면을 쓴다.
        structured output이 형식을 강제한다. 그래서 재요청하지 않는다.
        """
        prompt_name: PromptName = "expand_image" if target == "image" else "expand_video"
        user_content: Any = text
        if image is not None:
            user_content = [
                {"type": "text", "text": text},
                {"type": "image_url", "image_url": {"url": _png_data_url(image)}},
            ]
        messages: list[Message] = [
            {"role": "system", "content": load_system_prompt(prompt_name, board=self._board)},
            {"role": "user", "content": user_content},
        ]
        responses: list[str] = []
        response_text = self._complete(
            messages,
            prompt_name=prompt_name,
            response_format=_response_format(EXPANSION_SCHEMA_NAME, expansion_json_schema(target)),
            responses=responses,
        )
        try:
            return parse_expansion_response(extract_json_object(response_text), target)
        except (JsonExtractionError, ValidationError) as exc:
            raise ReasonerOutputError(
                f"Reasoner output is not a valid {target} prompt. "
                f"Error: {str(exc)[:ERROR_EXCERPT_CHARS]}",
                responses=responses,
            ) from exc

    def _request_layout(
        self, prompt_name: PromptName, *, user_content: Any, check_placement: bool
    ) -> Layout:
        messages: list[Message] = [
            {"role": "system", "content": load_system_prompt(prompt_name, board=self._board)},
            {"role": "user", "content": user_content},
        ]
        response_format, convert = self._formats[prompt_name]
        last_error = ""
        responses: list[str] = []
        # 배치 규칙만 어긴 레이아웃. 마지막 시도의 형식이 틀리면 이 값을 반환한다.
        fallback: Layout | None = None
        for attempt in range(1, MAX_ATTEMPTS + 1):
            text = self._complete(
                messages,
                prompt_name=prompt_name,
                response_format=response_format,
                responses=responses,
            )
            try:
                layout = convert(extract_json_object(text), self._board)
            except (JsonExtractionError, ValidationError) as exc:
                last_error = str(exc)
                logger.warning(
                    "Reasoner output is not a valid layout.",
                    extra={"prompt_name": prompt_name, "attempt": attempt},
                )
                feedback = _format_feedback(last_error)
            else:
                issues = check_layout(layout).issues if check_placement else []
                if not issues or attempt == MAX_ATTEMPTS:
                    return layout
                fallback = layout
                logger.warning(
                    "Reasoner layout breaks the placement rules.",
                    extra={"prompt_name": prompt_name, "attempt": attempt, "issues": len(issues)},
                )
                feedback = _placement_feedback(issues)
            messages += [
                {"role": "assistant", "content": text},
                {"role": "user", "content": feedback},
            ]
        if fallback is not None:
            return fallback
        raise ReasonerOutputError(
            f"Reasoner output is not a valid layout after {MAX_ATTEMPTS} attempts. "
            f"Last error: {last_error[:ERROR_EXCERPT_CHARS]}",
            responses=responses,
        )

    def _complete(
        self,
        messages: list[Message],
        *,
        prompt_name: PromptName,
        response_format: dict[str, Any],
        responses: list[str],
    ) -> str:
        """요청 1회를 보내고 응답 텍스트를 `responses`에 추가한 뒤 반환한다."""
        started = time.monotonic()
        self._request_count += 1
        response = self._client.chat.completions.create(
            model=self._model,
            messages=messages,
            max_tokens=self._max_tokens,
            seed=SAMPLING_SEED,
            response_format=response_format,
            **SAMPLING_PARAMS,
            extra_body=VLLM_SAMPLING_PARAMS,
        )
        choice = response.choices[0]
        text = choice.message.content or ""
        responses.append(text)
        logger.info(
            "Reasoner response received.",
            extra={
                "prompt_name": prompt_name,
                "elapsed_s": round(time.monotonic() - started, 1),
                "completion_tokens": response.usage.completion_tokens if response.usage else None,
                "finish_reason": choice.finish_reason,
            },
        )
        # 잘린 응답은 다시 요청해도 다시 잘린다. 그래서 재요청하지 않는다.
        if choice.finish_reason == FINISH_REASON_LENGTH:
            raise ReasonerOutputError(
                f"Reasoner output reached the limit of {self._max_tokens} tokens.",
                responses=responses,
                truncated=True,
            )
        if not text:
            raise ReasonerOutputError("Reasoner returned an empty response.", responses=responses)
        return text


def _response_format(name: str, schema: dict[str, Any]) -> dict[str, Any]:
    return {
        "type": "json_schema",
        "json_schema": {"name": name, "schema": schema, "strict": True},
    }


def _format_feedback(error: str) -> str:
    return (
        "Your output is not a valid layout. "
        f"Error: {error[:ERROR_EXCERPT_CHARS]} "
        "Write the complete JSON object again and no other text."
    )


def _placement_feedback(issues: list[Issue]) -> str:
    problems = "\n".join(f"- {issue.message}" for issue in issues)
    return (
        "Your layout breaks the placement rules. "
        "Part P001 is the first part in your list, P002 is the second part, and so on.\n"
        f"Problems:\n{problems}\n"
        "Move or remove parts to correct all problems. "
        "Write the complete JSON object again and no other text."
    )


def _png_data_url(image: Image.Image) -> str:
    buffer = io.BytesIO()
    image.save(buffer, format="PNG")
    encoded = base64.b64encode(buffer.getvalue()).decode("ascii")
    return f"data:image/png;base64,{encoded}"
