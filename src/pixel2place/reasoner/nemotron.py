"""Token Factory의 모델 접근과 짧은 Nemotron 응답을 검증한다.

이 모듈은 HTTP 클라이언트의 수명과 호출 예산을 소유한다.
Cosmos 계획, agent 도구 조율과 로봇 제어는 소유하지 않는다.
"""

from datetime import UTC, datetime
from time import perf_counter

from openai import DefaultHttpxClient, OpenAI
from openai.pagination import SyncPage
from openai.types import Model
from openai.types.chat import ChatCompletion
from pydantic import BaseModel, ConfigDict

from pixel2place.config import TokenFactorySettings

VERIFICATION_PROMPT = "Reply with exactly READY."


class NemotronVerification(BaseModel):
    """인증 정보를 제외한 접속 검증 결과와 호출 조건."""

    model_config = ConfigDict(frozen=True)

    verified_at: datetime
    base_url: str
    available_models: tuple[str, ...]
    requested_model: str
    served_model: str
    completion_id: str
    prompt: str
    response_text: str
    finish_reason: str
    max_tokens: int
    timeout_s: float
    max_retries: int
    latency_s: float
    prompt_tokens: int | None
    completion_tokens: int | None
    reasoning_tokens: int | None


def verify_nemotron(settings: TokenFactorySettings) -> NemotronVerification:
    """모델 목록을 확인한 뒤 모델 요청을 1회 실행한다. 오류는 호출자에게 전달한다."""
    with create_nemotron_client(settings) as client:
        available = require_nemotron_model(client, model=settings.nemotron_model)
        started = perf_counter()
        response = client.chat.completions.create(
            model=settings.nemotron_model,
            messages=[{"role": "user", "content": VERIFICATION_PROMPT}],
            max_tokens=settings.nemotron_max_tokens,
        )
        latency_s = perf_counter() - started
        if not isinstance(response, ChatCompletion):
            raise ValueError("Token Factory returned an invalid completion object.")
        completion = ChatCompletion.model_validate(response.model_dump(warnings=False))

    if not completion.choices:
        raise ValueError("Nemotron returned no response choices.")
    choice = completion.choices[0]
    if choice.finish_reason != "stop":
        raise ValueError("Nemotron did not finish the response. Check the token limit.")
    if not choice.message.content or not choice.message.content.strip():
        raise ValueError("Nemotron returned an empty verification response.")
    if completion.model != settings.nemotron_model:
        raise ValueError("Nemotron returned a different model ID.")
    usage = completion.usage
    details = usage.completion_tokens_details if usage is not None else None
    return NemotronVerification(
        verified_at=datetime.now(UTC),
        base_url=str(settings.nebius_base_url),
        available_models=available,
        requested_model=settings.nemotron_model,
        served_model=completion.model,
        completion_id=completion.id,
        prompt=VERIFICATION_PROMPT,
        response_text=choice.message.content,
        finish_reason=choice.finish_reason,
        max_tokens=settings.nemotron_max_tokens,
        timeout_s=settings.nemotron_timeout_s,
        max_retries=0,
        latency_s=latency_s,
        prompt_tokens=usage.prompt_tokens if usage is not None else None,
        completion_tokens=usage.completion_tokens if usage is not None else None,
        reasoning_tokens=details.reasoning_tokens if details is not None else None,
    )


def create_nemotron_client(settings: TokenFactorySettings) -> OpenAI:
    """인증과 재시도 방침을 고정한 SDK 클라이언트를 만든다. 호출자가 닫는다."""
    return OpenAI(
        base_url=str(settings.nebius_base_url),
        api_key=settings.nebius_api_key.get_secret_value(),
        timeout=settings.nemotron_timeout_s,
        max_retries=0,
        http_client=DefaultHttpxClient(follow_redirects=False),
    )


def require_nemotron_model(client: OpenAI, *, model: str) -> tuple[str, ...]:
    """계정 모델 목록을 검증한다. 요청할 Nemotron이 없으면 중단한다."""
    page = client.models.list()
    if not isinstance(page, SyncPage):
        raise ValueError("Token Factory returned an invalid model catalog.")
    # SDK의 느슨한 응답 생성과 별도로 외부 데이터 계약을 검사한다.
    catalog = SyncPage[Model].model_validate(page.model_dump(warnings=False))
    available = tuple(
        entry.id
        for entry in catalog.data
        if entry.id.startswith("nvidia/") and "nemotron" in entry.id.lower()
    )
    if model not in available:
        raise ValueError("The configured Nemotron model is not available in this account.")
    return available
