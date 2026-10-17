"""`openai.OpenAI`의 `chat.completions.create`만 흉내 내는 Fake."""

from dataclasses import dataclass, field
from types import SimpleNamespace
from typing import Any


@dataclass
class _Completions:
    responses: list[str | None]
    finish_reason: str = "stop"
    completion_tokens: int = 1
    calls: list[dict[str, Any]] = field(default_factory=list)

    def create(self, *, model: str, messages: list[dict[str, Any]], **kwargs: Any) -> Any:
        # 호출 뒤 messages가 바뀌어도 기록이 남도록 복사한다.
        self.calls.append({"model": model, "messages": list(messages), **kwargs})
        content = self.responses.pop(0)
        choice = SimpleNamespace(
            message=SimpleNamespace(content=content), finish_reason=self.finish_reason
        )
        usage = SimpleNamespace(completion_tokens=self.completion_tokens)
        return SimpleNamespace(choices=[choice], usage=usage)


class FakeOpenAI:
    """준비한 응답 텍스트를 순서대로 돌려준다."""

    def __init__(self, responses: list[str | None], *, finish_reason: str = "stop") -> None:
        self.completions = _Completions(responses=list(responses), finish_reason=finish_reason)
        self.chat = SimpleNamespace(completions=self.completions)

    @property
    def calls(self) -> list[dict[str, Any]]:
        return self.completions.calls
