"""모델 응답 텍스트에서 JSON 객체를 꺼낸다."""

import json
import re
from typing import Any

# 모델은 답 앞에 <think>...</think> 블록을 쓸 수 있다.
_THINK_BLOCK = re.compile(r"<think>.*?</think>", re.DOTALL | re.IGNORECASE)
_CODE_FENCE = re.compile(r"```(?:json)?\s*(.*?)```", re.DOTALL | re.IGNORECASE)


class JsonExtractionError(ValueError):
    """응답 텍스트에 JSON 객체가 없다."""


def extract_json_object(text: str) -> dict[str, Any]:
    """응답에서 첫 번째 JSON 객체를 찾아 dict로 반환한다.

    순서:
    1. `<think>` 블록을 지운다.
    2. 코드 펜스가 있으면 펜스 안의 텍스트를 쓴다.
    3. 첫 `{`부터 시작하는 JSON 객체를 디코드한다.
    """
    body = _THINK_BLOCK.sub("", text)
    fence = _CODE_FENCE.search(body)
    if fence is not None:
        body = fence.group(1)

    decoder = json.JSONDecoder()
    start = body.find("{")
    while start != -1:
        try:
            value, _ = decoder.raw_decode(body, start)
        except json.JSONDecodeError:
            start = body.find("{", start + 1)
            continue
        if isinstance(value, dict):
            return value
        start = body.find("{", start + 1)
    raise JsonExtractionError("Response text contains no JSON object.")
