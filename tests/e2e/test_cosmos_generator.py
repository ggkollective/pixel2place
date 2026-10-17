"""실제 Cosmos 생성 서버와 VLM 서버를 호출한다. `.env`나 환경 변수에 두 서버 설정이 필요하다.

영상 생성은 약 3분이 걸린다. 그래서 e2e에 넣지 않는다. `preview-video` CLI로 확인한다.
"""

import json

import pytest
from pydantic import ValidationError

from pixel2place.config import GeneratorSettings, Settings
from pixel2place.generation.cosmos_client import CosmosGenerator, ImageSize
from pixel2place.reasoner.client import CosmosReasoner
from pixel2place.reasoner.expansion import ImagePrompt

E2E_PROMPT = "An industrial kitting tray with blue and gray parts."
E2E_SIZE = ImageSize(width=256, height=256)
E2E_STEPS = 20


@pytest.fixture
def generator() -> CosmosGenerator:
    try:
        settings = GeneratorSettings()  # pyright: ignore[reportCallIssue]
    except ValidationError as exc:
        missing = ", ".join(str(error["loc"][0]).upper() for error in exc.errors())
        pytest.skip(f"E2E test needs {missing}.")
    return CosmosGenerator.from_settings(settings)


@pytest.mark.e2e
class TestCosmosGenerator:
    """Cosmos 생성 서버와 지시문 확장 e2e 테스트."""

    def test_expand_and_generate_image(self, generator):
        """확장한 지시문으로 요청한 크기의 이미지를 만든다."""
        reasoner = CosmosReasoner.from_settings(Settings())  # pyright: ignore[reportCallIssue]
        expanded = reasoner.expand_prompt(E2E_PROMPT, target="image")
        assert ImagePrompt.model_validate(json.loads(expanded)).subjects
        image = generator.generate_image(
            expanded, size=E2E_SIZE, steps=E2E_STEPS, guidance=6.0, seed=0
        )
        assert image.size == (E2E_SIZE.width, E2E_SIZE.height)
