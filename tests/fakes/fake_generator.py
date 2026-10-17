"""Cosmos 생성 서버를 흉내 내는 `httpx2.MockTransport` handler."""

import base64
import io

import httpx2
from PIL import Image

from pixel2place.generation.cosmos_client import CosmosGenerator

FAKE_BASE_URL = "http://generator.test:8100/v1"
FAKE_MODEL = "fake-cosmos-generator"
FAKE_MP4_BYTES = b"\x00\x00\x00\x18ftypmp42"


def image_response(image: Image.Image) -> httpx2.Response:
    """이미지 1장을 담은 text-to-image 성공 응답을 만든다."""
    buffer = io.BytesIO()
    image.save(buffer, format="PNG")
    encoded = base64.b64encode(buffer.getvalue()).decode("ascii")
    return httpx2.Response(200, json={"data": [{"b64_json": encoded}]})


class FakeGeneratorServer:
    """준비한 응답 1개를 돌려주고 받은 요청을 기록한다."""

    def __init__(self, response: httpx2.Response) -> None:
        self.response = response
        self.requests: list[httpx2.Request] = []

    def __call__(self, request: httpx2.Request) -> httpx2.Response:
        request.read()
        self.requests.append(request)
        return self.response

    def generator(self) -> CosmosGenerator:
        client = httpx2.Client(base_url=FAKE_BASE_URL, transport=httpx2.MockTransport(self))
        return CosmosGenerator(client=client, model=FAKE_MODEL)
