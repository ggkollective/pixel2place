import json
import logging

import httpx2
import pytest
from PIL import Image
from pydantic import ValidationError

from pixel2place.config import GeneratorSettings
from pixel2place.generation.cosmos_client import (
    CosmosGenerator,
    GeneratorError,
    ImageSize,
)
from tests.fakes.fake_generator import (
    FAKE_BASE_URL,
    FAKE_MODEL,
    FAKE_MP4_BYTES,
    FakeGeneratorServer,
    image_response,
)

MODEL = FAKE_MODEL
BASE_URL = FAKE_BASE_URL
SIZE = ImageSize(width=64, height=48)


@pytest.mark.unit
class TestImageSize:
    """ImageSize 테스트."""

    def test_parse_and_format(self):
        """`WIDTHxHEIGHT`를 읽고 같은 형식으로 쓴다."""
        size = ImageSize.parse(" 640x480 ")
        assert size == ImageSize(width=640, height=480)
        assert str(size) == "640x480"

    @pytest.mark.parametrize("text", ["640", "640x", "x480", "640*480", "0x480", "-1x2"])
    def test_rejects_bad_format(self, text):
        """형식이 틀리거나 값이 0이면 ValueError다."""
        with pytest.raises(ValueError):
            ImageSize.parse(text)


@pytest.mark.unit
class TestGenerateImage:
    """text-to-image 요청 테스트."""

    def test_sends_json_and_decodes_image(self, caplog):
        """vLLM 인자를 JSON 본문에 넣고 b64_json 이미지를 디코드한다."""
        expected = Image.new("RGB", (64, 48), "orange")
        server = FakeGeneratorServer(image_response(expected))
        caplog.set_level(logging.INFO)
        image = server.generator().generate_image(
            "a tray", size=SIZE, steps=20, guidance=6.0, seed=3
        )
        assert image.size == (64, 48)
        assert image.convert("RGB").getpixel((0, 0)) == expected.getpixel((0, 0))
        request = server.requests[0]
        assert str(request.url) == f"{BASE_URL}/images/generations"
        assert json.loads(request.content) == {
            "model": MODEL,
            "prompt": "a tray",
            "size": "64x48",
            "num_inference_steps": 20,
            "guidance_scale": 6.0,
            "seed": 3,
        }
        assert "Generator response received." in caplog.messages

    def test_error_response_raises_with_detail(self):
        """성공 응답이 아니면 상태 코드와 detail을 담은 GeneratorError다."""
        server = FakeGeneratorServer(httpx2.Response(400, json={"detail": "Model mismatch."}))
        with pytest.raises(GeneratorError, match="HTTP 400.*Model mismatch") as exc_info:
            server.generator().generate_image("a tray", size=SIZE, steps=1, guidance=1.0, seed=0)
        assert exc_info.value.status_code == 400

    def test_error_response_reads_openai_error_message(self):
        """OpenAI 형식 오류 응답은 `error.message`를 쓴다."""
        payload = {"error": {"message": "Model mismatch.", "code": 400}}
        server = FakeGeneratorServer(httpx2.Response(400, json=payload))
        with pytest.raises(GeneratorError, match=r"Detail: Model mismatch\.$"):
            server.generator().generate_image("a tray", size=SIZE, steps=1, guidance=1.0, seed=0)

    def test_error_response_without_json_uses_text(self):
        """오류 본문이 JSON이 아니면 본문 텍스트를 쓴다."""
        server = FakeGeneratorServer(httpx2.Response(502, text="Bad gateway"))
        with pytest.raises(GeneratorError, match="HTTP 502.*Bad gateway"):
            server.generator().generate_image("a tray", size=SIZE, steps=1, guidance=1.0, seed=0)


@pytest.mark.unit
class TestGenerateVideo:
    """image-to-video 요청 테스트."""

    def test_sends_multipart_and_returns_mp4(self):
        """첫 프레임을 PNG 파일로 보내고 MP4 바이트를 반환한다."""
        server = FakeGeneratorServer(httpx2.Response(200, content=FAKE_MP4_BYTES))
        frame = Image.new("RGB", (64, 48), "white")
        video = server.generator().generate_video(
            frame, "place a tile", size=SIZE, num_frames=9, fps=24, steps=5, guidance=6.0, seed=1
        )
        assert video == FAKE_MP4_BYTES
        request = server.requests[0]
        assert str(request.url) == f"{BASE_URL}/videos/sync"
        assert request.headers["accept"] == "video/mp4"
        assert request.headers["content-type"].startswith("multipart/form-data")
        body = request.content
        for name, value in [
            ("model", MODEL),
            ("prompt", "place a tile"),
            ("size", "64x48"),
            ("num_frames", "9"),
            ("fps", "24"),
            ("num_inference_steps", "5"),
            ("guidance_scale", "6.0"),
            ("seed", "1"),
        ]:
            assert f'name="{name}"\r\n\r\n{value}\r\n'.encode() in body, name
        assert b'name="input_reference"; filename="first_frame.png"' in body
        assert b"\x89PNG" in body

    def test_error_response_raises(self):
        """영상 요청이 실패하면 GeneratorError다."""
        server = FakeGeneratorServer(
            httpx2.Response(504, json={"detail": "Video generation timed out."})
        )
        with pytest.raises(GeneratorError, match="video generation.*timed out"):
            server.generator().generate_video(
                Image.new("RGB", (8, 8)),
                "p",
                size=SIZE,
                num_frames=1,
                fps=1,
                steps=1,
                guidance=1.0,
                seed=0,
            )


@pytest.mark.unit
class TestGeneratorSettings:
    """생성 서버 설정 테스트."""

    def test_from_settings_builds_client(self):
        """설정의 base URL, timeout, 모델 ID로 클라이언트를 만든다."""
        settings = GeneratorSettings(
            generator_base_url=BASE_URL, cosmos_generator_model=MODEL, generator_timeout_s=12.0
        )
        generator = CosmosGenerator.from_settings(settings)
        assert str(generator._client.base_url) == f"{BASE_URL}/"
        assert generator._client.timeout == httpx2.Timeout(12.0)
        assert generator._model == MODEL

    @pytest.mark.parametrize("base_url", ["", "   "])
    def test_rejects_blank_base_url(self, monkeypatch, tmp_path, base_url):
        """빈 base URL은 설정 오류다."""
        monkeypatch.chdir(tmp_path)
        monkeypatch.setenv("GENERATOR_BASE_URL", base_url)
        monkeypatch.setenv("COSMOS_GENERATOR_MODEL", MODEL)
        with pytest.raises(ValidationError, match="generator_base_url"):
            GeneratorSettings()  # pyright: ignore[reportCallIssue]
