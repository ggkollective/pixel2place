"""Cosmos 생성 서버 클라이언트.

생성 서버(vLLM-Omni)는 OpenAI 호환 경로를 제공한다. vLLM 전용 인자가 많다.
그래서 `openai` SDK 대신 `httpx2.Client` 1개로 두 경로를 호출한다.

- text-to-image: `POST images/generations` (JSON)
- image-to-video: `POST videos/sync` (multipart). 응답은 MP4 바이트다.
"""

import base64
import io
import logging
import re
import time
from dataclasses import dataclass
from typing import Any

import httpx2
from PIL import Image

from pixel2place.config import GeneratorSettings

logger = logging.getLogger(__name__)

IMAGES_PATH = "images/generations"
VIDEOS_PATH = "videos/sync"
VIDEO_MEDIA_TYPE = "video/mp4"
ERROR_EXCERPT_CHARS = 500
_SIZE_PATTERN = re.compile(r"(\d+)x(\d+)")


class GeneratorError(RuntimeError):
    """생성 서버가 성공 응답을 반환하지 않았다."""

    def __init__(self, message: str, *, status_code: int) -> None:
        super().__init__(message)
        self.status_code = status_code


@dataclass(frozen=True)
class ImageSize:
    """생성 이미지·영상의 크기. 서버에는 `WIDTHxHEIGHT` 문자열로 보낸다."""

    width: int
    height: int

    @classmethod
    def parse(cls, text: str) -> "ImageSize":
        """`640x640` 형식의 문자열을 읽는다. 형식이 틀리거나 값이 0이면 `ValueError`를 낸다."""
        match = _SIZE_PATTERN.fullmatch(text.strip())
        if match is None:
            raise ValueError(f"Size must have the format WIDTHxHEIGHT, for example 640x640: {text}")
        size = cls(width=int(match[1]), height=int(match[2]))
        if size.width == 0 or size.height == 0:
            raise ValueError(f"Width and height must be more than 0: {text}")
        return size

    def __str__(self) -> str:
        return f"{self.width}x{self.height}"


class CosmosGenerator:
    """Cosmos3 생성 서버로 이미지와 영상을 만든다."""

    def __init__(self, *, client: httpx2.Client, model: str) -> None:
        self._client = client
        self._model = model

    @classmethod
    def from_settings(cls, settings: GeneratorSettings) -> "CosmosGenerator":
        client = httpx2.Client(
            base_url=settings.generator_base_url,
            timeout=settings.generator_timeout_s,
        )
        return cls(client=client, model=settings.cosmos_generator_model)

    def generate_image(
        self, prompt: str, *, size: ImageSize, steps: int, guidance: float, seed: int
    ) -> Image.Image:
        """텍스트 지시문으로 이미지 1장을 만든다."""
        body = {
            "model": self._model,
            "prompt": prompt,
            "size": str(size),
            "num_inference_steps": steps,
            "guidance_scale": guidance,
            "seed": seed,
        }
        response = self._post(IMAGES_PATH, kind="image", json=body)
        encoded: str = response.json()["data"][0]["b64_json"]
        image = Image.open(io.BytesIO(base64.b64decode(encoded)))
        image.load()
        return image

    def generate_video(
        self,
        first_frame: Image.Image,
        prompt: str,
        *,
        size: ImageSize,
        num_frames: int,
        fps: int,
        steps: int,
        guidance: float,
        seed: int,
    ) -> bytes:
        """첫 프레임 이미지와 동작 지시문으로 영상을 만든다. MP4 바이트를 반환한다."""
        form = {
            "model": self._model,
            "prompt": prompt,
            "size": str(size),
            "num_frames": str(num_frames),
            "fps": str(fps),
            "num_inference_steps": str(steps),
            "guidance_scale": str(guidance),
            "seed": str(seed),
        }
        frame = io.BytesIO()
        first_frame.save(frame, format="PNG")
        files = {"input_reference": ("first_frame.png", frame.getvalue(), "image/png")}
        response = self._post(
            VIDEOS_PATH,
            kind="video",
            data=form,
            files=files,
            headers={"Accept": VIDEO_MEDIA_TYPE},
        )
        return response.content

    def _post(self, path: str, *, kind: str, **kwargs: Any) -> httpx2.Response:
        """요청 1회를 보낸다. 성공 응답이 아니면 `GeneratorError`를 낸다."""
        started = time.monotonic()
        response = self._client.post(path, **kwargs)
        extra = {
            "kind": kind,
            "status_code": response.status_code,
            "elapsed_s": round(time.monotonic() - started, 1),
        }
        if response.is_success:
            logger.info("Generator response received.", extra=extra)
            return response
        logger.warning("Generator request failed.", extra=extra)
        raise GeneratorError(
            f"Generator returned HTTP {response.status_code} for {kind} generation. "
            f"Detail: {_error_detail(response)[:ERROR_EXCERPT_CHARS]}",
            status_code=response.status_code,
        )


def _error_detail(response: httpx2.Response) -> str:
    """오류 응답의 메시지를 읽는다.

    서버는 `{"error": {"message": ...}}` 또는 `{"detail": ...}` 형식을 쓴다.
    두 형식이 아니면 본문 텍스트를 쓴다.
    """
    try:
        payload = response.json()
    except ValueError:
        return response.text
    if isinstance(payload, dict):
        error = payload.get("error")
        if isinstance(error, dict) and "message" in error:
            return str(error["message"])
        if "detail" in payload:
            return str(payload["detail"])
    return response.text
