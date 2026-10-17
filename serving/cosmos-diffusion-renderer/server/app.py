"""Cosmos DiffusionRenderer HTTP 서버.

Inverse 처리 흐름(프레임 → G-buffer)과 Forward 처리 흐름(G-buffer + 조명 → RGB)을 상주시킨다.
전처리와 후처리는 upstream 모델 실행 스크립트와 같은 dataset 클래스와 함수를 쓴다.
그래서 같은 입력과 seed이면 upstream CLI와 같은 결과를 낸다.
"""

import asyncio
import base64
import gc
import io
import logging
import os
import tempfile
import time
import zipfile
from collections.abc import AsyncIterator, Callable
from contextlib import asynccontextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Annotated, Any

import numpy as np
from cosmos_predict1.diffusion.inference.diffusion_renderer_pipeline import (
    DiffusionRendererPipeline,
)
from cosmos_predict1.diffusion.inference.diffusion_renderer_utils.dataloader_utils import (
    dict_collation_fn,
)
from cosmos_predict1.diffusion.inference.diffusion_renderer_utils.dataset_inference import (
    VideoFramesDataset,
    VideoGBufferDataset,
)
from cosmos_predict1.diffusion.inference.diffusion_renderer_utils.rendering_utils import (
    GBUFFER_INDEX_MAPPING,
    envmap_vec,
)
from cosmos_predict1.diffusion.inference.diffusion_renderer_utils.utils_env_proj import (
    process_environment_map,
)
from cosmos_predict1.diffusion.inference.inference_forward_renderer import ENV_LIGHT_PATH_LIST
from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.responses import JSONResponse, Response
from huggingface_hub import snapshot_download
from PIL import Image, UnidentifiedImageError

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s %(message)s")
logger = logging.getLogger("diffusion_renderer.server")

# upstream이 기대하는 체크포인트 디렉터리. 각 항목은 HF 캐시 snapshot을 가리키는 symlink다.
CHECKPOINT_DIR = Path("/opt/diffusion-renderer/checkpoints")
INVERSE_DIR = "Diffusion_Renderer_Inverse_Cosmos_7B"
FORWARD_DIR = "Diffusion_Renderer_Forward_Cosmos_7B"
TOKENIZER_DIR = "Cosmos-Tokenize1-CV8x8x8-720p"

# upstream CLI 기본값과 같다.
GBUFFER_PASSES = ("basecolor", "normal", "depth", "roughness", "metallic")
DEFAULT_HEIGHT = 704
DEFAULT_WIDTH = 1280
DEFAULT_NUM_STEPS = 15
DEFAULT_SEED = 1000
# 영상 tokenizer는 57프레임 단위로 인코딩한다. 그래서 프레임 수는 1(이미지) 또는 57(영상)이다.
MAX_FRAMES = 57
# tokenizer는 공간을 8배, DiT patch는 2배 줄인다.
SIZE_MULTIPLE = 16
MAX_SIDE = 1280
MAX_NUM_STEPS = 100
RESPONSE_FORMATS = ("zip", "b64_json")
ENVMAP_SUFFIXES = (".hdr", ".exr")


@dataclass(frozen=True)
class ModelSource:
    """HF 저장소 1개와 upstream 체크포인트 디렉터리 이름."""

    repo: str
    revision: str
    directory: str


@dataclass
class RenderResult:
    """pass 이름별 출력 프레임(T, H, W, 3) uint8 배열."""

    frames_by_pass: dict[str, np.ndarray]


class Renderers:
    """두 처리 흐름을 들고 GPU 작업을 한 번에 1건만 실행한다."""

    def __init__(self, inverse: DiffusionRendererPipeline, forward: DiffusionRendererPipeline):
        self.inverse = inverse
        self.forward = forward
        self._lock = asyncio.Lock()

    async def run(self, job: Callable[[], RenderResult]) -> RenderResult:
        async with self._lock:
            return await asyncio.to_thread(_run_without_grad, job)


def _generate(
    pipeline: DiffusionRendererPipeline, data_batch: dict[str, Any], seed: int, **kwargs: Any
) -> np.ndarray:
    """전역 RNG를 seed로 맞춘 뒤 생성한다.

    upstream `generate_samples_from_batch`는 seed 인자를 쓰지 않고 전역 RNG로 초기 노이즈를 만든다.
    호출마다 seed를 다시 맞춰서 같은 입력과 seed이면 같은 결과를 낸다.
    """
    import torch

    torch.manual_seed(seed)
    return pipeline.generate_video(data_batch=data_batch, seed=seed, **kwargs)


def _run_without_grad(job: Callable[[], RenderResult]) -> RenderResult:
    """작업 스레드에서 autograd를 끄고 job을 실행한다.

    upstream은 import할 때 메인 스레드의 grad만 끈다. torch의 grad 모드는 스레드마다 따로다.
    작업 스레드에서 grad가 켜져 있으면 요청 1건이 노드 메모리를 40 GiB 이상 더 쓴다.
    """
    import torch

    with torch.no_grad():
        return job()


def _model_sources() -> list[ModelSource]:
    return [
        ModelSource(os.environ["INVERSE_MODEL"], os.environ["INVERSE_REVISION"], INVERSE_DIR),
        ModelSource(os.environ["FORWARD_MODEL"], os.environ["FORWARD_REVISION"], FORWARD_DIR),
        ModelSource(os.environ["TOKENIZER_MODEL"], os.environ["TOKENIZER_REVISION"], TOKENIZER_DIR),
    ]


def _link_checkpoints(sources: list[ModelSource]) -> None:
    """고정 revision을 HF 캐시에 받고 upstream 체크포인트 경로에 연결한다."""
    CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)
    for source in sources:
        snapshot = Path(snapshot_download(source.repo, revision=source.revision))
        link = CHECKPOINT_DIR / source.directory
        if link.is_symlink() or link.exists():
            link.unlink()
        link.symlink_to(snapshot, target_is_directory=True)
        logger.info("Checkpoint %s is at %s.", source.repo, snapshot)


def _load_pipeline(directory: str) -> DiffusionRendererPipeline:
    """처리 흐름 1개를 load하고 fp32 체크포인트가 쓴 임시 메모리를 돌려준다."""
    import torch

    pipeline = DiffusionRendererPipeline(
        checkpoint_dir=str(CHECKPOINT_DIR),
        checkpoint_name=directory,
        guidance=0.0,
        num_steps=DEFAULT_NUM_STEPS,
        height=DEFAULT_HEIGHT,
        width=DEFAULT_WIDTH,
        fps=24,
        num_video_frames=MAX_FRAMES,
        seed=DEFAULT_SEED,
    )
    gc.collect()
    torch.cuda.empty_cache()
    return pipeline


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:
    started = time.monotonic()
    sources = _model_sources()
    await asyncio.to_thread(_link_checkpoints, sources)
    inverse = await asyncio.to_thread(_load_pipeline, INVERSE_DIR)
    forward = await asyncio.to_thread(_load_pipeline, FORWARD_DIR)
    app.state.renderers = Renderers(inverse, forward)
    app.state.sources = sources
    logger.info("Both pipelines are ready after %.0f s.", time.monotonic() - started)
    yield


app = FastAPI(title="Cosmos DiffusionRenderer", lifespan=lifespan)


def _check_request(
    num_frames: int, height: int, width: int, num_steps: int, response_format: str
) -> None:
    """모든 경로에 공통인 입력 규칙을 검사한다. 규칙을 어기면 400을 낸다."""
    if num_frames not in (1, MAX_FRAMES):
        raise HTTPException(400, f"Frame count must be 1 or {MAX_FRAMES}.")
    for name, value in (("height", height), ("width", width)):
        if value < SIZE_MULTIPLE or value > MAX_SIDE or value % SIZE_MULTIPLE != 0:
            raise HTTPException(
                400,
                f"The {name} must be a multiple of {SIZE_MULTIPLE} from {SIZE_MULTIPLE} "
                f"to {MAX_SIDE}.",
            )
    if num_steps < 1 or num_steps > MAX_NUM_STEPS:
        raise HTTPException(400, f"The num_steps value must be from 1 to {MAX_NUM_STEPS}.")
    if response_format not in RESPONSE_FORMATS:
        raise HTTPException(400, f"The response_format must be one of {list(RESPONSE_FORMATS)}.")
    if response_format == "b64_json" and num_frames != 1:
        raise HTTPException(400, "Use response_format zip for more than 1 frame.")


async def _write_png(upload: UploadFile, path: Path) -> None:
    """업로드 이미지를 RGB PNG로 저장한다. 이미지가 아니면 400을 낸다."""
    data = await upload.read()
    try:
        with Image.open(io.BytesIO(data)) as image:
            image.convert("RGB").save(path)
    except (UnidentifiedImageError, OSError) as error:
        raise HTTPException(400, f"File {upload.filename} is not a readable image.") from error


def _single_batch(dataset: Any) -> dict[str, Any]:
    if len(dataset) != 1:
        raise RuntimeError(f"Expected 1 input clip. Found {len(dataset)}.")
    return dict_collation_fn([dataset[0]])


def _encode_png(frame: np.ndarray) -> bytes:
    buffer = io.BytesIO()
    Image.fromarray(frame).save(buffer, format="PNG")
    return buffer.getvalue()


def _respond(result: RenderResult, response_format: str) -> Response:
    if response_format == "b64_json":
        data = [
            {
                "pass": name,
                "frame": 0,
                "b64_json": base64.b64encode(_encode_png(frames[0])).decode(),
            }
            for name, frames in result.frames_by_pass.items()
        ]
        return JSONResponse({"data": data})
    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, "w", compression=zipfile.ZIP_STORED) as archive:
        for name, frames in result.frames_by_pass.items():
            for index, frame in enumerate(frames):
                archive.writestr(f"{name}/{index:04d}.png", _encode_png(frame))
    return Response(buffer.getvalue(), media_type="application/zip")


def _light_label(envmap: UploadFile | None, envmap_preset: int | None) -> str:
    if envmap is not None:
        return "upload"
    if envmap_preset is not None:
        return f"preset {envmap_preset}"
    return "random"


@app.get("/health")
async def health() -> dict[str, str]:
    return {"status": "ready"}


@app.get("/v1/models")
async def models() -> dict[str, list[dict[str, str]]]:
    return {
        "data": [
            {"id": source.repo, "revision": source.revision, "directory": source.directory}
            for source in app.state.sources
        ]
    }


@app.post("/v1/inverse")
async def inverse(
    frames: Annotated[list[UploadFile], File()],
    passes: Annotated[str, Form()] = ",".join(GBUFFER_PASSES),
    height: Annotated[int, Form()] = DEFAULT_HEIGHT,
    width: Annotated[int, Form()] = DEFAULT_WIDTH,
    num_steps: Annotated[int, Form()] = DEFAULT_NUM_STEPS,
    seed: Annotated[int, Form()] = DEFAULT_SEED,
    normalize_normal: Annotated[bool, Form()] = False,
    response_format: Annotated[str, Form()] = "zip",
) -> Response:
    """프레임에서 G-buffer를 추정한다. 프레임은 업로드 순서대로 쓴다."""
    _check_request(len(frames), height, width, num_steps, response_format)
    pass_list = [name.strip() for name in passes.split(",") if name.strip()]
    unknown = [name for name in pass_list if name not in GBUFFER_PASSES]
    if not pass_list or unknown:
        raise HTTPException(400, f"Each pass must be one of {list(GBUFFER_PASSES)}.")

    renderers: Renderers = app.state.renderers
    with tempfile.TemporaryDirectory() as tmp:
        clip_dir = Path(tmp) / "clip"
        clip_dir.mkdir()
        for index, upload in enumerate(frames):
            await _write_png(upload, clip_dir / f"{index:04d}.png")

        def job() -> RenderResult:
            dataset = VideoFramesDataset(
                root_dir=tmp,
                sample_n_frames=len(frames),
                group_mode="folder",
                resolution=[height, width],
            )
            data_batch = _single_batch(dataset)
            renderers.inverse.num_steps = num_steps
            outputs = {}
            for name in pass_list:
                data_batch["context_index"].fill_(GBUFFER_INDEX_MAPPING[name])
                outputs[name] = _generate(
                    renderers.inverse,
                    data_batch,
                    seed,
                    normalize_normal=(name == "normal" and normalize_normal),
                )
            return RenderResult(outputs)

        started = time.monotonic()
        result = await renderers.run(job)
    logger.info(
        "Inverse request finished. frames=%d passes=%s size=%dx%d steps=%d time=%.1f s",
        len(frames),
        ",".join(pass_list),
        width,
        height,
        num_steps,
        time.monotonic() - started,
    )
    return _respond(result, response_format)


@app.post("/v1/forward")
async def forward(
    basecolor: Annotated[list[UploadFile], File()],
    normal: Annotated[list[UploadFile], File()],
    depth: Annotated[list[UploadFile], File()],
    roughness: Annotated[list[UploadFile], File()],
    metallic: Annotated[list[UploadFile], File()],
    envmap: Annotated[UploadFile | None, File()] = None,
    envmap_preset: Annotated[int | None, Form()] = None,
    rotate_light: Annotated[bool, Form()] = False,
    height: Annotated[int, Form()] = DEFAULT_HEIGHT,
    width: Annotated[int, Form()] = DEFAULT_WIDTH,
    num_steps: Annotated[int, Form()] = DEFAULT_NUM_STEPS,
    seed: Annotated[int, Form()] = DEFAULT_SEED,
    response_format: Annotated[str, Form()] = "zip",
) -> Response:
    """G-buffer와 조명으로 RGB를 렌더링한다. 조명이 없으면 seed로 무작위 조명을 쓴다."""
    gbuffers = {
        "basecolor": basecolor,
        "normal": normal,
        "depth": depth,
        "roughness": roughness,
        "metallic": metallic,
    }
    counts = {len(files) for files in gbuffers.values()}
    if len(counts) != 1:
        raise HTTPException(400, "Each G-buffer pass must have the same frame count.")
    num_frames = counts.pop()
    _check_request(num_frames, height, width, num_steps, response_format)
    if envmap is not None and envmap_preset is not None:
        raise HTTPException(400, "Send envmap or envmap_preset. Do not send both.")
    if envmap_preset is not None and not 0 <= envmap_preset < len(ENV_LIGHT_PATH_LIST):
        raise HTTPException(
            400, f"The envmap_preset must be from 0 to {len(ENV_LIGHT_PATH_LIST) - 1}."
        )
    envmap_suffix = Path(envmap.filename or "").suffix.lower() if envmap is not None else ""
    if envmap is not None and envmap_suffix not in ENVMAP_SUFFIXES:
        raise HTTPException(400, f"The envmap file must have a suffix in {list(ENVMAP_SUFFIXES)}.")

    renderers: Renderers = app.state.renderers
    with tempfile.TemporaryDirectory() as tmp:
        gbuffer_root = Path(tmp) / "gbuffers"
        clip_dir = gbuffer_root / "clip"
        clip_dir.mkdir(parents=True)
        for name, files in gbuffers.items():
            for index, upload in enumerate(files):
                await _write_png(upload, clip_dir / f"0000.{index:04d}.{name}.png")
        envmap_path: str | None = None
        if envmap is not None:
            envmap_path = str(Path(tmp) / f"envmap{envmap_suffix}")
            Path(envmap_path).write_bytes(await envmap.read())
        elif envmap_preset is not None:
            envmap_path = ENV_LIGHT_PATH_LIST[envmap_preset]

        def job() -> RenderResult:
            import torch

            dataset = VideoGBufferDataset(
                root_dir=str(gbuffer_root),
                sample_n_frames=num_frames,
                resolution=(height, width),
            )
            data_batch = _single_batch(dataset)
            if envmap_path is not None:
                device = torch.device("cuda")
                envlight = process_environment_map(
                    envmap_path,
                    resolution=(height, width),
                    num_frames=num_frames,
                    fixed_pose=True,
                    rotate_envlight=rotate_light,
                    env_format=["proj"],
                    device=device,
                )
                env_ldr = envlight["env_ldr"].unsqueeze(0).permute(0, 4, 1, 2, 3) * 2 - 1
                env_log = envlight["env_log"].unsqueeze(0).permute(0, 4, 1, 2, 3) * 2 - 1
                env_nrm = envmap_vec([height, width], device=device)
                env_nrm = (
                    env_nrm.unsqueeze(0).unsqueeze(0).permute(0, 4, 1, 2, 3).expand_as(env_ldr)
                )
                data_batch["env_ldr"] = env_ldr
                data_batch["env_log"] = env_log
                data_batch["env_nrm"] = env_nrm
            renderers.forward.num_steps = num_steps
            rgb = _generate(renderers.forward, data_batch, seed)
            return RenderResult({"rgb": rgb})

        started = time.monotonic()
        try:
            result = await renderers.run(job)
        except (OSError, ValueError) as error:
            if envmap is None:
                raise
            raise HTTPException(400, "The envmap file is not a readable HDR image.") from error
    logger.info(
        "Forward request finished. frames=%d light=%s size=%dx%d steps=%d time=%.1f s",
        num_frames,
        _light_label(envmap, envmap_preset),
        width,
        height,
        num_steps,
        time.monotonic() - started,
    )
    return _respond(result, response_format)
