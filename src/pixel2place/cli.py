"""pixel2place 명령줄 인터페이스."""

import json
import math
from contextlib import nullcontext
from enum import StrEnum
from pathlib import Path
from typing import Annotated

import typer
from openai import APIConnectionError, APIError, APIStatusError, APITimeoutError, OpenAIError
from PIL import Image, UnidentifiedImageError
from pydantic import ValidationError

from pixel2place.agent.cell import run_agent_cell
from pixel2place.agent.contracts import ToolState
from pixel2place.agent.eval import MAX_REPEATS, EvalConfig, prepare_eval, run_eval
from pixel2place.agent.eval_cases import Feature, load_saved_pairs
from pixel2place.agent.inference import REQUEST_RETRY
from pixel2place.agent.intake import run_intake
from pixel2place.agent.shift import ShiftItemReport, load_shift, run_shift
from pixel2place.agent.transition_probe import run_transition_probe
from pixel2place.benchmark import generate_layouts, run_parser_benchmark
from pixel2place.cell.bench import (
    ALL_LAYOUTS,
    BenchLayout,
    CellBenchmark,
    Shard,
    run_cell_benchmark,
)
from pixel2place.cell.calibration import (
    CalibrationError,
    calibrate_board,
    read_report,
    write_calibration,
)
from pixel2place.cell.camera import PinholeCamera
from pixel2place.cell.capture import CAPTURE_FILE, compile_capture
from pixel2place.cell.check import (
    DEFAULT_TOLERANCE,
    CheckTolerance,
    PartCheck,
    ResultMismatchError,
    check_result,
)
from pixel2place.cell.config import CellConfig
from pixel2place.cell.episodes import (
    EVAL_MAX_STEPS,
    SLOT_BOARD_ERROR,
    SLOTS,
    PolicyEvaluation,
    evaluate_policy,
    record_episodes,
)
from pixel2place.cell.image_check import ImageCheckError, PartImageCheck, check_image, draw_overlay
from pixel2place.cell.result import SimResult
from pixel2place.cell.rework import ReworkError, rework_scene
from pixel2place.cell.scene import BoardOffset, CameraOffset, JointErrorLevel, SimScene
from pixel2place.cell.sim import (
    CAMERA_FILE,
    SimRunError,
    SimSetupError,
    run_capture,
    run_place_task,
)
from pixel2place.cell.task import Task, TaskCompileError, compile_task
from pixel2place.config import (
    DEFAULT_JUDGMENT_MODEL,
    DEFAULT_NEMOTRON_MODEL,
    DemoSettings,
    GeneratorSettings,
    Settings,
    SimSettings,
    TokenFactorySettings,
)
from pixel2place.feasibility.checker import apply_report, check_layout
from pixel2place.generation.cosmos_client import CosmosGenerator, ImageSize
from pixel2place.generation.synthetic import SyntheticRenderer
from pixel2place.layout.schema import Board, Cell, Layout
from pixel2place.logconfig import setup_logging
from pixel2place.order import OrderNotFeasibleError, order_instruction, parse_order
from pixel2place.parsing.base import LayoutParser
from pixel2place.parsing.grid import GridImageParser
from pixel2place.pipeline import JSON_INDENT, repair_plan, run_cell, run_roundtrip, write_json
from pixel2place.reasoner.client import CosmosReasoner
from pixel2place.reasoner.expansion import ExpansionTarget
from pixel2place.reasoner.nemotron import verify_nemotron
from pixel2place.reasoner.nemotron_tools import DEFAULT_TOOL_MAX_TOKENS, verify_nemotron_tools

app = typer.Typer(
    no_args_is_help=True,
    help="Turn work orders and layout images into robot placement instructions.",
)

EXIT_INFEASIBLE = 1
EXIT_USAGE = 2
# 확장한 지시문은 출력 파일 옆에 `<출력 파일 이름>.prompt.json`으로 저장한다.
# 예: `tray.png.prompt.json`
PROMPT_FILE_SUFFIX = ".prompt.json"
# `sim-rework`가 출력 디렉터리에 쓰는 재작업 장면
REWORK_SCENE_FILE = "scene.json"


@app.command("verify-nemotron")
def verify_nemotron_command(
    output: Annotated[
        Path | None, typer.Option("--output", "-o", dir_okay=False, help="Save verification JSON.")
    ] = None,
) -> None:
    """Check Token Factory model access and run one short Nemotron request."""
    if output is not None and output.exists():
        typer.echo("Verification output already exists. Use a new path.", err=True)
        raise typer.Exit(EXIT_USAGE)
    try:
        settings = TokenFactorySettings()  # pyright: ignore[reportCallIssue]
    except ValidationError as exc:
        names = ", ".join(str(error["loc"][0]).upper() for error in exc.errors())
        typer.echo(f"Token Factory settings are not valid: {names}. Set them in .env.", err=True)
        raise typer.Exit(EXIT_USAGE) from exc
    try:
        report = verify_nemotron(settings)
    except APITimeoutError as exc:
        typer.echo("Token Factory request timed out. No retry was made.", err=True)
        raise typer.Exit(EXIT_INFEASIBLE) from exc
    except APIConnectionError as exc:
        typer.echo("Token Factory connection failed. No retry was made.", err=True)
        raise typer.Exit(EXIT_INFEASIBLE) from exc
    except APIStatusError as exc:
        typer.echo(f"Token Factory request failed (HTTP {exc.status_code}).", err=True)
        raise typer.Exit(EXIT_INFEASIBLE) from exc
    except APIError as exc:
        typer.echo("Token Factory returned an invalid API response.", err=True)
        raise typer.Exit(EXIT_INFEASIBLE) from exc
    except ValueError as exc:
        typer.echo(
            "Nemotron verification failed. Check model access, response and token limit.", err=True
        )
        raise typer.Exit(EXIT_INFEASIBLE) from exc
    if output is not None:
        output.parent.mkdir(parents=True, exist_ok=True)
        try:
            # 검증 중 같은 경로에 파일이 생겨도 기존 측정 근거를 보존한다.
            with output.open("x", encoding="utf-8") as stream:
                stream.write(report.model_dump_json(indent=JSON_INDENT) + "\n")
        except FileExistsError as exc:
            typer.echo("Verification output already exists. Use a new path.", err=True)
            raise typer.Exit(EXIT_USAGE) from exc
        typer.echo(str(output))
    typer.echo(
        f"Verified {report.served_model}: {report.latency_s:.3f}s, finish={report.finish_reason}."
    )


@app.command(
    "verify-nemotron-tools", help="Check tool selection and result reading on Token Factory."
)
def verify_nemotron_tools_command(
    output: Annotated[
        Path | None,
        typer.Option("--output", "-o", dir_okay=False, help="Save success or failure JSON."),
    ] = None,
    model: Annotated[
        str | None, typer.Option("--model", help="Use this Nemotron model for this command.")
    ] = None,
    max_tokens: Annotated[
        int,
        typer.Option(
            "--max-tokens",
            min=1,
            max=4096,
            help="Output token limit, including tokens for model reasoning.",
        ),
    ] = DEFAULT_TOOL_MAX_TOKENS,
) -> None:
    """Nemotron이 도구를 고르고 반환한 receipt를 읽는지 검증한다."""
    if output is not None and output.exists():
        typer.echo("Verification output already exists. Use a new path.", err=True)
        raise typer.Exit(EXIT_USAGE)
    try:
        if model is None:
            settings = TokenFactorySettings()  # pyright: ignore[reportCallIssue]
        else:
            settings = TokenFactorySettings(nemotron_model=model)  # pyright: ignore[reportCallIssue]
    except ValidationError as exc:
        names = ", ".join(str(error["loc"][0]).upper() for error in exc.errors())
        typer.echo(f"Token Factory settings are not valid: {names}. Set them in .env.", err=True)
        raise typer.Exit(EXIT_USAGE) from exc
    try:
        if output is not None:
            output.parent.mkdir(parents=True, exist_ok=True)
        # 출력 경로를 먼저 확보해서 충돌 뒤 유료 호출이 발생하지 않게 한다.
        with output.open("x", encoding="utf-8") if output is not None else nullcontext() as stream:
            report = verify_nemotron_tools(settings, max_tokens=max_tokens)
            if stream is not None:
                stream.write(report.model_dump_json(indent=JSON_INDENT) + "\n")
    except FileExistsError as exc:
        typer.echo("Verification output already exists. Use a new path.", err=True)
        raise typer.Exit(EXIT_USAGE) from exc
    except OSError as exc:
        typer.echo("Could not create or write the verification output.", err=True)
        raise typer.Exit(EXIT_INFEASIBLE) from exc
    if output is not None:
        typer.echo(str(output))
    if not report.passed:
        typer.echo(
            f"Nemotron tool verification failed: {report.error_code}. "
            f"requests={report.request_count}. No retry was made.",
            err=True,
        )
        raise typer.Exit(EXIT_INFEASIBLE)
    latency = sum(record.latency_s for record in report.completions)
    typer.echo(
        f"Verified tools on {report.requested_model}: requests={report.request_count}, "
        f"latency={latency:.3f}s, receipt=matched."
    )


class ParserName(StrEnum):
    """parse 단계에 쓰는 판독기. grid는 Cosmos 서버 없이 동작한다."""

    GRID = "grid"
    COSMOS = "cosmos"


ParserOption = Annotated[
    ParserName,
    typer.Option(
        "--parser", help="Read images with grid (pixel checks) or cosmos (model request)."
    ),
]


def _parse_size(value: str | ImageSize) -> ImageSize:
    # typer는 기본값 `ImageSize`도 이 함수에 넘긴다.
    if isinstance(value, ImageSize):
        return value
    try:
        return ImageSize.parse(value)
    except ValueError as exc:
        raise typer.BadParameter(str(exc)) from exc


SizeOption = Annotated[
    ImageSize,
    typer.Option("--size", parser=_parse_size, help="Output size as WIDTHxHEIGHT."),
]
GuidanceOption = Annotated[
    float,
    typer.Option(
        min=0.0,
        help="How closely generation follows the instruction (classifier-free guidance scale).",
    ),
]
SeedOption = Annotated[int, typer.Option(min=0, help="Seed for the generation.")]
NoExpandOption = Annotated[
    bool,
    typer.Option(
        "--no-expand", help="Send the instruction as written. Do not expand it with Cosmos."
    ),
]
DEFAULT_SIZE = ImageSize(width=640, height=640)

LayoutPath = Annotated[
    Path, typer.Argument(exists=True, dir_okay=False, help="Path to a layout.json file.")
]
TaskPath = Annotated[
    Path, typer.Argument(exists=True, dir_okay=False, help="Path to a task.json file.")
]
CellOption = Annotated[
    Path | None,
    typer.Option("--cell", exists=True, dir_okay=False, help="Cell config JSON path."),
]
SceneOption = Annotated[
    Path | None,
    typer.Option(
        "--scene", exists=True, dir_okay=False, help="Scene JSON path with the actual board pose."
    ),
]
UsdOption = Annotated[Path | None, typer.Option("--usd", dir_okay=False, help="SO-101 USD path.")]
# bench-cell이 만든 무작위 레이아웃의 이름. 출력 디렉터리 이름에 쓴다.
RANDOM_LAYOUT_FORMAT = "random_{index:02d}"
MM_PER_M = 1000.0


def _parse_board_error(value: str) -> BoardOffset:
    """`X_MM,Y_MM,YAW_DEG`를 보드 오차 한계로 읽는다. 값은 0 이상이다."""
    fields = value.split(",")
    try:
        x_mm, y_mm, yaw_deg = (float(field) for field in fields)
    except ValueError as exc:
        raise typer.BadParameter(f"Board error {value!r} is not X_MM,Y_MM,YAW_DEG.") from exc
    if min(x_mm, y_mm, yaw_deg) < 0:
        raise typer.BadParameter(f"Board error {value!r} has a negative value.")
    return BoardOffset(x=x_mm / MM_PER_M, y=y_mm / MM_PER_M, yaw=math.radians(yaw_deg))


def _parse_joint_error(value: str) -> JointErrorLevel:
    """`SIGMA_DEG,TAU_S`를 관절 센서 오차의 크기와 지속 시간으로 읽는다.

    `SIGMA_DEG`는 0 이상이다. `TAU_S`는 0보다 큰 수 또는 `inf`다. `inf`는 실행 동안 같은 오차다.
    """
    fields = value.split(",")
    try:
        sigma_deg, tau_s = (float(field) for field in fields)
    except ValueError as exc:
        raise typer.BadParameter(f"Joint error {value!r} is not SIGMA_DEG,TAU_S.") from exc
    if not math.isfinite(sigma_deg) or sigma_deg < 0:
        raise typer.BadParameter(f"Joint error {value!r} needs SIGMA_DEG of 0 or more.")
    if math.isnan(tau_s) or tau_s <= 0:
        raise typer.BadParameter(f"Joint error {value!r} needs TAU_S more than 0.")
    return JointErrorLevel(sigma=math.radians(sigma_deg), tau=None if math.isinf(tau_s) else tau_s)


def _parse_camera_error(value: str) -> CameraOffset:
    """`POS_MM,ANGLE_DEG`를 카메라 오차 한계로 읽는다. 한계는 모든 축에 같다. 값은 0 이상이다."""
    fields = value.split(",")
    try:
        position_mm, angle_deg = (float(field) for field in fields)
    except ValueError as exc:
        raise typer.BadParameter(f"Camera error {value!r} is not POS_MM,ANGLE_DEG.") from exc
    if min(position_mm, angle_deg) < 0:
        raise typer.BadParameter(f"Camera error {value!r} has a negative value.")
    position = position_mm / MM_PER_M
    angle = math.radians(angle_deg)
    return CameraOffset(position=(position, position, position), rotation=(angle, angle, angle))


def _parse_shard(value: str | Shard) -> Shard:
    # typer는 기본값 `Shard`도 이 함수에 넘긴다.
    if isinstance(value, Shard):
        return value
    try:
        return Shard.parse(value)
    except ValueError as exc:
        raise typer.BadParameter(str(exc)) from exc


ShardOption = Annotated[
    Shard,
    typer.Option(
        "--shard", parser=_parse_shard, help="Run only the layouts with index % N == K, as K/N."
    ),
]


@app.callback()
def main(verbose: Annotated[bool, typer.Option("--verbose", "-v")] = False) -> None:
    setup_logging(verbose=verbose)


@app.command()
def check(layout_path: LayoutPath) -> None:
    """Check the placement rules of a layout. Exit 1 if the layout is not feasible."""
    layout = _read_layout(layout_path)
    report = check_layout(layout)
    for issue in report.issues:
        typer.echo(f"{issue.code} {issue.part_id or '-'}: {issue.message}")
    typer.echo("feasible" if report.feasible else "not feasible")
    if not report.feasible:
        raise typer.Exit(EXIT_INFEASIBLE)


@app.command("compose-order")
def compose_order(
    order_path: Annotated[
        Path, typer.Argument(exists=True, dir_okay=False, help="Path to a work order JSON file.")
    ],
) -> None:
    """Write the work instruction of a board order or a kit order.

    Exit 1 if the board order is not feasible.
    """
    try:
        order = parse_order(order_path.read_text(encoding="utf-8"))
    except ValidationError:
        typer.echo("Work order is not valid. Check the JSON file.", err=True)
        raise typer.Exit(EXIT_USAGE) from None
    try:
        instruction = order_instruction(order)
    except OrderNotFeasibleError as exc:
        # stdout에는 지시문만 쓴다. 셸에서 다른 명령의 인자로 바로 쓴다.
        for issue in exc.issues:
            typer.echo(f"{issue.code} {issue.part_id or '-'}: {issue.message}", err=True)
        raise typer.Exit(EXIT_INFEASIBLE) from None
    typer.echo(instruction)


@app.command()
def render(
    layout_path: LayoutPath,
    output: Annotated[Path, typer.Option("--output", "-o", help="Output PNG path.")],
) -> None:
    """Draw a layout as a top-down board image."""
    image = SyntheticRenderer().render(_read_layout(layout_path))
    image.save(_with_parent(output))
    typer.echo(str(output))


@app.command()
def plan(
    prompt: str,
    output: Annotated[Path, typer.Option("--output", "-o", help="Output JSON path.")],
) -> None:
    """Make a part layout from an instruction with Cosmos.

    The command moves or removes parts that overlap or are outside the board.
    """
    layout, _ = repair_plan(_reasoner().plan(prompt))
    _write_layout(output, layout)


@app.command()
def parse(
    image_path: Annotated[Path, typer.Argument(exists=True, dir_okay=False)],
    output: Annotated[Path, typer.Option("--output", "-o", help="Output JSON path.")],
    parser: ParserOption = ParserName.GRID,
) -> None:
    """Read a layout from a top-down board image."""
    with Image.open(image_path) as image:
        layout = _parser(parser).parse(image.convert("RGB"))
    _write_layout(output, apply_report(layout, check_layout(layout)))


@app.command()
def run(
    prompt: str,
    out_dir: Annotated[Path, typer.Option("--out-dir", "-o", help="Output directory.")],
    parser: ParserOption = ParserName.GRID,
) -> None:
    """Make a layout from an instruction. Draw it, read the image and save all outputs."""
    result = run_roundtrip(
        prompt,
        out_dir=out_dir,
        reasoner=_reasoner(),
        parser=_parser(parser),
        renderer=SyntheticRenderer(),
    )
    typer.echo(f"cell_accuracy={result.comparison.cell_accuracy:.3f} out_dir={out_dir}")


@app.command("bench-parser")
def bench_parser(
    out_dir: Annotated[Path, typer.Option("--out-dir", "-o", help="Output directory.")],
    count: Annotated[int, typer.Option(min=1, help="Number of generated layouts.")] = 24,
    seed: Annotated[int, typer.Option(help="Seed for the generated layouts.")] = 0,
    parser: ParserOption = ParserName.GRID,
) -> None:
    """Check how well the image reader identifies generated layouts. Save all outputs."""
    summary = run_parser_benchmark(
        generate_layouts(count=count, seed=seed),
        seed=seed,
        parser=_parser(parser),
        renderer=SyntheticRenderer(),
        out_dir=out_dir,
    )
    typer.echo(
        f"mean_cell_accuracy={summary.mean_cell_accuracy:.3f} "
        f"exact={summary.exact_match_rate:.3f} "
        f"part_count_match={summary.part_count_match_rate:.3f} "
        f"failures={summary.failures} out_dir={out_dir}"
    )


@app.command("generate-image")
def generate_image(
    prompt: str,
    output: Annotated[Path, typer.Option("--output", "-o", help="Output PNG path.")],
    size: SizeOption = DEFAULT_SIZE,
    steps: Annotated[int, typer.Option(min=1, help="Number of denoising steps.")] = 50,
    guidance: GuidanceOption = 6.0,
    seed: SeedOption = 0,
    no_expand: NoExpandOption = False,
) -> None:
    """Make an image from an instruction with Cosmos."""
    generator = _generator()
    text = prompt if no_expand else _expand(prompt, output=output, target="image")
    image = generator.generate_image(text, size=size, steps=steps, guidance=guidance, seed=seed)
    image.save(_with_parent(output), format="PNG")
    typer.echo(str(output))


@app.command("preview-video")
def preview_video(
    image_path: Annotated[
        Path, typer.Argument(exists=True, dir_okay=False, help="First frame image.")
    ],
    prompt: str,
    output: Annotated[Path, typer.Option("--output", "-o", help="Output MP4 path.")],
    size: SizeOption = DEFAULT_SIZE,
    num_frames: Annotated[int, typer.Option(min=1, help="Number of video frames.")] = 93,
    fps: Annotated[int, typer.Option(min=1, help="Frames per second.")] = 24,
    steps: Annotated[int, typer.Option(min=1, help="Number of denoising steps.")] = 35,
    guidance: GuidanceOption = 6.0,
    seed: SeedOption = 0,
    no_expand: NoExpandOption = False,
) -> None:
    """Make a short video from the first image and a motion description."""
    generator = _generator()
    with Image.open(image_path) as image:
        first_frame = image.convert("RGB")
    text = (
        prompt if no_expand else _expand(prompt, output=output, target="video", image=first_frame)
    )
    video = generator.generate_video(
        first_frame,
        text,
        size=size,
        num_frames=num_frames,
        fps=fps,
        steps=steps,
        guidance=guidance,
        seed=seed,
    )
    _with_parent(output).write_bytes(video)
    typer.echo(str(output))


@app.command("compile-task")
def compile_task_command(
    layout_path: LayoutPath,
    output: Annotated[Path, typer.Option("--output", "-o", help="Output task JSON path.")],
    cell_path: CellOption = None,
) -> None:
    """Compile a layout into SO-101 pick-and-place waypoints. Exit 1 if it is not possible."""
    try:
        task = compile_task(_read_layout(layout_path), _read_cell(cell_path))
    except TaskCompileError as exc:
        typer.echo(str(exc), err=True)
        raise typer.Exit(EXIT_INFEASIBLE) from exc
    write_json(_with_parent(output), task)
    typer.echo(f"parts={len(task.parts)} output={output}")


@app.command("sim-place")
def sim_place(
    task_path: TaskPath,
    output: Annotated[Path, typer.Option("--output", "-o", help="Output directory.")],
    usd: UsdOption = None,
    scene_path: SceneOption = None,
    present: Annotated[
        bool,
        typer.Option("--present", help="Write present.mp4 from the presentation camera."),
    ] = False,
) -> None:
    """Run a task in Isaac Sim. Write result.json, cell.mp4 and top-down images.

    With --scene, the runner puts the board at the scene pose. The task keeps its own board pose.
    With --present, the runner also writes a 1280x720 video. The run takes more time.
    """
    try:
        task = Task.model_validate_json(task_path.read_text(encoding="utf-8"))
        if scene_path is not None:
            SimScene.model_validate_json(scene_path.read_text(encoding="utf-8"))
        settings = SimSettings()  # pyright: ignore[reportCallIssue]
    except ValidationError as exc:
        typer.echo(f"Task, scene or Isaac settings are not valid: {exc}", err=True)
        raise typer.Exit(EXIT_USAGE) from exc
    try:
        result = run_place_task(
            task_path,
            out_dir=output,
            settings=settings,
            usd_path=usd,
            scene_path=scene_path,
            present=present,
        )
    except SimSetupError as exc:
        typer.echo(str(exc), err=True)
        raise typer.Exit(EXIT_USAGE) from exc
    except SimRunError as exc:
        typer.echo(str(exc), err=True)
        raise typer.Exit(EXIT_INFEASIBLE) from exc
    typer.echo(f"parts={len(result.parts)}/{len(task.parts)} output={output}")


@app.command("sim-rework")
def sim_rework(
    task_path: TaskPath,
    result_path: Annotated[
        Path,
        typer.Argument(exists=True, dir_okay=False, help="result.json of the first run."),
    ],
    output: Annotated[Path, typer.Option("--output", "-o", help="Output directory.")],
    parts: Annotated[
        list[int],
        typer.Option(
            "--part", min=1, help="Placement sequence of a part to place again. Repeat for more."
        ),
    ],
    usd: UsdOption = None,
    scene_path: SceneOption = None,
) -> None:
    """Place parts again in Isaac Sim on the board of a first run.

    The runner puts the other parts at the poses of the first result. It does not move them.
    Give the scene of the first run with --scene. The command writes the new scene to scene.json.
    The output directory must not be the directory of the first result or of the first scene.
    """
    inputs = [result_path, *([scene_path] if scene_path is not None else [])]
    if any(path.resolve().parent == output.resolve() for path in inputs):
        typer.echo("Output directory holds the first run. Use a different directory.", err=True)
        raise typer.Exit(EXIT_USAGE)
    try:
        task = Task.model_validate_json(task_path.read_text(encoding="utf-8"))
        first = SimResult.model_validate_json(result_path.read_text(encoding="utf-8"))
        scene = None
        if scene_path is not None:
            scene = SimScene.model_validate_json(scene_path.read_text(encoding="utf-8"))
        rework = rework_scene(scene, task=task, result=first, sequences=parts)
        settings = SimSettings()  # pyright: ignore[reportCallIssue]
    except (ValidationError, ReworkError) as exc:
        typer.echo(f"Task, result, scene or Isaac settings are not valid: {exc}", err=True)
        raise typer.Exit(EXIT_USAGE) from exc
    output.mkdir(parents=True, exist_ok=True)
    rework_path = output / REWORK_SCENE_FILE
    write_json(rework_path, rework)
    try:
        result = run_place_task(
            task_path, out_dir=output, settings=settings, usd_path=usd, scene_path=rework_path
        )
    except SimSetupError as exc:
        typer.echo(str(exc), err=True)
        raise typer.Exit(EXIT_USAGE) from exc
    except SimRunError as exc:
        typer.echo(str(exc), err=True)
        raise typer.Exit(EXIT_INFEASIBLE) from exc
    typer.echo(
        f"parts={len(result.parts)}/{len(task.parts)} reworked={len(set(parts))} output={output}"
    )


@app.command("sim-capture")
def sim_capture(
    output: Annotated[Path, typer.Option("--output", "-o", help="Output directory.")],
    cell_path: CellOption = None,
    scene_path: SceneOption = None,
    usd: UsdOption = None,
) -> None:
    """Capture a top-down image of the empty cell in Isaac Sim for the board calibration.

    Write capture.json, topdown_start.png, camera.json and camera_truth.json.
    With --scene, the runner puts the board and the top-down camera at the scene poses.
    """
    try:
        cell = _read_cell(cell_path)
        if scene_path is not None:
            SimScene.model_validate_json(scene_path.read_text(encoding="utf-8"))
        settings = SimSettings()  # pyright: ignore[reportCallIssue]
    except ValidationError as exc:
        typer.echo(f"Cell config, scene or Isaac settings are not valid: {exc}", err=True)
        raise typer.Exit(EXIT_USAGE) from exc
    try:
        capture = compile_capture(cell, Board())
    except TaskCompileError as exc:
        typer.echo(str(exc), err=True)
        raise typer.Exit(EXIT_INFEASIBLE) from exc
    output.mkdir(parents=True, exist_ok=True)
    capture_path = output / CAPTURE_FILE
    write_json(capture_path, capture)
    try:
        captured = run_capture(
            capture_path, out_dir=output, settings=settings, usd_path=usd, scene_path=scene_path
        )
    except SimSetupError as exc:
        typer.echo(str(exc), err=True)
        raise typer.Exit(EXIT_USAGE) from exc
    except SimRunError as exc:
        typer.echo(str(exc), err=True)
        raise typer.Exit(EXIT_INFEASIBLE) from exc
    typer.echo(f"image={captured.image_path} camera={output / CAMERA_FILE} output={output}")


@app.command("calibrate-board")
def calibrate_board_command(
    image_path: Annotated[
        Path,
        typer.Argument(exists=True, dir_okay=False, help="Top-down image of the empty board."),
    ],
    camera_path: Annotated[
        Path,
        typer.Argument(
            exists=True, dir_okay=False, help="camera.json with the intrinsics and nominal pose."
        ),
    ],
    output: Annotated[Path, typer.Option("--output", "-o", help="Output cell config JSON path.")],
    cell_path: CellOption = None,
) -> None:
    """Measure the board pose in a top-down image and write the cell config with that pose.

    Also write <name>.report.json and <name>.overlay.png next to the output.
    Exit 1 if the command cannot find the board pose.
    """
    try:
        cell = _read_cell(cell_path)
        camera = PinholeCamera.model_validate_json(camera_path.read_text(encoding="utf-8"))
    except ValidationError as exc:
        typer.echo(f"Cell config or camera is not valid: {exc}", err=True)
        raise typer.Exit(EXIT_USAGE) from exc
    board = Board()
    with Image.open(image_path) as image:
        rgb = image.convert("RGB")
    try:
        calibration = calibrate_board(rgb, camera, cell=cell, board=board)
    except CalibrationError as exc:
        typer.echo(str(exc), err=True)
        raise typer.Exit(EXIT_INFEASIBLE) from exc
    write_calibration(calibration, rgb, output=_with_parent(output), board=board)
    report = calibration.report
    found = sum(marker.found_px is not None for marker in report.markers)
    offset = report.offset
    typer.echo(
        f"markers={found}/{len(report.markers)} pins={report.pins_used}/{len(report.pins)} "
        f"offset_mm={offset.x * MM_PER_M:+.2f},{offset.y * MM_PER_M:+.2f} "
        f"yaw_deg={math.degrees(offset.yaw):+.3f} board_shift_mm={report.board_shift_mm:.2f} "
        f"output={output}"
    )


@app.command("sim-check")
def sim_check(
    task_path: TaskPath,
    result_path: Annotated[
        Path, typer.Argument(exists=True, dir_okay=False, help="Path to a result.json file.")
    ],
    output: Annotated[
        Path | None, typer.Option("--output", "-o", help="Output check JSON path.")
    ] = None,
    position_ratio: Annotated[
        float, typer.Option(min=0.0, help="Position tolerance as a ratio of the cell size.")
    ] = DEFAULT_TOLERANCE.position_ratio,
    yaw_deg: Annotated[
        float, typer.Option(min=0.0, help="Yaw tolerance in degrees after part symmetry.")
    ] = DEFAULT_TOLERANCE.yaw_deg,
    tilt_deg: Annotated[
        float, typer.Option(min=0.0, help="Tilt tolerance in degrees.")
    ] = DEFAULT_TOLERANCE.tilt_deg,
    height_mm: Annotated[
        float, typer.Option(min=0.0, help="Height tolerance in mm from the seated height.")
    ] = DEFAULT_TOLERANCE.height_mm,
) -> None:
    """Compare the part poses in result.json with the task targets.

    Exit 1 if a part is out of tolerance.
    """
    tolerance = CheckTolerance(
        position_ratio=position_ratio, yaw_deg=yaw_deg, tilt_deg=tilt_deg, height_mm=height_mm
    )
    try:
        task = Task.model_validate_json(task_path.read_text(encoding="utf-8"))
        result = SimResult.model_validate_json(result_path.read_text(encoding="utf-8"))
        report = check_result(task, result, tolerance=tolerance)
    except (ValidationError, ResultMismatchError) as exc:
        typer.echo(f"Task or result is not valid: {exc}", err=True)
        raise typer.Exit(EXIT_USAGE) from exc
    for part in report.parts:
        typer.echo(_format_part_check(part))
    if output:
        write_json(_with_parent(output), report)
    verdict = "success" if report.passed else "failure"
    typer.echo(f"passed={report.passed_count}/{len(report.parts)} {verdict}")
    if not report.passed:
        raise typer.Exit(EXIT_INFEASIBLE)


@app.command("image-check")
def image_check_command(
    task_path: TaskPath,
    image_path: Annotated[
        Path,
        typer.Argument(
            exists=True, dir_okay=False, help="Top-down image of the board after the placement."
        ),
    ],
    camera_path: Annotated[
        Path | None,
        typer.Option("--camera", exists=True, dir_okay=False, help="camera.json path."),
    ] = None,
    calibration_path: Annotated[
        Path | None,
        typer.Option(
            "--calibration",
            exists=True,
            dir_okay=False,
            help="Board calibration report JSON path. The command uses the camera of the report.",
        ),
    ] = None,
    scene_path: SceneOption = None,
    output: Annotated[
        Path | None, typer.Option("--output", "-o", help="Output report JSON path.")
    ] = None,
    overlay: Annotated[
        Path | None, typer.Option("--overlay", help="Output overlay PNG path.")
    ] = None,
) -> None:
    """Read the placed parts in a top-down image and compare them with the task.

    Give the camera with --camera or with --calibration.
    With --scene, the command projects the cells onto the actual board pose of the scene.
    Exit 1 if a part does not match or if a cell that is not in the task is occupied.
    """
    if (camera_path is None) == (calibration_path is None):
        typer.echo("Give --camera or --calibration. Do not give the two options.", err=True)
        raise typer.Exit(EXIT_USAGE)
    try:
        task = Task.model_validate_json(task_path.read_text(encoding="utf-8"))
        camera = _image_check_camera(camera_path=camera_path, calibration_path=calibration_path)
        board = None
        if scene_path is not None:
            board = SimScene.model_validate_json(scene_path.read_text(encoding="utf-8")).board
        with Image.open(image_path) as image:
            rgb = image.convert("RGB")
        report = check_image(task, rgb, camera, board=board)
    except (ValidationError, UnidentifiedImageError, ImageCheckError) as exc:
        typer.echo(f"Task, camera, scene or image is not valid: {exc}", err=True)
        raise typer.Exit(EXIT_USAGE) from exc
    for part in report.parts:
        typer.echo(_format_part_image_check(part))
    if report.unexpected_cells:
        typer.echo(f"unexpected_cells={_cells_text(report.unexpected_cells)}")
    if output:
        write_json(_with_parent(output), report)
    if overlay:
        drawn = draw_overlay(rgb, report, task=task, camera=camera, board=board)
        drawn.save(_with_parent(overlay))
    verdict = "success" if report.passed else "failure"
    typer.echo(
        f"match={report.match_count} shadowed={report.shadowed_count} "
        f"mismatch={report.mismatch_count} unexpected={len(report.unexpected_cells)} {verdict}"
    )
    if not report.passed:
        raise typer.Exit(EXIT_INFEASIBLE)


@app.command("agent-cell", help="Coordinate layout, calibration and execution with Nemotron.")
def agent_cell_command(
    prompt: str,
    out_dir: Annotated[Path, typer.Option("--out-dir", "-o", help="New output directory.")],
    cell_path: Annotated[
        Path | None, typer.Option("--cell", help="Nominal cell JSON.", dir_okay=False)
    ] = None,
    scene_path: Annotated[
        Path | None, typer.Option("--scene", help="Simulation scene JSON.", dir_okay=False)
    ] = None,
    review: Annotated[
        bool,
        typer.Option(
            "--review/--no-review", help="Let the judgment model review the prepared layout."
        ),
    ] = True,
    disposition: Annotated[
        bool,
        typer.Option(
            "--disposition/--no-disposition",
            help="Let the model release, rework or reject the result from the camera check.",
        ),
    ] = True,
) -> None:
    """고정한 입력에서 Nemotron 조율을 실행한다. 출하한 실행만 exit 0이다.

    계획 검토는 기본으로 켠다. `--no-review`는 준비한 계획을 검토 없이 실행한다.
    처분은 기본으로 켠다. 출하는 참값 판정과 카메라 판정의 통과를 요구한다.
    `--no-disposition`은 참값 판정만으로 완료한다. 재작업은 없다.
    """
    if out_dir.exists() or out_dir.is_symlink():
        typer.echo("Run output already exists. Use a new directory.", err=True)
        raise typer.Exit(EXIT_USAGE)
    try:
        cell = _read_cell(cell_path)
        scene = (
            SimScene.model_validate_json(scene_path.read_text(encoding="utf-8"))
            if scene_path is not None
            else SimScene()
        )
        token_settings = TokenFactorySettings(  # pyright: ignore[reportCallIssue]
            nemotron_model=DEFAULT_NEMOTRON_MODEL
        )
        cosmos_settings = Settings()  # pyright: ignore[reportCallIssue]
        sim_settings = SimSettings()  # pyright: ignore[reportCallIssue]
    except (OSError, ValidationError):
        typer.echo("Run inputs or settings are not valid. Check JSON files and .env.", err=True)
        raise typer.Exit(EXIT_USAGE) from None
    try:
        report = run_agent_cell(
            prompt,
            out_dir=out_dir,
            cell=cell,
            scene=scene,
            token_settings=token_settings,
            cosmos_settings=cosmos_settings,
            sim_settings=sim_settings,
            review=review,
            disposition=disposition,
        )
    except (FileExistsError, SimSetupError, ValidationError, ValueError):
        typer.echo("Run input, output directory or execution files are not valid.", err=True)
        raise typer.Exit(EXIT_USAGE) from None
    except OSError:
        typer.echo("Could not create or write the run records.", err=True)
        raise typer.Exit(EXIT_INFEASIBLE) from None
    typer.echo(
        f"state={report.state} error_code={report.error_code} "
        f"disposition={report.disposition} place_runs={report.calls.place} "
        f"nemotron_requests={report.calls.nemotron} output={out_dir}"
    )
    if report.state != ToolState.COMPLETED:
        raise typer.Exit(EXIT_INFEASIBLE)


@app.command("agent-shift", help="Run the work orders of a shift one after the other.")
def agent_shift_command(
    shift_path: Annotated[
        Path, typer.Argument(exists=True, dir_okay=False, help="Path to a shift JSON file.")
    ],
    out_dir: Annotated[Path, typer.Option("--out-dir", "-o", help="New output directory.")],
    review: Annotated[
        bool,
        typer.Option(
            "--review/--no-review", help="Let the judgment model review each prepared layout."
        ),
    ] = True,
    disposition: Annotated[
        bool,
        typer.Option(
            "--disposition/--no-disposition",
            help="Let the model release, rework or reject each result from the camera check.",
        ),
    ] = True,
) -> None:
    """교대 파일의 주문을 차례로 실행한다. 모든 주문을 출하해야 exit 0이다.

    주문마다 `agent-cell`과 같은 실행을 1번 한다. 완료하지 못한 주문 뒤에도 계속한다.
    """
    if out_dir.exists() or out_dir.is_symlink():
        typer.echo("Shift output already exists. Use a new directory.", err=True)
        raise typer.Exit(EXIT_USAGE)
    try:
        orders = load_shift(shift_path)
        token_settings = TokenFactorySettings(  # pyright: ignore[reportCallIssue]
            nemotron_model=DEFAULT_NEMOTRON_MODEL
        )
        cosmos_settings = Settings()  # pyright: ignore[reportCallIssue]
        sim_settings = SimSettings()  # pyright: ignore[reportCallIssue]
    # 주문의 배치 규칙 위반과 결함의 부품 순서 오류도 `ValueError`다.
    except (OSError, ValueError):
        typer.echo(
            "Shift file, order files or settings are not valid. Check JSON files and .env.",
            err=True,
        )
        raise typer.Exit(EXIT_USAGE) from None
    try:
        report = run_shift(
            orders,
            out_dir=out_dir,
            cell=CellConfig(),
            token_settings=token_settings,
            cosmos_settings=cosmos_settings,
            sim_settings=sim_settings,
            review=review,
            disposition=disposition,
            on_item=lambda item: typer.echo(_format_shift_item(item)),
        )
    except (FileExistsError, SimSetupError, ValidationError, ValueError):
        typer.echo("Shift input, output directory or execution files are not valid.", err=True)
        raise typer.Exit(EXIT_USAGE) from None
    except OSError:
        typer.echo("Could not create or write the shift records.", err=True)
        raise typer.Exit(EXIT_INFEASIBLE) from None
    totals = report.totals
    if len(report.items) < totals.orders:
        typer.echo(
            f"A user interrupt stopped the shift after {len(report.items)} of "
            f"{totals.orders} orders.",
            err=True,
        )
    typer.echo(
        f"orders={totals.orders} completed={totals.completed_orders} "
        f"changeovers={totals.changeovers} wall_s={totals.wall_time_s:.1f} "
        f"first_pass={totals.first_pass_orders}/{len(report.items)} "
        f"rejected={totals.rejected_orders} output={out_dir}"
    )
    if totals.completed_orders != totals.orders:
        raise typer.Exit(EXIT_INFEASIBLE)


def _format_shift_item(item: ShiftItemReport) -> str:
    """주문 1건의 결과 1줄. 값이 없는 지표는 `-`로 쓴다."""
    passed = "-" if item.passed_parts is None else item.passed_parts
    robot = "-" if item.robot_time_s is None else f"{item.robot_time_s:.1f}"
    return (
        f"{item.directory} state={item.state} error_code={item.error_code} "
        f"parts={passed}/{item.part_count} robot_s={robot} wall_s={item.wall_time_s:.1f} "
        f"model_s={item.model_time_s:.1f} nemotron_requests={item.nemotron_requests} "
        f"retries={item.request_retries} disposition={item.disposition}"
    )


@app.command("intake-order", help="Read a work order from free text with Nemotron.")
def intake_order_command(
    text: str,
    out_dir: Annotated[Path, typer.Option("--out-dir", "-o", help="New output directory.")],
    model: Annotated[
        str | None, typer.Option("--model", help="Use this Nemotron model for this command.")
    ] = None,
) -> None:
    """자유 문장에서 주문을 읽고 지시문을 쓴다. 수락한 주문만 exit 0이다."""
    if out_dir.exists() or out_dir.is_symlink():
        typer.echo("Intake output already exists. Use a new directory.", err=True)
        raise typer.Exit(EXIT_USAGE)
    try:
        settings = TokenFactorySettings(  # pyright: ignore[reportCallIssue]
            nemotron_model=model or DEFAULT_JUDGMENT_MODEL
        )
        record = run_intake(text, out_dir=out_dir, settings=settings, retry=REQUEST_RETRY)
    except (ValidationError, FileExistsError):
        typer.echo("Intake text, settings or output directory are not valid.", err=True)
        raise typer.Exit(EXIT_USAGE) from None
    except OSError:
        typer.echo("Could not create or write the intake records.", err=True)
        raise typer.Exit(EXIT_INFEASIBLE) from None
    reading = record.reading
    if reading.instruction is None:
        detail = reading.reject_reason or ",".join(reading.issues) or "-"
        typer.echo(f"outcome={reading.outcome} detail={detail} output={out_dir}", err=True)
        raise typer.Exit(EXIT_INFEASIBLE)
    # stdout에는 지시문만 쓴다. 셸에서 agent-cell의 인자로 바로 쓴다.
    typer.echo(reading.instruction)


@app.command("eval-judgment", help="Measure one judgment feature on offline cases.")
def eval_judgment_command(
    feature: Annotated[Feature, typer.Option("--feature", help="Judgment to measure.")],
    model: Annotated[str, typer.Option("--model", help="Nemotron model ID.")],
    out_dir: Annotated[Path, typer.Option("--out-dir", "-o", help="New output directory.")],
    data_dir: Annotated[
        Path | None,
        typer.Option(
            "--data",
            help="Directory of saved plans for the review feature. "
            "Each subdirectory has order.json and layout.json.",
            exists=True,
            file_okay=False,
        ),
    ] = None,
    seed: Annotated[int, typer.Option(help="Seed of the generated cases.")] = 0,
    limit: Annotated[int | None, typer.Option(min=1, help="Use only this number of cases.")] = None,
    repeats: Annotated[
        int,
        typer.Option(
            min=1, max=MAX_REPEATS, help="Number of runs for each case of the repeated subset."
        ),
    ] = 1,
    dry_run: Annotated[
        bool, typer.Option("--dry-run", help="Write the cases and the cost estimate only.")
    ] = False,
    disposition: Annotated[
        bool,
        typer.Option(
            "--disposition/--no-disposition",
            help="Use the instruction and state of a run with release, rework or rejection. "
            "Applies to the review and coordination features.",
        ),
    ] = True,
) -> None:
    """시드로 만든 사례에서 판단 기능 1개를 잰다. 모든 기준을 통과해야 exit 0이다.

    계획 검토와 조율은 처분을 켠 실행의 지시문으로 잰다. `--no-disposition`은 끈 지시문을 쓴다.
    접수는 그 설정을 쓰지 않는다. 처분은 항상 켠 지시문을 쓴다.
    """
    if out_dir.exists() or out_dir.is_symlink():
        typer.echo("Evaluation output already exists. Use a new directory.", err=True)
        raise typer.Exit(EXIT_USAGE)
    if data_dir is not None and feature != Feature.REVIEW:
        typer.echo("--data applies only to the review feature.", err=True)
        raise typer.Exit(EXIT_USAGE)
    try:
        config = EvalConfig(
            feature=feature,
            model=model,
            seed=seed,
            limit=limit,
            repeats=repeats,
            disposition=feature == Feature.DISPOSITION
            or (disposition and feature != Feature.INTAKE),
        )
        saved = load_saved_pairs(data_dir) if data_dir is not None else []
        settings = None
        if not dry_run:
            settings = TokenFactorySettings(nemotron_model=model)  # pyright: ignore[reportCallIssue]
    except (OSError, ValueError):
        typer.echo("Evaluation options, saved plans or settings are not valid.", err=True)
        raise typer.Exit(EXIT_USAGE) from None
    try:
        prepared = prepare_eval(config, out_dir=out_dir, saved=saved, dry_run=dry_run)
    except (FileExistsError, ValueError):
        typer.echo("Saved plans or the output directory are not valid.", err=True)
        raise typer.Exit(EXIT_USAGE) from None
    except OSError:
        typer.echo("Could not write the evaluation records.", err=True)
        raise typer.Exit(EXIT_INFEASIBLE) from None
    manifest = prepared.manifest
    typer.echo(
        f"cases={manifest.case_count} requests={manifest.request_count} "
        f"input_tokens={manifest.estimate.input_tokens} "
        f"expected_usd={_usd(manifest.estimate.expected_usd)} "
        f"max_usd={_usd(manifest.estimate.max_usd)} cases_sha256={manifest.cases_sha256}"
    )
    if settings is None:
        return
    try:
        report = run_eval(prepared, out_dir=out_dir, settings=settings)
    except (OSError, OpenAIError, ValueError):
        typer.echo("Could not read the model catalog or write the evaluation records.", err=True)
        raise typer.Exit(EXIT_INFEASIBLE) from None
    for metric in report.metrics:
        verdict = {True: "pass", False: "fail", None: "-"}[metric.passed]
        typer.echo(
            f"{metric.name}={metric.count}/{metric.total} "
            f"criterion={metric.criterion or '-'} {verdict}"
        )
    if report.consistency is not None:
        typer.echo(
            f"consistency={report.consistency.agreed}/{report.consistency.cases} "
            f"runs={report.consistency.runs}"
        )
    typer.echo(
        f"requests={report.completed_requests}/{report.request_count} stopped={report.stopped} "
        f"cost_usd={_usd(report.usage.cost_usd)} passed={report.passed} output={out_dir}"
    )
    if not report.passed:
        raise typer.Exit(EXIT_INFEASIBLE)


def _usd(value: float | None) -> str:
    return "unknown" if value is None else f"{value:.4f}"


@app.command(
    "verify-agent-transition", help="Compare the second model request without running tools."
)
def verify_agent_transition_command(
    source: Annotated[Path, typer.Option("--run", help="Directory of a saved cell run.")],
    out_dir: Annotated[Path, typer.Option("--out-dir", "-o", help="New probe output directory.")],
    confirm: Annotated[
        bool,
        typer.Option(
            "--confirm",
            help="Check the current instruction and strict tool schemas in two requests.",
        ),
    ] = False,
) -> None:
    """네 조건의 모델 응답을 비교한다. 조립 도구는 실행하지 않는다."""
    if out_dir.exists() or out_dir.is_symlink():
        typer.echo("Probe output already exists. Use a new directory.", err=True)
        raise typer.Exit(EXIT_USAGE)
    try:
        settings = TokenFactorySettings(nemotron_model=DEFAULT_NEMOTRON_MODEL)  # pyright: ignore[reportCallIssue]
        report = run_transition_probe(
            source=source, out_dir=out_dir, settings=settings, confirm=confirm
        )
    except (ValidationError, ValueError, FileExistsError):
        typer.echo("Probe source or settings are not valid.", err=True)
        raise typer.Exit(EXIT_USAGE) from None
    except (OSError, OpenAIError):
        typer.echo("Could not complete or record the model probe.", err=True)
        raise typer.Exit(EXIT_INFEASIBLE) from None
    for result in report.results:
        typer.echo(
            f"case={result.case} accepted={result.accepted} "
            f"tool={result.selected_tool} reason={result.reason}"
        )
    if any(not result.accepted for result in report.results):
        raise typer.Exit(EXIT_INFEASIBLE)


@app.command("demo-serve", help="Serve the assembly demo on the DEMO_HOST address.")
def demo_serve(
    root: Annotated[Path, typer.Option("--runs", help="Directory for saved web runs.")] = Path(
        ".local-artifacts/demo/runs"
    ),
    port: Annotated[int, typer.Option(min=1024, max=65535)] = 8765,
) -> None:
    """기존 CLI를 실행하는 웹 서버를 `DEMO_HOST` 주소에서 연다."""
    from pixel2place.demo.server import serve

    try:
        settings = DemoSettings()
    except ValidationError:
        typer.echo("DEMO_HOST is not valid. Set one IPv4 address in .env.", err=True)
        raise typer.Exit(EXIT_USAGE) from None
    try:
        serve(root=root, workspace=Path.cwd(), host=str(settings.demo_host), port=port)
    except (OSError, ValueError):
        typer.echo(
            "Could not start the demo server. Check the run directory and network.", err=True
        )
        raise typer.Exit(EXIT_USAGE) from None


@app.command("run-cell")
def run_cell_command(
    prompt: str,
    out_dir: Annotated[Path, typer.Option("--out-dir", "-o", help="Output directory.")],
    parser: ParserOption = ParserName.GRID,
    cell_path: CellOption = None,
) -> None:
    """Plan a layout from an instruction. Place the parts in Isaac Sim. Check the result.

    Exit 1 if the task cannot be compiled, the runner fails or a part is out of tolerance.
    """
    cell = _read_cell(cell_path)
    try:
        sim_settings = SimSettings()  # pyright: ignore[reportCallIssue]
        reasoner = _reasoner()
    except ValidationError as exc:
        typer.echo(f"Reasoner or Isaac settings are not valid: {exc}", err=True)
        raise typer.Exit(EXIT_USAGE) from exc
    try:
        result = run_cell(
            prompt,
            out_dir=out_dir,
            reasoner=reasoner,
            parser=reasoner if parser == ParserName.COSMOS else GridImageParser(),
            renderer=SyntheticRenderer(),
            cell=cell,
            sim_settings=sim_settings,
        )
    except SimSetupError as exc:
        typer.echo(str(exc), err=True)
        raise typer.Exit(EXIT_USAGE) from exc
    except (TaskCompileError, SimRunError) as exc:
        typer.echo(str(exc), err=True)
        raise typer.Exit(EXIT_INFEASIBLE) from exc
    check = result.check
    for part in check.parts:
        typer.echo(_format_part_check(part))
    typer.echo(
        f"cell_accuracy={result.roundtrip.comparison.cell_accuracy:.3f} "
        f"passed={check.passed_count}/{len(check.parts)} output={out_dir}"
    )
    if not check.passed:
        raise typer.Exit(EXIT_INFEASIBLE)


@app.command("bench-cell")
def bench_cell(
    out_dir: Annotated[Path, typer.Option("--out-dir", "-o", help="Output directory.")],
    count: Annotated[int, typer.Option(min=0, help="Number of generated layouts.")] = 30,
    seed: Annotated[
        int, typer.Option(help="Seed for the generated layouts and the feeder offsets.")
    ] = 0,
    layout_dirs: Annotated[
        list[Path] | None,
        typer.Option(
            "--layouts",
            exists=True,
            file_okay=False,
            help="Directory of layout JSON files to add. Give it again to add more directories.",
        ),
    ] = None,
    repeats: Annotated[
        int,
        typer.Option(
            min=1, help="Runs per layout. Runs after the first move each feeder part a little."
        ),
    ] = 1,
    shard: ShardOption = ALL_LAYOUTS,
    jobs: Annotated[int, typer.Option(min=1, help="Number of Isaac runs at the same time.")] = 1,
    cell_path: CellOption = None,
    board_error: Annotated[
        BoardOffset | None,
        typer.Option(
            "--board-error",
            parser=_parse_board_error,
            metavar="X_MM,Y_MM,YAW_DEG",
            help="Limits of the board placement error. Each run moves the actual board at random.",
        ),
    ] = None,
    calibrate: Annotated[
        bool,
        typer.Option(
            "--calibrate",
            help=(
                "Before each run, capture the empty cell, measure the board pose "
                "and compile the task for that pose."
            ),
        ),
    ] = False,
    camera_error: Annotated[
        CameraOffset | None,
        typer.Option(
            "--camera-error",
            parser=_parse_camera_error,
            metavar="POS_MM,ANGLE_DEG",
            help="Limits of the top-down camera pose error on each axis. Needs --calibrate.",
        ),
    ] = None,
) -> None:
    """Run many layouts in the Isaac Sim cell and measure the part failure rate.

    The command skips runs that have a check for the same task and scene.
    With --calibrate, it also skips captures for the same capture task and scene.
    Exit 1 if a run did not finish. Parts out of tolerance do not change the exit code.
    """
    if camera_error is not None and not calibrate:
        typer.echo("--camera-error needs --calibrate.", err=True)
        raise typer.Exit(EXIT_USAGE)
    try:
        layouts = _bench_layouts(count=count, seed=seed, layout_dirs=layout_dirs or [])
        sim_settings = SimSettings()  # pyright: ignore[reportCallIssue]
    except ValidationError as exc:
        typer.echo(f"Layouts or Isaac settings are not valid: {exc}", err=True)
        raise typer.Exit(EXIT_USAGE) from exc
    if not layouts:
        typer.echo("There are no layouts. Give --count or --layouts.", err=True)
        raise typer.Exit(EXIT_USAGE)
    try:
        summary = run_cell_benchmark(
            layouts,
            out_dir=out_dir,
            cell=_read_cell(cell_path),
            sim_settings=sim_settings,
            repeats=repeats,
            seed=seed,
            shard=shard,
            jobs=jobs,
            board_error=board_error,
            camera_error=camera_error,
            calibrate=calibrate,
        )
    except (SimSetupError, TaskCompileError) as exc:
        typer.echo(str(exc), err=True)
        raise typer.Exit(EXIT_USAGE) from exc
    for line in _bench_failure_lines(summary):
        typer.echo(line)
    total = summary.total
    calibration = ""
    if calibrate:
        max_error = summary.max_calibration_error_mm
        calibration = (
            f"calibration_error_max_mm={'none' if max_error is None else f'{max_error:.2f}'} "
            f"order_changed_runs={summary.order_changed_runs} "
        )
    typer.echo(
        f"parts_passed={total.parts - total.failed}/{total.parts} "
        f"failure_rate={total.failure_rate:.3f} "
        f"runs_passed={summary.passed_runs}/{len(summary.runs)} "
        f"errors={summary.errors} {calibration}out_dir={out_dir}"
    )
    if summary.errors:
        raise typer.Exit(EXIT_INFEASIBLE)


@app.command("record-episodes")
def record_episodes_command(
    out_dir: Annotated[Path, typer.Option("--out-dir", "-o", help="Output directory.")],
    per_slot: Annotated[int, typer.Option(min=1, help="Number of episodes for each slot.")] = 100,
    seed: Annotated[
        int, typer.Option(help="Seed for the board errors and the feeder offsets.")
    ] = 0,
    shard: ShardOption = ALL_LAYOUTS,
    jobs: Annotated[int, typer.Option(min=1, help="Number of Isaac runs at the same time.")] = 1,
    cell_path: CellOption = None,
    board_error: Annotated[
        BoardOffset | None,
        typer.Option(
            "--board-error",
            parser=_parse_board_error,
            metavar="X_MM,Y_MM,YAW_DEG",
            help="Limits of the board placement error. The default is 6,10,3.",
        ),
    ] = None,
) -> None:
    """Record demonstrations of the slot task in the Isaac Sim cell.

    Each episode places one square tile in one slot. The board moves at random in each episode.
    The scripted controller knows the actual board pose.
    The command skips episodes that have a check and all record files for the same task and scene.
    Exit 1 if an episode did not finish.
    """
    try:
        sim_settings = SimSettings()  # pyright: ignore[reportCallIssue]
    except ValidationError as exc:
        typer.echo(f"Isaac settings are not valid: {exc}", err=True)
        raise typer.Exit(EXIT_USAGE) from exc
    try:
        summary = record_episodes(
            out_dir=out_dir,
            cell=_read_cell(cell_path),
            sim_settings=sim_settings,
            per_slot=per_slot,
            seed=seed,
            board_error=SLOT_BOARD_ERROR if board_error is None else board_error,
            shard=shard,
            jobs=jobs,
        )
    except (SimSetupError, TaskCompileError) as exc:
        typer.echo(str(exc), err=True)
        raise typer.Exit(EXIT_USAGE) from exc
    for line in _bench_failure_lines(summary):
        typer.echo(line)
    typer.echo(
        f"episodes_passed={summary.passed_runs}/{len(summary.runs)} slots={len(SLOTS)} "
        f"errors={summary.errors} out_dir={out_dir}"
    )
    if summary.errors:
        raise typer.Exit(EXIT_INFEASIBLE)


@app.command("eval-policy")
def eval_policy_command(
    out_dir: Annotated[Path, typer.Option("--out-dir", "-o", help="Output directory.")],
    policy: Annotated[
        str,
        typer.Option(
            "--policy",
            help=(
                "Controller: scripted, scripted-calibrated, replay:DIR or HOST:PORT. "
                "DIR is an output directory of record-episodes. "
                "HOST:PORT is the address of a controller server."
            ),
        ),
    ],
    per_slot: Annotated[int, typer.Option(min=1, help="Number of episodes for each slot.")] = 25,
    seed: Annotated[
        int, typer.Option(help="Seed for the board errors and the feeder offsets.")
    ] = 1,
    shard: ShardOption = ALL_LAYOUTS,
    jobs: Annotated[int, typer.Option(min=1, help="Number of Isaac runs at the same time.")] = 1,
    cell_path: CellOption = None,
    board_error: Annotated[
        BoardOffset | None,
        typer.Option(
            "--board-error",
            parser=_parse_board_error,
            metavar="X_MM,Y_MM,YAW_DEG",
            help="Limits of the board placement error. The default is 6,10,3.",
        ),
    ] = None,
    max_steps: Annotated[
        int, typer.Option(min=1, help="Frame limit for a run with a learned controller.")
    ] = EVAL_MAX_STEPS,
    joint_error: Annotated[
        JointErrorLevel | None,
        typer.Option(
            "--joint-error",
            parser=_parse_joint_error,
            metavar="SIGMA_DEG,TAU_S",
            help=(
                "Joint sensor error of the arm. SIGMA_DEG is the standard deviation of each "
                "joint. TAU_S is the correlation time. TAU_S inf gives a constant error."
            ),
        ),
    ] = None,
) -> None:
    """Run one controller on the slot task in the Isaac Sim cell and measure the pass count.

    Scenes with the same seed, board error and joint error are the same for all controllers.
    The command skips episodes that have a check and all record files for the same task and scene.
    Exit 1 if an episode did not finish. Episodes out of tolerance do not change the exit code.
    """
    try:
        sim_settings = SimSettings()  # pyright: ignore[reportCallIssue]
    except ValidationError as exc:
        typer.echo(f"Isaac settings are not valid: {exc}", err=True)
        raise typer.Exit(EXIT_USAGE) from exc
    try:
        evaluation = evaluate_policy(
            policy,
            out_dir=out_dir,
            cell=_read_cell(cell_path),
            sim_settings=sim_settings,
            per_slot=per_slot,
            seed=seed,
            board_error=SLOT_BOARD_ERROR if board_error is None else board_error,
            shard=shard,
            jobs=jobs,
            max_steps=max_steps,
            joint_error=joint_error,
        )
    except (SimSetupError, TaskCompileError, ValueError) as exc:
        typer.echo(str(exc), err=True)
        raise typer.Exit(EXIT_USAGE) from exc
    for line in _evaluation_lines(evaluation):
        typer.echo(line)
    if evaluation.errors:
        raise typer.Exit(EXIT_INFEASIBLE)


def _evaluation_lines(evaluation: PolicyEvaluation) -> list[str]:
    """통과하지 못한 에피소드마다 한 줄, 구간마다 한 줄과 요약 한 줄을 만든다."""
    lines = []
    for episode in evaluation.episodes:
        if episode.passed:
            continue
        reason = "fail" if episode.error_kind is None else f"error {episode.error_kind}"
        placed = "none" if episode.placed_slot is None else str(episode.placed_slot)
        lines.append(
            f"{episode.layout} r{episode.repeat} {reason} "
            f"slot_shift_mm={episode.slot_shift_mm:.1f} placed_slot={placed}"
        )
    lines += [
        f"slot {name} passed={count.passed}/{count.episodes}"
        for name, count in evaluation.by_slot.items()
    ]
    lines += [
        f"slot_shift_mm {name} passed={count.passed}/{count.episodes}"
        for name, count in evaluation.by_shift.items()
    ]
    total = evaluation.total
    lines.append(
        f"controller={evaluation.controller} passed={total.passed}/{total.episodes} "
        f"wrong_slot={evaluation.wrong_slot} errors={evaluation.errors}"
    )
    return lines


def _bench_layouts(*, count: int, seed: int, layout_dirs: list[Path]) -> list[BenchLayout]:
    """무작위 레이아웃 다음에 디렉터리의 레이아웃 파일을 이름 순서로 붙인다."""
    layouts = [
        BenchLayout(RANDOM_LAYOUT_FORMAT.format(index=index), layout)
        for index, layout in enumerate(generate_layouts(count=count, seed=seed))
    ]
    for directory in layout_dirs:
        layouts += [
            BenchLayout(path.stem, _read_layout(path)) for path in sorted(directory.glob("*.json"))
        ]
    return layouts


def _bench_failure_lines(summary: CellBenchmark) -> list[str]:
    """끝나지 못한 실행과 실패한 부품마다 한 줄을 만든다."""
    lines = []
    for run in summary.runs:
        name = f"{run.layout} r{run.repeat}"
        if run.error is not None:
            first_line = run.error.partition("\n")[0]
            lines.append(f"{name} error {run.error_kind} {first_line}")
        lines += [
            f"{name} {part.part_id} seq={part.sequence} {part.type} "
            f"neighbor_sides={part.neighbor_sides} "
            f"clearance_mm={_clearance_text(part.clearance_mm)} "
            f"fail {','.join(part.failures)} stage={part.failed_stage}"
            for part in run.parts
            if not part.passed
        ]
    return lines


def _clearance_text(clearance_mm: float | None) -> str:
    return "none" if clearance_mm is None else f"{clearance_mm:.2f}"


def _format_part_check(part: PartCheck) -> str:
    verdict = "pass" if part.passed else f"fail {','.join(part.failures)} stage={part.failed_stage}"
    return (
        f"{part.part_id} seq={part.sequence} {part.type} "
        f"position_mm={part.position_error_mm:.1f} yaw_deg={part.yaw_error_deg:+.1f} "
        f"tilt_deg={part.tilt_deg:.1f} height_mm={part.height_error_mm:+.1f} {verdict}"
    )


def _image_check_camera(
    *, camera_path: Path | None, calibration_path: Path | None
) -> PinholeCamera:
    """`--camera`의 카메라. 없으면 `--calibration` 보고서의 추정 카메라다."""
    if camera_path is not None:
        return PinholeCamera.model_validate_json(camera_path.read_text(encoding="utf-8"))
    assert calibration_path is not None
    return read_report(calibration_path).camera


def _format_part_image_check(part: PartImageCheck) -> str:
    pins = "hidden" if part.pins_hidden else "visible"
    return (
        f"{part.part_id} seq={part.sequence} {part.type} {part.color} "
        f"cells={_cells_text(part.cells)} coverage={part.coverage:.2f} "
        f"shadow_coverage={part.shadow_coverage:.2f} pins={pins} {part.status}"
    )


def _cells_text(cells: list[Cell]) -> str:
    return ",".join(f"({row},{col})" for row, col in cells)


def _generator() -> CosmosGenerator:
    """생성 서버 설정이 없으면 없는 변수 이름을 stderr에 쓰고 exit 2로 끝낸다."""
    try:
        settings = GeneratorSettings()  # pyright: ignore[reportCallIssue]
    except ValidationError as exc:
        names = ", ".join(str(error["loc"][0]).upper() for error in exc.errors())
        typer.echo(f"Generator settings are not valid: {names}. Set them in .env.", err=True)
        raise typer.Exit(EXIT_USAGE) from exc
    return CosmosGenerator.from_settings(settings)


def _expand(
    prompt: str, *, output: Path, target: ExpansionTarget, image: Image.Image | None = None
) -> str:
    """VLM으로 지시문을 확장하고 결과를 출력 파일 옆에 저장한다."""
    expanded = _reasoner().expand_prompt(prompt, target=target, image=image)
    prompt_path = _with_parent(output.with_name(output.name + PROMPT_FILE_SUFFIX))
    prompt_path.write_text(
        json.dumps(json.loads(expanded), indent=JSON_INDENT) + "\n", encoding="utf-8"
    )
    typer.echo(str(prompt_path))
    return expanded


def _reasoner() -> CosmosReasoner:
    # 필수 환경 변수가 없으면 pydantic이 ValidationError를 낸다.
    return CosmosReasoner.from_settings(Settings())  # pyright: ignore[reportCallIssue]


def _parser(name: ParserName) -> LayoutParser:
    if name == ParserName.COSMOS:
        return _reasoner()
    return GridImageParser()


def _read_cell(path: Path | None) -> CellConfig:
    if path is None:
        return CellConfig()
    return CellConfig.model_validate_json(path.read_text(encoding="utf-8"))


def _read_layout(path: Path) -> Layout:
    return Layout.model_validate_json(path.read_text(encoding="utf-8"))


def _write_layout(path: Path, layout: Layout) -> None:
    write_json(_with_parent(path), layout)
    typer.echo(str(path))


def _with_parent(path: Path) -> Path:
    """출력 파일의 부모 디렉터리를 만들고 같은 경로를 반환한다."""
    path.parent.mkdir(parents=True, exist_ok=True)
    return path
