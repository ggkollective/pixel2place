"""판독기 벤치마크.

코드로 배치 가능한 레이아웃을 만들고 그린다. 판독기가 그 이미지를 읽은 결과를 정답과 비교한다.
배치 계획 모델을 거치지 않으므로 판독기 성능만 측정한다.
"""

import logging
import random
import statistics
import time
from collections.abc import Callable
from pathlib import Path

from pydantic import BaseModel, ConfigDict

from pixel2place.feasibility.checker import check_layout
from pixel2place.generation.base import LayoutRenderer
from pixel2place.layout.parts import place
from pixel2place.layout.schema import Board, Cell, Color, Layout, Part, PartType, Rotation
from pixel2place.parsing.base import LayoutParser
from pixel2place.parsing.grid import GridParseError
from pixel2place.pipeline import compare_layouts, keep_raw_responses, write_json
from pixel2place.reasoner.client import ReasonerOutputError

logger = logging.getLogger(__name__)

# 표본 i의 부품 수 구간은 PART_COUNT_BANDS[i % 3]이다. 양 끝을 포함한다.
PART_COUNT_BANDS: tuple[tuple[int, int], ...] = ((2, 4), (5, 8), (9, 14))
# 부품 1개를 놓을 때 무작위 위치를 고르는 최대 횟수.
MAX_PLACEMENT_TRIES = 200
SAMPLE_DIR_FORMAT = "{index:03d}"
TRUTH_FILE = "truth.json"
RAW_IMAGE_FILE = "raw_image.png"
LAYOUT_FILE = "layout.json"
SUMMARY_FILE = "summary.json"
PART_ID_FORMAT = "P{index:03d}"


class SampleResult(BaseModel):
    """표본 1개의 결과. 판독기가 실패하면 `cell_accuracy`는 0.0이고 `error`에 메시지가 있다."""

    model_config = ConfigDict(frozen=True)

    index: int
    part_count: int
    parsed_part_count: int
    cell_accuracy: float
    issue_count: int
    elapsed_s: float
    error: str | None = None


class BandSummary(BaseModel):
    """부품 수 구간 1개의 평균."""

    model_config = ConfigDict(frozen=True)

    min_parts: int
    max_parts: int
    count: int
    mean_cell_accuracy: float


class ParserBenchmark(BaseModel):
    """벤치마크 1회의 요약. `summary.json`에 그대로 저장한다."""

    model_config = ConfigDict(frozen=True)

    parser: str
    """평가한 판독기 클래스 이름."""
    count: int
    seed: int
    mean_cell_accuracy: float
    median_cell_accuracy: float
    exact_match_rate: float
    """`cell_accuracy == 1.0`인 표본의 비율."""
    part_count_match_rate: float
    failures: int
    mean_elapsed_s: float
    bands: list[BandSummary]
    samples: list[SampleResult]


def generate_layouts(*, count: int, seed: int, board: Board | None = None) -> list[Layout]:
    """배치 규칙을 지키는 무작위 레이아웃을 만든다. 같은 인자는 같은 레이아웃을 만든다."""
    rng = random.Random(seed)
    board = board or Board()
    layouts = []
    for index in range(count):
        low, high = PART_COUNT_BANDS[index % len(PART_COUNT_BANDS)]
        layouts.append(_random_layout(rng, board=board, target=rng.randint(low, high)))
    return layouts


def _random_layout(rng: random.Random, *, board: Board, target: int) -> Layout:
    occupied: set[Cell] = set()
    parts: list[Part] = []
    for _ in range(target):
        part = _random_part(rng, board=board, occupied=occupied, index=len(parts) + 1)
        # 빈 자리를 찾지 못하면 부품 수를 목표보다 적게 둔다.
        if part is None:
            break
        occupied.update(part.cells)
        parts.append(part)
    palette = sorted({part.color for part in parts})
    return Layout(board=board, palette=palette, parts=parts)


def _random_part(
    rng: random.Random, *, board: Board, occupied: set[Cell], index: int
) -> Part | None:
    part_type = rng.choice(list(PartType))
    rotation = rng.choice(list(Rotation))
    color = rng.choice(list(Color))
    for _ in range(MAX_PLACEMENT_TRIES):
        origin = (rng.randrange(board.rows), rng.randrange(board.cols))
        cells = place(part_type, rotation, origin=origin)
        inside = all(0 <= row < board.rows and 0 <= col < board.cols for row, col in cells)
        if inside and occupied.isdisjoint(cells):
            return Part(
                part_id=PART_ID_FORMAT.format(index=index),
                type=part_type,
                color=color,
                cells=cells,
                rotation_deg=rotation,
            )
    return None


def run_parser_benchmark(
    layouts: list[Layout],
    *,
    seed: int,
    parser: LayoutParser,
    renderer: LayoutRenderer,
    out_dir: Path,
) -> ParserBenchmark:
    """레이아웃마다 이미지를 그리고 판독기 결과를 정답과 비교한다. 결과 파일을 `out_dir`에 쓴다."""
    samples = []
    for index, truth in enumerate(layouts):
        sample_dir = out_dir / SAMPLE_DIR_FORMAT.format(index=index)
        sample_dir.mkdir(parents=True, exist_ok=True)
        samples.append(
            _run_sample(index, truth, parser=parser, renderer=renderer, sample_dir=sample_dir)
        )
        logger.info(
            "Parser benchmark sample finished.",
            extra={"index": index, "cell_accuracy": samples[-1].cell_accuracy},
        )
    summary = _summarize(samples, seed=seed, parser=type(parser).__name__)
    write_json(out_dir / SUMMARY_FILE, summary)
    return summary


def _run_sample(
    index: int,
    truth: Layout,
    *,
    parser: LayoutParser,
    renderer: LayoutRenderer,
    sample_dir: Path,
) -> SampleResult:
    write_json(sample_dir / TRUTH_FILE, truth)
    image = renderer.render(truth)
    image.save(sample_dir / RAW_IMAGE_FILE)
    started = time.monotonic()
    try:
        with keep_raw_responses(sample_dir, stage="parse"):
            parsed = parser.parse(image)
    except (ReasonerOutputError, GridParseError) as exc:
        return SampleResult(
            index=index,
            part_count=len(truth.parts),
            parsed_part_count=0,
            cell_accuracy=0.0,
            issue_count=0,
            elapsed_s=round(time.monotonic() - started, 1),
            error=str(exc),
        )
    elapsed_s = round(time.monotonic() - started, 1)
    write_json(sample_dir / LAYOUT_FILE, parsed)
    comparison = compare_layouts(planned=truth, parsed=parsed)
    return SampleResult(
        index=index,
        part_count=len(truth.parts),
        parsed_part_count=len(parsed.parts),
        cell_accuracy=comparison.cell_accuracy,
        issue_count=len(check_layout(parsed).issues),
        elapsed_s=elapsed_s,
    )


def _summarize(samples: list[SampleResult], *, seed: int, parser: str) -> ParserBenchmark:
    accuracies = [sample.cell_accuracy for sample in samples]
    bands = []
    for low, high in PART_COUNT_BANDS:
        in_band = [s.cell_accuracy for s in samples if low <= s.part_count <= high]
        if in_band:
            bands.append(
                BandSummary(
                    min_parts=low,
                    max_parts=high,
                    count=len(in_band),
                    mean_cell_accuracy=statistics.fmean(in_band),
                )
            )
    return ParserBenchmark(
        parser=parser,
        count=len(samples),
        seed=seed,
        mean_cell_accuracy=statistics.fmean(accuracies),
        median_cell_accuracy=statistics.median(accuracies),
        exact_match_rate=_rate(samples, lambda s: s.cell_accuracy == 1.0),
        part_count_match_rate=_rate(samples, lambda s: s.parsed_part_count == s.part_count),
        failures=sum(1 for sample in samples if sample.error is not None),
        mean_elapsed_s=statistics.fmean(sample.elapsed_s for sample in samples),
        bands=bands,
        samples=samples,
    )


def _rate(samples: list[SampleResult], predicate: Callable[[SampleResult], bool]) -> float:
    return sum(1 for sample in samples if predicate(sample)) / len(samples)
