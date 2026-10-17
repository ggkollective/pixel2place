"""지시문에서 시작하는 처리 흐름.

- round-trip: 지시문 -> 레이아웃 -> 이미지 -> 레이아웃
- 셀 실행: round-trip -> 작업 -> Isaac Sim 셀 -> 판정
"""

import json
import logging
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path

from pydantic import BaseModel, ConfigDict

from pixel2place.cell.check import CheckReport, check_result
from pixel2place.cell.config import CellConfig
from pixel2place.cell.sim import require_sim_files, run_place_task
from pixel2place.cell.task import compile_task
from pixel2place.config import SimSettings
from pixel2place.feasibility.checker import FeasibilityReport, apply_report, check_layout
from pixel2place.feasibility.repair import Repair, repair_layout
from pixel2place.generation.base import LayoutRenderer
from pixel2place.layout.schema import Cell, Color, Layout
from pixel2place.parsing.base import LayoutParser
from pixel2place.reasoner.client import CosmosReasoner, ReasonerOutputError

logger = logging.getLogger(__name__)

RAW_IMAGE_FILE = "raw_image.png"
NORMALIZED_IMAGE_FILE = "normalized_image.png"
PLAN_LAYOUT_FILE = "plan_layout.json"
LAYOUT_FILE = "layout.json"
REPORT_FILE = "report.json"
TASK_FILE = "task.json"
SIM_DIR = "sim"
CHECK_FILE = "check.json"
IMAGE_CHECK_FILE = "image_check.json"
# 실패한 단계의 원시 응답 파일. `{stage}`는 plan 또는 parse다.
RAW_RESPONSE_FILE = "{stage}_raw_response.txt"
RAW_RESPONSE_SEPARATOR = "\n\n----- next response -----\n\n"
JSON_INDENT = 2


class LayoutComparison(BaseModel):
    """계획 레이아웃과 파싱 레이아웃의 셀 단위 비교 결과."""

    model_config = ConfigDict(frozen=True)

    cell_accuracy: float
    """두 레이아웃 중 한쪽이라도 채운 셀에서 색이 같은 셀의 비율."""
    planned_part_count: int
    parsed_part_count: int


class RoundTripResult(BaseModel):
    """round-trip 1회의 결과. `report.json`에 그대로 저장한다."""

    model_config = ConfigDict(frozen=True)

    prompt: str
    parser: str
    """parse 단계에 쓴 판독기 클래스 이름."""
    plan_feasibility: FeasibilityReport
    """수리 전 planner 출력의 검사 결과. 모델 출력 품질을 나타낸다."""
    plan_repairs: list[Repair]
    """planner 출력에 적용한 수리 기록."""
    parse_feasibility: FeasibilityReport
    comparison: LayoutComparison


class CellRunResult(BaseModel):
    """셀 실행 1회의 결과."""

    model_config = ConfigDict(frozen=True)

    roundtrip: RoundTripResult
    check: CheckReport


def compare_layouts(*, planned: Layout, parsed: Layout) -> LayoutComparison:
    """두 레이아웃의 셀 점유와 색을 비교한다."""
    planned_cells = _color_by_cell(planned)
    parsed_cells = _color_by_cell(parsed)
    union = planned_cells.keys() | parsed_cells.keys()
    matched = sum(1 for cell in union if planned_cells.get(cell) == parsed_cells.get(cell))
    return LayoutComparison(
        cell_accuracy=matched / len(union) if union else 1.0,
        planned_part_count=len(planned.parts),
        parsed_part_count=len(parsed.parts),
    )


def repair_plan(planned: Layout) -> tuple[Layout, list[Repair]]:
    """배치 계획 모델 레이아웃의 배치 위반을 고치고 검사 결과를 기록한다.

    `notes`에는 수리 뒤 검사 결과와 수리 기록을 쓴다.
    수리 기록 줄의 형식은 `"<MOVED|REMOVED>: <메시지>"`다.
    """
    result = repair_layout(planned)
    layout = apply_report(result.layout, check_layout(result.layout))
    if result.repairs:
        logger.info("Repaired planned layout.", extra={"repairs": len(result.repairs)})
    notes = [*layout.notes, *(f"{repair.action}: {repair.message}" for repair in result.repairs)]
    return layout.model_copy(update={"notes": notes}), result.repairs


def run_roundtrip(
    prompt: str,
    *,
    out_dir: Path,
    reasoner: CosmosReasoner,
    parser: LayoutParser,
    renderer: LayoutRenderer,
) -> RoundTripResult:
    """round-trip을 실행하고 결과 파일을 `out_dir`에 쓴다.

    단계:
    1. reasoner가 지시문에서 레이아웃을 만든다. 배치 위반은 `repair_plan`이 고친다.
    2. renderer가 레이아웃을 `raw_image.png`로 그린다.
    3. 판독기가 이미지에서 레이아웃을 다시 읽는다.
    4. 두 레이아웃을 검사하고 비교한다.

    Cosmos reasoner가 실패하면 원시 응답을 `{stage}_raw_response.txt`에 쓰고 예외를 다시 낸다.
    """
    out_dir.mkdir(parents=True, exist_ok=True)

    with keep_raw_responses(out_dir, stage="plan"):
        planned = reasoner.plan(prompt)
    plan_report = check_layout(planned)
    planned, plan_repairs = repair_plan(planned)
    write_json(out_dir / PLAN_LAYOUT_FILE, planned)
    logger.info("Planned layout.", extra={"parts": len(planned.parts)})

    raw_image = renderer.render(planned)
    raw_image.save(out_dir / RAW_IMAGE_FILE)

    with keep_raw_responses(out_dir, stage="parse"):
        parsed = parser.parse(raw_image)
    parse_report = check_layout(parsed)
    parsed = apply_report(parsed, parse_report)
    write_json(out_dir / LAYOUT_FILE, parsed)
    renderer.render(parsed).save(out_dir / NORMALIZED_IMAGE_FILE)

    result = RoundTripResult(
        prompt=prompt,
        parser=type(parser).__name__,
        plan_feasibility=plan_report,
        plan_repairs=plan_repairs,
        parse_feasibility=parse_report,
        comparison=compare_layouts(planned=planned, parsed=parsed),
    )
    write_json(out_dir / REPORT_FILE, result)
    logger.info(
        "Round trip finished.",
        extra={"cell_accuracy": result.comparison.cell_accuracy, "out_dir": str(out_dir)},
    )
    return result


def run_cell(
    prompt: str,
    *,
    out_dir: Path,
    reasoner: CosmosReasoner,
    parser: LayoutParser,
    renderer: LayoutRenderer,
    cell: CellConfig,
    sim_settings: SimSettings,
) -> CellRunResult:
    """지시문에서 만든 레이아웃을 Isaac Sim 셀에서 놓고 판정한다. 결과 파일은 `out_dir`에 쓴다.

    단계:
    1. Isaac 실행 파일을 확인한다. 파일이 없으면 Reasoner를 부르기 전에 `SimSetupError`를 낸다.
    2. `run_roundtrip`으로 레이아웃을 계획하고, 이미지를 그리고, 이미지에서 레이아웃을 읽는다.
    3. 읽은 레이아웃(`layout.json`)을 작업으로 컴파일하고 `task.json`에 쓴다.
    4. `sim/`에 Isaac 셀 실행 결과를 쓴다.
    5. 부품 자세를 판정하고 `check.json`에 쓴다.

    컴파일 실패(`TaskCompileError`)와 실행 실패(`SimRunError`)는 그대로 올린다.
    """
    require_sim_files(sim_settings)
    roundtrip = run_roundtrip(
        prompt, out_dir=out_dir, reasoner=reasoner, parser=parser, renderer=renderer
    )
    parsed = Layout.model_validate_json((out_dir / LAYOUT_FILE).read_text(encoding="utf-8"))
    task = compile_task(parsed, cell)
    task_path = out_dir / TASK_FILE
    write_json(task_path, task)
    logger.info("Compiled task.", extra={"parts": len(task.parts)})

    result = run_place_task(task_path, out_dir=out_dir / SIM_DIR, settings=sim_settings)
    check = check_result(task, result)
    write_json(out_dir / CHECK_FILE, check)
    logger.info(
        "Cell run finished.",
        extra={"passed": check.passed_count, "parts": len(check.parts), "out_dir": str(out_dir)},
    )
    return CellRunResult(roundtrip=roundtrip, check=check)


@contextmanager
def keep_raw_responses(out_dir: Path, *, stage: str) -> Iterator[None]:
    """`ReasonerOutputError`가 나면 원시 응답을 `{stage}_raw_response.txt`에 쓰고 다시 낸다."""
    try:
        yield
    except ReasonerOutputError as exc:
        path = out_dir / RAW_RESPONSE_FILE.format(stage=stage)
        path.write_text(RAW_RESPONSE_SEPARATOR.join(exc.responses), encoding="utf-8")
        logger.info("Wrote raw reasoner responses.", extra={"path": str(path)})
        raise


def _color_by_cell(layout: Layout) -> dict[Cell, Color]:
    return {cell: part.color for part in layout.parts for cell in part.cells}


def write_json(path: Path, model: BaseModel) -> None:
    """모델을 들여쓴 JSON 파일로 쓴다."""
    payload = model.model_dump(mode="json")
    path.write_text(json.dumps(payload, indent=JSON_INDENT) + "\n", encoding="utf-8")
