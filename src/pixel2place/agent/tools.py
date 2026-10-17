"""기존 처리 흐름을 호출하는 레이아웃 준비, 보정과 셀 실행 도구."""

import logging
from pathlib import Path
from typing import Annotated, Literal, NamedTuple

from openai import OpenAIError
from PIL import Image
from pydantic import BaseModel, Field, TypeAdapter, ValidationError

from pixel2place.agent.artifacts import ArtifactError, ArtifactStore, write_record
from pixel2place.agent.contracts import (
    ArtifactRef,
    CalibrateBoardArguments,
    CalibrationResult,
    CallId,
    Contract,
    ErrorCode,
    ErrorResult,
    ErrorSummary,
    ExecuteCellArguments,
    ExecutionResult,
    LayoutResult,
    OpaqueId,
    PrepareLayoutArguments,
    ReplanLayoutArguments,
    ReviewReason,
    ReworkCellArguments,
    ToolName,
    ToolResult,
    ToolState,
)
from pixel2place.agent.inspection import (
    checked_error,
    inspect_layout,
    inspected_summary,
    review_summary,
    rework_targets,
    summarize_calibration,
    summarize_execution,
)
from pixel2place.agent.runtime_contracts import AgentMetadata
from pixel2place.cell.calibration import (
    REPORT_SUFFIX,
    CalibrationError,
    calibrate_board,
    read_report,
    write_calibration,
)
from pixel2place.cell.capture import CAPTURE_FILE, compile_capture
from pixel2place.cell.check import ResultMismatchError, check_result
from pixel2place.cell.config import CellConfig
from pixel2place.cell.image_check import ImageCheckReport, check_image
from pixel2place.cell.result import SimResult
from pixel2place.cell.rework import ReworkError, rework_scene
from pixel2place.cell.scene import SimScene, placement_of
from pixel2place.cell.sim import (
    CAMERA_FILE,
    CAMERA_TRUTH_FILE,
    CAPTURE_IMAGE_FILE,
    FINAL_IMAGE_FILE,
    RESULT_FILE,
    SimRunError,
    SimSetupError,
    require_sim_files,
    run_capture,
    run_place_task,
)
from pixel2place.cell.task import Task, TaskCompileError, compile_task
from pixel2place.config import SimSettings
from pixel2place.feasibility.checker import IssueCode
from pixel2place.generation.synthetic import SyntheticRenderer
from pixel2place.layout.parts import canonical_cols
from pixel2place.layout.schema import Board, Layout, PartType
from pixel2place.parsing.grid import GridImageParser, GridParseError
from pixel2place.pipeline import (
    CHECK_FILE,
    IMAGE_CHECK_FILE,
    LAYOUT_FILE,
    NORMALIZED_IMAGE_FILE,
    PLAN_LAYOUT_FILE,
    RAW_IMAGE_FILE,
    REPORT_FILE,
    SIM_DIR,
    TASK_FILE,
    run_roundtrip,
    write_json,
)
from pixel2place.reasoner.client import CosmosReasoner, ReasonerOutputError

logger = logging.getLogger(__name__)

MAX_LAYOUT_ATTEMPTS = 2
SIM_TIMEOUT_S = 1800.0
EXECUTION_DIR = "execution"
"""실행 디렉터리 안에서 작업, 첫 배치의 결과와 판정을 두는 디렉터리."""
REWORK_DIR = "rework"
"""`EXECUTION_DIR` 안에서 재작업 실행의 결과와 판정을 두는 디렉터리."""
REWORK_SCENE_FILE = "rework_scene.json"
_RECT_COLS = canonical_cols(PartType.RECT_TILE)
RETRY_FEEDBACK = (
    "The previous layout preparation failed with error_code={code}.\n"
    "Create a valid layout for the original goal above.\n"
    "Keep all required parts and conditions in the goal.\n"
    "Correct the layout format and placement rules."
)
# 계획 검토의 재계획에서 원래 목표 뒤에 붙인다. 모델은 Cosmos에 문장을 줄 수 없다.
REVIEW_FEEDBACK = {
    ReviewReason.PART_SET_MISMATCH: (
        "The previous layout did not have the part types, colors and quantities "
        "of the goal above.\n"
        "Create a new layout with exactly the parts of the goal."
    ),
    ReviewReason.POSITION_MISMATCH: (
        "The previous layout did not put each part at the position that the goal above gives.\n"
        "Create a new layout. Put each part at its position in the goal."
    ),
    # 부품이 쓰는 col의 수와 row의 길이를 쓴다. 규칙만 쓴 문장은 planner가 따르지 못한다.
    ReviewReason.ARRANGEMENT_MISMATCH: (
        "The previous layout did not obey the arrangement rule of the goal above.\n"
        "Create a new layout. "
        "Parts of the same type must have the same row and different cols. "
        f"A rectangular tile uses {_RECT_COLS} cols. A row has {Board().cols} cols."
    ),
}


class _Goal(Contract):
    prompt: Annotated[str, Field(min_length=1, max_length=4000, pattern=r"\S")]


class ToolRun(Contract):
    """참값이 없는 고정 실행 입력."""

    schema_version: Literal[1] = 1
    run_id: OpaqueId
    prompt: str
    cell: CellConfig
    agent: AgentMetadata | None = None


class ToolLedger(Contract):
    """외부 작업 전에 기록하는 호출과 소비 상태. 자동 재개에 쓰지 않는다."""

    schema_version: Literal[1] = 1
    run_id: OpaqueId
    state: ToolState = ToolState.CREATED
    layout_attempts: int = 0
    capture_consumed: bool = False
    place_consumed: bool = False
    in_flight: ToolName | None = None
    call_ids: tuple[CallId, ...] = ()
    result_count: int = 0
    layout_id: OpaqueId | None = None
    calibration_id: OpaqueId | None = None
    check_id: OpaqueId | None = None
    check_passed: bool | None = None
    error_code: ErrorCode | None = None
    rework_consumed: bool = False
    image_check_id: OpaqueId | None = None
    """처분을 켠 실행의 카메라 판정. 처분을 끈 실행은 None이다."""
    image_check_passed: bool | None = None
    rework_targets: tuple[str, ...] = ()
    """`rework_cell`이 받는 부품 ID. 카메라 판정에서 구한다. 모델에 전달하지 않는다."""


class _ImageCheck(NamedTuple):
    """등록한 카메라 판정의 ID와 보고서."""

    id: str
    report: ImageCheckReport


class AssemblyTools:
    """한 실행의 도구와 산출물을 소유한다. 실패 뒤 캡처와 배치를 반복하지 않는다."""

    def __init__(
        self,
        *,
        out_dir: Path,
        prompt: str,
        cell: CellConfig,
        scene: SimScene,
        reasoner: CosmosReasoner,
        sim_settings: SimSettings,
        agent: AgentMetadata | None = None,
    ) -> None:
        goal = _Goal(prompt=prompt)
        cell = CellConfig.model_validate_json(cell.model_dump_json())
        scene = SimScene.model_validate_json(scene.model_dump_json())
        python_sh, usd = require_sim_files(sim_settings)
        self._board = Board()
        # board=None이면 보정 후 실제 보드까지 움직인다. 최초 자세로 고정한다.
        scene = scene.model_copy(update={"board": scene.board or placement_of(cell, self._board)})
        self._prompt = goal.prompt
        self._review = agent is not None and agent.review
        self._disposition = agent is not None and agent.disposition
        self._reasoner = reasoner
        self._initial_cosmos_requests = reasoner.request_count
        self._sim_settings = sim_settings.model_copy(
            update={
                "sim_timeout_s": SIM_TIMEOUT_S,
                "isaac_sim_path": python_sh.parent.resolve(),
                "so101_usd_path": usd.resolve(),
            }
        )
        self._store = ArtifactStore(out_dir)
        self._ledger = ToolLedger(run_id=self._store.run_id)
        self._recording_failed = False
        root = self.root
        (root / "evaluation").mkdir()
        (root / "tool_results").mkdir()
        write_record(
            root / "run.json",
            ToolRun(run_id=self.run_id, prompt=goal.prompt, cell=cell, agent=agent),
        )
        write_record(root / "cell.json", cell)
        self._scene_path = root / "evaluation" / "scene.json"
        write_record(self._scene_path, scene)
        inputs = self._store.register(
            "inputs", (root / "cell.json", self._scene_path, root / "run.json")
        )
        self._inputs_id = inputs.id
        self._save_ledger()

    @property
    def root(self) -> Path:
        return self._store.root

    @property
    def run_id(self) -> str:
        return self._store.run_id

    @property
    def ledger(self) -> ToolLedger:
        return self._ledger

    @property
    def review(self) -> bool:
        """계획 검토를 켠 실행인지. 준비 결과의 부품 목록과 `replan_layout`을 정한다."""
        return self._review

    @property
    def disposition(self) -> bool:
        """처분을 켠 실행인지. 실행 결과의 카메라 판정, `rework_cell`과 출하 규칙을 정한다."""
        return self._disposition

    @property
    def cosmos_requests(self) -> int:
        """실패한 요청을 포함한 이 실행의 Cosmos 호출 수."""
        return self._reasoner.request_count - self._initial_cosmos_requests

    def finish(self, *, completed: bool, error_code: ErrorCode | None) -> None:
        """실행 종료를 기록한다. 출하 규칙이 허용해야 완료를 받는다."""
        if self._recording_failed:
            raise OSError("Tool record is unavailable.")
        if completed and not self.release_allowed():
            raise ArtifactError("Check did not pass.")
        self._ledger = self._ledger.model_copy(
            update={
                "state": ToolState.COMPLETED if completed else ToolState.ABORTED,
                "error_code": error_code,
                "in_flight": None,
            }
        )
        self._save_ledger()

    def prepare_layout(
        self, arguments: PrepareLayoutArguments, *, tool_call_id: str
    ) -> LayoutResult | ErrorResult:
        """원문을 Cosmos에 전달한다. 재계획은 코드가 보존한 오류만 추가한다."""
        tool = ToolName.PREPARE_LAYOUT
        expected = ToolState.LAYOUT_RETRYABLE if arguments.retry else ToolState.CREATED
        failure = self._begin(tool, tool_call_id, expected)
        if failure is not None:
            return failure
        prompt = self._prompt
        if arguments.retry:
            assert self._ledger.error_code is not None
            prompt += "\n\n" + RETRY_FEEDBACK.format(code=self._ledger.error_code.value)
        return self._plan(tool, tool_call_id, prompt)

    def replan_layout(
        self, arguments: ReplanLayoutArguments, *, tool_call_id: str
    ) -> LayoutResult | ErrorResult:
        """모델이 거부한 계획을 다시 만든다. Cosmos에는 원문과 사유별 고정 문장을 전달한다.

        계획 검토를 켠 실행에서만 쓴다. 준비 시도 한도는 `prepare_layout`과 함께 쓴다.
        실패하면 중단한다. 이전 계획으로 돌아가지 않는다.
        """
        tool = ToolName.REPLAN_LAYOUT
        failure = self._begin(tool, tool_call_id, ToolState.LAYOUT_READY)
        if failure is not None:
            return failure
        if not self._review or arguments.layout_id != self._ledger.layout_id:
            return self._fail(tool, tool_call_id, ErrorCode.PROTOCOL_ERROR)
        prompt = f"{self._prompt}\n\n{REVIEW_FEEDBACK[arguments.reason]}"
        return self._plan(tool, tool_call_id, prompt)

    def _plan(self, tool: ToolName, tool_call_id: str, prompt: str) -> LayoutResult | ErrorResult:
        """준비 시도 1회를 쓴다. 계획, 렌더, 파싱과 검사를 실행하고 레이아웃을 등록한다."""
        if self._ledger.layout_attempts >= MAX_LAYOUT_ATTEMPTS:
            return self._fail(tool, tool_call_id, ErrorCode.PROTOCOL_ERROR)
        self._ledger = self._ledger.model_copy(
            update={"layout_attempts": self._ledger.layout_attempts + 1}
        )
        self._save_ledger()
        attempt_dir = self.root / "attempts" / f"{self._ledger.layout_attempts:02d}"
        attempt_dir.mkdir(parents=True, exist_ok=False)
        try:
            result = run_roundtrip(
                prompt,
                out_dir=attempt_dir,
                reasoner=self._reasoner,
                parser=GridImageParser(board=self._board),
                renderer=SyntheticRenderer(),
            )
            planned = self._read_layout(attempt_dir / PLAN_LAYOUT_FILE)
            parsed = self._read_layout(attempt_dir / LAYOUT_FILE)
        except ReasonerOutputError as exc:
            code = ErrorCode.REASONER_TRUNCATED if exc.truncated else ErrorCode.LAYOUT_INVALID
            return self._fail(tool, tool_call_id, code)
        except (GridParseError, ValidationError):
            return self._fail(tool, tool_call_id, ErrorCode.LAYOUT_INVALID)
        except OpenAIError:
            return self._fail(tool, tool_call_id, ErrorCode.REASONER_ERROR)
        inspection = inspect_layout(
            planned, parsed, board=self._board, repair_count=len(result.plan_repairs)
        )
        if inspection.error_code is not None:
            return self._fail(
                tool,
                tool_call_id,
                inspection.error_code,
                violations=inspection.violations,
                cell_accuracy=inspection.summary.cell_accuracy,
            )
        record = self._store.register(
            "layout",
            (
                attempt_dir / LAYOUT_FILE,
                attempt_dir / PLAN_LAYOUT_FILE,
                attempt_dir / REPORT_FILE,
                attempt_dir / RAW_IMAGE_FILE,
                attempt_dir / NORMALIZED_IMAGE_FILE,
            ),
        )
        self._ledger = self._ledger.model_copy(update={"layout_id": record.id})
        # 셀은 파싱한 레이아웃을 실행한다. 모델도 같은 레이아웃을 본다.
        summary = (
            review_summary(inspection.summary, layout=parsed)
            if self._review
            else inspection.summary
        )
        response = LayoutResult(
            run_id=self.run_id,
            tool_call_id=tool_call_id,
            artifacts=(ArtifactRef(id=record.id, kind="layout"),),
            summary=summary,
        )
        self._end(response, ToolState.LAYOUT_READY)
        return response

    def calibrate_board(
        self, arguments: CalibrateBoardArguments, *, tool_call_id: str
    ) -> CalibrationResult | ErrorResult:
        """공칭 카메라와 캡처 이미지로 보정한다. 장면 참값을 추정에 쓰지 않는다."""
        tool = ToolName.CALIBRATE_BOARD
        failure = self._begin(tool, tool_call_id, ToolState.LAYOUT_READY)
        if failure is not None:
            return failure
        try:
            self._require_layout(arguments.layout_id)
        except ArtifactError:
            return self._fail(tool, tool_call_id, ErrorCode.PROTOCOL_ERROR)
        cell = self._read_cell()
        capture_dir = self.root / "capture"
        calibration_dir = self.root / "calibration"
        capture_dir.mkdir(exist_ok=False)
        calibration_dir.mkdir(exist_ok=False)
        try:
            capture_task = compile_capture(cell, self._board)
        except TaskCompileError:
            return self._fail(tool, tool_call_id, ErrorCode.COMPILE_FAILED)
        capture_path = capture_dir / CAPTURE_FILE
        write_json(capture_path, capture_task)
        self._ledger = self._ledger.model_copy(update={"capture_consumed": True})
        self._save_ledger()
        try:
            capture = run_capture(
                capture_path,
                out_dir=capture_dir,
                settings=self._sim_settings,
                scene_path=self._scene_path,
            )
            # 카메라 참값은 평가 디렉터리에만 둔다. 추정에는 capture.camera를 전달한다.
            truth_path = self.root / "evaluation" / CAMERA_TRUTH_FILE
            (capture_dir / CAMERA_TRUTH_FILE).replace(truth_path)
            with Image.open(capture.image_path) as image:
                calibration = calibrate_board(image, capture.camera, cell=cell, board=self._board)
                output = calibration_dir / "cell.json"
                write_calibration(calibration, image, output=output, board=self._board)
        except (SimRunError, SimSetupError):
            return self._fail(tool, tool_call_id, ErrorCode.RUNNER_ERROR)
        except (CalibrationError, ValidationError):
            return self._fail(tool, tool_call_id, ErrorCode.CALIBRATION_FAILED)
        record = self._store.register(
            "calibration",
            (
                output,
                calibration_dir / "cell.report.json",
                calibration_dir / "cell.overlay.png",
                capture_path,
                capture_dir / CAPTURE_IMAGE_FILE,
                capture_dir / CAMERA_FILE,
                truth_path,
            ),
            layout_id=arguments.layout_id,
        )
        self._ledger = self._ledger.model_copy(update={"calibration_id": record.id})
        response = CalibrationResult(
            run_id=self.run_id,
            tool_call_id=tool_call_id,
            artifacts=(ArtifactRef(id=record.id, kind="calibration"),),
            summary=summarize_calibration(calibration.report),
        )
        self._end(response, ToolState.CALIBRATED)
        return response

    def execute_cell(
        self, arguments: ExecuteCellArguments, *, tool_call_id: str
    ) -> ExecutionResult | ErrorResult:
        """보정한 셀로 순서를 다시 계산하고 배치 1회와 판정을 실행한다.

        최종 이미지의 카메라 판정을 따로 기록한다. 시점은 도구 결과를 기록한 뒤다.
        처분을 켠 실행은 도구 결과의 앞에서 판독한다. 결과의 요약에 카메라 판정이 있다.
        """
        tool = ToolName.EXECUTE_CELL
        failure = self._begin(tool, tool_call_id, ToolState.CALIBRATED)
        if failure is not None:
            return failure
        try:
            layout = self._require_layout(arguments.layout_id)
            if arguments.calibration_id != self._ledger.calibration_id:
                raise ArtifactError("Calibration is not current.")
            cell_path = self._store.require(
                arguments.calibration_id, "calibration", layout_id=arguments.layout_id
            )
        except ArtifactError:
            return self._fail(tool, tool_call_id, ErrorCode.PROTOCOL_ERROR)
        cell = CellConfig.model_validate_json(cell_path.read_text(encoding="utf-8"))
        execution_dir = self.root / EXECUTION_DIR
        execution_dir.mkdir(exist_ok=False)
        try:
            task = compile_task(layout, cell)
        except TaskCompileError:
            return self._fail(tool, tool_call_id, ErrorCode.COMPILE_FAILED)
        write_json(execution_dir / TASK_FILE, task)
        self._ledger = self._ledger.model_copy(update={"place_consumed": True})
        self._save_ledger()
        return self._place_and_check(
            tool,
            tool_call_id,
            task=task,
            cell_path=cell_path,
            run_dir=execution_dir,
            scene_path=self._scene_path,
        )

    def rework_cell(
        self, arguments: ReworkCellArguments, *, tool_call_id: str
    ) -> ExecutionResult | ErrorResult:
        """카메라 판정이 빈 칸으로 읽은 부품을 다시 놓는다. 작업 전체를 다시 판정한다.

        처분을 켠 실행에서만 쓴다. 실행마다 1회다.
        부품 목록은 카메라 판정의 재작업 대상과 같아야 한다. 다르면 `protocol_error`다.
        다른 부품은 첫 배치의 최종 자세에 둔다. 작업은 첫 배치와 같은 `task.json`이다.
        판정과 카메라 판정을 새로 등록한다. 실패하면 중단한다.
        """
        tool = ToolName.REWORK_CELL
        failure = self._begin(tool, tool_call_id, ToolState.CHECKED)
        if failure is not None:
            return failure
        targets = set(arguments.part_ids)
        if len(targets) != len(arguments.part_ids) or targets != set(self._ledger.rework_targets):
            return self._fail(tool, tool_call_id, ErrorCode.PROTOCOL_ERROR)
        ledger = self._ledger
        assert ledger.layout_id is not None and ledger.calibration_id is not None
        assert ledger.check_id is not None and ledger.image_check_id is not None
        try:
            self._require_layout(ledger.layout_id)
            cell_path = self._store.require(
                ledger.calibration_id, "calibration", layout_id=ledger.layout_id
            )
            # 작업과 첫 배치의 결과는 이 등록의 파일이다. 검사한 뒤에 읽는다.
            self._store.require(ledger.check_id, "check")
            self._store.require(ledger.image_check_id, "image_check")
        except ArtifactError:
            return self._fail(tool, tool_call_id, ErrorCode.PROTOCOL_ERROR)
        execution_dir = self.root / EXECUTION_DIR
        task = Task.model_validate_json((execution_dir / TASK_FILE).read_text(encoding="utf-8"))
        placed = SimResult.model_validate_json(
            (execution_dir / SIM_DIR / RESULT_FILE).read_text(encoding="utf-8")
        )
        scene = SimScene.model_validate_json(self._scene_path.read_text(encoding="utf-8"))
        try:
            rework = rework_scene(
                scene,
                task=task,
                result=placed,
                sequences=[part.sequence for part in task.parts if part.part_id in targets],
            )
        except ReworkError:
            return self._fail(tool, tool_call_id, ErrorCode.RESULT_MISMATCH)
        rework_dir = execution_dir / REWORK_DIR
        rework_dir.mkdir(exist_ok=False)
        # 재작업 장면은 놓인 부품의 참값 자세를 갖는다. 평가 디렉터리에만 둔다.
        scene_path = self.root / "evaluation" / REWORK_SCENE_FILE
        self._persist(scene_path, rework)
        self._ledger = self._ledger.model_copy(update={"rework_consumed": True})
        self._save_ledger()
        return self._place_and_check(
            tool,
            tool_call_id,
            task=task,
            cell_path=cell_path,
            run_dir=rework_dir,
            scene_path=scene_path,
        )

    def _place_and_check(
        self,
        tool: ToolName,
        tool_call_id: str,
        *,
        task: Task,
        cell_path: Path,
        run_dir: Path,
        scene_path: Path,
    ) -> ExecutionResult | ErrorResult:
        """배치 1회를 실행하고 판정한다. 판정을 등록하고 `checked`로 간다.

        작업은 `execution/task.json`이다. 배치 결과, 판정과 카메라 판정은 `run_dir`에 쓴다.
        배치는 발표 영상도 쓴다. 데모 화면이 그 영상을 보여 준다.
        `scene_path`가 재작업 장면이면 그 파일도 판정의 근거로 등록한다.
        """
        task_path = self.root / EXECUTION_DIR / TASK_FILE
        try:
            result = run_place_task(
                task_path,
                out_dir=run_dir / SIM_DIR,
                settings=self._sim_settings,
                scene_path=scene_path,
                present=True,
            )
        except (SimRunError, SimSetupError):
            return self._fail(tool, tool_call_id, ErrorCode.RUNNER_ERROR)
        except ValidationError:
            return self._fail(tool, tool_call_id, ErrorCode.RESULT_MISMATCH)
        try:
            check = check_result(task, result)
        except ResultMismatchError:
            return self._fail(tool, tool_call_id, ErrorCode.RESULT_MISMATCH)
        check_path = run_dir / CHECK_FILE
        write_json(check_path, check)
        evidence = (check_path, task_path, run_dir / SIM_DIR / RESULT_FILE)
        if scene_path != self._scene_path:
            evidence = (*evidence, scene_path)
        record = self._store.register("check", evidence)
        self._ledger = self._ledger.model_copy(
            update={"check_id": record.id, "check_passed": check.passed}
        )
        summary = summarize_execution(check)
        error = None if check.passed else ErrorCode.CHECK_FAILED
        if self._disposition:
            image_check = self._record_image_check(task, cell_path=cell_path, run_dir=run_dir)
            report = None if image_check is None else image_check.report
            # 재작업은 실행마다 1회다. 재작업 뒤에는 대상을 두지 않는다.
            targets = (
                () if report is None or self._ledger.rework_consumed else rework_targets(report)
            )
            self._ledger = self._ledger.model_copy(
                update={
                    "image_check_id": None if image_check is None else image_check.id,
                    "image_check_passed": None if report is None else report.passed,
                    "rework_targets": targets,
                }
            )
            summary = inspected_summary(summary, report=report)
            error = checked_error(check_passed=check.passed, report=report)
        response = ExecutionResult(
            run_id=self.run_id,
            tool_call_id=tool_call_id,
            artifacts=(ArtifactRef(id=record.id, kind="check"),),
            summary=summary,
        )
        self._end(response, ToolState.CHECKED, error=error)
        if not self._disposition:
            self._record_image_check(task, cell_path=cell_path, run_dir=run_dir)
        return response

    def _record_image_check(
        self, task: Task, *, cell_path: Path, run_dir: Path
    ) -> _ImageCheck | None:
        """최종 top-down 이미지를 판독하고 보고서를 `image_check`로 등록한다.

        카메라는 `cell_path` 옆의 보정 보고에 있는 추정 카메라다. 칸은 `task.cell`에 투영한다.
        처분을 끈 실행은 판독을 기록에만 쓴다. 도구 결과, 상태와 완료 조건을 바꾸지 않는다.
        최종 이미지가 없거나 판독이 실패하면 None을 돌려준다.
        """
        image_path = run_dir / SIM_DIR / FINAL_IMAGE_FILE
        if not image_path.is_file():
            return None
        output = run_dir / IMAGE_CHECK_FILE
        try:
            camera = read_report(cell_path.with_name(cell_path.stem + REPORT_SUFFIX)).camera
            with Image.open(image_path) as image:
                report = check_image(task, image, camera)
            write_json(output, report)
            record = self._store.register("image_check", (output, image_path))
        # 판독의 오류는 참값 판정의 결과를 바꾸면 안 된다. 처분을 켠 실행은 보고서 없이 간다.
        # 이 자리를 빠져나간 예외는 `AgentRuntime.run`에서 실행을 오류로 끝낸다.
        # `OSError`는 기록 오류, `ArtifactError`는 프로토콜 오류가 된다.
        # 다른 예외는 실행 보고를 쓰기 전에 빠져나간다. 그래서 예외의 종류를 가리지 않는다.
        except Exception:
            logger.warning(
                "Image check failed. The run continues without the image check record.",
                exc_info=True,
                extra={"run_id": self.run_id},
            )
            return None
        return _ImageCheck(id=record.id, report=report)

    def verified_check_passed(self) -> bool:
        """판정 근거가 바뀌지 않았는지 확인한다. 최종 판단에 쓴다."""
        if self._ledger.state != ToolState.CHECKED or self._ledger.check_id is None:
            raise ArtifactError("Check is unavailable.")
        self._store.require(self._inputs_id, "inputs")
        assert self._ledger.layout_id is not None
        assert self._ledger.calibration_id is not None
        self._require_layout(self._ledger.layout_id)
        self._store.require(
            self._ledger.calibration_id, "calibration", layout_id=self._ledger.layout_id
        )
        self._store.require(self._ledger.check_id, "check")
        return self._ledger.check_passed is True

    def release_allowed(self) -> bool:
        """출하(완료)를 허용하는지 정한다. 최종 판단의 검증과 종료 기록이 함께 쓴다.

        판정 근거가 바뀌지 않았고 참값 판정이 통과해야 한다.
        처분을 켠 실행은 카메라 판정의 통과도 요구한다. 보고서가 없으면 허용하지 않는다.
        """
        passed = self.verified_check_passed()
        if not self._disposition:
            return passed
        if self._ledger.image_check_id is None:
            return False
        self._store.require(self._ledger.image_check_id, "image_check")
        return passed and self._ledger.image_check_passed is True

    def _begin(self, tool: ToolName, call_id: str, expected: ToolState) -> ErrorResult | None:
        if self._recording_failed:
            raise OSError("Tool record is unavailable.")
        TypeAdapter(CallId).validate_python(call_id)
        if (
            self._ledger.state != expected
            or self._ledger.in_flight is not None
            or call_id in self._ledger.call_ids
            or (tool == ToolName.CALIBRATE_BOARD and self._ledger.capture_consumed)
            or (tool == ToolName.EXECUTE_CELL and self._ledger.place_consumed)
            or (
                tool == ToolName.REWORK_CELL
                and (not self._disposition or self._ledger.rework_consumed)
            )
        ):
            return self._fail(tool, call_id, ErrorCode.PROTOCOL_ERROR)
        try:
            self._store.require(self._inputs_id, "inputs")
        except ArtifactError:
            return self._fail(tool, call_id, ErrorCode.PROTOCOL_ERROR)
        self._ledger = self._ledger.model_copy(
            update={"in_flight": tool, "call_ids": (*self._ledger.call_ids, call_id)}
        )
        self._save_ledger()
        return None

    def _fail(
        self,
        tool: ToolName,
        call_id: str,
        code: ErrorCode,
        *,
        violations: tuple[IssueCode, ...] = (),
        cell_accuracy: float | None = None,
    ) -> ErrorResult:
        recoverable = (
            tool == ToolName.PREPARE_LAYOUT
            and code in (ErrorCode.LAYOUT_INVALID, ErrorCode.ROUNDTRIP_MISMATCH)
            and self._ledger.layout_attempts == 1
        )
        result = ErrorResult(
            run_id=self.run_id,
            tool_call_id=call_id,
            error_code=code,
            recoverable=recoverable,
            summary=ErrorSummary(
                tool=tool,
                layout_attempt=self._ledger.layout_attempts,
                violations=violations,
                cell_accuracy=cell_accuracy,
            ),
        )
        state = ToolState.LAYOUT_RETRYABLE if recoverable else ToolState.ABORTED
        self._end(result, state, error=code)
        return result

    def _end(self, result: ToolResult, state: ToolState, *, error: ErrorCode | None = None) -> None:
        count = self._ledger.result_count + 1
        self._persist(self.root / "tool_results" / f"{count:02d}.json", result)
        self._ledger = self._ledger.model_copy(
            update={"state": state, "error_code": error, "in_flight": None, "result_count": count}
        )
        self._save_ledger()

    def _persist(self, path: Path, model: BaseModel) -> None:
        try:
            write_record(path, model)
        except OSError:
            self._recording_failed = True
            raise

    def _save_ledger(self) -> None:
        self._persist(self.root / "tool_state.json", self._ledger)

    def _require_layout(self, layout_id: str) -> Layout:
        if layout_id != self._ledger.layout_id:
            raise ArtifactError("Layout is not current.")
        return self._read_layout(self._store.require(layout_id, "layout"))

    def _read_cell(self) -> CellConfig:
        path = self._store.require(self._inputs_id, "inputs")
        return CellConfig.model_validate_json(path.read_text(encoding="utf-8"))

    @staticmethod
    def _read_layout(path: Path) -> Layout:
        return Layout.model_validate_json(path.read_text(encoding="utf-8"))
