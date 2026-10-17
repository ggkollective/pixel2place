"""CLI 작업 프로세스와 산출물 읽기 경계.

단일 실행, 입력 저장, 프로세스 중단과 이력 수명을 소유한다.
셀의 예약도 소유한다. 예약 중에는 예약한 쪽의 실행만 시작한다.
HTTP 요청과 모델·Isaac의 단계 실행은 소유하지 않는다.
주문의 큐와 실행 순서는 소유하지 않는다(`pixel2place.demo.shifts`).
"""

import fcntl
import json
import os
import re
import shutil
import signal
import subprocess
import sys
from collections.abc import Sequence
from contextlib import suppress
from datetime import UTC, datetime
from pathlib import Path
from threading import RLock
from time import monotonic, sleep
from uuid import uuid4

from pydantic import ValidationError

from pixel2place.agent.artifacts import write_record
from pixel2place.agent.contracts import ToolName, ToolState
from pixel2place.agent.fixture import FixtureOffset, fixture_scene
from pixel2place.agent.intake import INTAKE_FILE
from pixel2place.agent.runtime_contracts import AgentEvent, AgentReport
from pixel2place.agent.shift import ShiftItemReport, ShiftOrder, item_report
from pixel2place.agent.tools import EXECUTION_DIR, REWORK_DIR
from pixel2place.cell.check import CheckReport
from pixel2place.cell.config import CellConfig
from pixel2place.cell.image_check import ImageCheckReport
from pixel2place.demo.contracts import (
    AssetName,
    JobRecord,
    ReferenceView,
    RunRequest,
    RunSummary,
    RunView,
    ScenePreset,
)
from pixel2place.demo.intake import IntakeDesk, OrderReader, read_with_token_factory
from pixel2place.demo.orders import match_order
from pixel2place.layout.schema import Layout
from pixel2place.order import order_instruction, parse_order
from pixel2place.pipeline import CHECK_FILE, IMAGE_CHECK_FILE

SHUTDOWN_GRACE_S = 5.0
TERMINATE_GRACE_S = 2.0
MAX_HISTORY = 30
_ID = re.compile(r"^[a-f0-9]{32}$")
# 화면 자료의 후보 파일. 앞의 후보부터 찾는다.
# 발표 영상이 없는 실행은 사선 카메라 영상을 준다.
_ASSETS: dict[AssetName, tuple[str, ...]] = {
    "capture": ("capture/topdown_start.png",),
    "calibration": ("calibration/cell.overlay.png",),
    "video": ("execution/sim/present.mp4", "execution/sim/cell.mp4"),
    "result": ("execution/sim/topdown_final.png",),
    "rework_video": ("execution/rework/sim/present.mp4", "execution/rework/sim/cell.mp4"),
    "rework_result": ("execution/rework/sim/topdown_final.png",),
}
# 영상은 실행이 끝난 뒤에 준다. 실행기가 쓰는 중인 파일은 주지 않는다.
VIDEO_ASSETS: frozenset[AssetName] = frozenset({"video", "rework_video"})
_REFERENCE = {
    "before": "look/nocal/examples/003_repetitive_panel_pocket/r0",
    "after": "look/cal/examples/003_repetitive_panel_pocket/r0",
}
# 오차 값이 없는 기록은 장면 프리셋만 저장했다. 그 프리셋의 오차다.
_PRESET_FIXTURES: dict[ScenePreset, FixtureOffset] = {
    "nominal": FixtureOffset(),
    "board_offset": FixtureOffset(x_mm=3.0, y_mm=5.0, yaw_deg=1.5),
}


def layout_attempts(events: Sequence[AgentEvent]) -> int:
    """끝난 준비 시도의 수. 첫 준비, 형식 오류 재시도와 검토의 재계획을 함께 센다.

    가장 큰 번호의 시도 디렉터리에 현재 레이아웃이 있다.
    """
    planning = (ToolName.PREPARE_LAYOUT, ToolName.REPLAN_LAYOUT)
    return sum(
        event.kind == "tool_result" and event.call is not None and event.call.name in planning
        for event in events
    )


def read_check(path: Path) -> CheckReport:
    """저장한 계산 필드를 다시 계산한다. 원본 판정과 다르면 거부한다."""
    saved = json.loads(path.read_text())
    if not isinstance(saved, dict) or not isinstance(saved.get("parts"), list):
        raise ValueError("Saved check fields are not valid.")
    if any(not isinstance(part, dict) for part in saved["parts"]):
        raise ValueError("Saved part checks are not valid.")
    passed = saved.pop("passed")
    passed_count = saved.pop("passed_count")
    part_passed = [part.pop("passed") for part in saved["parts"]]
    check = CheckReport.model_validate_json(json.dumps(saved))
    if (
        type(passed) is not bool
        or type(passed_count) is not int
        or any(type(value) is not bool for value in part_passed)
        or passed != check.passed
        or passed_count != check.passed_count
        or part_passed != [part.passed for part in check.parts]
    ):
        raise ValueError("Saved check fields do not match the part checks.")
    return check


def read_image_check(path: Path) -> ImageCheckReport | None:
    """저장한 카메라 판정을 읽는다. 파일이 없거나 형식이 틀리면 None이다.

    카메라 판정은 화면 표시에만 쓴다. 이 파일의 오류는 실행 조회를 실패로 만들지 않는다.
    """
    if not path.is_file():
        return None
    try:
        return ImageCheckReport.model_validate_json(path.read_bytes())
    except ValidationError:
        return None


def _verdict_directories(execution: Path) -> tuple[Path, Path]:
    """첫 배치의 판정 디렉터리와 마지막 배치의 판정 디렉터리.

    재작업의 판정이 있으면 마지막은 재작업 디렉터리다. 없으면 둘은 같다.
    """
    rework = execution / REWORK_DIR
    return execution, rework if (rework / CHECK_FILE).is_file() else execution


def _read_verdict(directory: Path) -> CheckReport | None:
    path = directory / CHECK_FILE
    return read_check(path) if path.is_file() else None


def _asset_file(root: Path, name: AssetName) -> Path | None:
    """화면 자료의 파일. 내용이 있는 첫 후보다. 그런 후보가 없으면 None이다."""
    for relative in _ASSETS.get(name, ()):
        path = root / relative
        if path.is_file() and path.stat().st_size:
            return path
    return None


class RunBusyError(ValueError):
    """한 실행 또는 한 서버만 실행 루트를 소유할 수 있다."""


class JobManager:
    """프로세스는 한 번 시작한다. 서버 종료 때 프로세스 그룹을 정리한다."""

    def __init__(
        self, *, root: Path, workspace: Path, read_order: OrderReader = read_with_token_factory
    ) -> None:
        root.mkdir(parents=True, exist_ok=True)
        self.root = root.resolve()
        self.workspace = workspace.resolve(strict=True)
        # 디렉터리 이름이 실행 ID의 형식이 아니다. 실행 기록의 복원이 건너뛴다.
        self.intakes = IntakeDesk(root=self.root / "intakes", read=read_order)
        self._lock = RLock()
        self._file_lock = (self.root / ".server.lock").open("a")
        self._records: dict[str, JobRecord] = {}
        self._processes: dict[str, subprocess.Popen[bytes]] = {}
        self._closed = False
        self._reserved = False
        try:
            fcntl.flock(self._file_lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
            self._restore()
        except BlockingIOError:
            self._file_lock.close()
            raise RunBusyError("Another demo server owns the run directory.") from None
        except (OSError, ValueError):
            self._file_lock.close()
            raise

    def _restore(self) -> None:
        for directory in self.root.iterdir():
            if not _ID.fullmatch(directory.name) or not directory.is_dir():
                continue
            record = JobRecord.model_validate_json((directory / "job.json").read_text())
            if record.id != directory.name:
                raise ValueError("Run identity does not match its directory.")
            if record.status in ("running", "cancelling"):
                if record.worker_pid is not None:
                    try:
                        os.kill(record.worker_pid, 0)
                    except ProcessLookupError:
                        pass
                    else:
                        raise RunBusyError("An interrupted worker is still stopping.")
                record = record.model_copy(
                    update={"status": "interrupted", "finished_at": datetime.now(UTC)}
                )
                write_record(directory / "job.json", record)
            self._records[record.id] = record

    def _directory(self, run_id: str) -> Path:
        if run_id not in self._records:
            raise KeyError(run_id)
        directory = (self.root / run_id).resolve(strict=True)
        if directory.parent != self.root:
            raise ValueError("Run directory is outside the run root.")
        return directory

    def _save(self, record: JobRecord) -> None:
        write_record(self.root / record.id / "job.json", record)
        self._records[record.id] = record

    def _require_idle(self) -> None:
        """실행 중인 실행이 있거나 서버가 멈추는 중이면 거부한다. 잠금 안에서 부른다."""
        if self._closed:
            raise RunBusyError("The demo server is stopping.")
        for run_id in self._records:
            self._refresh(run_id)
        if any(record.status in ("running", "cancelling") for record in self._records.values()):
            raise RunBusyError("An assembly run is already active.")

    def reserve(self) -> None:
        """셀을 주문의 큐 1개에 예약한다. 예약 중에는 `queued` 실행만 시작한다.

        Raises:
            RunBusyError: 실행 중인 실행 또는 다른 예약이 있을 때. 서버가 멈추는 중일 때.
        """
        with self._lock:
            self._require_idle()
            if self._reserved:
                raise RunBusyError("A shift already has the cell.")
            self._reserved = True

    def release(self) -> None:
        """셀의 예약을 푼다. 실행 중인 실행은 바꾸지 않는다."""
        with self._lock:
            self._reserved = False

    def start(self, request: RunRequest, *, queued: bool = False) -> RunView:
        """지시문, 주문과 장면을 저장한다. 셸 없이 기존 CLI를 실행한다.

        주문은 지시문으로 바꾼 뒤에 실행한다. CLI는 지시문 원문만 받는다.
        접수로 읽은 주문은 그 접수의 기록을 실행 디렉터리에 복사한다.
        부품의 결함은 장면에만 쓴다. CLI와 모델은 결함을 받지 않는다.
        `queued`는 셀을 예약한 큐의 실행이다. 예약 중에는 다른 실행을 거부한다.

        Raises:
            OrderNotFeasibleError: 주문이 배치 규칙을 어길 때. 실행 기록을 만들지 않는다.
            IntakeMismatchError: 주문이 수락한 접수의 주문이 아닐 때. 실행 기록을 만들지 않는다.
            RunBusyError: 실행 중인 실행이 있거나 다른 쪽이 셀을 예약했을 때.
        """
        order = request.order
        prompt = order_instruction(order) if order is not None else request.prompt
        if prompt is None:
            raise ValueError("The run request has no goal.")
        intake = (
            self.intakes.accepted_record(request.intake_id, order=order)
            if request.intake_id is not None and order is not None
            else None
        )
        fixture = request.fixture
        # 빈 결함은 결함이 없는 실행이다. 장면과 기록에 쓰지 않는다.
        faults = request.faults
        if faults is not None and not faults.sequences:
            faults = None
        with self._lock:
            self._require_idle()
            if self._reserved and not queued:
                raise RunBusyError("A shift has the cell.")
            run_id = uuid4().hex
            directory = self.root / run_id
            directory.mkdir(exist_ok=False)
            write_record(directory / "request.json", RunRequest(prompt=prompt, fixture=fixture))
            if order is not None:
                write_record(directory / "order.json", order)
            if intake is not None:
                shutil.copyfile(intake, directory / INTAKE_FILE)
            write_record(
                directory / "scene.json", fixture_scene(CellConfig(), fixture, faults=faults)
            )
            record = JobRecord(
                id=run_id,
                prompt=prompt,
                scene="nominal" if fixture.is_nominal else "board_offset",
                status="running",
                created_at=datetime.now(UTC),
                fixture=fixture,
                ordered_parts=order.part_count if order is not None else None,
                order_kind=order.kind if order is not None else None,
                intake_id=request.intake_id,
                faults=faults,
            )
            self._save(record)
            try:
                with (directory / "worker.log").open("wb") as log:
                    process = subprocess.Popen(
                        [
                            sys.executable,
                            "-m",
                            "pixel2place.demo.worker",
                            "--parent",
                            str(os.getpid()),
                            "--job",
                            str(directory),
                        ],
                        cwd=self.workspace,
                        stdout=log,
                        stderr=subprocess.STDOUT,
                        start_new_session=True,
                    )
            except OSError:
                self._save(
                    record.model_copy(
                        update={"status": "aborted", "finished_at": datetime.now(UTC)}
                    )
                )
                raise
            self._processes[run_id] = process
            try:
                self._save(record.model_copy(update={"worker_pid": process.pid}))
            except OSError:
                self._stop(process)
                raise
            return self.get(run_id)

    def _refresh(self, run_id: str) -> None:
        process = self._processes.get(run_id)
        if process is None or process.poll() is None:
            return
        exit_code = process.returncode
        self._stop(process)
        record = self._records[run_id]
        report_path = self._directory(run_id) / "run" / "agent_report.json"
        status = "aborted"
        disposition = None
        if record.status == "cancelling":
            status = "cancelled"
        elif report_path.is_file():
            report = AgentReport.model_validate_json(report_path.read_text())
            disposition = report.disposition
            if exit_code == 0 and report.state == ToolState.COMPLETED:
                status = "completed"
        self._save(
            record.model_copy(
                update={
                    "status": status,
                    "finished_at": datetime.now(UTC),
                    "exit_code": exit_code,
                    "disposition": disposition,
                }
            )
        )
        del self._processes[run_id]

    def list(self) -> tuple[RunSummary, ...]:
        """최근 실행을 반환한다. 모델이나 프로세스를 다시 실행하지 않는다."""
        with self._lock:
            for run_id in self._records:
                self._refresh(run_id)
            records = sorted(self._records.values(), key=lambda item: item.created_at, reverse=True)
            return tuple(self._summary(item) for item in records[:MAX_HISTORY])

    @staticmethod
    def _summary(record: JobRecord) -> RunSummary:
        fields = record.model_dump(exclude={"worker_pid", "exit_code"})
        if record.fixture is None:
            fields["fixture"] = _PRESET_FIXTURES[record.scene].model_dump()
        if record.ordered_parts is not None and record.order_kind is None:
            fields["order_kind"] = "board"
        return RunSummary.model_validate(fields, strict=True)

    def summary(self, run_id: str) -> RunSummary:
        """실행 1건의 입력과 프로세스 상태. 이벤트와 산출물은 읽지 않는다."""
        with self._lock:
            self._directory(run_id)
            self._refresh(run_id)
            return self._summary(self._records[run_id])

    def order_report(self, run_id: str, *, name: str) -> ShiftItemReport | None:
        """끝난 주문 실행의 지표. 계산은 `agent-shift`의 교대 보고서와 같다.

        wall 시간은 실행 기록의 시작부터 끝까지다.
        끝나지 않은 실행, 주문이 없는 실행과 실행 보고서가 없는 실행은 None이다.

        Args:
            run_id: 실행 ID. 지표의 `directory`에도 쓴다.
            name: 지표에 쓰는 주문의 이름.
        """
        with self._lock:
            directory = self._directory(run_id)
            self._refresh(run_id)
            record = self._records[run_id]
            order_path = directory / "order.json"
            report_path = directory / "run" / "agent_report.json"
            if record.finished_at is None or not order_path.is_file() or not report_path.is_file():
                return None
            planned = ShiftOrder(
                name=name,
                order=parse_order(order_path.read_text()),
                fixture=self._summary(record).fixture,
                prompt=record.prompt,
                faults=record.faults,
            )
            return item_report(
                planned,
                directory=run_id,
                root=directory / "run",
                report=AgentReport.model_validate_json(report_path.read_text()),
                wall_time_s=(record.finished_at - record.created_at).total_seconds(),
            )

    def get(self, run_id: str) -> RunView:
        """완성한 이벤트와 화면용 자료를 읽는다. 원시 로그와 인증값은 제외한다."""
        with self._lock:
            directory = self._directory(run_id)
            self._refresh(run_id)
            record = self._records[run_id]
            root = directory / "run"
            events_path = root / "events.jsonl"
            events: tuple[AgentEvent, ...] = ()
            if events_path.is_file():
                text = events_path.read_text()
                lines = text.splitlines() if text.endswith("\n") else text.splitlines()[:-1]
                events = tuple(AgentEvent.model_validate_json(line) for line in lines)
            report_path = root / "agent_report.json"
            report = (
                AgentReport.model_validate_json(report_path.read_text())
                if report_path.is_file()
                else None
            )
            prepared = [event for event in events if event.to_state == ToolState.LAYOUT_READY]
            layout_path = root / "attempts" / f"{layout_attempts(events):02d}" / "layout.json"
            layout = (
                Layout.model_validate_json(layout_path.read_text())
                if prepared and layout_path.is_file()
                else None
            )
            first, last = _verdict_directories(root / EXECUTION_DIR)
            check = _read_verdict(last) if report is not None else None
            image_check = read_image_check(last / IMAGE_CHECK_FILE) if report is not None else None
            reworked = report is not None and first != last
            order_path = directory / "order.json"
            order = parse_order(order_path.read_text()) if order_path.is_file() else None
            available: list[AssetName] = []
            if layout is not None:
                available.append("layout")
            for name in _ASSETS:
                if _asset_file(root, name) is not None and (
                    name not in VIDEO_ASSETS or record.status not in ("running", "cancelling")
                ):
                    available.append(name)
            return RunView(
                **self._summary(record).model_dump(),
                runtime_state=report.state
                if report
                else events[-1].to_state
                if events
                else ToolState.CREATED,
                error_code="user_cancelled"
                if record.status == "cancelled"
                else report.error_code
                if report
                else (
                    "server_interrupted"
                    if record.status == "interrupted"
                    else "cli_failed"
                    if record.status == "aborted"
                    else None
                ),
                events=events,
                report=report,
                layout=layout,
                check=check,
                image_check=image_check,
                first_check=_read_verdict(first) if reworked else None,
                first_image_check=read_image_check(first / IMAGE_CHECK_FILE) if reworked else None,
                assets=tuple(available),
                order=order,
                order_match=match_order(order, planned=layout)
                if order is not None and layout is not None
                else None,
            )

    def asset(self, run_id: str, name: AssetName) -> Path:
        """등록한 화면 자료만 반환한다. symlink가 실행 경계를 벗어나면 거부한다."""
        with self._lock:
            view = self.get(run_id)
            if name not in view.assets:
                raise KeyError(name)
            root = self._directory(run_id) / "run"
            if name == "layout":
                attempt = layout_attempts(view.events)
                path = root / "attempts" / f"{attempt:02d}" / "normalized_image.png"
            else:
                path = _asset_file(root, name)
                if path is None:
                    raise KeyError(name)
            path = path.resolve(strict=True)
            if not path.is_relative_to(root.resolve()) or not path.is_file():
                raise KeyError(name)
            return path

    def _reference_root(self, variant: str) -> Path:
        if variant not in _REFERENCE:
            raise KeyError(variant)
        base = self.workspace / ".local-artifacts/measurements/board-calibration"
        root = (base / _REFERENCE[variant]).resolve(strict=True)
        if not root.is_relative_to(base.resolve()):
            raise KeyError(variant)
        return root

    def reference(self, variant: str) -> ReferenceView:
        """기존 보정 측정은 새 작업 실행과 합치지 않는다."""
        root = self._reference_root(variant)
        return ReferenceView(
            variant=variant,  # pyright: ignore[reportArgumentType]
            layout=Layout.model_validate_json((root.parent / "layout.json").read_text()),
            check=read_check(root / "check.json"),
            description="Recorded calibration benchmark. The after run also has camera offsets.",
        )

    def reference_asset(self, *, variant: str, name: str) -> Path:
        root = self._reference_root(variant)
        relative = {"video": "sim/cell.mp4", "result": "sim/topdown_final.png"}.get(name)
        if relative is None:
            raise KeyError(name)
        path = (root / relative).resolve(strict=True)
        if not path.is_relative_to(root) or not path.is_file():
            raise KeyError(name)
        return path

    @staticmethod
    def _stop(process: subprocess.Popen[bytes]) -> None:
        for action, timeout in (
            (signal.SIGINT, SHUTDOWN_GRACE_S),
            (signal.SIGTERM, TERMINATE_GRACE_S),
            (signal.SIGKILL, TERMINATE_GRACE_S),
        ):
            with suppress(ProcessLookupError):
                os.killpg(process.pid, action)
            deadline = monotonic() + timeout
            while monotonic() < deadline:
                process.poll()
                try:
                    os.killpg(process.pid, 0)
                except ProcessLookupError:
                    process.wait(timeout=TERMINATE_GRACE_S)
                    return
                sleep(0.1)
        try:
            process.wait(timeout=TERMINATE_GRACE_S)
        except subprocess.TimeoutExpired:
            raise OSError("The assembly worker did not stop.") from None

    def cancel(self, run_id: str) -> RunView:
        """중단은 전체 그룹에 전달한다. 이미 끝난 실행은 바꾸지 않는다."""
        with self._lock:
            self._directory(run_id)
            self._refresh(run_id)
            process = self._processes.get(run_id)
            if process is not None:
                try:
                    self._save(self._records[run_id].model_copy(update={"status": "cancelling"}))
                finally:
                    self._stop(process)
                self._refresh(run_id)
            return self.get(run_id)

    def close(self) -> None:
        """작업을 중단한 뒤 서버의 실행 루트 잠금을 해제한다."""
        with self._lock:
            if self._closed:
                return
            self._closed = True
            try:
                for run_id in tuple(self._processes):
                    self.cancel(run_id)
            finally:
                try:
                    for process in self._processes.values():
                        self._stop(process)
                finally:
                    self._file_lock.close()
