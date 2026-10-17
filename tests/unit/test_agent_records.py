"""저장한 실행 기록을 현재 계약 모델로 읽는다.

`tests/data/agent_v1/`은 schema_version 1의 실행 프로그램이 Fake로 쓴 기록이다. 다시 만들지 않는다.
계약을 바꾼 뒤에도 이 기록이 읽혀야 한다. 데모 서버는 같은 모델로 기존 실행을 읽는다.

- `completed`, `retry_check_failed`: 계획 검토와 처분이 없는 계약의 실행.
- `rework_completed`: 계획 검토와 처분을 켠 실행. 공급되지 않은 부품 1개를 재작업으로 놓았다.
"""

from pathlib import Path

import pytest
from pydantic import TypeAdapter

from pixel2place.agent.artifacts import Manifest
from pixel2place.agent.contracts import (
    ErrorCode,
    ErrorResult,
    ExecutionResult,
    ExecutionSummary,
    InspectedExecutionSummary,
    LayoutResult,
    LayoutSummary,
    ToolResult,
    ToolState,
)
from pixel2place.agent.prompts import SYSTEM_PROMPT
from pixel2place.agent.runtime_contracts import AgentEvent, AgentReport, Disposition
from pixel2place.agent.tools import ToolLedger, ToolRun

pytestmark = pytest.mark.unit
RECORDS = Path(__file__).parents[1] / "data" / "agent_v1"
OLD_RUNS = ("completed", "retry_check_failed")
RUNS = (*OLD_RUNS, "rework_completed")


def _events(run: str) -> list[AgentEvent]:
    lines = (RECORDS / run / "events.jsonl").read_text(encoding="utf-8").splitlines()
    return [AgentEvent.model_validate_json(line) for line in lines]


class TestStoredRecords:
    """모든 기록 파일의 형식과 두 실행의 종료 상태를 검사한다."""

    @pytest.mark.parametrize("run", RUNS)
    def test_every_record_file_is_valid(self, run: str) -> None:
        """입력, 장부, manifest, 이벤트, 도구 결과와 보고가 strict 모델을 통과한다."""
        root = RECORDS / run
        inputs = ToolRun.model_validate_json((root / "run.json").read_text(encoding="utf-8"))
        ledger = ToolLedger.model_validate_json(
            (root / "tool_state.json").read_text(encoding="utf-8")
        )
        manifest = Manifest.model_validate_json(
            (root / "manifest.json").read_text(encoding="utf-8")
        )
        report = AgentReport.model_validate_json(
            (root / "agent_report.json").read_text(encoding="utf-8")
        )
        events = _events(run)
        results = [
            TypeAdapter(ToolResult).validate_json(path.read_text(encoding="utf-8"))
            for path in sorted((root / "tool_results").glob("*.json"))
        ]
        run_ids = {inputs.run_id, ledger.run_id, manifest.run_id, report.run_id}
        run_ids |= {event.run_id for event in events} | {result.run_id for result in results}
        assert len(run_ids) == 1, f"The records have more than one run ID: {run_ids}"
        assert [event.sequence for event in events] == list(range(1, len(events) + 1))
        assert results == [event.result for event in events if event.kind == "tool_result"]
        assert ledger.state == report.state == events[-1].to_state

    @pytest.mark.parametrize("run", OLD_RUNS)
    def test_records_are_review_off_runs_with_the_current_prompt(self, run: str) -> None:
        """기록의 지시문은 검토를 끈 지시문과 같다. 준비 결과에는 부품 목록이 없다."""
        inputs = ToolRun.model_validate_json(
            (RECORDS / run / "run.json").read_text(encoding="utf-8")
        )
        assert inputs.agent is not None and inputs.agent.review is False
        assert inputs.agent.system_prompt == SYSTEM_PROMPT
        layouts = [e.result for e in _events(run) if isinstance(e.result, LayoutResult)]
        assert layouts and all(type(result.summary) is LayoutSummary for result in layouts)

    def test_completed_run_has_four_inferences(self) -> None:
        """정상 실행은 도구 3개와 완료 판단을 기록한다."""
        root = RECORDS / "completed"
        report = AgentReport.model_validate_json((root / "agent_report.json").read_text())
        assert report.state == ToolState.COMPLETED and report.error_code is None
        assert report.calls.nemotron == 4 and report.check_passed is True
        calls = [e.call.name for e in _events("completed") if e.kind == "tool_result" and e.call]
        assert calls == ["prepare_layout", "calibrate_board", "execute_cell"]

    @pytest.mark.parametrize(
        ("run", "disposition"),
        [("completed", Disposition.RELEASED), ("retry_check_failed", Disposition.REJECTED)],
    )
    def test_records_without_disposition_fields_read_as_disposition_off(
        self, run: str, disposition: Disposition
    ) -> None:
        """처분의 필드가 없는 기록은 처분을 끈 실행으로 읽힌다. 재작업은 없다."""
        root = RECORDS / run
        inputs = ToolRun.model_validate_json((root / "run.json").read_text(encoding="utf-8"))
        assert inputs.agent is not None and inputs.agent.disposition is False
        assert inputs.agent.budget.max_place_runs == 1
        ledger = ToolLedger.model_validate_json(
            (root / "tool_state.json").read_text(encoding="utf-8")
        )
        assert not ledger.rework_consumed and ledger.image_check_id is None
        assert ledger.rework_targets == ()
        report = AgentReport.model_validate_json((root / "agent_report.json").read_text())
        assert report.reworked is False and report.disposition == disposition
        executions = [e.result for e in _events(run) if isinstance(e.result, ExecutionResult)]
        assert executions and all(type(result.summary) is ExecutionSummary for result in executions)

    def test_retry_run_keeps_the_error_result_and_the_failed_check(self) -> None:
        """재계획 실행은 복구 가능한 오류 결과와 판정 실패를 기록한다."""
        root = RECORDS / "retry_check_failed"
        report = AgentReport.model_validate_json((root / "agent_report.json").read_text())
        assert report.state == ToolState.ABORTED and report.error_code == ErrorCode.CHECK_FAILED
        assert report.calls.nemotron == 5 and report.check_passed is False
        results = [e.result for e in _events("retry_check_failed") if e.kind == "tool_result"]
        assert isinstance(results[0], ErrorResult) and results[0].recoverable
        states = [e.to_state for e in _events("retry_check_failed") if e.kind == "tool_result"]
        assert states == [
            ToolState.LAYOUT_RETRYABLE,
            ToolState.LAYOUT_READY,
            ToolState.CALIBRATED,
            ToolState.CHECKED,
        ]

    def test_rework_run_keeps_both_checks_and_the_rework_call(self) -> None:
        """재작업 실행의 기록은 실패한 첫 판정, 재작업 호출과 통과한 두 번째 판정을 갖는다."""
        root = RECORDS / "rework_completed"
        inputs = ToolRun.model_validate_json((root / "run.json").read_text(encoding="utf-8"))
        assert inputs.agent is not None and inputs.agent.review and inputs.agent.disposition
        assert inputs.agent.budget.max_place_runs == 2
        report = AgentReport.model_validate_json((root / "agent_report.json").read_text())
        assert report.state == ToolState.COMPLETED and report.reworked
        assert report.disposition == Disposition.REWORKED_RELEASED
        assert report.calls.nemotron == 5 and report.calls.place == 2
        events = _events("rework_completed")
        calls = [e.call.name for e in events if e.kind == "tool_result" and e.call]
        assert calls == ["prepare_layout", "calibrate_board", "execute_cell", "rework_cell"]
        first, second = (e.result for e in events if isinstance(e.result, ExecutionResult))
        assert isinstance(first.summary, InspectedExecutionSummary)
        assert isinstance(second.summary, InspectedExecutionSummary)
        assert first.summary.inspection is not None and second.summary.inspection is not None
        assert not first.summary.passed and not first.summary.inspection.passed
        assert [(part.status, part.empty) for part in first.summary.inspection.parts] == [
            ("mismatch", True)
        ]
        assert second.summary.passed and second.summary.inspection.passed
        assert second.artifacts[0].id == report.check_id != first.artifacts[0].id
        ledger = ToolLedger.model_validate_json(
            (root / "tool_state.json").read_text(encoding="utf-8")
        )
        assert ledger.rework_consumed and ledger.image_check_passed is True
        manifest = Manifest.model_validate_json(
            (root / "manifest.json").read_text(encoding="utf-8")
        )
        kinds = [record.kind for record in manifest.artifacts]
        assert kinds.count("check") == kinds.count("image_check") == 2
