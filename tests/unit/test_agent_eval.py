"""오프라인 평가의 요청, 채점, 기준 판정과 CLI를 HTTP Fake로 검사한다."""

import hashlib
import json
from collections import Counter
from collections.abc import Iterator, Sequence
from pathlib import Path
from typing import Any

import pytest
from pydantic import ValidationError
from typer.testing import CliRunner

from pixel2place.agent.audit import AUDIT_FILE
from pixel2place.agent.contracts import (
    CalibrationResult,
    ErrorResult,
    ExecutionResult,
    InspectedExecutionSummary,
    LayoutResult,
    ReviewedLayoutSummary,
)
from pixel2place.agent.eval import (
    BYTES_PER_TOKEN,
    CASES_FILE,
    CONSISTENCY_CASES,
    EVAL_FILE,
    EXPECTED_OUTPUT_TOKENS,
    MAX_CONSECUTIVE_FAILURES,
    PRICES_USD,
    REPORT_FILE,
    RESULTS_FILE,
    EvalConfig,
    EvalManifest,
    EvalReport,
    EvalResult,
    PreparedEval,
    ResponseStatus,
    prepare_eval,
    run_eval,
)
from pixel2place.agent.eval_cases import (
    DispositionCase,
    DispositionKind,
    EvalCase,
    Feature,
    IntakeCase,
    IntakeGroup,
    ReviewAction,
    ReviewCase,
    ReviewGroup,
    SavedPair,
    build_cases,
    select_cases,
)
from pixel2place.agent.intake import INTAKE_SYSTEM_PROMPT, RejectReason
from pixel2place.agent.prompts import REVIEW_SYSTEM_PROMPT, system_prompt
from pixel2place.cli import app
from pixel2place.config import DEFAULT_NEMOTRON_MODEL, TokenFactorySettings
from pixel2place.layout.schema import Color, PartType
from pixel2place.order import Arrangement, BoardOrder, KitItem, KitOrder
from pixel2place.reasoner import nemotron
from tests.factories import make_layout, make_part
from tests.fakes.fake_judgment_model import (
    FAKE_JUDGMENT_KEY,
    FAKE_JUDGMENT_REASONING,
    FakeJudgmentModel,
    Reply,
    Responder,
    coordination_reply,
    correct_replies,
    review_reply,
    runtime_state,
)

pytestmark = pytest.mark.unit
runner = CliRunner()
ULTRA = "nvidia/Nemotron-3-Ultra-550b-a55b"
DISPOSITION_PROMPT = system_prompt(review=True, disposition=True)
COMPLETE = Reply(content='{"decision":"complete","reason_code":"check_passed"}')
KIT = KitOrder(
    kind="kit",
    items=(KitItem(type=PartType.SQUARE_TILE, count=2), KitItem(type=PartType.RECT_TILE, count=1)),
    arrangement=Arrangement.TYPE_ROWS,
)
KIT_LAYOUT = make_layout(
    make_part("P001", PartType.SQUARE_TILE, origin=(1, 1)),
    make_part("P002", PartType.SQUARE_TILE, origin=(1, 4)),
    make_part("P003", PartType.RECT_TILE, origin=(5, 2), color=Color.GRAY),
)
FREE_TEXT = Reply(content="The plan is correct.")


@pytest.fixture
def api(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Iterator[FakeJudgmentModel]:
    """SDK의 HTTP 클라이언트만 대체한다. 환경은 테스트 값으로 고정한다."""
    fake = FakeJudgmentModel()
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("NEBIUS_API_KEY", FAKE_JUDGMENT_KEY)
    monkeypatch.setenv("NEBIUS_BASE_URL", "https://api.tokenfactory.nebius.com/v1/")
    monkeypatch.setattr(nemotron, "DefaultHttpxClient", fake.make_client)
    yield fake
    for client in fake.clients:
        client.close()


def _config(feature: Feature, **values: Any) -> EvalConfig:
    """기능 1개의 평가 조건. 처분 기능은 처분을 켠 지시문을 쓴다."""
    values.setdefault("disposition", feature == Feature.DISPOSITION)
    return EvalConfig(feature=feature, model=DEFAULT_NEMOTRON_MODEL, **values)


def _run(
    api: FakeJudgmentModel,
    out_dir: Path,
    config: EvalConfig,
    *,
    responder: Responder | None = None,
    saved: Sequence[SavedPair] = (),
) -> tuple[PreparedEval, EvalReport]:
    """사례를 준비하고 Fake 모델로 평가한다. 응답기가 없으면 정답대로 답한다."""
    prepared = prepare_eval(config, out_dir=out_dir, saved=saved, dry_run=False)
    api.responder = responder or correct_replies(prepared.cases)
    settings = TokenFactorySettings()  # pyright: ignore[reportCallIssue]
    return prepared, run_eval(prepared, out_dir=out_dir, settings=settings)


def _metrics(report: EvalReport) -> dict[str, tuple[int, int, bool | None]]:
    return {metric.name: (metric.count, metric.total, metric.passed) for metric in report.metrics}


def _results(out_dir: Path) -> list[EvalResult]:
    lines = (out_dir / RESULTS_FILE).read_text().splitlines()
    return [EvalResult.model_validate_json(line) for line in lines]


def _bodies(api: FakeJudgmentModel) -> list[dict[str, Any]]:
    return [json.loads(request.content) for request in api.model_posts]


def _override(
    cases: Sequence[EvalCase], replies: dict[str, Reply], *, key: str = "text"
) -> Responder:
    """고른 접수 문장에는 정한 응답을 준다. 나머지는 정답대로 답한다."""
    correct = correct_replies(cases)

    def respond(body: dict[str, Any]) -> Reply:
        return replies.get(body["messages"][-1]["content"]) or correct(body)

    return respond


def _texts(cases: Sequence[EvalCase], group: IntakeGroup) -> list[IntakeCase]:
    return [case for case in cases if isinstance(case, IntakeCase) and case.group == group]


class TestPrepareEval:
    """요청을 보내기 전의 사례, 평가 조건과 비용 어림을 검사한다."""

    def test_dry_run_writes_cases_and_conditions_without_requests(
        self, tmp_path: Path, api: FakeJudgmentModel
    ) -> None:
        """사례와 평가 조건만 쓴다. 네트워크를 쓰지 않는다. 인증값을 기록하지 않는다."""
        out_dir = tmp_path / "eval"
        prepared = prepare_eval(_config(Feature.INTAKE), out_dir=out_dir, dry_run=True)
        assert sorted(path.name for path in out_dir.iterdir()) == [CASES_FILE, EVAL_FILE]
        assert not api.requests
        manifest = EvalManifest.model_validate_json((out_dir / EVAL_FILE).read_text())
        assert manifest == prepared.manifest and manifest.dry_run
        assert manifest.case_count == manifest.request_count == len(prepared.cases) == 130
        cases_bytes = (out_dir / CASES_FILE).read_bytes()
        assert manifest.cases_sha256 == hashlib.sha256(cases_bytes).hexdigest()
        assert len(cases_bytes.splitlines()) == 130
        assert (manifest.max_tokens, manifest.timeout_s, manifest.sdk_retries) == (2048, 120.0, 0)
        assert FAKE_JUDGMENT_KEY not in (out_dir / EVAL_FILE).read_text()
        assert str(tmp_path) not in (out_dir / EVAL_FILE).read_text()

    def test_same_config_makes_the_same_cases(self, tmp_path: Path) -> None:
        """같은 시드와 한도는 같은 SHA-256을 만든다. 평가 ID는 실행마다 다르다."""
        config = _config(Feature.REVIEW, seed=5, limit=12)
        first = prepare_eval(config, out_dir=tmp_path / "a", dry_run=True).manifest
        second = prepare_eval(config, out_dir=tmp_path / "b", dry_run=True).manifest
        assert first.cases_sha256 == second.cases_sha256 and first.eval_id != second.eval_id
        other = prepare_eval(
            _config(Feature.REVIEW, seed=6, limit=12), out_dir=tmp_path / "c", dry_run=True
        ).manifest
        assert other.cases_sha256 != first.cases_sha256

    @pytest.mark.parametrize(
        ("limit", "repeats", "requests"),
        [(None, 1, 130), (None, 3, 130 + 2 * CONSISTENCY_CASES), (3, 1, 3), (3, 3, 9)],
    )
    def test_repeats_add_runs_of_the_repeated_subset(
        self, tmp_path: Path, limit: int | None, repeats: int, requests: int
    ) -> None:
        """모든 사례는 1번 실행한다. 반복 사례 20개는 남은 횟수만큼 더 실행한다."""
        config = _config(Feature.INTAKE, limit=limit, repeats=repeats)
        manifest = prepare_eval(config, out_dir=tmp_path / "eval", dry_run=True).manifest
        assert manifest.request_count == requests

    def test_estimate_uses_the_request_size_and_the_model_price(self, tmp_path: Path) -> None:
        """입력 토큰은 요청 본문의 크기에서 어림한다. 비용은 모델 가격으로 계산한다."""
        config = EvalConfig(feature=Feature.INTAKE, model=ULTRA, limit=4)
        prepared = prepare_eval(config, out_dir=tmp_path / "eval", dry_run=True)
        estimate = prepared.manifest.estimate
        texts = sum(
            len(case.text.encode()) for case in prepared.cases if isinstance(case, IntakeCase)
        )
        assert estimate.input_tokens > (4 * len(INTAKE_SYSTEM_PROMPT) + texts) // BYTES_PER_TOKEN
        input_price, output_price = PRICES_USD[ULTRA]
        tokens = estimate.input_tokens
        assert estimate.expected_usd == pytest.approx(
            (tokens * input_price + 4 * EXPECTED_OUTPUT_TOKENS * output_price) / 1e6
        )
        assert estimate.max_usd == pytest.approx(
            (tokens * input_price + 4 * 2048 * output_price) / 1e6
        )

    def test_model_without_a_price_has_no_cost(self, tmp_path: Path) -> None:
        """가격표에 없는 모델은 입력 토큰만 어림한다."""
        config = EvalConfig(feature=Feature.INTAKE, model="nvidia/nemotron-new", limit=2)
        estimate = prepare_eval(config, out_dir=tmp_path / "eval", dry_run=True).manifest.estimate
        assert estimate.input_tokens > 0
        assert estimate.expected_usd is None and estimate.max_usd is None

    @pytest.mark.parametrize(
        ("feature", "prompt", "tools"),
        [
            (
                Feature.INTAKE,
                INTAKE_SYSTEM_PROMPT,
                ["submit_kit_order", "submit_board_order", "reject_request"],
            ),
            (Feature.REVIEW, REVIEW_SYSTEM_PROMPT, ["calibrate_board", "replan_layout"]),
            (Feature.COORDINATION, REVIEW_SYSTEM_PROMPT, ["prepare_layout", "execute_cell"]),
            (Feature.DISPOSITION, DISPOSITION_PROMPT, ["rework_cell"]),
        ],
    )
    def test_manifest_records_the_prompt_and_the_tools(
        self, tmp_path: Path, feature: Feature, prompt: str, tools: list[str]
    ) -> None:
        """평가 조건에 실행 프로그램의 지시문과 strict 도구 정의가 있다."""
        manifest = prepare_eval(
            _config(feature, limit=3), out_dir=tmp_path / "eval", dry_run=True
        ).manifest
        assert manifest.system_prompt == prompt
        functions: list[Any] = [tool["function"] for tool in manifest.tools]
        assert [function["name"] for function in functions] == tools
        assert all(function["strict"] is True for function in functions)

    @pytest.mark.parametrize("feature", [Feature.REVIEW, Feature.COORDINATION])
    def test_disposition_setting_selects_the_prompt_of_the_run(
        self, tmp_path: Path, feature: Feature
    ) -> None:
        """처분을 켠 평가는 처분을 켠 실행의 지시문으로 요청을 만든다. 사례는 같다."""
        off = prepare_eval(_config(feature, limit=3), out_dir=tmp_path / "off", dry_run=True)
        on = prepare_eval(
            _config(feature, limit=3, disposition=True), out_dir=tmp_path / "on", dry_run=True
        )
        assert off.manifest.system_prompt == REVIEW_SYSTEM_PROMPT
        assert on.manifest.system_prompt == DISPOSITION_PROMPT
        assert on.manifest.cases_sha256 == off.manifest.cases_sha256
        assert on.manifest.estimate.input_tokens > off.manifest.estimate.input_tokens

    @pytest.mark.parametrize(
        ("feature", "disposition"), [(Feature.DISPOSITION, False), (Feature.INTAKE, True)]
    )
    def test_config_rejects_a_disposition_setting_that_the_feature_cannot_use(
        self, feature: Feature, disposition: bool
    ) -> None:
        """처분 기능은 처분을 켠 지시문만 쓴다. 접수는 그 설정을 쓰지 않는다."""
        with pytest.raises(ValidationError, match="does not fit the feature"):
            EvalConfig(feature=feature, model=DEFAULT_NEMOTRON_MODEL, disposition=disposition)

    def test_existing_output_is_not_overwritten(self, tmp_path: Path) -> None:
        """기존 출력 디렉터리에는 쓰지 않는다."""
        (tmp_path / "eval").mkdir()
        with pytest.raises(FileExistsError):
            prepare_eval(_config(Feature.INTAKE), out_dir=tmp_path / "eval", dry_run=True)
        assert not list((tmp_path / "eval").iterdir())

    @pytest.mark.parametrize(
        "values",
        [{"repeats": 0}, {"repeats": 6}, {"limit": 0}, {"model": ""}, {"model": "a b"}],
    )
    def test_config_rejects_values_outside_the_limits(self, values: dict[str, Any]) -> None:
        """반복 횟수, 한도와 모델 ID의 범위를 검사한다."""
        with pytest.raises(ValidationError):
            EvalConfig(**{"feature": Feature.INTAKE, "model": DEFAULT_NEMOTRON_MODEL, **values})


class TestReviewRequests:
    """검토 요청이 실행 프로그램의 `layout_ready` 요청과 같은 계약인지 검사한다."""

    @pytest.fixture
    def sent(
        self, tmp_path: Path, api: FakeJudgmentModel
    ) -> list[tuple[ReviewCase, dict[str, Any]]]:
        """묶음마다 5개의 사례와 그 요청 본문."""
        prepared, _ = _run(api, tmp_path / "eval", _config(Feature.REVIEW, limit=15))
        cases = [case for case in prepared.cases if isinstance(case, ReviewCase)]
        return list(zip(cases, _bodies(api), strict=True))

    def _find(
        self, sent: list[tuple[ReviewCase, dict[str, Any]]], case_id: str
    ) -> tuple[ReviewCase, dict[str, Any]]:
        return next((case, body) for case, body in sent if case.case_id == case_id)

    def test_first_attempt_offers_calibration_and_replan(
        self, sent: list[tuple[ReviewCase, dict[str, Any]]], api: FakeJudgmentModel
    ) -> None:
        """요청은 검토 지시문, 상태, 목표와 부품 목록이 있는 준비 결과를 보낸다."""
        case, body = self._find(sent, "review-correct-001")
        system, goal, call, result = body["messages"]
        assert system["content"].startswith(REVIEW_SYSTEM_PROMPT + "\nCurrent runtime state: ")
        state = runtime_state(body)
        assert state["state"] == "layout_ready" and state["layout_attempts"] == 1
        assert state["error_code"] is None and state["calibration_id"] is None
        assert goal == {"role": "user", "content": case.goal}
        assert [tool["function"]["name"] for tool in body["tools"]] == [
            "calibrate_board",
            "replan_layout",
        ]
        assert all(tool["function"]["strict"] is True for tool in body["tools"])
        assert body["tool_choice"] == "auto" and body["max_tokens"] == 2048
        function = call["tool_calls"][0]["function"]
        assert function == {"name": "prepare_layout", "arguments": '{"retry":false}'}
        prepared = LayoutResult.model_validate_json(result["content"])
        assert result["tool_call_id"] == call["tool_calls"][0]["id"] == prepared.tool_call_id
        assert prepared.artifacts[0].id == state["layout_id"]
        assert isinstance(prepared.summary, ReviewedLayoutSummary)
        assert [part.part_id for part in prepared.summary.parts] == [
            part.part_id for part in case.layout.parts
        ]
        assert prepared.summary.cell_accuracy == 1.0 and prepared.summary.repair_count == 0
        assert api.model_posts[0].extensions["timeout"]["read"] == 120

    def test_format_retry_history_offers_only_calibration(
        self, sent: list[tuple[ReviewCase, dict[str, Any]]]
    ) -> None:
        """형식 오류 재시도 뒤의 요청은 준비 오류와 두 번째 준비 결과를 보낸다."""
        case, body = self._find(sent, "review-last_attempt-001")
        assert case.layout_attempts == 2 and case.rejected is None
        assert [tool["function"]["name"] for tool in body["tools"]] == ["calibrate_board"]
        state = runtime_state(body)
        assert state["layout_attempts"] == 2
        _, _, first, failed, second, result = body["messages"]
        assert first["tool_calls"][0]["function"]["arguments"] == '{"retry":false}'
        error = ErrorResult.model_validate_json(failed["content"])
        assert error.error_code == "layout_invalid" and error.recoverable
        assert error.summary.layout_attempt == 1
        assert second["tool_calls"][0]["function"] == {
            "name": "prepare_layout",
            "arguments": '{"retry":true}',
        }
        assert first["tool_calls"][0]["id"] != second["tool_calls"][0]["id"]
        prepared = LayoutResult.model_validate_json(result["content"])
        assert prepared.artifacts[0].id == state["layout_id"]

    def test_replan_history_shows_the_rejected_plan_and_the_reason(
        self, sent: list[tuple[ReviewCase, dict[str, Any]]]
    ) -> None:
        """재계획 뒤의 요청은 거부한 첫 계획, 재계획 호출과 두 번째 계획을 보낸다."""
        case, body = self._find(sent, "review-last_attempt-005")
        assert case.layout_attempts == 2 and case.rejected is not None
        assert [tool["function"]["name"] for tool in body["tools"]] == ["calibrate_board"]
        _, _, _, first_result, replan, second_result = body["messages"]
        first = LayoutResult.model_validate_json(first_result["content"])
        second = LayoutResult.model_validate_json(second_result["content"])
        assert isinstance(first.summary, ReviewedLayoutSummary)
        assert len(first.summary.parts) == len(case.rejected.layout.parts)
        function = replan["tool_calls"][0]["function"]
        assert function["name"] == "replan_layout"
        assert json.loads(function["arguments"]) == {
            "layout_id": first.artifacts[0].id,
            "reason": case.rejected.reason.value,
        }
        assert second.artifacts[0].id == runtime_state(body)["layout_id"]
        assert second.artifacts[0].id != first.artifacts[0].id

    def test_same_case_makes_the_same_request(
        self, tmp_path: Path, api: FakeJudgmentModel, sent: list[tuple[ReviewCase, dict[str, Any]]]
    ) -> None:
        """요청의 ID는 사례 ID에서 만든다. 다시 실행해도 요청 본문이 같다."""
        first = [body for _, body in sent]
        _run(api, tmp_path / "again", _config(Feature.REVIEW, limit=15))
        assert _bodies(api)[len(first) :] == first


class TestIntakeEval:
    """접수 평가의 채점과 기준 판정을 검사한다."""

    def test_correct_answers_pass_every_criterion(
        self, tmp_path: Path, api: FakeJudgmentModel
    ) -> None:
        """정답대로 답하면 모든 기준을 통과한다. 요청은 모델 목록 1회와 사례마다 1회다."""
        out_dir = tmp_path / "eval"
        prepared, report = _run(api, out_dir, _config(Feature.INTAKE))
        assert report.passed and not report.stopped and report.consistency is None
        assert _metrics(report) == {
            "contract_valid": (130, 130, True),
            "kit_order_match": (40, 40, True),
            "board_order_match": (30, 30, True),
            "false_accept": (0, 48, True),
            "non_assembly_accept": (0, 8, True),
            "reject_reason_match": (48, 48, None),
            "korean_match": (12, 12, None),
        }
        assert [request.method for request in api.requests] == ["GET"] + ["POST"] * 130
        assert all(client.is_closed for client in api.clients)
        results = _results(out_dir)
        assert [result.case_id for result in results] == [case.case_id for case in prepared.cases]
        assert [result.inference for result in results] == list(range(1, 131))
        assert all(result.correct and result.status == ResponseStatus.VALID for result in results)
        assert EvalReport.model_validate_json((out_dir / REPORT_FILE).read_text()) == report
        assert report.eval_id == prepared.manifest.eval_id
        assert report.completed_requests == report.request_count == 130
        usage = report.usage
        assert (usage.prompt_tokens, usage.completion_tokens, usage.reasoning_tokens) == (
            130 * 100,
            130 * 50,
            130 * 20,
        )
        input_price, output_price = PRICES_USD[DEFAULT_NEMOTRON_MODEL]
        assert usage.cost_usd == pytest.approx((13000 * input_price + 6500 * output_price) / 1e6)

    def test_records_have_no_key_and_no_reasoning(
        self, tmp_path: Path, api: FakeJudgmentModel
    ) -> None:
        """기록에 인증값과 reasoning 원문이 없다. 관찰 기록은 요청마다 4줄이다."""
        out_dir = tmp_path / "eval"
        prepared, _ = _run(api, out_dir, _config(Feature.INTAKE, limit=4))
        for path in out_dir.iterdir():
            text = path.read_text()
            assert FAKE_JUDGMENT_KEY not in text and FAKE_JUDGMENT_REASONING not in text, path
        audit = [json.loads(line) for line in (out_dir / AUDIT_FILE).read_text().splitlines()]
        assert Counter(record["inference"] for record in audit) == dict.fromkeys(range(1, 5), 4)
        assert {record["run_id"] for record in audit} == {prepared.manifest.eval_id}

    @pytest.mark.parametrize(("wrong", "passed"), [(2, True), (3, False)])
    def test_kit_criterion_allows_two_wrong_orders(
        self, tmp_path: Path, api: FakeJudgmentModel, wrong: int, passed: bool
    ) -> None:
        """키트 주문은 38/40 이상이 기준이다. 다른 수량의 주문은 일치가 아니다."""
        cases = build_cases(Feature.INTAKE, seed=0)
        other = Reply(
            tool="submit_kit_order",
            arguments={
                "kind": "kit",
                "items": [{"type": "square_tile", "count": 6}],
                "arrangement": "type_rows",
            },
        )
        # 정사각 6개의 주문이 정답이 아닌 문장만 고른다.
        texts = [
            case.text
            for case in _texts(cases, IntakeGroup.KIT)
            if isinstance(case.order, KitOrder) and case.order.part_count < 6
        ]
        replies = dict.fromkeys(texts[:wrong], other)
        _, report = _run(
            api, tmp_path / "eval", _config(Feature.INTAKE), responder=_override(cases, replies)
        )
        assert _metrics(report)["kit_order_match"] == (40 - wrong, 40, passed)
        assert _metrics(report)["contract_valid"] == (130, 130, True)
        assert report.passed is passed

    def test_board_order_in_another_part_order_matches(
        self, tmp_path: Path, api: FakeJudgmentModel
    ) -> None:
        """부품의 순서가 달라도 셀이 같으면 같은 주문이다."""
        cases = build_cases(Feature.INTAKE, seed=0)
        replies = {}
        for case in _texts(cases, IntakeGroup.BOARD):
            assert isinstance(case.order, BoardOrder)
            reordered = BoardOrder(parts=tuple(reversed(case.order.parts)))
            replies[case.text] = Reply(
                tool="submit_board_order", arguments=reordered.model_dump(mode="json")
            )
        _, report = _run(
            api, tmp_path / "eval", _config(Feature.INTAKE), responder=_override(cases, replies)
        )
        assert _metrics(report)["board_order_match"] == (30, 30, True)

    @pytest.mark.parametrize(("accepted", "passed"), [(2, True), (3, False)])
    def test_false_accept_criterion_allows_two_accepted_requests(
        self, tmp_path: Path, api: FakeJudgmentModel, accepted: int, passed: bool
    ) -> None:
        """범위 밖 요청의 수락은 2/48 이하가 기준이다. 수락한 요청은 사유도 틀린 것이다."""
        cases = build_cases(Feature.INTAKE, seed=0)
        submit = Reply(tool="submit_kit_order", arguments=KIT.model_dump(mode="json"))
        texts = [
            case.text
            for case in _texts(cases, IntakeGroup.OUT_OF_SCOPE)
            if case.reject_reason == RejectReason.TOO_MANY_PARTS
        ]
        replies = dict.fromkeys(texts[:accepted], submit)
        _, report = _run(
            api, tmp_path / "eval", _config(Feature.INTAKE), responder=_override(cases, replies)
        )
        metrics = _metrics(report)
        assert metrics["false_accept"] == (accepted, 48, passed)
        assert metrics["non_assembly_accept"] == (0, 8, True)
        assert metrics["reject_reason_match"] == (48 - accepted, 48, None)
        assert report.passed is passed

    def test_accepted_non_assembly_request_fails(
        self, tmp_path: Path, api: FakeJudgmentModel
    ) -> None:
        """조립이 아닌 요청을 1건이라도 수락하면 기준에 못 미친다."""
        cases = build_cases(Feature.INTAKE, seed=0)
        text = next(
            case.text
            for case in _texts(cases, IntakeGroup.OUT_OF_SCOPE)
            if case.reject_reason == RejectReason.NOT_ASSEMBLY_REQUEST
        )
        submit = Reply(tool="submit_kit_order", arguments=KIT.model_dump(mode="json"))
        _, report = _run(
            api,
            tmp_path / "eval",
            _config(Feature.INTAKE),
            responder=_override(cases, {text: submit}),
        )
        metrics = _metrics(report)
        assert metrics["false_accept"] == (1, 48, True)
        assert metrics["non_assembly_accept"] == (1, 8, False)
        assert not report.passed

    def test_another_reject_reason_is_not_a_false_accept(
        self, tmp_path: Path, api: FakeJudgmentModel
    ) -> None:
        """다른 사유의 거절은 사유 일치만 낮춘다. 기준은 통과한다."""
        cases = build_cases(Feature.INTAKE, seed=0)
        reject = Reply(tool="reject_request", arguments={"reason": "ambiguous_request"})
        replies = dict.fromkeys(
            (case.text for case in _texts(cases, IntakeGroup.OUT_OF_SCOPE)), reject
        )
        _, report = _run(
            api, tmp_path / "eval", _config(Feature.INTAKE), responder=_override(cases, replies)
        )
        metrics = _metrics(report)
        assert metrics["false_accept"] == (0, 48, True)
        assert metrics["reject_reason_match"] == (8, 48, None)
        assert report.passed

    def test_responses_outside_the_contract_fail_the_contract_criterion(
        self, tmp_path: Path, api: FakeJudgmentModel
    ) -> None:
        """자유 텍스트, 계약 밖의 주문과 잘린 응답은 계약에 맞는 응답이 아니다."""
        cases = select_cases(build_cases(Feature.INTAKE, seed=0), limit=8)
        texts = [case.text for case in cases if isinstance(case, IntakeCase)]
        too_many = {
            "kind": "kit",
            "items": [{"type": "square_tile", "count": 6}, {"type": "rect_tile", "count": 6}],
            "arrangement": "type_rows",
        }
        replies = {
            texts[0]: FREE_TEXT,
            texts[1]: Reply(tool="submit_kit_order", arguments=too_many),
            texts[2]: Reply(tool="reject_request", arguments={"reason": "a"}, finish="length"),
        }
        out_dir = tmp_path / "eval"
        _, report = _run(
            api, out_dir, _config(Feature.INTAKE, limit=8), responder=_override(cases, replies)
        )
        results = _results(out_dir)
        assert [(result.status, result.failure, result.answer) for result in results[:3]] == [
            (ResponseStatus.INVALID, "protocol_error", None),
            (ResponseStatus.INVALID, "order_invalid", None),
            (ResponseStatus.INVALID, "model_error", None),
        ]
        assert not any(result.correct for result in results[:3])
        assert _metrics(report)["contract_valid"] == (5, 8, False)
        assert not report.passed

    def test_infeasible_board_order_is_valid_and_wrong(
        self, tmp_path: Path, api: FakeJudgmentModel
    ) -> None:
        """배치 규칙을 어긴 보드 주문은 계약에 맞는 응답이다. 정답은 아니다."""
        cases = select_cases(build_cases(Feature.INTAKE, seed=0), limit=2)
        overlap = {
            "kind": "board",
            "parts": [
                {"type": "square_tile", "color": "blue", "rotation_deg": 0, "row": 1, "col": 1},
                {"type": "rect_tile", "color": "gray", "rotation_deg": 0, "row": 1, "col": 1},
            ],
        }
        board = next(case for case in cases if isinstance(case, IntakeCase) and case.order)
        replies = {board.text: Reply(tool="submit_board_order", arguments=overlap)}
        out_dir = tmp_path / "eval"
        _run(api, out_dir, _config(Feature.INTAKE, limit=2), responder=_override(cases, replies))
        result = _results(out_dir)[0]
        assert (result.status, result.answer) == (ResponseStatus.VALID, "order_not_feasible")
        assert not result.correct and result.order is not None


class TestReviewEval:
    """계획 검토 평가의 채점과 기준 판정을 검사한다."""

    def test_correct_answers_pass_every_criterion(
        self, tmp_path: Path, api: FakeJudgmentModel
    ) -> None:
        """정답대로 답하면 모든 기준을 통과한다. 저장한 계획이 없으면 그 기준은 판정하지 않는다."""
        out_dir = tmp_path / "eval"
        _, report = _run(api, out_dir, _config(Feature.REVIEW))
        assert report.passed
        assert _metrics(report) == {
            "contract_valid": (200, 200, True),
            "false_detection": (0, 96, True),
            "saved_false_detection": (0, 0, None),
            "detected_part_set_mismatch": (28, 28, True),
            "detected_position_mismatch": (28, 28, True),
            "detected_arrangement_mismatch": (28, 28, True),
            "wrong_plan_not_accepted": (84, 84, None),
            "last_attempt_match": (20, 20, None),
        }
        answers = Counter(result.answer for result in _results(out_dir))
        assert answers == {
            "calibrate_board": 106,
            "replan_layout:part_set_mismatch": 28,
            "replan_layout:position_mismatch": 28,
            "replan_layout:arrangement_mismatch": 28,
            "abort:plan_rejected": 10,
        }
        audit = [json.loads(line) for line in (out_dir / AUDIT_FILE).read_text().splitlines()]
        assert audit[0]["kind"] == "request" and audit[0]["data"]["case_id"] == "review-correct-001"
        assert audit[3] == {**audit[3], "kind": "validation", "data": {"accepted": True}}

    def test_model_that_always_proceeds_detects_nothing(
        self, tmp_path: Path, api: FakeJudgmentModel
    ) -> None:
        """항상 보정을 고르면 오검출은 없다. 검출률 기준에 못 미친다."""

        def proceed(body: dict[str, Any]) -> Reply:
            return review_reply(ReviewAction.PROCEED, runtime_state(body))

        _, report = _run(api, tmp_path / "eval", _config(Feature.REVIEW), responder=proceed)
        metrics = _metrics(report)
        assert metrics["false_detection"] == (0, 96, True)
        assert metrics["detected_part_set_mismatch"] == (0, 28, False)
        assert metrics["detected_position_mismatch"] == (0, 28, False)
        assert metrics["detected_arrangement_mismatch"] == (0, 28, False)
        assert metrics["wrong_plan_not_accepted"] == (0, 84, None)
        assert metrics["last_attempt_match"] == (10, 20, None)
        assert not report.passed

    def test_model_that_always_replans_fails_the_false_detection_criterion(
        self, tmp_path: Path, api: FakeJudgmentModel
    ) -> None:
        """올바른 계획의 재계획은 오검출이다. 다른 사유의 재계획은 검출이 아니다."""

        def replan(body: dict[str, Any]) -> Reply:
            state = runtime_state(body)
            action = ReviewAction.REPLAN if state["layout_attempts"] == 1 else ReviewAction.PROCEED
            return review_reply(action, state, reason="part_set_mismatch")

        saved = [SavedPair("kit-ok", KIT, KIT_LAYOUT)]
        _, report = _run(
            api, tmp_path / "eval", _config(Feature.REVIEW), responder=replan, saved=saved
        )
        metrics = _metrics(report)
        assert metrics["contract_valid"] == (200, 200, True)
        assert metrics["false_detection"] == (96, 96, False)
        assert metrics["saved_false_detection"] == (1, 1, False)
        assert metrics["detected_part_set_mismatch"] == (28, 28, True)
        assert metrics["detected_position_mismatch"] == (0, 28, False)
        assert metrics["wrong_plan_not_accepted"] == (84, 84, None)

    @pytest.mark.parametrize(("wrong", "passed"), [(1, True), (2, False)])
    def test_false_detection_criterion_allows_one_case(
        self, tmp_path: Path, api: FakeJudgmentModel, wrong: int, passed: bool
    ) -> None:
        """올바른 계획의 오검출은 1/96 이하가 기준이다. 자발적 중단도 오검출이다."""
        cases = build_cases(Feature.REVIEW, seed=0)
        correct = correct_replies(cases)
        goals = Counter(case.goal for case in cases if isinstance(case, ReviewCase))
        # 목표가 1번만 나오는 올바른 계획을 고른다. 다른 사례의 답을 바꾸지 않는다.
        targets = [
            case.goal
            for case in cases
            if isinstance(case, ReviewCase)
            and case.group == ReviewGroup.CORRECT
            and goals[case.goal] == 1
        ][:wrong]

        def respond(body: dict[str, Any]) -> Reply:
            first_attempt = runtime_state(body)["layout_attempts"] == 1
            if first_attempt and body["messages"][1]["content"] in targets:
                return Reply(content='{"decision":"abort","reason_code":"agent_aborted"}')
            return correct(body)

        _, report = _run(api, tmp_path / "eval", _config(Feature.REVIEW), responder=respond)
        assert _metrics(report)["false_detection"] == (wrong, 96, passed)
        assert report.passed is passed

    @pytest.mark.parametrize(
        ("reply", "failure"),
        [
            (FREE_TEXT, "invalid_action"),
            (Reply(content='{"decision":"abort"}'), "invalid_arguments"),
            (
                Reply(content='{"decision":"complete","reason_code":"check_passed"}'),
                "unverified_completion",
            ),
            (Reply(tool="calibrate_board", arguments={"layout_id": "0" * 32}), "invalid_layout_id"),
            (Reply(tool="execute_cell", arguments={}), "tool_not_allowed"),
            (Reply(tool="calibrate_board", arguments="{"), "invalid_action"),
            (Reply(tool="calibrate_board", arguments={}, finish="length"), "token_limit"),
        ],
    )
    def test_response_outside_the_contract_is_invalid(
        self, tmp_path: Path, api: FakeJudgmentModel, reply: Reply, failure: str
    ) -> None:
        """실행 프로그램의 검증을 통과하지 못한 응답은 답이 없다. 오검출로 세지 않는다."""
        out_dir = tmp_path / "eval"
        _, report = _run(
            api, out_dir, _config(Feature.REVIEW, limit=1), responder=lambda body: reply
        )
        result = _results(out_dir)[0]
        assert result.status == ResponseStatus.INVALID and result.answer is None
        assert result.failure == failure and not result.correct
        assert _metrics(report)["contract_valid"] == (0, 1, False)
        assert _metrics(report)["false_detection"] == (0, 1, True)
        validation = json.loads((out_dir / AUDIT_FILE).read_text().splitlines()[-1])
        assert validation["data"]["accepted"] is False
        assert validation["data"]["reason"] == result.failure

    def test_replan_on_the_last_attempt_is_not_offered(
        self, tmp_path: Path, api: FakeJudgmentModel
    ) -> None:
        """준비 시도가 2회인 요청에서 재계획은 계약 위반이다."""

        def replan(body: dict[str, Any]) -> Reply:
            return review_reply(
                ReviewAction.REPLAN, runtime_state(body), reason="position_mismatch"
            )

        out_dir = tmp_path / "eval"
        _run(api, out_dir, _config(Feature.REVIEW, limit=3), responder=replan)
        last = _results(out_dir)[2]
        assert last.case_id == "review-last_attempt-001"
        assert (last.status, last.failure) == (ResponseStatus.INVALID, "tool_not_allowed")


def _tool_names(body: dict[str, Any]) -> list[str]:
    return [tool["function"]["name"] for tool in body.get("tools", [])]


def _last_result(body: dict[str, Any]) -> str:
    return body["messages"][-1]["content"]


class TestCoordinationEval:
    """조율 평가의 요청, 채점과 기준 판정을 검사한다."""

    def test_requests_are_the_runtime_requests_of_each_state(
        self, tmp_path: Path, api: FakeJudgmentModel
    ) -> None:
        """상태마다 실행 프로그램과 같은 대화, 상태와 도구 1개를 보낸다."""
        prepared, _ = _run(api, tmp_path / "eval", _config(Feature.COORDINATION))
        sent = dict(zip([case.case_id for case in prepared.cases], _bodies(api), strict=True))
        created = sent["coordination-created-001"]
        assert [message["role"] for message in created["messages"]] == ["system", "user"]
        assert created["messages"][0]["content"].startswith(REVIEW_SYSTEM_PROMPT + "\nCurrent")
        assert _tool_names(created) == ["prepare_layout"] and created["tool_choice"] == "auto"
        state = runtime_state(created)
        assert state["state"] == "created" and state["layout_id"] is None
        assert "rework_runs" not in state
        retryable = sent["coordination-layout_retryable-001"]
        failed = ErrorResult.model_validate_json(_last_result(retryable))
        assert failed.recoverable and failed.error_code.value == "layout_invalid"
        assert runtime_state(retryable)["error_code"] == "layout_invalid"
        assert _tool_names(retryable) == ["prepare_layout"]
        calibrated = sent["coordination-calibrated-001"]
        assert _tool_names(calibrated) == ["execute_cell"]
        state = runtime_state(calibrated)
        result = CalibrationResult.model_validate_json(_last_result(calibrated))
        assert result.artifacts[0].id == state["calibration_id"] != state["layout_id"]
        prepared_layout = LayoutResult.model_validate_json(calibrated["messages"][3]["content"])
        assert prepared_layout.artifacts[0].id == state["layout_id"]

    def test_correct_answers_pass_every_criterion(
        self, tmp_path: Path, api: FakeJudgmentModel
    ) -> None:
        """정답대로 답하면 기준을 통과한다. 처분을 켠 평가는 그 지시문과 상태를 보낸다."""
        out_dir = tmp_path / "eval"
        _, report = _run(api, out_dir, _config(Feature.COORDINATION, disposition=True))
        assert report.passed
        assert _metrics(report) == {
            "contract_valid": (30, 30, True),
            "tool_match": (30, 30, True),
            "tool_match_created": (10, 10, None),
            "tool_match_layout_retryable": (10, 10, None),
            "tool_match_calibrated": (10, 10, None),
        }
        answers = Counter(result.answer for result in _results(out_dir))
        assert answers == {"prepare_layout": 20, "execute_cell": 10}
        for body in _bodies(api):
            assert body["messages"][0]["content"].startswith(DISPOSITION_PROMPT + "\nCurrent")
            assert runtime_state(body)["rework_runs"] == 0

    @pytest.mark.parametrize(("aborted", "passed"), [(1, True), (2, False)])
    def test_tool_criterion_allows_one_abort(
        self, tmp_path: Path, api: FakeJudgmentModel, aborted: int, passed: bool
    ) -> None:
        """중단은 계약과 같다. 정답은 아니다. 30개 중 1개까지 기준을 통과한다."""
        count = 0

        def respond(body: dict[str, Any]) -> Reply:
            nonlocal count
            count += 1
            if count <= aborted:
                return Reply(content='{"decision":"abort","reason_code":"agent_aborted"}')
            return coordination_reply(runtime_state(body))

        out_dir = tmp_path / "eval"
        _, report = _run(api, out_dir, _config(Feature.COORDINATION), responder=respond)
        metrics = _metrics(report)
        assert metrics["contract_valid"] == (30, 30, True)
        assert metrics["tool_match"] == (30 - aborted, 30, passed)
        assert _results(out_dir)[0].answer == "abort:agent_aborted"
        assert report.passed is passed

    def test_another_tool_is_outside_the_contract(
        self, tmp_path: Path, api: FakeJudgmentModel
    ) -> None:
        """제공하지 않은 도구의 호출은 계약 위반이다."""

        def respond(body: dict[str, Any]) -> Reply:
            return Reply(tool="calibrate_board", arguments={"layout_id": "0" * 32})

        out_dir = tmp_path / "eval"
        _, report = _run(api, out_dir, _config(Feature.COORDINATION, limit=3), responder=respond)
        assert _metrics(report)["contract_valid"] == (0, 3, False)
        assert {result.failure for result in _results(out_dir)} == {"tool_not_allowed"}


class TestDispositionEval:
    """처분 평가의 요청, 채점과 기준 판정을 검사한다."""

    @pytest.fixture
    def cases(self) -> list[DispositionCase]:
        cases = build_cases(Feature.DISPOSITION, seed=0)
        return [case for case in cases if isinstance(case, DispositionCase)]

    def _kind(self, cases: list[DispositionCase], kind: DispositionKind) -> DispositionCase:
        return next(case for case in cases if case.kind == kind)

    def test_requests_are_the_runtime_requests_of_the_checked_state(
        self, tmp_path: Path, api: FakeJudgmentModel, cases: list[DispositionCase]
    ) -> None:
        """요청은 처분 지시문, 검사 요약과 재작업 도구를 실행 프로그램과 같은 조건으로 보낸다."""
        _run(api, tmp_path / "eval", _config(Feature.DISPOSITION))
        sent = dict(zip([case.case_id for case in cases], _bodies(api), strict=True))
        for body in sent.values():
            assert body["messages"][0]["content"].startswith(DISPOSITION_PROMPT + "\nCurrent")
            assert runtime_state(body)["state"] == "checked"
            assert "rework_targets" not in json.dumps(body)
        missing = self._kind(cases, DispositionKind.MISSING_TWO)
        body = sent[missing.case_id]
        state = runtime_state(body)
        assert _tool_names(body) == ["rework_cell"] and body["tool_choice"] == "auto"
        assert (state["error_code"], state["rework_runs"]) == ("check_failed", 0)
        assert len(body["messages"]) == 8
        result = ExecutionResult.model_validate_json(_last_result(body))
        assert result.artifacts[0].id == state["check_id"]
        assert isinstance(result.summary, InspectedExecutionSummary)
        inspection = result.summary.inspection
        assert inspection is not None and not result.summary.passed
        assert tuple(part.part_id for part in inspection.parts) == missing.expected_parts
        assert all(part.empty for part in inspection.parts)
        assert [part.part_id for part in result.summary.failed_parts] == list(
            missing.expected_parts
        )
        released = sent[self._kind(cases, DispositionKind.ALL_MATCH).case_id]
        assert "tools" not in released and "tool_choice" not in released
        assert runtime_state(released)["error_code"] is None
        wrong_color = sent[self._kind(cases, DispositionKind.WRONG_COLOR).case_id]
        assert runtime_state(wrong_color)["error_code"] == "inspection_rejected"
        assert _tool_names(wrong_color) == ["rework_cell"]
        no_report = sent[self._kind(cases, DispositionKind.NO_CAMERA_REPORT).case_id]
        summary = json.loads(_last_result(no_report))["summary"]
        assert summary["inspection"] is None and summary["passed"] is True
        assert runtime_state(no_report)["error_code"] == "inspection_rejected"

    def test_after_rework_request_shows_the_rework_and_offers_no_tool(
        self, tmp_path: Path, api: FakeJudgmentModel, cases: list[DispositionCase]
    ) -> None:
        """재작업 뒤의 요청은 재작업의 호출과 두 번째 결과를 갖는다. 재작업을 주지 않는다."""
        _run(api, tmp_path / "eval", _config(Feature.DISPOSITION))
        sent = dict(zip([case.case_id for case in cases], _bodies(api), strict=True))
        case = self._kind(cases, DispositionKind.REWORK_STILL_MISSING)
        body = sent[case.case_id]
        state = runtime_state(body)
        assert "tools" not in body and state["rework_runs"] == 1
        assert state["error_code"] == "check_failed" and len(body["messages"]) == 10
        calls = [m["tool_calls"][0]["function"] for m in body["messages"] if "tool_calls" in m]
        assert [call["name"] for call in calls] == [
            "prepare_layout",
            "calibrate_board",
            "execute_cell",
            "rework_cell",
        ]
        first = ExecutionResult.model_validate_json(body["messages"][7]["content"])
        assert isinstance(first.summary, InspectedExecutionSummary)
        assert first.summary.inspection is not None
        reworked = [part.part_id for part in first.summary.inspection.parts]
        assert json.loads(calls[3]["arguments"]) == {"part_ids": reworked}
        second = ExecutionResult.model_validate_json(_last_result(body))
        assert second.artifacts[0].id == state["check_id"] != first.artifacts[0].id

    def test_correct_answers_pass_every_criterion(
        self, tmp_path: Path, api: FakeJudgmentModel
    ) -> None:
        """정답대로 답하면 모든 기준을 통과한다."""
        out_dir = tmp_path / "eval"
        _, report = _run(api, out_dir, _config(Feature.DISPOSITION))
        assert report.passed
        assert _metrics(report) == {
            "format_valid": (100, 100, True),
            "release_match": (30, 30, True),
            "rework_match": (32, 32, True),
            "reject_match": (38, 38, True),
            "false_release": (0, 70, True),
            "wrong_rework_parts": (0, 70, None),
            "needless_rework": (0, 38, None),
            "after_rework_match": (14, 14, None),
        }
        answers = Counter((result.answer or "").split(":")[0] for result in _results(out_dir))
        assert answers == {"complete": 30, "rework_cell": 32, "abort": 38}
        assert all(result.status == ResponseStatus.VALID for result in _results(out_dir))

    def test_model_that_always_releases_fails_the_false_release_criterion(
        self, tmp_path: Path, api: FakeJudgmentModel
    ) -> None:
        """출하할 수 없는 실행의 완료는 실행 프로그램이 거부한다. 형식은 맞고 판단이 틀린 답이다."""
        out_dir = tmp_path / "eval"
        _, report = _run(api, out_dir, _config(Feature.DISPOSITION), responder=lambda _: COMPLETE)
        metrics = _metrics(report)
        assert metrics["format_valid"] == (100, 100, True)
        assert metrics["release_match"] == (30, 30, True)
        assert metrics["false_release"] == (70, 70, False)
        assert metrics["rework_match"] == (0, 32, False)
        assert metrics["reject_match"] == (0, 38, False)
        statuses = Counter(result.status for result in _results(out_dir))
        assert statuses == {ResponseStatus.VALID: 30, ResponseStatus.INVALID: 70}
        assert not report.passed

    def test_model_that_always_rejects_releases_and_reworks_nothing(
        self, tmp_path: Path, api: FakeJudgmentModel
    ) -> None:
        """항상 중단하면 불량의 정답만 맞는다. 통과한 실행의 중단은 자발적 중단이다."""

        def reject(body: dict[str, Any]) -> Reply:
            code = runtime_state(body)["error_code"] or "agent_aborted"
            return Reply(content=json.dumps({"decision": "abort", "reason_code": code}))

        _, report = _run(api, tmp_path / "eval", _config(Feature.DISPOSITION), responder=reject)
        metrics = _metrics(report)
        assert metrics["format_valid"] == (100, 100, True)
        assert metrics["release_match"] == (0, 30, False)
        assert metrics["rework_match"] == (0, 32, False)
        assert metrics["reject_match"] == (38, 38, True)
        assert metrics["false_release"] == (0, 70, True)

    def test_voluntary_abort_is_not_a_rejection(
        self, tmp_path: Path, api: FakeJudgmentModel
    ) -> None:
        """`agent_aborted`의 중단은 계약과 같다. 불량의 정답은 현재 오류 코드의 중단이다."""
        abort = Reply(content='{"decision":"abort","reason_code":"agent_aborted"}')
        _, report = _run(
            api, tmp_path / "eval", _config(Feature.DISPOSITION), responder=lambda _: abort
        )
        metrics = _metrics(report)
        assert metrics["format_valid"] == (100, 100, True)
        assert metrics["reject_match"] == (0, 38, False)

    def test_another_part_list_is_rejected_by_the_runtime(
        self, tmp_path: Path, api: FakeJudgmentModel, cases: list[DispositionCase]
    ) -> None:
        """실행 프로그램은 재작업 대상이 아닌 부품 목록을 거부한다. 정답으로 세지 않는다."""
        correct = correct_replies(cases)

        def respond(body: dict[str, Any]) -> Reply:
            reply = correct(body)
            if reply.tool != "rework_cell":
                return reply
            return Reply(tool="rework_cell", arguments={"part_ids": ["P999"]})

        out_dir = tmp_path / "eval"
        _, report = _run(api, out_dir, _config(Feature.DISPOSITION), responder=respond)
        metrics = _metrics(report)
        assert metrics["format_valid"] == (100, 100, True)
        assert metrics["rework_match"] == (0, 32, False)
        assert metrics["wrong_rework_parts"] == (32, 70, None)
        assert metrics["release_match"] == (30, 30, True)
        failures = Counter(result.failure for result in _results(out_dir))
        assert failures == {None: 68, "invalid_rework_parts": 32}

    def test_rework_of_an_assembly_with_a_covered_mismatch_is_needless(
        self, tmp_path: Path, api: FakeJudgmentModel, cases: list[DispositionCase]
    ) -> None:
        """실행 프로그램은 빈 칸의 재작업을 받는다. 덮인 불일치가 남으면 정답은 불량이다."""
        correct = correct_replies(cases)

        def respond(body: dict[str, Any]) -> Reply:
            reply = correct(body)
            inspection = json.loads(_last_result(body))["summary"]["inspection"]
            if reply.tool is not None or inspection is None or "tools" not in body:
                return reply
            empty = [part["part_id"] for part in inspection["parts"] if part["empty"]]
            if not empty or inspection["unexpected_cell_count"]:
                return reply
            return Reply(tool="rework_cell", arguments={"part_ids": empty})

        out_dir = tmp_path / "eval"
        _, report = _run(api, out_dir, _config(Feature.DISPOSITION), responder=respond)
        metrics = _metrics(report)
        assert metrics["format_valid"] == (100, 100, True)
        assert metrics["needless_rework"] == (8, 38, None)
        assert metrics["reject_match"] == (30, 38, False)
        assert metrics["wrong_rework_parts"] == (0, 70, None)

    @pytest.mark.parametrize(
        ("metric", "wrong", "expected"),
        [
            ("release_match", 1, (29, 30, True)),
            ("release_match", 2, (28, 30, False)),
            ("rework_match", 3, (29, 32, True)),
            ("rework_match", 4, (28, 32, False)),
            ("reject_match", 3, (35, 38, True)),
            ("reject_match", 4, (34, 38, False)),
        ],
    )
    def test_match_criteria_allow_a_small_number_of_wrong_answers(
        self,
        tmp_path: Path,
        api: FakeJudgmentModel,
        cases: list[DispositionCase],
        metric: str,
        wrong: int,
        expected: tuple[int, int, bool],
    ) -> None:
        """출하는 95%, 재작업과 불량은 90%가 기준이다. 틀린 답은 자발적 중단으로 만든다."""
        correct = correct_replies(cases)
        target = {"release_match": "complete", "rework_match": "rework", "reject_match": "abort"}
        count = 0

        def respond(body: dict[str, Any]) -> Reply:
            nonlocal count
            reply = correct(body)
            kind = "rework" if reply.tool else json.loads(reply.content or "{}")["decision"]
            if kind != target[metric]:
                return reply
            count += 1
            if count > wrong:
                return reply
            return Reply(content='{"decision":"abort","reason_code":"agent_aborted"}')

        _, report = _run(api, tmp_path / "eval", _config(Feature.DISPOSITION), responder=respond)
        assert _metrics(report)[metric] == expected

    @pytest.mark.parametrize(("released", "passed"), [(1, True), (2, False)])
    def test_false_release_criterion_allows_one_case(
        self,
        tmp_path: Path,
        api: FakeJudgmentModel,
        cases: list[DispositionCase],
        released: int,
        passed: bool,
    ) -> None:
        """출하할 수 없는 실행 70개 중 1개의 완료까지 기준을 통과한다."""
        correct = correct_replies(cases)
        count = 0

        def respond(body: dict[str, Any]) -> Reply:
            nonlocal count
            reply = correct(body)
            if reply.tool == "rework_cell":
                count += 1
                if count <= released:
                    return COMPLETE
            return reply

        _, report = _run(api, tmp_path / "eval", _config(Feature.DISPOSITION), responder=respond)
        assert _metrics(report)["false_release"] == (released, 70, passed)

    def test_free_text_is_outside_the_format(self, tmp_path: Path, api: FakeJudgmentModel) -> None:
        """자유 텍스트는 형식 위반이다. 형식의 기준에 못 미친다."""
        out_dir = tmp_path / "eval"
        _, report = _run(
            api, out_dir, _config(Feature.DISPOSITION, limit=4), responder=lambda _: FREE_TEXT
        )
        assert _metrics(report)["format_valid"] == (0, 4, False)
        assert {result.status for result in _results(out_dir)} == {ResponseStatus.INVALID}


class TestStopAndRepeats:
    """응답 없는 요청의 처리, 반복 실행과 모델 목록 확인을 검사한다."""

    def test_five_failures_in_a_row_stop_the_evaluation(
        self, tmp_path: Path, api: FakeJudgmentModel
    ) -> None:
        """HTTP 오류가 5번 이어지면 남은 요청을 보내지 않는다. 보고는 미통과다."""
        out_dir = tmp_path / "eval"
        _, report = _run(
            api,
            out_dir,
            _config(Feature.INTAKE, limit=12),
            responder=lambda body: Reply(status=503),
        )
        assert len(api.model_posts) == MAX_CONSECUTIVE_FAILURES == 5
        assert report.stopped and not report.passed
        assert (report.completed_requests, report.request_count) == (5, 12)
        results = _results(out_dir)
        assert all(result.status == ResponseStatus.NO_RESPONSE for result in results)
        assert all(result.failure == "model_error" for result in results)
        assert FAKE_JUDGMENT_KEY not in (out_dir / AUDIT_FILE).read_text()

    @pytest.mark.parametrize("feature", list(Feature))
    def test_separate_failures_do_not_stop_the_evaluation(
        self, tmp_path: Path, api: FakeJudgmentModel, feature: Feature
    ) -> None:
        """이어지지 않은 오류는 세지 않는다. 다른 응답 모델도 응답 없음이다."""
        cases = select_cases(build_cases(feature, seed=0), limit=12)
        correct = correct_replies(cases)
        count = 0

        def respond(body: dict[str, Any]) -> Reply:
            nonlocal count
            count += 1
            if count % 5 == 0:
                return correct(body)
            if count % 2 == 0:
                return Reply(tool="reject_request", arguments={}, model="nvidia/Nemotron-other")
            return Reply(status=503)

        out_dir = tmp_path / "eval"
        _, report = _run(api, out_dir, _config(feature, limit=12), responder=respond)
        assert not report.stopped and report.completed_requests == 12
        statuses = Counter(result.status for result in _results(out_dir))
        assert statuses == {ResponseStatus.NO_RESPONSE: 10, ResponseStatus.VALID: 2}
        # 처분 기능은 형식의 지표에 다른 이름을 쓴다.
        name = "format_valid" if feature == Feature.DISPOSITION else "contract_valid"
        assert _metrics(report)[name] == (2, 12, False)
        assert report.usage.prompt_tokens == 200

    def test_repeated_cases_with_the_same_answers_agree(
        self, tmp_path: Path, api: FakeJudgmentModel
    ) -> None:
        """반복 사례 20개를 3번 실행한다. 답이 같으면 모두 일치다. 지표는 첫 실행만 쓴다."""
        out_dir = tmp_path / "eval"
        prepared, report = _run(api, out_dir, _config(Feature.INTAKE, limit=24, repeats=3))
        assert report.request_count == len(api.model_posts) == 24 + 2 * CONSISTENCY_CASES
        assert report.consistency is not None
        assert report.consistency.model_dump() == {"cases": 20, "runs": 3, "agreed": 20}
        results = _results(out_dir)
        assert Counter(result.run for result in results) == {1: 24, 2: 20, 3: 20}
        repeated = [case.case_id for case in select_cases(prepared.cases, limit=20)]
        assert [result.case_id for result in results[24:44]] == repeated
        assert _metrics(report)["contract_valid"] == (64, 64, True)
        assert sum(total for _, total, _ in list(_metrics(report).values())[1:3]) == 12

    def test_another_answer_in_a_later_run_is_a_disagreement(
        self, tmp_path: Path, api: FakeJudgmentModel
    ) -> None:
        """나중 실행의 답이 다르면 그 사례는 불일치다. 첫 실행의 지표는 그대로다."""
        cases = select_cases(build_cases(Feature.INTAKE, seed=0), limit=24)
        correct = correct_replies(cases)
        seen: Counter[str] = Counter()
        kit = _texts(cases, IntakeGroup.KIT)[0].text
        board = _texts(cases, IntakeGroup.BOARD)[0].text

        def respond(body: dict[str, Any]) -> Reply:
            text = body["messages"][-1]["content"]
            seen[text] += 1
            if text == kit and seen[text] == 3:
                return Reply(tool="reject_request", arguments={"reason": "ambiguous_request"})
            if text == board and seen[text] == 2:
                return FREE_TEXT
            return correct(body)

        _, report = _run(
            api, tmp_path / "eval", _config(Feature.INTAKE, limit=24, repeats=3), responder=respond
        )
        assert report.consistency is not None and report.consistency.agreed == 18
        assert _metrics(report)["kit_order_match"] == (6, 6, True)
        assert _metrics(report)["contract_valid"] == (63, 64, False)

    def test_run_without_a_response_is_not_compared(
        self, tmp_path: Path, api: FakeJudgmentModel
    ) -> None:
        """응답 없는 실행이 있는 사례는 일치에서 뺀다. HTTP 오류는 모델의 답이 아니다."""
        cases = select_cases(build_cases(Feature.INTAKE, seed=0), limit=8)
        correct = correct_replies(cases)
        seen: Counter[str] = Counter()
        failing = {case.text for case in cases[:2] if isinstance(case, IntakeCase)}

        def respond(body: dict[str, Any]) -> Reply:
            text = body["messages"][-1]["content"]
            seen[text] += 1
            if text in failing and seen[text] == 2:
                return Reply(status=503)
            return correct(body)

        _, report = _run(
            api, tmp_path / "eval", _config(Feature.INTAKE, limit=8, repeats=2), responder=respond
        )
        assert report.consistency is not None
        assert report.consistency.model_dump() == {"cases": 6, "runs": 2, "agreed": 6}
        assert report.completed_requests == 16 and not report.stopped

    def test_model_outside_the_catalog_sends_no_inference(
        self, tmp_path: Path, api: FakeJudgmentModel
    ) -> None:
        """계정 목록에 모델이 없으면 모델 요청하지 않는다. 보고를 쓰지 않는다."""
        out_dir = tmp_path / "eval"
        prepared = prepare_eval(_config(Feature.INTAKE, limit=3), out_dir=out_dir, dry_run=False)
        api.models = [ULTRA]
        with pytest.raises(ValueError, match="not available"):
            run_eval(
                prepared,
                out_dir=out_dir,
                settings=TokenFactorySettings(),  # pyright: ignore[reportCallIssue]
            )
        assert not api.model_posts and not (out_dir / REPORT_FILE).exists()
        assert all(client.is_closed for client in api.clients)


class TestEvalJudgmentCLI:
    """`eval-judgment`의 종료 코드, 표준 출력과 기록 파일을 검사한다."""

    def _pairs(self, root: Path) -> Path:
        directory = root / "pairs" / "kit-ok"
        directory.mkdir(parents=True)
        (directory / "order.json").write_text(KIT.model_dump_json())
        (directory / "layout.json").write_text(KIT_LAYOUT.model_dump_json())
        return root / "pairs"

    def test_dry_run_needs_no_key_and_sends_no_request(
        self, tmp_path: Path, api: FakeJudgmentModel, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """`--dry-run`은 인증 키 없이 사례와 비용 어림을 쓴다. exit 0이다."""
        monkeypatch.delenv("NEBIUS_API_KEY")
        out_dir = tmp_path / "eval"
        result = runner.invoke(
            app,
            ["eval-judgment", "--feature", "intake", "--model", ULTRA, "-o", str(out_dir)]
            + ["--dry-run", "--repeats", "3"],
        )
        assert result.exit_code == 0, result.output
        assert "cases=130 requests=170 " in result.stdout
        assert "expected_usd=0." in result.stdout and "cases_sha256=" in result.stdout
        assert sorted(path.name for path in out_dir.iterdir()) == [CASES_FILE, EVAL_FILE]
        assert not api.requests

    def test_saved_plans_are_used_for_the_review_feature(
        self, tmp_path: Path, api: FakeJudgmentModel
    ) -> None:
        """`--data`의 계획은 검토 사례에 들어간다. 평가 조건은 쌍의 수만 기록한다."""
        out_dir = tmp_path / "eval"
        result = runner.invoke(
            app,
            ["eval-judgment", "--feature", "review", "--model", DEFAULT_NEMOTRON_MODEL]
            + ["-o", str(out_dir), "--data", str(self._pairs(tmp_path)), "--dry-run"],
        )
        assert result.exit_code == 0, result.output
        manifest = EvalManifest.model_validate_json((out_dir / EVAL_FILE).read_text())
        assert manifest.saved_pairs == 1 and manifest.case_count == 200
        first = json.loads((out_dir / CASES_FILE).read_text().splitlines()[0])
        assert first["saved_name"] == "kit-ok"

    @pytest.mark.parametrize(
        ("feature", "options", "disposition", "prompt"),
        [
            ("review", [], True, DISPOSITION_PROMPT),
            ("review", ["--no-disposition"], False, REVIEW_SYSTEM_PROMPT),
            ("coordination", [], True, DISPOSITION_PROMPT),
            ("disposition", ["--no-disposition"], True, DISPOSITION_PROMPT),
            ("intake", [], False, INTAKE_SYSTEM_PROMPT),
        ],
    )
    def test_disposition_option_selects_the_prompt(
        self,
        tmp_path: Path,
        api: FakeJudgmentModel,
        feature: str,
        options: list[str],
        disposition: bool,
        prompt: str,
    ) -> None:
        """계획 검토와 조율은 기본으로 처분을 켠 지시문을 쓴다. 처분은 항상 켠다. 접수는 끈다."""
        out_dir = tmp_path / "eval"
        result = runner.invoke(
            app,
            ["eval-judgment", "--feature", feature, "--model", DEFAULT_NEMOTRON_MODEL]
            + ["-o", str(out_dir), "--dry-run", "--limit", "4", *options],
        )
        assert result.exit_code == 0, result.output
        manifest = EvalManifest.model_validate_json((out_dir / EVAL_FILE).read_text())
        assert manifest.config.disposition is disposition
        assert manifest.system_prompt == prompt and not api.requests

    def test_disposition_run_reports_each_metric(
        self, tmp_path: Path, api: FakeJudgmentModel
    ) -> None:
        """처분 평가는 사례 100개를 실행하고 지표마다 1줄을 쓴다. 정답이면 exit 0이다."""
        api.responder = correct_replies(build_cases(Feature.DISPOSITION, seed=0))
        out_dir = tmp_path / "eval"
        result = runner.invoke(
            app,
            ["eval-judgment", "--feature", "disposition", "--model", DEFAULT_NEMOTRON_MODEL]
            + ["-o", str(out_dir)],
        )
        assert result.exit_code == 0, result.output
        assert "cases=100 requests=100 " in result.stdout
        assert "format_valid=100/100 criterion=>= 99/100 pass" in result.stdout
        assert "rework_match=32/32 criterion=>= 90/100 pass" in result.stdout
        assert "false_release=0/70 criterion=<= 2/100 pass" in result.stdout
        assert "passed=True" in result.stdout

    def test_passed_criteria_are_exit_0(self, tmp_path: Path, api: FakeJudgmentModel) -> None:
        """기준을 통과하면 exit 0이다. `--model`의 모델로 요청한다."""
        api.models = [DEFAULT_NEMOTRON_MODEL, ULTRA]
        api.responder = correct_replies(select_cases(build_cases(Feature.INTAKE, seed=0), limit=6))
        out_dir = tmp_path / "eval"
        result = runner.invoke(
            app,
            ["eval-judgment", "--feature", "intake", "--model", ULTRA, "-o", str(out_dir)]
            + ["--limit", "6"],
        )
        assert result.exit_code == 0, result.output
        assert "kit_order_match=2/2 criterion=>= 38/40 pass" in result.stdout
        assert "korean_match=1/1 criterion=- -" in result.stdout
        assert "requests=6/6 stopped=False" in result.stdout and "passed=True" in result.stdout
        assert {body["model"] for body in _bodies(api)} == {ULTRA}
        report = EvalReport.model_validate_json((out_dir / REPORT_FILE).read_text())
        assert report.passed and report.model == ULTRA
        assert all(client.is_closed for client in api.clients)

    def test_failed_criterion_is_exit_1(self, tmp_path: Path, api: FakeJudgmentModel) -> None:
        """기준에 못 미치면 exit 1이다. 보고는 쓴다."""
        api.responder = lambda body: Reply(
            tool="reject_request", arguments={"reason": "ambiguous_request"}
        )
        out_dir = tmp_path / "eval"
        result = runner.invoke(
            app,
            ["eval-judgment", "--feature", "intake", "--model", DEFAULT_NEMOTRON_MODEL]
            + ["-o", str(out_dir), "--limit", "6", "--seed", "0"],
        )
        assert result.exit_code == 1, result.output
        assert "kit_order_match=0/2 criterion=>= 38/40 fail" in result.stdout
        assert "passed=False" in result.stdout and (out_dir / REPORT_FILE).exists()

    def test_review_run_with_repeats_reports_consistency(
        self, tmp_path: Path, api: FakeJudgmentModel
    ) -> None:
        """검토 평가도 같은 명령으로 실행한다. 반복하면 일치 수를 쓴다."""
        api.responder = correct_replies(select_cases(build_cases(Feature.REVIEW, seed=2), limit=6))
        out_dir = tmp_path / "eval"
        result = runner.invoke(
            app,
            ["eval-judgment", "--feature", "review", "--model", DEFAULT_NEMOTRON_MODEL]
            + ["-o", str(out_dir), "--limit", "6", "--seed", "2", "--repeats", "2"],
        )
        assert result.exit_code == 0, result.output
        assert "consistency=6/6 runs=2" in result.stdout
        assert len(api.model_posts) == 12

    def test_unlisted_model_is_exit_1_without_inference(
        self, tmp_path: Path, api: FakeJudgmentModel
    ) -> None:
        """계정 목록에 모델이 없으면 모델 요청하지 않는다."""
        api.models = []
        result = runner.invoke(
            app,
            ["eval-judgment", "--feature", "intake", "--model", DEFAULT_NEMOTRON_MODEL]
            + ["-o", str(tmp_path / "eval"), "--limit", "3"],
        )
        assert result.exit_code == 1 and not api.model_posts
        assert "model catalog" in result.stderr

    @pytest.mark.parametrize(
        "arguments",
        [
            ["--feature", "intake", "--data", "pairs", "--dry-run"],
            ["--feature", "review", "--data", "empty", "--dry-run"],
            ["--feature", "review", "--data", "missing", "--dry-run"],
            ["--feature", "review", "--data", "partial", "--dry-run"],
            ["--feature", "review", "--data", "overlap", "--dry-run"],
            ["--feature", "intake", "--repeats", "6", "--dry-run"],
            ["--feature", "intake", "--limit", "0", "--dry-run"],
            ["--feature", "diagnosis", "--dry-run"],
            ["--feature", "intake", "--model", "openai/other"],
        ],
    )
    def test_usage_error_is_exit_2_without_request(
        self, tmp_path: Path, api: FakeJudgmentModel, arguments: list[str]
    ) -> None:
        """옵션, 저장한 계획 또는 모델 ID가 틀리면 외부 요청 전에 거부한다."""
        self._pairs(tmp_path)
        (tmp_path / "empty").mkdir()
        # 주문만 있는 쌍과 두 부품이 겹친 계획의 쌍이다.
        (tmp_path / "partial" / "a").mkdir(parents=True)
        (tmp_path / "partial" / "a" / "order.json").write_text(KIT.model_dump_json())
        overlap = make_layout(
            make_part("P001", PartType.SQUARE_TILE, origin=(0, 0)),
            make_part("P002", PartType.RECT_TILE, origin=(0, 0)),
        )
        (tmp_path / "overlap" / "a").mkdir(parents=True)
        (tmp_path / "overlap" / "a" / "order.json").write_text(KIT.model_dump_json())
        (tmp_path / "overlap" / "a" / "layout.json").write_text(overlap.model_dump_json())
        out_dir = tmp_path / "eval"
        result = runner.invoke(
            app,
            ["eval-judgment", "--model", DEFAULT_NEMOTRON_MODEL, "-o", str(out_dir), *arguments],
        )
        assert result.exit_code == 2, result.output
        assert not api.requests and not out_dir.exists()

    def test_existing_output_is_exit_2(self, tmp_path: Path, api: FakeJudgmentModel) -> None:
        """기존 출력 디렉터리를 덮어쓰지 않는다."""
        out_dir = tmp_path / "eval"
        out_dir.mkdir()
        result = runner.invoke(
            app,
            ["eval-judgment", "--feature", "intake", "--model", DEFAULT_NEMOTRON_MODEL]
            + ["-o", str(out_dir), "--dry-run"],
        )
        assert result.exit_code == 2 and not list(out_dir.iterdir())

    def test_missing_key_is_exit_2_before_the_output(
        self, tmp_path: Path, api: FakeJudgmentModel, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """인증 키가 없으면 출력 디렉터리를 만들지 않는다."""
        monkeypatch.delenv("NEBIUS_API_KEY")
        out_dir = tmp_path / "eval"
        result = runner.invoke(
            app,
            ["eval-judgment", "--feature", "intake", "--model", DEFAULT_NEMOTRON_MODEL]
            + ["-o", str(out_dir)],
        )
        assert result.exit_code == 2 and not out_dir.exists() and not api.requests
