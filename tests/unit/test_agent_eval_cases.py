"""오프라인 평가 사례의 수, 정답과 결정성을 검사한다. 모델을 호출하지 않는다."""

import hashlib
import json
from collections import Counter
from collections.abc import Sequence
from pathlib import Path

import pytest
from pydantic import ValidationError

from pixel2place.agent.contracts import ReviewReason, ToolName
from pixel2place.agent.eval_cases import (
    COORDINATION_CASES_PER_STATE,
    INTAKE_BOARD_CASES,
    INTAKE_BOARD_MAX_PARTS,
    INTAKE_KIT_CASES,
    INTAKE_REJECTS_PER_REASON,
    REVIEW_CORRECT_CASES,
    REVIEW_LAST_ATTEMPT_CASES,
    REVIEW_WRONG_PER_REASON,
    CoordinationCase,
    CoordinationState,
    DispositionAction,
    DispositionCase,
    DispositionGroup,
    DispositionKind,
    EvalCase,
    Feature,
    IntakeCase,
    IntakeGroup,
    PartOutcome,
    PlaceRun,
    ReviewAction,
    ReviewCase,
    ReviewGroup,
    SavedPair,
    build_cases,
    build_coordination_cases,
    build_disposition_cases,
    build_intake_cases,
    build_review_cases,
    cases_jsonl,
    expected_disposition,
    expected_review,
    kit_orders,
    load_saved_pairs,
    order_key,
    place_reports,
    plan_mismatch,
    select_cases,
)
from pixel2place.agent.inspection import rework_targets
from pixel2place.agent.intake import MAX_TEXT_CHARS, RejectReason
from pixel2place.cell.check import CheckFailure
from pixel2place.cell.image_check import PartStatus
from pixel2place.feasibility.checker import check_layout
from pixel2place.layout.schema import Color, Layout, PartType, Rotation
from pixel2place.order import (
    Arrangement,
    BoardOrder,
    KitItem,
    KitOrder,
    OrderPart,
    order_instruction,
)
from tests.factories import make_layout, make_part

pytestmark = pytest.mark.unit

KIT = KitOrder(
    kind="kit",
    items=(KitItem(type=PartType.SQUARE_TILE, count=2), KitItem(type=PartType.RECT_TILE, count=2)),
    arrangement=Arrangement.TYPE_ROWS,
)
BOARD = BoardOrder(
    parts=(
        OrderPart(type=PartType.SQUARE_TILE, color=Color.BLUE, row=1, col=1),
        OrderPart(type=PartType.L_TILE, color=Color.ORANGE, row=3, col=4),
    )
)


def _kit_layout(*, rect_rows: tuple[int, int] = (5, 5), squares: int = 2) -> Layout:
    parts = [
        make_part(f"P00{index + 1}", PartType.SQUARE_TILE, origin=(1, 2 * index))
        for index in range(squares)
    ]
    parts += [
        make_part(f"P01{index}", PartType.RECT_TILE, origin=(row, 3 * index), color=Color.GRAY)
        for index, row in enumerate(rect_rows)
    ]
    return make_layout(*parts)


def _board_layout(*, square: tuple[int, int] = (1, 1), l_color: Color = Color.ORANGE) -> Layout:
    return make_layout(
        make_part("P001", PartType.SQUARE_TILE, origin=square),
        make_part("P002", PartType.L_TILE, origin=(3, 4), color=l_color),
    )


def _sha(cases: Sequence[EvalCase]) -> str:
    return hashlib.sha256(cases_jsonl(cases).encode()).hexdigest()


@pytest.fixture(scope="module")
def cases() -> list[ReviewCase]:
    """시드 0의 검토 사례. 사례는 바꿀 수 없는 값이라서 모듈에서 함께 쓴다."""
    return build_review_cases(seed=0)


def _write_pair(root: Path, name: str, order: KitOrder | BoardOrder, layout: Layout) -> None:
    directory = root / name
    directory.mkdir(parents=True)
    (directory / "order.json").write_text(order.model_dump_json())
    (directory / "layout.json").write_text(layout.model_dump_json())


class TestOrderKey:
    """뜻이 같은 주문의 비교를 검사한다."""

    def test_item_order_does_not_change_the_key(self) -> None:
        """키트 항목의 순서가 달라도 같은 주문이다."""
        swapped = KIT.model_copy(update={"items": tuple(reversed(KIT.items))})
        assert order_key(swapped) == order_key(KIT)

    def test_symmetric_rotation_does_not_change_the_key(self) -> None:
        """정사각을 돌려도 셀이 같다. 같은 주문이다."""
        rotated = BoardOrder(
            parts=(
                BOARD.parts[1],
                BOARD.parts[0].model_copy(update={"rotation_deg": Rotation.DEG_90}),
            )
        )
        assert order_key(rotated) == order_key(BOARD)

    @pytest.mark.parametrize(
        "change",
        [{"row": 2}, {"color": Color.GRAY}, {"type": PartType.RECT_TILE}],
    )
    def test_another_cell_color_or_type_changes_the_key(self, change: dict[str, object]) -> None:
        """위치, 색 또는 종류가 다르면 다른 주문이다."""
        other = BoardOrder(parts=(BOARD.parts[0].model_copy(update=change), BOARD.parts[1]))
        assert order_key(other) != order_key(BOARD)

    def test_another_count_changes_the_key(self) -> None:
        """키트 수량이 다르면 다른 주문이다. 보드 주문과도 다르다."""
        other = KIT.model_copy(update={"items": (KitItem(type=PartType.SQUARE_TILE, count=3),)})
        assert order_key(other) != order_key(KIT) and order_key(KIT) != order_key(BOARD)


class TestPlanMismatch:
    """주문과 계획의 결정적 비교를 검사한다."""

    @pytest.mark.parametrize(
        ("layout", "expected"),
        [
            (_kit_layout(), None),
            (_kit_layout(squares=3), ReviewReason.PART_SET_MISMATCH),
            (_kit_layout(rect_rows=(5, 6)), ReviewReason.ARRANGEMENT_MISMATCH),
            # 사유의 순서는 지시문과 같다. 부품 구성이 배치 방식보다 먼저다.
            (_kit_layout(rect_rows=(5, 6), squares=1), ReviewReason.PART_SET_MISMATCH),
        ],
    )
    def test_kit_order_compares_counts_then_arrangement(
        self, layout: Layout, expected: ReviewReason | None
    ) -> None:
        """키트 주문은 종류와 색의 수량을 먼저 본다. 그 뒤에 배치 방식을 본다."""
        assert plan_mismatch(KIT, layout=layout) == expected

    @pytest.mark.parametrize(
        ("layout", "expected"),
        [
            (_board_layout(), None),
            (_board_layout(square=(1, 2)), ReviewReason.POSITION_MISMATCH),
            (_board_layout(l_color=Color.WHITE), ReviewReason.PART_SET_MISMATCH),
            (_board_layout(square=(1, 2), l_color=Color.WHITE), ReviewReason.PART_SET_MISMATCH),
        ],
    )
    def test_board_order_compares_the_part_set_then_positions(
        self, layout: Layout, expected: ReviewReason | None
    ) -> None:
        """보드 주문은 부품 구성을 먼저 본다. 그 뒤에 셀을 본다."""
        assert plan_mismatch(BOARD, layout=layout) == expected

    @pytest.mark.parametrize(
        ("mismatch", "attempts", "expected"),
        [
            (None, 1, ReviewAction.PROCEED),
            (ReviewReason.PART_SET_MISMATCH, 1, ReviewAction.REPLAN),
            (ReviewReason.POSITION_MISMATCH, 1, ReviewAction.REPLAN),
            (ReviewReason.ARRANGEMENT_MISMATCH, 1, ReviewAction.REPLAN),
            (None, 2, ReviewAction.PROCEED),
            (ReviewReason.PART_SET_MISMATCH, 2, ReviewAction.REJECT),
            (ReviewReason.POSITION_MISMATCH, 2, ReviewAction.PROCEED),
            (ReviewReason.ARRANGEMENT_MISMATCH, 2, ReviewAction.PROCEED),
        ],
    )
    def test_expected_action_follows_the_review_rules(
        self, mismatch: ReviewReason | None, attempts: int, expected: ReviewAction
    ) -> None:
        """마지막 시도에서는 부품 구성이 다를 때만 거부한다."""
        assert expected_review(mismatch, layout_attempts=attempts) == expected


class TestIntakeCases:
    """접수 사례의 수, 정답과 결정성을 검사한다."""

    def test_groups_have_the_planned_sizes(self) -> None:
        """키트 40, 보드 30, 범위 밖 48과 한국어 12개를 만든다. ID는 겹치지 않는다."""
        cases = build_intake_cases(seed=0)
        counts = Counter(case.group for case in cases)
        assert counts == {
            IntakeGroup.KIT: INTAKE_KIT_CASES,
            IntakeGroup.BOARD: INTAKE_BOARD_CASES,
            IntakeGroup.OUT_OF_SCOPE: len(RejectReason) * INTAKE_REJECTS_PER_REASON,
            IntakeGroup.KOREAN: 12,
        }, counts
        assert len({case.case_id for case in cases}) == len(cases) == 130
        assert all(len(case.text) <= MAX_TEXT_CHARS for case in cases)

    def test_kit_cases_cover_every_quantity_combination(self) -> None:
        """키트 계약이 받는 24개 조합이 모두 있다. 정답은 계약을 통과한 주문이다."""
        cases = [case for case in build_intake_cases(seed=0) if case.group == IntakeGroup.KIT]
        keys = {order_key(case.order) for case in cases if case.order is not None}
        assert len(kit_orders()) == 24 and keys == {order_key(order) for order in kit_orders()}
        assert all(case.reject_reason is None for case in cases)

    def test_kit_orders_keep_each_part_type_in_one_row(self) -> None:
        """모든 조합에서 한 종류의 수량은 한 row에 들어간다. 직사각은 4개 이하다."""
        rects = {
            item.count
            for order in kit_orders()
            for item in order.items
            if item.type == PartType.RECT_TILE
        }
        assert rects == {1, 2, 3, 4}, rects

    def test_too_many_cases_include_the_part_type_limit(self) -> None:
        """부품 초과 문장 2개는 합계가 한도 안이고 직사각의 수량만 한도를 넘는다."""
        texts = [
            case.text
            for case in build_intake_cases(seed=0)
            if case.reject_reason == RejectReason.TOO_MANY_PARTS
            and case.group == IntakeGroup.OUT_OF_SCOPE
        ]
        assert "Next kit for the cell: 1 square and 5 rectangles." in texts, texts
        assert any(
            text in ("Kit order: 5 rectangular tiles.", "Kit order: 6 rectangular tiles.")
            for text in texts
        ), texts

    def test_board_cases_have_feasible_short_orders(self) -> None:
        """보드 정답은 배치 규칙을 지킨다. 부품 수는 문장 한도 이하다."""
        cases = [case for case in build_intake_cases(seed=0) if case.group == IntakeGroup.BOARD]
        for case in cases:
            assert isinstance(case.order, BoardOrder), case.case_id
            assert 1 <= case.order.part_count <= INTAKE_BOARD_MAX_PARTS, case.case_id
            assert order_instruction(case.order), case.case_id
            # 문장은 모든 부품의 위치를 쓴다.
            assert case.text.count("row") >= case.order.part_count, case.text

    def test_out_of_scope_cases_have_each_reason_eight_times(self) -> None:
        """범위 밖 요청은 사유마다 8개다. 정답 주문은 없다."""
        cases = [
            case for case in build_intake_cases(seed=0) if case.group == IntakeGroup.OUT_OF_SCOPE
        ]
        reasons = Counter(case.reject_reason for case in cases)
        assert reasons == dict.fromkeys(RejectReason, INTAKE_REJECTS_PER_REASON), reasons
        assert all(case.order is None for case in cases)
        # 사유를 번갈아 둔다. 앞의 6개가 모든 사유를 쓴다.
        assert {case.reject_reason for case in cases[: len(RejectReason)]} == set(RejectReason)

    def test_same_seed_makes_the_same_bytes(self) -> None:
        """같은 시드는 같은 SHA-256을 만든다. 다른 시드는 다른 사례를 만든다."""
        assert _sha(build_intake_cases(seed=3)) == _sha(build_intake_cases(seed=3))
        assert _sha(build_intake_cases(seed=3)) != _sha(build_intake_cases(seed=4))

    def test_case_needs_exactly_one_answer(self) -> None:
        """정답이 없거나 2개인 사례는 만들 수 없다."""
        with pytest.raises(ValidationError):
            IntakeCase(case_id="a", group=IntakeGroup.KIT, text="Text.")
        with pytest.raises(ValidationError):
            IntakeCase(
                case_id="a",
                group=IntakeGroup.KIT,
                text="Text.",
                order=KIT,
                reject_reason=RejectReason.TOO_MANY_PARTS,
            )

    def test_cases_round_trip_through_json(self) -> None:
        """기록한 사례 1줄은 같은 사례로 읽힌다."""
        cases = build_intake_cases(seed=0)
        lines = cases_jsonl(cases).splitlines()
        assert [IntakeCase.model_validate_json(line) for line in lines] == cases


class TestReviewCases:
    """검토 사례의 수, 정답과 결정성을 검사한다."""

    def test_groups_have_the_planned_sizes(self, cases: list[ReviewCase]) -> None:
        """올바른 계획 96, 틀린 계획 84와 마지막 시도 20개를 만든다."""
        counts = Counter(case.group for case in cases)
        assert counts == {
            ReviewGroup.CORRECT: REVIEW_CORRECT_CASES,
            ReviewGroup.WRONG: len(ReviewReason) * REVIEW_WRONG_PER_REASON,
            ReviewGroup.LAST_ATTEMPT: REVIEW_LAST_ATTEMPT_CASES,
        }, counts
        assert len({case.case_id for case in cases}) == len(cases) == 200

    def test_every_answer_agrees_with_the_deterministic_comparison(
        self, cases: list[ReviewCase]
    ) -> None:
        """사례의 사유와 정답 행동은 결정적 비교의 결과다. 모든 계획은 배치 규칙을 지킨다."""
        for case in cases:
            mismatch = plan_mismatch(case.order, layout=case.layout)
            assert case.mismatch == mismatch, case.case_id
            assert case.expected == expected_review(
                mismatch, layout_attempts=case.layout_attempts
            ), case.case_id
            assert case.goal == order_instruction(case.order), case.case_id
            assert check_layout(case.layout).feasible, case.case_id

    def test_correct_plans_match_and_wrong_plans_cover_each_reason(
        self, cases: list[ReviewCase]
    ) -> None:
        """올바른 계획은 사유가 없다. 틀린 계획은 사유마다 28개이고 재계획이 정답이다."""
        correct = [case for case in cases if case.group == ReviewGroup.CORRECT]
        wrong = [case for case in cases if case.group == ReviewGroup.WRONG]
        assert all(case.mismatch is None and case.layout_attempts == 1 for case in correct)
        assert {type(case.order) for case in correct} == {KitOrder, BoardOrder}
        reasons = Counter(case.mismatch for case in wrong)
        assert reasons == dict.fromkeys(ReviewReason, REVIEW_WRONG_PER_REASON), reasons
        assert all(case.expected == ReviewAction.REPLAN for case in wrong)
        assert {case.mismatch for case in wrong[: len(ReviewReason)]} == set(ReviewReason)

    def test_last_attempt_cases_reject_only_another_part_set(self, cases: list[ReviewCase]) -> None:
        """마지막 시도는 준비 시도가 2회다. 절반은 부품 구성이 달라서 거부가 정답이다."""
        last = [case for case in cases if case.group == ReviewGroup.LAST_ATTEMPT]
        assert all(case.layout_attempts == 2 for case in last)
        expected = Counter(case.expected for case in last)
        assert expected == {ReviewAction.PROCEED: 10, ReviewAction.REJECT: 10}, expected
        # 위치 또는 배치 방식만 다른 계획은 진행이 정답이다.
        minor = [
            case for case in last if case.mismatch not in (None, ReviewReason.PART_SET_MISMATCH)
        ]
        assert minor and all(case.expected == ReviewAction.PROCEED for case in minor)

    def test_last_attempt_cases_have_both_histories(self, cases: list[ReviewCase]) -> None:
        """이전 경로는 형식 오류 재시도와 재계획이다. 거부한 첫 계획은 그 사유로 주문과 다르다."""
        last = [case for case in cases if case.group == ReviewGroup.LAST_ATTEMPT]
        replanned = [case for case in last if case.rejected is not None]
        assert 0 < len(replanned) < len(last)
        for case in replanned:
            assert case.rejected is not None
            reason = plan_mismatch(case.order, layout=case.rejected.layout)
            assert reason == case.rejected.reason, case.case_id

    def test_same_seed_makes_the_same_bytes(self, cases: list[ReviewCase]) -> None:
        """같은 시드는 같은 SHA-256을 만든다. 다른 시드는 다른 사례를 만든다."""
        assert _sha(build_review_cases(seed=0)) == _sha(cases)
        assert _sha(build_review_cases(seed=1)) != _sha(cases)

    def test_cases_round_trip_through_json(self, cases: list[ReviewCase]) -> None:
        """기록한 사례 1줄은 같은 사례로 읽힌다."""
        lines = cases_jsonl(cases).splitlines()
        assert [ReviewCase.model_validate_json(line) for line in lines] == cases


class TestCoordinationCases:
    """조율 사례의 수, 정답 도구와 결정성."""

    def test_each_state_has_the_planned_cases_and_one_tool(self) -> None:
        """상태마다 같은 수의 사례가 있다. 정답은 그 상태가 허용하는 도구다."""
        cases = build_coordination_cases(seed=0)
        assert Counter(case.group for case in cases) == dict.fromkeys(
            CoordinationState, COORDINATION_CASES_PER_STATE
        )
        tools = {case.group: case.expected for case in cases}
        assert tools == {
            CoordinationState.CREATED: ToolName.PREPARE_LAYOUT,
            CoordinationState.LAYOUT_RETRYABLE: ToolName.PREPARE_LAYOUT,
            CoordinationState.CALIBRATED: ToolName.EXECUTE_CELL,
        }
        assert len({case.case_id for case in cases}) == len(cases)

    def test_only_calibrated_cases_have_a_prepared_layout(self) -> None:
        """준비의 앞과 준비 실패의 뒤에는 계획이 없다. 보정 뒤의 계획은 배치 규칙을 지킨다."""
        for case in build_coordination_cases(seed=0):
            if case.group == CoordinationState.CALIBRATED:
                assert case.layout is not None and check_layout(case.layout).feasible
            else:
                assert case.layout is None, case.case_id

    def test_same_seed_makes_the_same_bytes(self) -> None:
        """같은 시드는 같은 SHA-256을 만든다. 기록한 사례 1줄은 같은 사례로 읽힌다."""
        cases = build_coordination_cases(seed=0)
        assert _sha(build_cases(Feature.COORDINATION, seed=0)) == _sha(cases)
        assert _sha(build_coordination_cases(seed=1)) != _sha(cases)
        lines = cases_jsonl(cases).splitlines()
        assert [CoordinationCase.model_validate_json(line) for line in lines] == cases


LAYOUT = make_layout(
    make_part("P001", PartType.SQUARE_TILE, origin=(1, 1)),
    make_part("P002", PartType.RECT_TILE, origin=(3, 2), color=Color.GRAY),
    make_part("P003", PartType.SQUARE_TILE, origin=(5, 5), color=Color.WHITE),
)
PLACED, MISSING = PartOutcome.PLACED, PartOutcome.MISSING


class TestPlaceReports:
    """배치 결과에서 만든 두 보고서가 그 결과의 판정과 판독인지 검사한다."""

    def test_each_outcome_gives_its_check_and_reading(self) -> None:
        """부품의 결과마다 참값 판정의 통과, 카메라 판정의 상태와 빈 칸이 정해진다."""
        expected = {
            PartOutcome.PLACED: (True, PartStatus.MATCH, False),
            PartOutcome.SHADOWED: (True, PartStatus.SHADOWED, False),
            PartOutcome.MISSING: (False, PartStatus.MISMATCH, True),
            PartOutcome.WRONG_COLOR: (True, PartStatus.MISMATCH, False),
            PartOutcome.DISPLACED: (False, PartStatus.MISMATCH, True),
            PartOutcome.TURNED: (False, PartStatus.MATCH, False),
        }
        for outcome, (passed, status, empty) in expected.items():
            check, report = place_reports(LAYOUT, PlaceRun(outcomes=(outcome, PLACED, PLACED)))
            assert report is not None
            part, reading = check.parts[0], report.parts[0]
            assert (part.passed, reading.status, reading.empty) == (passed, status, empty), outcome
            assert part.part_id == reading.part_id == "P001" and part.sequence == 1
            assert check.passed is passed
            assert report.passed is (status is not PartStatus.MISMATCH)
        fallen, _ = place_reports(LAYOUT, PlaceRun(outcomes=(MISSING, PLACED, PLACED)))
        assert fallen.parts[0].fell and CheckFailure.FELL in fallen.parts[0].failures

    def test_unexpected_cell_fails_the_camera_check(self) -> None:
        """계획에 없는 점유 칸은 보고서에 있다. 카메라 판정은 실패다."""
        run = PlaceRun(outcomes=(PLACED, PLACED, PLACED), unexpected_cells=((7, 7),))
        check, report = place_reports(LAYOUT, run)
        assert check.passed and report is not None and not report.passed
        assert report.unexpected_cells == [(7, 7)]

    def test_run_without_a_camera_report_has_only_the_check(self) -> None:
        """카메라 판정의 보고서가 없는 배치는 참값 판정만 준다."""
        run = PlaceRun(outcomes=(PLACED, PLACED, PLACED), camera_report=False)
        check, report = place_reports(LAYOUT, run)
        assert check.passed and report is None

    def test_run_needs_one_outcome_for_each_part(self) -> None:
        """결과의 수가 부품의 수와 다르면 보고서를 만들지 않는다."""
        with pytest.raises(ValueError, match="one outcome for each part"):
            place_reports(LAYOUT, PlaceRun(outcomes=(PLACED,)))


class TestExpectedDisposition:
    """처분 지시문의 규칙이 정하는 정답."""

    @pytest.mark.parametrize(
        ("outcomes", "fields", "rework_runs", "expected", "parts"),
        [
            ((PLACED, PLACED, PLACED), {}, 0, DispositionAction.RELEASE, ()),
            ((PartOutcome.SHADOWED, PLACED, PLACED), {}, 0, DispositionAction.RELEASE, ()),
            ((PLACED, MISSING, PLACED), {}, 0, DispositionAction.REWORK, ("P002",)),
            ((MISSING, PLACED, MISSING), {}, 0, DispositionAction.REWORK, ("P001", "P003")),
            ((PLACED, MISSING, PLACED), {}, 1, DispositionAction.REJECT, ()),
            ((PartOutcome.WRONG_COLOR, PLACED, PLACED), {}, 0, DispositionAction.REJECT, ()),
            ((PartOutcome.WRONG_COLOR, MISSING, PLACED), {}, 0, DispositionAction.REJECT, ()),
            ((PartOutcome.TURNED, PLACED, PLACED), {}, 0, DispositionAction.REJECT, ()),
            (
                (PartOutcome.DISPLACED, PLACED, PLACED),
                {"unexpected_cells": ((1, 2),)},
                0,
                DispositionAction.REJECT,
                (),
            ),
            ((PLACED, PLACED, PLACED), {"camera_report": False}, 0, DispositionAction.REJECT, ()),
            ((PLACED, PLACED, PLACED), {}, 1, DispositionAction.RELEASE, ()),
        ],
    )
    def test_answer_follows_the_disposition_rules(
        self,
        outcomes: tuple[PartOutcome, ...],
        fields: dict[str, object],
        rework_runs: int,
        expected: DispositionAction,
        parts: tuple[str, ...],
    ) -> None:
        """두 판정이 통과하면 출하다. 빈 칸의 불일치만 있으면 재작업이다. 그 밖은 불량이다."""
        run = PlaceRun.model_validate({"outcomes": outcomes, **fields}, strict=False)
        check, report = place_reports(LAYOUT, run)
        answer = expected_disposition(
            check_passed=check.passed, report=report, rework_runs=rework_runs
        )
        assert answer == (expected, parts)


@pytest.fixture(scope="module")
def disposition_cases() -> list[DispositionCase]:
    """시드 0의 처분 사례. 사례는 바꿀 수 없는 값이라서 모듈에서 함께 쓴다."""
    return build_disposition_cases(seed=0)


class TestDispositionCases:
    """처분 사례의 수, 정답과 결정성."""

    @pytest.fixture
    def cases(self, disposition_cases: list[DispositionCase]) -> list[DispositionCase]:
        return disposition_cases

    def test_groups_and_kinds_have_the_planned_sizes(self, cases: list[DispositionCase]) -> None:
        """사례는 100개다. 묶음은 출하 24, 재작업 32, 불량 30과 재작업 뒤 14다."""
        assert Counter(case.group for case in cases) == {
            DispositionGroup.RELEASE: 24,
            DispositionGroup.REWORK: 32,
            DispositionGroup.REJECT: 30,
            DispositionGroup.AFTER_REWORK: 14,
        }
        assert set(Counter(case.kind for case in cases)) == set(DispositionKind)
        assert len({case.case_id for case in cases}) == len(cases) == 100

    def test_every_answer_agrees_with_the_disposition_rules(
        self, cases: list[DispositionCase]
    ) -> None:
        """정답은 사례의 마지막 배치에서 다시 계산한 값과 같다."""
        for case in cases:
            check, report = place_reports(case.layout, case.run)
            answer = expected_disposition(
                check_passed=check.passed, report=report, rework_runs=int(case.first is not None)
            )
            assert answer == (case.expected, case.expected_parts), case.case_id
            assert case.goal and check_layout(case.layout).feasible

    def test_first_pass_groups_have_their_answer(self, cases: list[DispositionCase]) -> None:
        """첫 배치 뒤의 묶음은 묶음의 이름이 정답이다. 그 사례에는 첫 배치의 기록이 없다."""
        for case in cases:
            if case.group == DispositionGroup.AFTER_REWORK:
                continue
            assert case.expected.value == case.group.value and case.first is None, case.case_id

    def test_rework_answers_are_the_runtime_targets(self, cases: list[DispositionCase]) -> None:
        """재작업의 정답 부품은 실행 프로그램의 재작업 대상이다. 모든 불일치 부품이 빈 칸이다."""
        reworked = [case for case in cases if case.expected == DispositionAction.REWORK]
        assert {len(case.expected_parts) for case in reworked} == {1, 2, 3}
        for case in reworked:
            _, report = place_reports(case.layout, case.run)
            assert report is not None and case.expected_parts == rework_targets(report)
            mismatched = [part for part in report.parts if part.status is PartStatus.MISMATCH]
            assert all(part.empty for part in mismatched) and not report.unexpected_cells
            assert len(case.expected_parts) < len(case.layout.parts), case.case_id

    def test_reject_cases_cover_each_reason_for_no_rework(
        self, cases: list[DispositionCase]
    ) -> None:
        """불량 사례는 덮인 불일치, 계획에 없는 칸, 없는 보고서와 자세만의 실패를 갖는다."""
        rejected = [case for case in cases if case.group == DispositionGroup.REJECT]
        assert Counter(case.kind for case in rejected) == {
            DispositionKind.WRONG_COLOR: 8,
            DispositionKind.MISSING_AND_WRONG_COLOR: 8,
            DispositionKind.DISPLACED: 6,
            DispositionKind.NO_CAMERA_REPORT: 4,
            DispositionKind.TURNED: 4,
        }
        for case in rejected:
            _, report = place_reports(case.layout, case.run)
            if case.kind == DispositionKind.NO_CAMERA_REPORT:
                assert report is None
            elif case.kind == DispositionKind.DISPLACED:
                assert report is not None and len(report.unexpected_cells) == 1
                used = {cell for part in case.layout.parts for cell in part.cells}
                assert report.unexpected_cells[0] not in used
            elif case.kind == DispositionKind.MISSING_AND_WRONG_COLOR:
                # 실행 프로그램은 이 재작업을 받는다. 지시문의 규칙은 불량이다.
                assert report is not None and rework_targets(report)

    def test_after_rework_cases_have_the_first_run(self, cases: list[DispositionCase]) -> None:
        """재작업 뒤의 사례는 재작업 대상이 있는 첫 배치를 갖는다. 정답은 출하 또는 불량이다."""
        after = [case for case in cases if case.group == DispositionGroup.AFTER_REWORK]
        assert Counter(case.expected for case in after) == {
            DispositionAction.RELEASE: 6,
            DispositionAction.REJECT: 8,
        }
        for case in after:
            assert case.first is not None
            _, first = place_reports(case.layout, case.first)
            assert first is not None and rework_targets(first), case.case_id

    def test_same_seed_makes_the_same_bytes(self, cases: list[DispositionCase]) -> None:
        """같은 시드는 같은 SHA-256을 만든다. 기록한 사례 1줄은 같은 사례로 읽힌다."""
        assert _sha(build_cases(Feature.DISPOSITION, seed=0)) == _sha(cases)
        assert _sha(build_disposition_cases(seed=1)) != _sha(cases)
        lines = cases_jsonl(cases).splitlines()
        assert [DispositionCase.model_validate_json(line) for line in lines] == cases

    def test_small_limit_takes_one_case_from_each_group(self, cases: list[DispositionCase]) -> None:
        """한도 4는 묶음마다 1개를 고른다. 묶음 안에서는 종류를 번갈아 둔다."""
        assert [case.group for case in select_cases(cases, limit=4)] == list(DispositionGroup)
        rework = [case.kind for case in cases if case.group == DispositionGroup.REWORK][:4]
        assert len(set(rework)) == 4, rework


class TestSavedPairs:
    """저장한 실제 계획을 읽고 사례에 넣는 경로를 검사한다."""

    def test_pairs_are_read_in_name_order(self, tmp_path: Path) -> None:
        """하위 디렉터리마다 주문과 레이아웃을 읽는다. 순서는 이름 순서다."""
        _write_pair(tmp_path, "b-board", BOARD, _board_layout())
        _write_pair(tmp_path, "a-kit", KIT, _kit_layout())
        (tmp_path / "notes.txt").write_text("ignored")
        pairs = load_saved_pairs(tmp_path)
        assert [pair.name for pair in pairs] == ["a-kit", "b-board"]
        assert pairs[0].order == KIT and pairs[1].layout == _board_layout()

    def test_empty_directory_is_rejected(self, tmp_path: Path) -> None:
        """쌍이 없는 디렉터리는 자료가 아니다."""
        with pytest.raises(ValueError, match="no saved pairs"):
            load_saved_pairs(tmp_path)

    def test_missing_or_invalid_file_is_rejected(self, tmp_path: Path) -> None:
        """파일이 없거나 형식이 틀리면 읽지 않는다."""
        (tmp_path / "a").mkdir()
        with pytest.raises(OSError):
            load_saved_pairs(tmp_path)
        (tmp_path / "a" / "order.json").write_text(json.dumps({"kind": "kit", "items": []}))
        (tmp_path / "a" / "layout.json").write_text(_kit_layout().model_dump_json())
        with pytest.raises(ValidationError):
            load_saved_pairs(tmp_path)

    def test_saved_plans_come_first_and_keep_their_names(self) -> None:
        """저장한 올바른 계획은 올바른 묶음의 앞에 있다. 배치 위반은 틀린 묶음에 있다."""
        saved = [
            SavedPair("kit-ok", KIT, _kit_layout()),
            SavedPair("kit-rows", KIT, _kit_layout(rect_rows=(5, 6))),
            SavedPair("board-ok", BOARD, _board_layout()),
        ]
        cases = build_review_cases(seed=0, saved=saved)
        correct = [case for case in cases if case.group == ReviewGroup.CORRECT]
        assert [case.saved_name for case in correct[:3]] == ["kit-ok", "board-ok", None]
        assert len(correct) == REVIEW_CORRECT_CASES
        violation = next(case for case in cases if case.saved_name == "kit-rows")
        assert violation.group == ReviewGroup.WRONG
        assert violation.mismatch == ReviewReason.ARRANGEMENT_MISMATCH
        reasons = Counter(case.mismatch for case in cases if case.group == ReviewGroup.WRONG)
        assert reasons[ReviewReason.ARRANGEMENT_MISMATCH] == REVIEW_WRONG_PER_REASON
        assert _sha(cases) == _sha(build_review_cases(seed=0, saved=saved))
        assert _sha(cases) != _sha(build_review_cases(seed=0))

    def test_same_order_and_plan_is_used_one_time(self) -> None:
        """같은 주문과 같은 계획의 쌍은 요청이 같다. 처음 1개만 쓴다."""
        saved = [SavedPair("first", KIT, _kit_layout()), SavedPair("second", KIT, _kit_layout())]
        names = [case.saved_name for case in build_review_cases(seed=0, saved=saved)]
        assert "first" in names and "second" not in names

    @pytest.mark.parametrize(
        ("order", "layout"),
        [
            # 두 부품이 겹친다. 준비 검사를 통과하지 못한다.
            (
                BOARD,
                make_layout(
                    make_part("P001", PartType.SQUARE_TILE, origin=(0, 0)),
                    make_part("P002", PartType.RECT_TILE, origin=(0, 0)),
                ),
            ),
            # 주문이 보드 밖에 부품을 둔다. 지시문이 없다.
            (
                BoardOrder(
                    parts=(OrderPart(type=PartType.RECT_TILE, color=Color.GRAY, row=0, col=7),)
                ),
                _board_layout(),
            ),
        ],
    )
    def test_invalid_saved_pair_is_rejected(self, order: BoardOrder, layout: Layout) -> None:
        """준비 검사를 통과하지 못한 계획과 지시문이 없는 주문은 사례가 아니다."""
        with pytest.raises(ValueError, match="Saved pair bad is not a valid plan"):
            build_review_cases(seed=0, saved=[SavedPair("bad", order, layout)])


class TestSelectCases:
    """사례를 줄일 때 모든 묶음을 쓰는지 검사한다."""

    def test_no_limit_keeps_the_order(self) -> None:
        """한도가 없거나 사례 수 이상이면 사례를 그대로 둔다."""
        cases = build_cases(Feature.INTAKE, seed=0)
        assert select_cases(cases, limit=None) == cases
        assert select_cases(cases, limit=len(cases)) == cases

    @pytest.mark.parametrize(
        ("feature", "groups"),
        [
            (Feature.INTAKE, [IntakeGroup.KIT, IntakeGroup.BOARD, IntakeGroup.OUT_OF_SCOPE]),
            (Feature.REVIEW, list(ReviewGroup)),
        ],
    )
    def test_small_limit_takes_one_case_from_each_group(
        self, feature: Feature, groups: list[object]
    ) -> None:
        """한도 3은 앞의 세 묶음에서 1개씩 고른다. 같은 한도는 같은 사례를 고른다."""
        cases = build_cases(feature, seed=0)
        picked = select_cases(cases, limit=3)
        assert [case.group for case in picked] == groups
        assert select_cases(cases, limit=3) == picked

    def test_limit_keeps_the_group_balance(self) -> None:
        """한도 20은 네 묶음에서 5개씩 고른다."""
        picked = select_cases(build_cases(Feature.INTAKE, seed=0), limit=20)
        assert Counter(case.group for case in picked) == dict.fromkeys(IntakeGroup, 5)
