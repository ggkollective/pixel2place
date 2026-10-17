"""실제 CLI와 HTTP Fake로 종료 코드와 고정 실행 조건을 검사한다."""

import json
import math
from collections.abc import Callable
from pathlib import Path
from typing import Any

import httpx2 as httpx
import pytest
from openai import OpenAI
from typer.testing import CliRunner

from pixel2place.agent import cell as agent_cell
from pixel2place.agent import shift
from pixel2place.agent.contracts import ErrorCode, ToolState
from pixel2place.agent.fixture import FixtureOffset, PartFaults, WrongColorFault
from pixel2place.agent.prompts import system_prompt
from pixel2place.agent.runtime_contracts import AgentReport, Disposition
from pixel2place.agent.shift import (
    SHIFT_REPORT_FILE,
    ShiftReport,
    load_shift,
    run_shift,
)
from pixel2place.agent.tools import ToolRun
from pixel2place.cell.config import CellConfig
from pixel2place.cell.scene import (
    BoardOffset,
    CameraOffset,
    SceneFaults,
    SimScene,
    WrongColor,
    displace_board,
    placement_of,
)
from pixel2place.cli import app
from pixel2place.config import (
    DEFAULT_JUDGMENT_MODEL,
    DEFAULT_NEMOTRON_MODEL,
    Settings,
    SimSettings,
    TokenFactorySettings,
)
from pixel2place.generation.synthetic import COLOR_RGB
from pixel2place.layout.schema import Board, Color, Layout, PartType
from pixel2place.order import BoardOrder, order_instruction, order_layout, parse_order
from pixel2place.reasoner import nemotron
from tests.factories import make_layout, make_part, make_planner_response_json
from tests.fakes.fake_agent_model import FAKE_AGENT_KEY, FAKE_REASONING, FakeAgentModel

pytestmark = [pytest.mark.unit, pytest.mark.usefixtures("fake_isaac")]
runner = CliRunner()
GOAL = "Place all four parts on the board. Keep their colors."
EXAMPLES = Path(__file__).parents[2] / "examples"
BOARD_ORDER = {
    "kind": "board",
    "parts": [
        {"type": "square_tile", "color": "blue", "row": 1, "col": 1},
        {"type": "rect_tile", "color": "gray", "row": 3, "col": 5},
    ],
}
KIT_ORDER = {
    "kind": "kit",
    "items": [{"type": "square_tile", "count": 2}, {"type": "rect_tile", "count": 1}],
    "arrangement": "type_rows",
}
SHIFTED = {"x_mm": 3, "y_mm": 5.0, "yaw_deg": 1.5}
THREE_ORDERS: list[dict[str, object]] = [
    {"name": "first", "order": "../orders/board.json"},
    {"name": "second", "order": "../orders/kit.json", "fixture": SHIFTED},
    {"name": "third", "order": "../orders/kit.json"},
]
WRONG_COLOR_SCENE = SimScene(
    faults=SceneFaults(
        wrong_color=(WrongColor(sequence=1, color=Color.ORANGE, color_rgb=COLOR_RGB[Color.ORANGE]),)
    )
)
# Fake 모델은 실행마다 모델 요청 4개를 받는다.
REQUESTS_PER_RUN = 4


def write_shift(root: Path, items: list[dict[str, object]]) -> Path:
    """주문 파일 2개와 교대 파일을 쓴다. 주문의 경로는 교대 파일 기준의 상대 경로다."""
    orders = root / "orders"
    orders.mkdir()
    (orders / "board.json").write_text(json.dumps(BOARD_ORDER))
    (orders / "kit.json").write_text(json.dumps(KIT_ORDER))
    path = root / "shifts" / "shift.json"
    path.parent.mkdir()
    path.write_text(json.dumps({"items": items}))
    return path


def read_shift_report(out_dir: Path) -> ShiftReport:
    """저장한 교대 보고서를 계약 모델로 다시 읽는다."""
    return ShiftReport.model_validate_json((out_dir / SHIFT_REPORT_FILE).read_text())


def answer_orders(api: FakeAgentModel) -> Callable[[httpx.Request], None]:
    """Cosmos Fake가 요청의 지시문에 맞는 계획을 답하게 한다.

    돌려준 함수는 `on_request`에 둔 함수다. 다른 동작을 더하는 테스트가 먼저 부른다.
    """
    board = parse_order(json.dumps(BOARD_ORDER))
    assert isinstance(board, BoardOrder)
    kit_layout = make_layout(
        make_part("P001", PartType.SQUARE_TILE, origin=(0, 0)),
        make_part("P002", PartType.SQUARE_TILE, origin=(0, 2)),
        make_part("P003", PartType.RECT_TILE, origin=(2, 0), color=Color.GRAY),
    )
    plans = {
        order_instruction(board): make_planner_response_json(order_layout(board)),
        order_instruction(parse_order(json.dumps(KIT_ORDER))): make_planner_response_json(
            kit_layout
        ),
    }

    def select(request: httpx.Request) -> None:
        if request.url.host == "cosmos.invalid":
            api.cosmos_response = plans[json.loads(request.content)["messages"][1]["content"]]

    api.on_request = select
    return select


@pytest.fixture
def api(monkeypatch: pytest.MonkeyPatch, valid_layout: Layout) -> FakeAgentModel:
    """환경은 테스트 값으로 고정한다. SDK의 HTTP 클라이언트만 대체한다."""
    fake = FakeAgentModel()
    fake.cosmos_response = make_planner_response_json(valid_layout)
    monkeypatch.setenv("NEBIUS_API_KEY", FAKE_AGENT_KEY)
    monkeypatch.setenv("NEBIUS_BASE_URL", "https://api.tokenfactory.nebius.com/v1/")
    monkeypatch.setenv("REASONER_BASE_URL", "https://cosmos.invalid/v1/")
    monkeypatch.setenv("COSMOS_REASONER_MODEL", "fake-cosmos")
    monkeypatch.setenv("REASONER_MAX_RETRIES", "3")
    monkeypatch.setenv("REASONER_TIMEOUT_S", "5")
    monkeypatch.setenv("NEMOTRON_TIMEOUT_S", "5")
    monkeypatch.setenv("NEMOTRON_MODEL", "nvidia/Nemotron-other")
    monkeypatch.setattr(nemotron, "DefaultHttpxClient", fake.make_client)

    def cosmos_client(**kwargs: Any) -> OpenAI:
        return OpenAI(**kwargs, http_client=fake.make_client())

    monkeypatch.setattr(agent_cell, "OpenAI", cosmos_client)
    return fake


class TestAgentCellCLI:
    """CLI 입력 검사와 완료·중단 결과를 검사한다."""

    @pytest.mark.parametrize("offset", [False, True])
    def test_cli_completes_nominal_and_offset_scene(
        self,
        tmp_path: Path,
        api: FakeAgentModel,
        offset: bool,
    ) -> None:
        """공칭·오차 장면을 캡처하고 보정한 뒤 완료하면 exit 0이다."""
        args = ["agent-cell", GOAL, "-o", str(tmp_path / "run")]
        if offset:
            cell = CellConfig()
            scene = SimScene(
                board=displace_board(
                    cell, Board(), offset=BoardOffset(x=-0.002, y=0.004, yaw=math.radians(-1))
                ),
                camera=CameraOffset(
                    position=(-0.006, 0.009, -0.004),
                    rotation=(math.radians(-0.7), math.radians(0.9), math.radians(-1)),
                ),
            )
            scene_path, cell_path = tmp_path / "scene.json", tmp_path / "cell.json"
            scene_path.write_text(scene.model_dump_json())
            cell_path.write_text(cell.model_dump_json())
            args += ["--scene", str(scene_path), "--cell", str(cell_path)]
        result = runner.invoke(app, args)
        assert result.exit_code == 0, result.output
        root = tmp_path / "run"
        report = AgentReport.model_validate_json((root / "agent_report.json").read_text())
        assert report.check_passed and report.calls.nemotron == 4
        run = ToolRun.model_validate_json((root / "run.json").read_text())
        assert run.agent is not None and run.agent.model == DEFAULT_NEMOTRON_MODEL
        # 계획 검토와 처분은 기본으로 켠다. `layout_ready`와 `checked`의 요청을 판단 모델로 보낸다.
        assert run.agent.review and run.agent.review_model == DEFAULT_JUDGMENT_MODEL
        assert run.agent.disposition and run.agent.disposition_model == DEFAULT_JUDGMENT_MODEL
        models = [json.loads(request.content)["model"] for request in api.model_posts]
        assert models == [
            DEFAULT_NEMOTRON_MODEL,
            DEFAULT_JUDGMENT_MODEL,
            DEFAULT_NEMOTRON_MODEL,
            DEFAULT_JUDGMENT_MODEL,
        ]
        assert run.agent.budget.cosmos_timeout_s == 900
        cosmos_request = next(r for r in api.requests if r.url.host == "cosmos.invalid")
        body = json.loads(cosmos_request.content)
        assert body["messages"][1]["content"] == GOAL and body["max_tokens"] == 4096
        assert cosmos_request.extensions["timeout"]["read"] == 900
        assert all(c.is_closed for c in api.clients)
        assert FAKE_AGENT_KEY not in result.output
        assert FAKE_REASONING not in result.output

    def test_no_review_and_no_disposition_run_without_the_judgment_model(
        self, tmp_path: Path, api: FakeAgentModel
    ) -> None:
        """검토와 처분을 끄면 모든 요청을 조율 모델로 보낸다. 판단 모델이 없어도 된다."""
        api.models = [DEFAULT_NEMOTRON_MODEL]
        root = tmp_path / "run"
        result = runner.invoke(
            app, ["agent-cell", GOAL, "-o", str(root), "--no-review", "--no-disposition"]
        )
        assert result.exit_code == 0, result.output
        run = ToolRun.model_validate_json((root / "run.json").read_text())
        assert run.agent is not None and not run.agent.review and not run.agent.disposition
        assert run.agent.review_model is None and run.agent.disposition_model is None
        models = {json.loads(request.content)["model"] for request in api.model_posts}
        assert models == {DEFAULT_NEMOTRON_MODEL}
        summary = json.loads(json.loads(api.model_posts[1].content)["messages"][-1]["content"])
        assert "parts" not in summary["summary"]

    def test_no_review_option_keeps_the_judgment_model_for_the_disposition(
        self, tmp_path: Path, api: FakeAgentModel
    ) -> None:
        """`--no-review`는 검토만 끈다. `checked`의 요청은 판단 모델로 보낸다."""
        root = tmp_path / "run"
        result = runner.invoke(app, ["agent-cell", GOAL, "-o", str(root), "--no-review"])
        assert result.exit_code == 0, result.output
        run = ToolRun.model_validate_json((root / "run.json").read_text())
        assert run.agent is not None and not run.agent.review and run.agent.disposition
        models = [json.loads(request.content)["model"] for request in api.model_posts]
        assert models == [DEFAULT_NEMOTRON_MODEL] * 3 + [DEFAULT_JUDGMENT_MODEL]

    def test_review_needs_the_judgment_model_in_the_catalog(
        self, tmp_path: Path, api: FakeAgentModel
    ) -> None:
        """검토는 계정 목록에 판단 모델이 있어야 한다. 없으면 모델 요청 없이 exit 1이다."""
        api.models = [DEFAULT_NEMOTRON_MODEL]
        result = runner.invoke(app, ["agent-cell", GOAL, "-o", str(tmp_path / "run")])
        assert result.exit_code == 1, result.output
        assert "error_code=model_error" in result.output and not api.model_posts

    def test_cli_reworks_a_missing_part_and_completes(
        self, tmp_path: Path, api: FakeAgentModel
    ) -> None:
        """처분은 기본으로 켠다. 공급되지 않은 부품을 재작업으로 놓으면 exit 0이다."""
        api.dispositions = [None]
        root, scene_path = tmp_path / "run", tmp_path / "scene.json"
        scene_path.write_text(json.dumps({"faults": {"missing": [2]}}))
        result = runner.invoke(
            app, ["agent-cell", GOAL, "-o", str(root), "--scene", str(scene_path)]
        )
        assert result.exit_code == 0, result.output
        assert "disposition=reworked_released place_runs=2 nemotron_requests=5" in result.output
        run = ToolRun.model_validate_json((root / "run.json").read_text())
        assert run.agent is not None and run.agent.disposition
        assert run.agent.budget.max_place_runs == 2
        assert run.agent.system_prompt == system_prompt(review=True, disposition=True)
        report = AgentReport.model_validate_json((root / "agent_report.json").read_text())
        assert report.reworked and report.disposition == Disposition.REWORKED_RELEASED
        models = [json.loads(request.content)["model"] for request in api.model_posts]
        assert models == [
            DEFAULT_NEMOTRON_MODEL,
            DEFAULT_JUDGMENT_MODEL,
            DEFAULT_NEMOTRON_MODEL,
            DEFAULT_JUDGMENT_MODEL,
            DEFAULT_JUDGMENT_MODEL,
        ]

    def test_cli_rejects_a_part_of_another_color(self, tmp_path: Path, api: FakeAgentModel) -> None:
        """다른 색의 부품은 참값 판정을 통과한다. 카메라 판정이 실패해서 exit 1이다."""
        root, scene_path = tmp_path / "run", tmp_path / "scene.json"
        scene_path.write_text(WRONG_COLOR_SCENE.model_dump_json())
        result = runner.invoke(
            app, ["agent-cell", GOAL, "-o", str(root), "--scene", str(scene_path)]
        )
        assert result.exit_code == 1, result.output
        assert "error_code=inspection_rejected disposition=rejected place_runs=1" in result.output
        report = AgentReport.model_validate_json((root / "agent_report.json").read_text())
        assert report.check_passed is True and not report.reworked

    def test_no_disposition_option_completes_from_the_pose_check(
        self, tmp_path: Path, api: FakeAgentModel
    ) -> None:
        """`--no-disposition`은 참값 판정만으로 완료한다. 다른 색의 부품도 exit 0이다."""
        root, scene_path = tmp_path / "run", tmp_path / "scene.json"
        scene_path.write_text(WRONG_COLOR_SCENE.model_dump_json())
        result = runner.invoke(
            app,
            ["agent-cell", GOAL, "-o", str(root), "--scene", str(scene_path), "--no-disposition"],
        )
        assert result.exit_code == 0, result.output
        assert "disposition=released place_runs=1" in result.output
        run = ToolRun.model_validate_json((root / "run.json").read_text())
        assert run.agent is not None and not run.agent.disposition
        assert run.agent.budget.max_place_runs == 1
        assert run.agent.system_prompt == system_prompt(review=True)
        checked = json.loads(api.model_posts[3].content)
        assert "inspection" not in json.loads(checked["messages"][-1]["content"])["summary"]

    @pytest.mark.parametrize("variant", ["abort", "http_503", "early_complete"])
    def test_cli_runtime_failure_returns_one(
        self,
        tmp_path: Path,
        api: FakeAgentModel,
        retry_waits: list[float],
        variant: str,
    ) -> None:
        """모델 중단이나 오류가 있으면 근거를 저장하고 exit 1이다.

        503은 같은 요청의 4회 연속 실패다. CLI는 3회 다시 보낸 뒤에 중단한다.
        """
        api.variant, api.http_failures = variant, 4
        result = runner.invoke(app, ["agent-cell", GOAL, "-o", str(tmp_path / "run")])
        assert result.exit_code == 1, result.output
        report = AgentReport.model_validate_json((tmp_path / "run/agent_report.json").read_text())
        retried = variant == "http_503"
        assert report.calls.retries == (3 if retried else 0)
        assert retry_waits == ([3.0, 9.0, 20.0] if retried else []), retry_waits
        assert all(c.is_closed for c in api.clients)

    @pytest.mark.parametrize("stage", [0, 1])
    def test_cli_sends_a_failed_request_again_and_completes(
        self,
        tmp_path: Path,
        api: FakeAgentModel,
        retry_waits: list[float],
        stage: int,
    ) -> None:
        """모델 목록 또는 모델 요청의 503이 1회이면 CLI는 다시 보내고 exit 0이다."""
        api.variant, api.variant_stage, api.http_failures = "http_503", stage, 1
        root = tmp_path / "run"
        result = runner.invoke(app, ["agent-cell", GOAL, "-o", str(root)])
        assert result.exit_code == 0, result.output
        run = ToolRun.model_validate_json((root / "run.json").read_text())
        assert run.agent is not None and run.agent.budget.request_retries == 3
        assert run.agent.budget.sdk_retries == 0
        report = AgentReport.model_validate_json((root / "agent_report.json").read_text())
        assert report.calls.retries == 1 and report.calls.nemotron == 4
        assert retry_waits == [3.0], retry_waits

    @pytest.mark.parametrize(
        "failure",
        [
            "existing",
            "blank",
            "long",
            "missing_runner",
            "invalid_scene",
            "invalid_cell",
            "missing_key",
            "blank_cosmos_model",
            "blank_cosmos_url",
            "invalid_cosmos_url",
        ],
    )
    def test_cli_preflight_failure_has_no_http_requests(
        self,
        tmp_path: Path,
        api: FakeAgentModel,
        monkeypatch: pytest.MonkeyPatch,
        failure: str,
    ) -> None:
        """입력이나 실행 설정이 틀리면 외부 요청 없이 exit 2이다."""
        root = tmp_path / "run"
        goal = " " if failure == "blank" else "x" * 4001 if failure == "long" else GOAL
        args = ["agent-cell", goal, "-o", str(root)]
        if failure == "existing":
            root.mkdir()
        elif failure == "missing_runner":
            (tmp_path / "isaac/python.sh").unlink()
        elif failure in ("invalid_scene", "invalid_cell"):
            path = tmp_path / "bad.json"
            path.write_text('{"extra": true}')
            args += ["--scene" if failure == "invalid_scene" else "--cell", str(path)]
        elif failure == "missing_key":
            monkeypatch.delenv("NEBIUS_API_KEY")
        elif failure == "blank_cosmos_model":
            monkeypatch.setenv("COSMOS_REASONER_MODEL", " ")
        elif failure == "blank_cosmos_url":
            monkeypatch.setenv("REASONER_BASE_URL", " ")
        elif failure == "invalid_cosmos_url":
            monkeypatch.setenv("REASONER_BASE_URL", "not-a-url")
        result = runner.invoke(app, args)
        assert result.exit_code == 2, result.output
        assert not api.requests
        assert all(c.is_closed for c in api.clients)
        assert FAKE_AGENT_KEY not in result.output


class TestAgentShiftCLI:
    """`agent-shift`의 연속 실행, 교대 보고서와 종료 코드를 검사한다."""

    def test_shift_runs_each_order_and_writes_the_report(
        self, tmp_path: Path, api: FakeAgentModel
    ) -> None:
        """주문 3개를 차례로 실행하고 주문별 지표와 합계를 보고한다. 모두 완료하면 exit 0이다."""
        answer_orders(api)
        shift_path = write_shift(tmp_path, THREE_ORDERS)
        out_dir = tmp_path / "shift"
        result = runner.invoke(app, ["agent-shift", str(shift_path), "-o", str(out_dir)])
        assert result.exit_code == 0, result.output
        report = read_shift_report(out_dir)
        assert [item.directory for item in report.items] == ["01-first", "02-second", "03-third"]
        board, kit = parse_order(json.dumps(BOARD_ORDER)), parse_order(json.dumps(KIT_ORDER))
        assert [item.order for item in report.items] == [board, kit, kit]
        assert [item.part_count for item in report.items] == [2, 3, 3]
        for item in report.items:
            root = out_dir / item.directory
            saved = AgentReport.model_validate_json((root / "agent_report.json").read_text())
            run = ToolRun.model_validate_json((root / "run.json").read_text())
            assert run.prompt == order_instruction(item.order)
            assert run.agent is not None and run.agent.review
            assert run.agent.budget.request_retries == 3
            assert item.state == saved.state == ToolState.COMPLETED and item.error_code is None
            assert item.passed_parts == item.part_count
            # 가짜 실행기의 `sim_time_s`는 1.0이다.
            assert item.robot_time_s == 1.0
            assert item.nemotron_requests == REQUESTS_PER_RUN and item.request_retries == 0
            assert 0 <= item.model_time_s < item.wall_time_s
        totals = report.totals
        assert (totals.orders, totals.completed_orders, totals.changeovers) == (3, 3, 1)
        assert totals.wall_time_s == sum(item.wall_time_s for item in report.items)
        scenes = [
            SimScene.model_validate_json(
                (out_dir / item.directory / "evaluation/scene.json").read_text()
            )
            for item in report.items
        ]
        cell = CellConfig()
        nominal = placement_of(cell, Board())
        offset = BoardOffset(x=0.003, y=0.005, yaw=math.radians(1.5))
        assert scenes[0].board == scenes[2].board == nominal
        assert scenes[1].board == displace_board(cell, Board(), offset=offset)
        assert report.items[1].fixture == FixtureOffset(x_mm=3.0, y_mm=5.0, yaw_deg=1.5)
        assert "01-first state=completed error_code=None parts=2/2 robot_s=1.0" in result.output
        assert "orders=3 completed=3 changeovers=1" in result.output
        assert len(api.model_posts) == 3 * REQUESTS_PER_RUN
        assert all(c.is_closed for c in api.clients)
        assert FAKE_AGENT_KEY not in result.output + (out_dir / SHIFT_REPORT_FILE).read_text()

    def test_shift_reports_dispositions_and_yields(
        self, tmp_path: Path, api: FakeAgentModel
    ) -> None:
        """결함이 없는 주문, 공급 누락과 다른 색 부품을 출하, 재작업 뒤 출하와 불량으로 보고한다."""
        answer_orders(api)
        # 재작업은 두 번째 주문에서 1번 고른다. 세 번째 주문은 대기열이 비어서 불량이 된다.
        api.dispositions = [None]
        items = [dict(item) for item in THREE_ORDERS]
        items[1]["faults"] = {"missing": [1]}
        items[2]["faults"] = {"wrong_color": [{"sequence": 1, "color": "orange"}]}
        out_dir = tmp_path / "shift"
        result = runner.invoke(
            app, ["agent-shift", str(write_shift(tmp_path, items)), "-o", str(out_dir)]
        )
        assert result.exit_code == 1, result.output
        report = read_shift_report(out_dir)
        first, reworked, rejected = report.items
        assert [item.disposition for item in report.items] == [
            Disposition.RELEASED,
            Disposition.REWORKED_RELEASED,
            Disposition.REJECTED,
        ]
        assert [item.reworked for item in report.items] == [False, True, False]
        assert first.faults is None and reworked.faults == PartFaults(missing=(1,))
        assert rejected.faults == PartFaults(
            wrong_color=(WrongColorFault(sequence=1, color=Color.ORANGE),)
        ), rejected.faults
        # 재작업 뒤의 판정이 부품 수의 근거다. 로봇 시간은 두 실행의 합이다.
        assert reworked.passed_parts == reworked.part_count == 3
        assert reworked.robot_time_s == 2.0 and reworked.nemotron_requests == 5
        assert rejected.error_code == ErrorCode.INSPECTION_REJECTED
        assert rejected.passed_parts == 3 and rejected.robot_time_s == 1.0
        totals = report.totals
        assert (totals.completed_orders, totals.first_pass_orders, totals.rejected_orders) == (
            2,
            1,
            1,
        )
        assert totals.first_pass_yield == pytest.approx(1 / 3)
        assert totals.final_yield == pytest.approx(2 / 3)
        assert "02-second state=completed" in result.output
        assert "disposition=reworked_released" in result.output
        assert "first_pass=1/3 rejected=1" in result.output
        scene = SimScene.model_validate_json(
            (out_dir / "02-second/evaluation/scene.json").read_text()
        )
        assert scene.faults == SceneFaults(missing=(1,))

    def test_no_disposition_option_applies_to_each_order(
        self, tmp_path: Path, api: FakeAgentModel
    ) -> None:
        """`--no-disposition`은 모든 주문의 처분을 끈다. 공급 누락은 재작업 없이 불량이다."""
        answer_orders(api)
        api.dispositions = [None]
        items = [dict(item) for item in THREE_ORDERS[:2]]
        items[1]["faults"] = {"missing": [1]}
        out_dir = tmp_path / "shift"
        args = ["agent-shift", str(write_shift(tmp_path, items)), "-o", str(out_dir)]
        result = runner.invoke(app, [*args, "--no-disposition"])
        assert result.exit_code == 1, result.output
        report = read_shift_report(out_dir)
        assert [item.disposition for item in report.items] == [
            Disposition.RELEASED,
            Disposition.REJECTED,
        ]
        assert report.items[1].error_code == ErrorCode.CHECK_FAILED
        assert not report.items[1].reworked and report.items[1].nemotron_requests == 4
        for item in report.items:
            run = ToolRun.model_validate_json((out_dir / item.directory / "run.json").read_text())
            assert run.agent is not None and not run.agent.disposition

    def test_shift_continues_after_an_order_that_did_not_complete(
        self, tmp_path: Path, api: FakeAgentModel, retry_waits: list[float]
    ) -> None:
        """두 번째 주문이 요청 실패로 중단해도 세 번째 주문을 실행한다. exit 1이다."""
        answer_orders(api)
        api.variant, api.http_failures = "http_503", 4
        api.variant_stage = REQUESTS_PER_RUN + 1
        shift_path = write_shift(tmp_path, THREE_ORDERS)
        out_dir = tmp_path / "shift"
        result = runner.invoke(app, ["agent-shift", str(shift_path), "-o", str(out_dir)])
        assert result.exit_code == 1, result.output
        report = read_shift_report(out_dir)
        assert [item.state for item in report.items] == [
            ToolState.COMPLETED,
            ToolState.ABORTED,
            ToolState.COMPLETED,
        ]
        failed = report.items[1]
        assert failed.error_code == ErrorCode.MODEL_ERROR
        assert failed.passed_parts is None and failed.robot_time_s is None
        assert failed.nemotron_requests == 1 and failed.request_retries == 3
        assert failed.model_time_s == 0 and retry_waits == [3.0, 9.0, 20.0]
        assert report.totals.orders == 3 and report.totals.completed_orders == 2
        assert "02-second state=aborted error_code=model_error parts=-/3 robot_s=-" in result.output
        assert "retries=3" in result.output and "orders=3 completed=2" in result.output

    def test_shift_stops_after_a_user_interrupt(self, tmp_path: Path, api: FakeAgentModel) -> None:
        """첫 주문에서 사용자가 인터럽트하면 다음 주문을 실행하지 않는다. exit 1이다."""

        select = answer_orders(api)

        def interrupt(request: httpx.Request) -> None:
            select(request)
            if len(api.model_posts) == 2:
                raise KeyboardInterrupt

        api.on_request = interrupt
        shift_path = write_shift(tmp_path, THREE_ORDERS)
        out_dir = tmp_path / "shift"
        result = runner.invoke(app, ["agent-shift", str(shift_path), "-o", str(out_dir)])
        assert result.exit_code == 1, result.output
        report = read_shift_report(out_dir)
        assert [item.directory for item in report.items] == ["01-first"]
        assert report.items[0].error_code == ErrorCode.AGENT_ABORTED
        assert report.totals.orders == 3 and report.totals.completed_orders == 0
        assert not (out_dir / "02-second").exists()
        assert "stopped the shift after 1 of 3 orders" in result.output

    def test_model_abort_does_not_stop_the_shift(self, tmp_path: Path, api: FakeAgentModel) -> None:
        """모델의 자발적 중단은 인터럽트가 아니다. 다음 주문을 실행한다."""
        answer_orders(api)
        api.variant, api.variant_stage = "abort", 1
        shift_path = write_shift(tmp_path, THREE_ORDERS[:2])
        out_dir = tmp_path / "shift"
        result = runner.invoke(app, ["agent-shift", str(shift_path), "-o", str(out_dir)])
        assert result.exit_code == 1, result.output
        report = read_shift_report(out_dir)
        assert [item.error_code for item in report.items] == [ErrorCode.AGENT_ABORTED, None]
        assert report.totals.completed_orders == 1

    def test_order_with_a_recording_error_has_no_event_values(
        self, tmp_path: Path, api: FakeAgentModel
    ) -> None:
        """이벤트 파일이 없는 주문은 이벤트의 지표 없이 보고한다. 다음 주문을 실행한다."""
        out_dir = tmp_path / "shift"
        select = answer_orders(api)

        def obstruct(request: httpx.Request) -> None:
            select(request)
            if len(api.model_posts) == 1:
                path = out_dir / "01-first" / "events.jsonl"
                if path.is_file():
                    path.unlink()
                    path.mkdir()

        api.on_request = obstruct
        shift_path = write_shift(tmp_path, THREE_ORDERS[:2])
        result = runner.invoke(app, ["agent-shift", str(shift_path), "-o", str(out_dir)])
        assert result.exit_code == 1, result.output
        first, second = read_shift_report(out_dir).items
        assert first.error_code == ErrorCode.RECORDING_ERROR
        assert first.model_time_s == 0 and first.passed_parts is None
        assert second.state == ToolState.COMPLETED

    def test_order_with_a_broken_event_line_is_reported(
        self, tmp_path: Path, api: FakeAgentModel
    ) -> None:
        """이벤트 파일에 읽지 못하는 줄이 있어도 그 주문을 보고하고 다음 주문을 실행한다."""
        out_dir = tmp_path / "shift"
        select = answer_orders(api)
        broken: list[Path] = []

        def corrupt(request: httpx.Request) -> None:
            select(request)
            path = out_dir / "01-first" / "events.jsonl"
            if not broken and path.is_file():
                with path.open("a", encoding="utf-8") as stream:
                    stream.write('{"schema_version": 1, "run_id"\n')
                broken.append(path)

        api.on_request = corrupt
        shift_path = write_shift(tmp_path, THREE_ORDERS[:2])
        result = runner.invoke(app, ["agent-shift", str(shift_path), "-o", str(out_dir)])
        assert result.exit_code == 0, result.output
        assert broken
        first, second = read_shift_report(out_dir).items
        assert first.state == second.state == ToolState.COMPLETED
        assert first.passed_parts == first.part_count
        assert first.model_time_s > 0

    def test_interrupt_outside_a_run_keeps_the_report(
        self, tmp_path: Path, api: FakeAgentModel, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """실행의 밖에서 받은 인터럽트는 교대를 멈춘다. 앞 주문의 보고서는 남는다. exit 1이다."""
        answer_orders(api)
        run_cell = shift.run_agent_cell
        calls: list[Path] = []

        def interrupt_second(prompt: str, **kwargs: Any) -> AgentReport:
            calls.append(kwargs["out_dir"])
            if len(calls) == 2:
                raise KeyboardInterrupt
            return run_cell(prompt, **kwargs)

        monkeypatch.setattr(shift, "run_agent_cell", interrupt_second)
        shift_path = write_shift(tmp_path, THREE_ORDERS)
        out_dir = tmp_path / "shift"
        result = runner.invoke(app, ["agent-shift", str(shift_path), "-o", str(out_dir)])
        assert result.exit_code == 1, result.output
        report = read_shift_report(out_dir)
        assert [item.directory for item in report.items] == ["01-first"]
        assert report.items[0].state == ToolState.COMPLETED
        assert len(calls) == 2
        assert "stopped the shift after 1 of 3 orders" in result.output

    def test_no_review_option_applies_to_each_order(
        self, tmp_path: Path, api: FakeAgentModel
    ) -> None:
        """`--no-review`는 모든 주문의 검토를 끈다. 처분도 끄면 판단 모델이 없어도 완료한다."""
        answer_orders(api)
        api.models = [DEFAULT_NEMOTRON_MODEL]
        shift_path = write_shift(tmp_path, THREE_ORDERS[:1])
        out_dir = tmp_path / "shift"
        args = ["agent-shift", str(shift_path), "-o", str(out_dir)]
        result = runner.invoke(app, [*args, "--no-review", "--no-disposition"])
        assert result.exit_code == 0, result.output
        run = ToolRun.model_validate_json((out_dir / "01-first/run.json").read_text())
        assert run.agent is not None and not run.agent.review and not run.agent.disposition

    @pytest.mark.parametrize(
        "failure",
        [
            "missing_shift",
            "bad_json",
            "no_items",
            "bad_name",
            "extra_field",
            "fixture_out_of_range",
            "missing_order",
            "fault_outside_order",
            "fault_with_unknown_color",
            "two_faults_for_one_part",
            "invalid_order",
            "infeasible_order",
            "existing",
            "missing_key",
            "missing_runner",
        ],
    )
    def test_shift_input_failure_has_no_http_requests(
        self,
        tmp_path: Path,
        api: FakeAgentModel,
        monkeypatch: pytest.MonkeyPatch,
        failure: str,
    ) -> None:
        """교대 파일, 주문 파일 또는 설정이 틀리면 외부 요청 없이 exit 2이다."""
        items: list[dict[str, object]] = [dict(item) for item in THREE_ORDERS]
        if failure == "no_items":
            items = []
        elif failure == "bad_name":
            items[1]["name"] = "Second order"
        elif failure == "extra_field":
            items[1]["command"] = "never execute"
        elif failure == "fixture_out_of_range":
            items[1]["fixture"] = {"x_mm": 0.0, "y_mm": 0.0, "yaw_deg": 15.0}
        elif failure == "missing_order":
            items[2]["order"] = "../orders/none.json"
        elif failure == "fault_outside_order":
            # 첫 주문의 부품은 2개다.
            items[0]["faults"] = {"missing": [3]}
        elif failure == "fault_with_unknown_color":
            items[0]["faults"] = {"wrong_color": [{"sequence": 1, "color": "red"}]}
        elif failure == "two_faults_for_one_part":
            items[0]["faults"] = {
                "missing": [1],
                "wrong_color": [{"sequence": 1, "color": "orange"}],
            }
        shift_path = write_shift(tmp_path, items)
        out_dir = tmp_path / "shift"
        if failure == "missing_shift":
            shift_path = tmp_path / "none.json"
        elif failure == "bad_json":
            shift_path.write_text("{")
        elif failure == "invalid_order":
            (tmp_path / "orders/kit.json").write_text('{"kind": "kit", "items": []}')
        elif failure == "infeasible_order":
            overlap = {"kind": "board", "parts": [BOARD_ORDER["parts"][0]] * 2}
            (tmp_path / "orders/board.json").write_text(json.dumps(overlap))
        elif failure == "existing":
            out_dir.mkdir()
        elif failure == "missing_key":
            monkeypatch.delenv("NEBIUS_API_KEY")
        elif failure == "missing_runner":
            (tmp_path / "isaac/python.sh").unlink()
        result = runner.invoke(app, ["agent-shift", str(shift_path), "-o", str(out_dir)])
        assert result.exit_code == 2, result.output
        assert not api.requests
        assert failure == "existing" or not out_dir.exists()
        assert FAKE_AGENT_KEY not in result.output

    def test_run_shift_rejects_an_existing_output_directory(
        self, tmp_path: Path, api: FakeAgentModel
    ) -> None:
        """출력 디렉터리가 이미 있으면 실행하지 않는다."""
        orders = load_shift(write_shift(tmp_path, THREE_ORDERS))
        out_dir = tmp_path / "shift"
        out_dir.mkdir()
        with pytest.raises(FileExistsError):
            run_shift(
                orders,
                out_dir=out_dir,
                cell=CellConfig(),
                token_settings=TokenFactorySettings(),  # pyright: ignore[reportCallIssue]
                cosmos_settings=Settings(),  # pyright: ignore[reportCallIssue]
                sim_settings=SimSettings(),  # pyright: ignore[reportCallIssue]
            )
        assert not api.requests and not list(out_dir.iterdir())

    def test_example_shift_has_the_four_demo_requests(self) -> None:
        """예시 교대는 화면의 현장 요청 카드 4개다. 마지막 주문만 지그 오차가 있다."""
        orders = load_shift(EXAMPLES / "shifts" / "mixed.json")
        assert [order.name for order in orders] == ["changeover", "kitting", "mixed", "fixture"]
        files = ["changeover.json", "kitting.json", "mixed.json", "changeover.json"]
        assert [order.order for order in orders] == [
            parse_order((EXAMPLES / "orders" / name).read_text()) for name in files
        ]
        assert [order.fixture for order in orders] == [
            FixtureOffset(),
            FixtureOffset(),
            FixtureOffset(),
            FixtureOffset(x_mm=3.0, y_mm=5.0, yaw_deg=1.5),
        ]
        assert all(order.prompt == order_instruction(order.order) for order in orders)
