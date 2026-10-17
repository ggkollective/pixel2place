"""웹 실행의 CLI 경계, 단일 실행, 중단과 화면 자료를 검사한다."""

import json
import math
import os
import re
import signal
import subprocess
import sys
from collections.abc import Iterator
from contextlib import suppress
from datetime import UTC, datetime
from pathlib import Path
from time import monotonic, sleep
from typing import Any

import pytest
from fastapi.testclient import TestClient
from typer.testing import CliRunner

from pixel2place.agent.contracts import ToolState
from pixel2place.agent.fixture import (
    FIXTURE_X_LIMIT_MM,
    FIXTURE_Y_LIMIT_MM,
    FIXTURE_YAW_LIMIT_DEG,
    FixtureOffset,
    fixture_scene,
)
from pixel2place.agent.intake import INTAKE_FILE, IntakeRecord
from pixel2place.agent.tools import AssemblyTools
from pixel2place.cell.config import CellConfig
from pixel2place.cell.scene import SimScene
from pixel2place.cell.task import MM_PER_M, BoardFrame, pan_distance
from pixel2place.cli import app
from pixel2place.config import DEFAULT_JUDGMENT_MODEL
from pixel2place.demo.contracts import (
    MAX_SHIFT_REQUEST_ORDERS,
    JobRecord,
    RunRequest,
    ShiftEntry,
    ShiftOrderRequest,
    ShiftRecord,
    ShiftRequest,
)
from pixel2place.demo.jobs import JobManager, RunBusyError
from pixel2place.demo.orders import FIRST_PART_MARGIN_M, supply_faults
from pixel2place.demo.server import MAX_BODY_BYTES, create_app
from pixel2place.demo.shifts import SHIFT_FILE, SHIFTS_DIR, ShiftDesk
from pixel2place.generation.synthetic import COLOR_RGB
from pixel2place.layout.parts import shape_of
from pixel2place.layout.schema import Board, Color, Layout, PartType, Rotation
from pixel2place.order import (
    MAX_KIT_PARTS,
    MAX_ORDER_PARTS,
    BoardOrder,
    compose_instruction,
    order_layout,
    parse_order,
)
from pixel2place.pipeline import CHECK_FILE, IMAGE_CHECK_FILE
from pixel2place.reasoner import nemotron
from tests.factories import make_layout, make_part, make_planner_response_json
from tests.fakes.fake_agent_model import FakeAgentModel
from tests.fakes.fake_judgment_model import FAKE_JUDGMENT_KEY, FakeJudgmentModel, Reply
from tests.unit.test_agent_runtime import GOAL, make_runtime

pytestmark = [pytest.mark.unit, pytest.mark.usefixtures("fake_isaac")]

SQUARE = {"type": "square_tile", "color": "blue", "row": 1, "col": 1}
RECT = {"type": "rect_tile", "color": "gray", "row": 3, "col": 5}
KIT = {
    "kind": "kit",
    "items": [
        {"type": "square_tile", "count": 2},
        {"type": "rect_tile", "count": 1},
    ],
    "arrangement": "type_rows",
}
KIT_INSTRUCTION = (
    "On an 8 by 8 board, place two blue square tiles and one gray rectangular tile. "
    "All rotations are 0 degrees. Put parts of the same type in the same row."
)
TEXT = "I need two square tiles and one rectangular tile."
SUBMIT_KIT = Reply(tool="submit_kit_order", arguments=KIT)
ORDERS = Path(__file__).parents[2] / "examples" / "orders"
MISSING_FIRST = {"missing": [1], "wrong_color": []}
ORANGE_FIRST = {"missing": [], "wrong_color": [{"sequence": 1, "color": "orange"}]}
KIT_PREVIEW = {
    "part_count": 3,
    "issues": [],
    "instruction": KIT_INSTRUCTION,
    "supply_faults": {"missing": MISSING_FIRST, "wrong_color": ORANGE_FIRST},
}
# 보정한 보드와 실제 보드의 셀 중심 거리의 측정 최댓값(mm)이다.
CALIBRATION_ERROR_MM = 0.25


class FakeWorker:
    """프로세스 시작 경계만 대체한다. CLI 산출물은 실제 실행 프로그램으로 만든다."""

    def __init__(self, arguments: list[str], **options: Any) -> None:
        self.arguments = arguments
        self.options = options
        self.pid = 999_999_999
        self.returncode: int | None = None

    def poll(self) -> int | None:
        return self.returncode


@pytest.fixture
def cell(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    """실행 관리자, 시작한 워커와 교대 창구. HTTP 서버와 서버의 주기 확인은 없다."""
    workers: list[FakeWorker] = []
    original_spawn = subprocess.Popen

    def start(arguments: list[str], **options: Any):
        if arguments[1:3] != ["-m", "pixel2place.demo.worker"]:
            return original_spawn(arguments, **options)
        worker = FakeWorker(arguments, **options)
        workers.append(worker)
        return worker

    def stop(worker: FakeWorker) -> None:
        if worker.returncode is None:
            worker.returncode = 130

    monkeypatch.setattr(subprocess, "Popen", start)
    monkeypatch.setattr(JobManager, "_stop", staticmethod(stop))
    manager = JobManager(root=tmp_path / "jobs", workspace=tmp_path)
    try:
        yield manager, workers, ShiftDesk(jobs=manager)
    finally:
        manager.close()


@pytest.fixture
def served(cell):
    """서버 1개의 클라이언트, 실행 관리자, 시작한 워커와 교대 창구."""
    manager, workers, desk = cell
    with TestClient(create_app(manager=manager, shifts=desk)) as client:
        yield client, manager, workers, desk
    assert manager._closed


@pytest.fixture
def web(served):
    client, manager, workers, _ = served
    return client, manager, workers


@pytest.fixture
def desk(served) -> ShiftDesk:
    return served[3]


@pytest.fixture
def intake_api(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Iterator[FakeJudgmentModel]:
    """접수의 환경을 테스트 값으로 고정한다. SDK의 HTTP 클라이언트만 대체한다.

    작업 디렉터리를 옮긴다. 저장소의 `.env`를 읽지 않는다.
    """
    fake = FakeJudgmentModel()
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("NEBIUS_API_KEY", FAKE_JUDGMENT_KEY)
    monkeypatch.setenv("NEBIUS_BASE_URL", "https://api.tokenfactory.nebius.com/v1/")
    monkeypatch.setattr(nemotron, "DefaultHttpxClient", fake.make_client)
    yield fake
    for client in fake.clients:
        client.close()


def test_http_start_uses_original_goal_and_single_worker(web) -> None:
    """실행을 요청하면 목표를 보존하고 중복 실행을 거부한다."""
    client, manager, workers = web
    fixture = {"x_mm": 3.0, "y_mm": -5.0, "yaw_deg": 1.5}
    response = client.post("/api/runs", json={"prompt": GOAL, "fixture": fixture})
    assert response.status_code == 202
    view = response.json()
    assert view["prompt"] == GOAL and view["status"] == "running"
    assert view["scene"] == "board_offset" and view["fixture"] == fixture
    assert view["order"] is None and view["ordered_parts"] is None
    assert view["order_kind"] is None
    assert view["runtime_state"] == "created" and view["assets"] == []
    assert "worker_pid" not in view
    directory = manager.root / view["id"]
    assert json.loads((directory / "request.json").read_text())["prompt"] == GOAL
    assert json.loads((directory / "scene.json").read_text())["board"] is not None
    assert workers[0].arguments[1:3] == ["-m", "pixel2place.demo.worker"]
    assert workers[0].options["start_new_session"] is True
    assert workers[0].options["cwd"] == manager.workspace
    assert client.post("/api/runs", json={"prompt": GOAL}).status_code == 409
    assert len(workers) == 1
    assert len(client.get("/api/runs").json()) == 1


@pytest.mark.parametrize(
    "body",
    [
        {"prompt": " "},
        {"prompt": "x" * 4001},
        {"prompt": GOAL, "scene": "other"},
        {"prompt": GOAL, "command": "private-key"},
        {},
        {"prompt": GOAL, "order": {"parts": [SQUARE]}},
        {"order": {"parts": []}},
        {"order": {"parts": [SQUARE, SQUARE]}},
        {"order": {"parts": [{**SQUARE, "col": 8}]}},
        {"order": {"parts": [{**SQUARE, "col": index % 8} for index in range(17)]}},
        {"order": {"parts": [{**SQUARE, "type": "hex_tile"}]}},
        {"order": {**KIT, "arrangement": "symmetric"}},
        {"order": {**KIT, "items": [{"type": "T_tile", "count": 1}]}},
        {"order": {**KIT, "items": [{"type": "square_tile", "count": 7}]}},
        {"order": {**KIT, "items": [{"type": "rect_tile", "count": 5}]}},
        {"order": {**KIT, "items": [{"type": "square_tile", "color": "white", "count": 1}]}},
        {"order": {**KIT, "parts": [SQUARE]}},
        {"prompt": GOAL, "fixture": {"x_mm": 3.1}},
        {"prompt": GOAL, "fixture": {"y_mm": -5.1}},
        {"prompt": GOAL, "fixture": {"yaw_deg": 1.6}},
        {"prompt": GOAL, "fixture": {"x_mm": "1"}},
    ],
)
def test_invalid_http_input_never_starts_worker(web, body: dict[str, object]) -> None:
    """입력이 계약이나 배치 규칙을 벗어나면 워커와 실행 기록을 만들지 않는다."""
    client, manager, workers = web
    response = client.post("/api/runs", json=body)
    assert response.status_code == 422, response.text
    assert not workers and client.get("/api/runs").json() == []
    assert not [path for path in manager.root.iterdir() if path.is_dir()]
    assert "private-key" not in response.text


def test_oversized_body_is_rejected_before_validation(web) -> None:
    """본문이 크기 한도를 넘으면 내용이 유효해도 워커를 시작하지 않는다."""
    client, _, workers = web
    body = json.dumps({"prompt": GOAL})
    headers = {"Content-Type": "application/json"}
    assert client.post("/api/runs", content=body, headers=headers).status_code == 202
    padded = body + " " * MAX_BODY_BYTES
    response = client.post("/api/runs", content=padded, headers=headers)
    assert response.status_code == 422 and len(workers) == 1, response.text
    preview = json.dumps({"parts": [SQUARE]}) + " " * MAX_BODY_BYTES
    assert client.post("/api/orders/preview", content=preview, headers=headers).status_code == 422


def test_page_scripts_are_served_and_use_existing_element_ids(web) -> None:
    """화면을 요청하면 스크립트 모듈을 전달하고 스크립트가 찾는 요소가 화면에 있다."""
    client, _, _ = web
    page = client.get("/")
    assert page.status_code == 200
    for name in ("app.js", "order.js", "http.js", "style.css"):
        script = client.get(f"/ui/{name}")
        assert script.status_code == 200, name
        # 템플릿 문자열로 만든 ID는 검사하지 않는다.
        for element_id in set(re.findall(r'\$\("([a-z0-9-]+)"\)', script.text)):
            assert f'id="{element_id}"' in page.text, f"{name}: #{element_id}"


def test_order_options_use_shape_table_and_verified_limits(web) -> None:
    """선택지를 요청하면 부품 형태 표와 검증한 오차 범위를 반환한다."""
    client, _, _ = web
    options = client.get("/api/orders/options").json()
    assert options["board"] == {"type": "grid", "rows": 8, "cols": 8, "cell_size_mm": 20.0}
    assert options["max_parts"] == MAX_ORDER_PARTS
    assert options["colors"] == ["blue", "gray", "white", "orange"]
    assert [part["type"] for part in options["parts"]] == [item.value for item in PartType]
    for part in options["parts"]:
        for shape in part["shapes"]:
            expected = shape_of(PartType(part["type"]), Rotation(shape["rotation_deg"]))
            assert {tuple(cell) for cell in shape["cells"]} == expected, f"{part['type']}: {shape}"
    assert options["kit"] == {
        "max_parts": MAX_KIT_PARTS,
        "parts": [
            {"type": "square_tile", "color": "blue", "max_count": 6},
            {"type": "rect_tile", "color": "gray", "max_count": 4},
        ],
        "arrangements": [{"value": "type_rows", "name": "One row for each part type"}],
    }
    assert options["fixture"] == {
        "x_mm": {"min": -3.0, "max": 3.0},
        "y_mm": {"min": -5.0, "max": 5.0},
        "yaw_deg": {"min": -1.5, "max": 1.5},
    }
    assert options["seconds_per_part"] > 0 and options["seconds_base"] > 0


def test_order_preview_returns_instruction_or_issues_without_run(web) -> None:
    """미리보기를 요청하면 지시문 또는 위반만 반환하고 실행을 시작하지 않는다."""
    client, manager, workers = web
    feasible = client.post("/api/orders/preview", json={"parts": [SQUARE, RECT]})
    assert feasible.status_code == 200, feasible.text
    assert feasible.json() == {
        "part_count": 2,
        "issues": [],
        "instruction": (
            "On an 8 by 8 board, place a blue square tile at row 1 col 1. "
            "Place a gray rectangular tile at row 3 col 5. All rotations are 0 degrees. "
            "Each row and col is the position of the part. Rows and cols start at 0."
        ),
        "supply_faults": {"missing": MISSING_FIRST, "wrong_color": ORANGE_FIRST},
    }
    blocked = client.post("/api/orders/preview", json={"parts": [SQUARE, {**RECT, "col": 7}]})
    assert blocked.status_code == 200 and blocked.json()["instruction"] is None
    assert blocked.json()["supply_faults"] is None, "배치할 수 없는 주문에는 공급 결함이 없다."
    assert [issue["code"] for issue in blocked.json()["issues"]] == ["OUT_OF_BOUNDS"]
    assert blocked.json()["issues"][0]["part_id"] == "P002"
    empty = client.post("/api/orders/preview", json={"parts": []}).json()
    assert empty["part_count"] == 0 and empty["issues"][0]["code"] == "EMPTY_LAYOUT"
    assert client.post("/api/orders/preview", json={"parts": [{}]}).status_code == 422
    kit = client.post("/api/orders/preview", json=KIT)
    assert kit.status_code == 200, kit.text
    assert kit.json() == KIT_PREVIEW
    outside = {**KIT, "items": [{"type": "L_tile", "count": 1}]}
    assert client.post("/api/orders/preview", json=outside).status_code == 422
    assert not workers and not list(manager.root.glob("*/"))


def test_http_start_with_order_saves_instruction_and_order(web) -> None:
    """주문으로 실행을 요청하면 서버가 만든 지시문을 CLI 입력으로 저장한다."""
    client, manager, workers = web
    order = {"kind": "board", "parts": [SQUARE, RECT]}
    response = client.post("/api/runs", json={"order": order})
    assert response.status_code == 202, response.text
    view = response.json()
    instruction = client.post("/api/orders/preview", json=order).json()["instruction"]
    assert view["prompt"] == instruction and view["ordered_parts"] == 2
    assert view["order_kind"] == "board"
    assert view["scene"] == "nominal"
    assert view["fixture"] == {"x_mm": 0.0, "y_mm": 0.0, "yaw_deg": 0.0}
    assert [part["type"] for part in view["order"]["parts"]] == ["square_tile", "rect_tile"]
    assert view["order_match"] is None, "레이아웃을 준비하기 전에는 비교가 없다."
    directory = manager.root / view["id"]
    saved = RunRequest.model_validate_json((directory / "request.json").read_text())
    assert saved.prompt == instruction and saved.order is None
    assert json.loads((directory / "order.json").read_text())["parts"][1]["col"] == 5
    assert json.loads((directory / "scene.json").read_text())["board"] is None
    assert len(workers) == 1 and client.get("/api/runs").json()[0]["ordered_parts"] == 2


def _order_parts(layout: Layout) -> list[dict[str, object]]:
    """레이아웃과 같은 부품을 가진 보드 주문의 부품 목록."""
    return [
        {
            "type": part.type.value,
            "color": part.color.value,
            "rotation_deg": int(part.rotation_deg),
            "row": min(row for row, _ in part.cells),
            "col": min(col for _, col in part.cells),
        }
        for part in layout.parts
    ]


def test_order_match_compares_order_with_prepared_layout(web, valid_layout: Layout) -> None:
    """레이아웃을 준비하면 주문과 비교한 결과를 표시하고 실행은 바꾸지 않는다."""
    client, manager, workers = web
    same = _order_parts(valid_layout)
    moved = [*same[:-1], {**same[-1], "row": 6}]
    for parts, expected in (
        (same, {"matches": True, "missing": 0, "extra": 0}),
        (moved, {"matches": False, "missing": 1, "extra": 1}),
        (same[:-1], {"matches": False, "missing": 0, "extra": 1}),
    ):
        view = client.post("/api/runs", json={"order": {"parts": parts}}).json()
        fake = FakeAgentModel()
        runtime, _, _ = make_runtime(manager.root / view["id"] / "run", valid_layout, fake)
        runtime.run()
        for sdk_client in fake.clients:
            sdk_client.close()
        workers[-1].returncode = 0
        result = client.get(f"/api/runs/{view['id']}").json()
        assert result["order_match"] == expected, f"{parts}: {result['order_match']}"
        assert result["status"] == "completed" and result["check"]["passed"]
    assert compose_instruction(valid_layout) == client.get("/api/runs").json()[-1]["prompt"]


def test_replanned_run_shows_and_compares_the_second_layout(web, valid_layout: Layout) -> None:
    """검토의 재계획 뒤에는 두 번째 계획이 현재 레이아웃이다. 화면과 주문 비교가 그 계획을 쓴다."""
    client, manager, workers = web
    second = make_layout(
        make_part("P001", PartType.SQUARE_TILE, origin=(1, 1)),
        make_part("P002", PartType.RECT_TILE, origin=(3, 5), color=Color.GRAY),
    )
    view = client.post("/api/runs", json={"order": {"parts": [SQUARE, RECT]}}).json()
    fake = FakeAgentModel()
    fake.replans = ["part_set_mismatch"]
    runtime, _, _ = make_runtime(
        manager.root / view["id"] / "run",
        valid_layout,
        fake,
        review=True,
        responses=[make_planner_response_json(valid_layout), make_planner_response_json(second)],
    )
    runtime.run()
    for sdk_client in fake.clients:
        sdk_client.close()
    workers[-1].returncode = 0
    result = client.get(f"/api/runs/{view['id']}").json()
    assert result["status"] == "completed", result
    assert len(result["layout"]["parts"]) == 2, result["layout"]
    assert result["order_match"] == {"matches": True, "missing": 0, "extra": 0}
    calls = [event["call"]["name"] for event in result["events"] if event["kind"] == "tool_result"]
    assert calls == ["prepare_layout", "replan_layout", "calibrate_board", "execute_cell"]
    image = manager.asset(view["id"], "layout")
    assert image.parent.name == "02", f"화면의 계획 이미지는 두 번째 시도여야 한다: {image}"


def test_kit_order_run_saves_instruction_and_compares_counts(web) -> None:
    """키트 주문으로 실행하면 수량 지시문을 저장하고 종류와 색별 수량만 비교한다."""
    client, manager, workers = web
    same = make_layout(
        make_part("P001", PartType.SQUARE_TILE, origin=(0, 0)),
        make_part("P002", PartType.SQUARE_TILE, origin=(0, 2)),
        make_part("P003", PartType.RECT_TILE, origin=(2, 0), color=Color.GRAY),
    )
    other = make_layout(
        make_part("P001", PartType.SQUARE_TILE, origin=(0, 0)),
        make_part("P002", PartType.RECT_TILE, origin=(2, 0), color=Color.GRAY),
        make_part("P003", PartType.RECT_TILE, origin=(4, 0), color=Color.WHITE),
    )
    for planned, expected in (
        (same, {"matches": True, "missing": 0, "extra": 0}),
        (other, {"matches": False, "missing": 1, "extra": 1}),
    ):
        response = client.post("/api/runs", json={"order": KIT})
        assert response.status_code == 202, response.text
        view = response.json()
        assert view["prompt"] == KIT_INSTRUCTION
        assert view["ordered_parts"] == 3 and view["order_kind"] == "kit"
        assert view["order"] == KIT and view["order_match"] is None
        directory = manager.root / view["id"]
        assert json.loads((directory / "order.json").read_text()) == KIT
        fake = FakeAgentModel()
        runtime, _, _ = make_runtime(directory / "run", planned, fake)
        runtime.run()
        for sdk_client in fake.clients:
            sdk_client.close()
        workers[-1].returncode = 0
        result = client.get(f"/api/runs/{view['id']}").json()
        assert result["order_match"] == expected, f"실제: {result['order_match']}"
        assert result["status"] == "completed", result["error_code"]
    assert [run["order_kind"] for run in client.get("/api/runs").json()] == ["kit", "kit"]


def test_intake_reads_an_order_and_returns_the_code_instruction(
    web, intake_api: FakeJudgmentModel
) -> None:
    """문장을 접수하면 판단 모델의 주문과 코드가 만든 지시문을 반환한다. 실행은 시작하지 않는다."""
    client, manager, workers = web
    intake_api.replies = [SUBMIT_KIT]
    response = client.post("/api/orders/intake", json={"text": TEXT})
    assert response.status_code == 200, response.text
    view = response.json()
    assert view["outcome"] == "accepted" and view["order"] == KIT
    assert view["preview"] == KIT_PREVIEW
    assert view["model"] == DEFAULT_JUDGMENT_MODEL and view["reject_reason"] is None
    body = json.loads(intake_api.model_posts[0].content)
    assert body["model"] == DEFAULT_JUDGMENT_MODEL and body["messages"][-1]["content"] == TEXT
    saved = (manager.root / "intakes" / view["intake_id"] / INTAKE_FILE).read_text()
    record = IntakeRecord.model_validate_json(saved)
    assert record.intake_id == view["intake_id"] and record.text == TEXT
    assert FAKE_JUDGMENT_KEY not in response.text + saved
    assert not workers and client.get("/api/runs").json() == []


@pytest.mark.parametrize(
    ("reply", "expected"),
    [
        (
            Reply(tool="reject_request", arguments={"reason": "too_many_parts"}),
            {"outcome": "rejected", "reject_reason": "too_many_parts", "order": None},
        ),
        (
            Reply(
                tool="submit_kit_order",
                arguments={**KIT, "items": [{"type": "rect_tile", "count": 5}]},
            ),
            {"outcome": "order_invalid", "reject_reason": None, "order": None},
        ),
        (Reply(content="Done."), {"outcome": "protocol_error", "reject_reason": None}),
    ],
)
def test_intake_without_an_order_has_no_instruction(
    web, intake_api: FakeJudgmentModel, reply: Reply, expected: dict[str, object]
) -> None:
    """거절, 계약 밖의 주문과 응답 형식의 오류는 주문과 지시문이 없는 결과다. HTTP는 200이다."""
    client, _, _ = web
    intake_api.replies = [reply]
    response = client.post("/api/orders/intake", json={"text": TEXT})
    assert response.status_code == 200, response.text
    view = response.json()
    assert view["preview"] is None and view["order"] is None
    assert {key: view[key] for key in expected} == expected, view
    assert FAKE_JUDGMENT_KEY not in response.text


@pytest.mark.parametrize(
    ("failures", "outcome", "waits"),
    [(1, "accepted", [3.0]), (4, "model_error", [3.0, 9.0, 20.0])],
)
def test_intake_sends_a_failed_request_again(
    web,
    intake_api: FakeJudgmentModel,
    retry_waits: list[float],
    failures: int,
    outcome: str,
    waits: list[float],
) -> None:
    """HTTP 500이 1회이면 다시 보내고 수락한다. 4회 연속이면 모델 오류다. HTTP는 200이다."""
    client, _, _ = web
    intake_api.replies = [Reply(status=500)] * failures + [SUBMIT_KIT]
    response = client.post("/api/orders/intake", json={"text": TEXT})
    assert response.status_code == 200, response.text
    view = response.json()
    assert view["outcome"] == outcome and view["reject_reason"] is None, view
    assert (view["order"] is None) == (outcome == "model_error")
    assert len(intake_api.model_posts) == min(failures + 1, 4) and retry_waits == waits
    assert FAKE_JUDGMENT_KEY not in response.text


def test_intake_of_infeasible_board_order_returns_the_issues(
    web, intake_api: FakeJudgmentModel
) -> None:
    """배치 규칙을 어긴 보드 주문은 위반 목록을 반환한다. 지시문은 없다."""
    client, _, _ = web
    overlap = {"kind": "board", "parts": [{**SQUARE, "rotation_deg": 0}] * 2}
    intake_api.replies = [Reply(tool="submit_board_order", arguments=overlap)]
    view = client.post("/api/orders/intake", json={"text": TEXT}).json()
    assert view["outcome"] == "order_not_feasible" and view["order"] == overlap
    assert view["preview"]["instruction"] is None
    assert [issue["code"] for issue in view["preview"]["issues"]] == ["OVERLAP"]


@pytest.mark.parametrize(
    ("body", "headers", "status"),
    [
        ({"text": " "}, {}, 422),
        ({"text": "x" * 4001}, {}, 422),
        ({}, {}, 422),
        ({"text": TEXT, "model": "other"}, {}, 422),
        ({"text": TEXT}, {"Origin": "https://example.com"}, 403),
    ],
)
def test_invalid_intake_request_sends_no_model_request(
    web,
    intake_api: FakeJudgmentModel,
    body: dict[str, object],
    headers: dict[str, str],
    status: int,
) -> None:
    """문장이 계약을 벗어나거나 출처가 다르면 모델 요청과 접수 기록을 만들지 않는다."""
    client, manager, _ = web
    response = client.post("/api/orders/intake", json=body, headers=headers)
    assert response.status_code == status, response.text
    assert not intake_api.requests and not (manager.root / "intakes").exists()


def test_intake_without_model_settings_is_not_available(
    web, intake_api: FakeJudgmentModel, monkeypatch: pytest.MonkeyPatch
) -> None:
    """서버에 Token Factory 설정이 없으면 접수는 503이다. 모델 요청과 기록이 없다."""
    client, manager, _ = web
    monkeypatch.delenv("NEBIUS_API_KEY")
    response = client.post("/api/orders/intake", json={"text": TEXT})
    assert response.status_code == 503, response.text
    assert response.json() == {"detail": "The order reading is not available on this server."}
    assert not intake_api.requests and not (manager.root / "intakes").exists()


def test_second_intake_is_rejected_while_one_is_active(web, intake_api: FakeJudgmentModel) -> None:
    """접수가 실행 중이면 다른 접수는 409다. 모델 요청을 겹쳐 보내지 않는다."""
    client, manager, _ = web
    intake_api.replies = [SUBMIT_KIT]
    with manager.intakes._lock:
        response = client.post("/api/orders/intake", json={"text": TEXT})
        assert response.status_code == 409 and not intake_api.requests, response.text
    assert client.post("/api/orders/intake", json={"text": TEXT}).status_code == 200


def test_run_with_intake_copies_the_intake_record(web, intake_api: FakeJudgmentModel) -> None:
    """접수한 주문으로 실행하면 접수 기록을 실행 디렉터리에 복사한다. 지시문은 코드가 만든다."""
    client, manager, workers = web
    intake_api.replies = [SUBMIT_KIT]
    intake = client.post("/api/orders/intake", json={"text": TEXT}).json()
    response = client.post(
        "/api/runs", json={"order": intake["order"], "intake_id": intake["intake_id"]}
    )
    assert response.status_code == 202, response.text
    view = response.json()
    assert view["intake_id"] == intake["intake_id"] and view["prompt"] == KIT_INSTRUCTION
    assert view["order"] == KIT and len(workers) == 1
    directory = manager.root / view["id"]
    saved = (manager.root / "intakes" / intake["intake_id"] / INTAKE_FILE).read_bytes()
    assert (directory / INTAKE_FILE).read_bytes() == saved
    assert json.loads((directory / "request.json").read_text())["prompt"] == KIT_INSTRUCTION
    assert client.get("/api/runs").json()[0]["intake_id"] == intake["intake_id"]


def test_run_rejects_an_order_that_is_not_the_intake_order(
    web, intake_api: FakeJudgmentModel
) -> None:
    """주문이 수락한 접수의 주문과 다르면 실행을 시작하지 않는다."""
    client, _, workers = web
    intake_api.replies = [SUBMIT_KIT, Reply(tool="reject_request", arguments={"reason": "a"})]
    accepted = client.post("/api/orders/intake", json={"text": TEXT}).json()["intake_id"]
    failed = client.post("/api/orders/intake", json={"text": TEXT}).json()
    assert failed["outcome"] == "protocol_error"
    other = {**KIT, "items": [{"type": "square_tile", "count": 6}]}
    bodies = [
        {"order": other, "intake_id": accepted},
        {"order": KIT, "intake_id": "0" * 32},
        {"order": KIT, "intake_id": failed["intake_id"]},
        {"order": KIT, "intake_id": "../intakes"},
        {"prompt": GOAL, "intake_id": accepted},
    ]
    for body in bodies:
        response = client.post("/api/runs", json=body)
        assert response.status_code == 422, f"{body}: {response.text}"
    assert not workers and client.get("/api/runs").json() == []
    assert client.post("/api/runs", json={"order": KIT, "intake_id": accepted}).status_code == 202


def test_cross_origin_request_is_rejected(web) -> None:
    """다른 출처에서 실행을 요청하면 요청을 거부한다."""
    client, _, workers = web
    response = client.post(
        "/api/runs", json={"prompt": GOAL}, headers={"Origin": "https://example.com"}
    )
    assert response.status_code == 403 and not workers
    assert client.get("/api/runs", headers={"Host": "example.com"}).status_code == 400


def test_terminal_run_and_assets_use_actual_cli_records(web, valid_layout: Layout) -> None:
    """CLI가 완료하면 실제 이벤트와 판정 및 등록한 영상만 전달한다."""
    client, manager, workers = web
    view = client.post("/api/runs", json={"prompt": GOAL}).json()
    fake = FakeAgentModel()
    runtime, tools, _ = make_runtime(manager.root / view["id"] / "run", valid_layout, fake)
    report = runtime.run()
    for sdk_client in fake.clients:
        sdk_client.close()
    workers[0].returncode = 0
    response = client.get(f"/api/runs/{view['id']}")
    result = response.json()
    assert result["status"] == result["runtime_state"] == "completed"
    assert result["report"]["run_id"] == report.run_id
    assert result["check"]["passed"] and result["layout"]["parts"]
    assert [
        event["inference"] for event in result["events"] if event["kind"] == "model_request"
    ] == [1, 2, 3, 4]
    layout = client.get(f"/api/runs/{view['id']}/assets/layout")
    assert layout.status_code == 200 and layout.headers["content-type"] == "image/png"
    video_path = tools.root / "execution" / "sim" / "present.mp4"
    video_path.write_bytes(b"test video content")
    video = client.get(f"/api/runs/{view['id']}/assets/video", headers={"Range": "bytes=0-3"})
    assert video.status_code == 206 and video.content == b"test"
    assert "authorization" not in response.text.lower()
    assert client.get(f"/api/runs/{view['id']}/assets/worker.log").status_code == 422


def _finish_run(web, layout: Layout, *, exit_code: int = 0) -> tuple[str, AssemblyTools]:
    """실행을 시작하고 Fake 실행 프로그램으로 끝낸다. 배치는 현재 `FAKE_ISAAC_MODE`를 따른다."""
    client, manager, workers = web
    run_id = client.post("/api/runs", json={"prompt": GOAL}).json()["id"]
    fake = FakeAgentModel()
    runtime, tools, _ = make_runtime(manager.root / run_id / "run", layout, fake)
    runtime.run()
    for sdk_client in fake.clients:
        sdk_client.close()
    workers[-1].returncode = exit_code
    return run_id, tools


class TestRunVideo:
    """`video` 자료는 발표 영상이다. 발표 영상이 없는 실행은 사선 카메라 영상이다."""

    def test_presentation_video_comes_first(self, web, valid_layout: Layout) -> None:
        """두 영상이 모두 있으면 발표 영상을 전달한다."""
        client, _, _ = web
        run_id, tools = _finish_run(web, valid_layout)
        (tools.root / "execution/sim/present.mp4").write_bytes(b"presentation video")
        (tools.root / "execution/sim/cell.mp4").write_bytes(b"cell video")
        video = client.get(f"/api/runs/{run_id}/assets/video")
        assert video.status_code == 200 and video.content == b"presentation video"

    def test_run_without_presentation_video_serves_cell_video(
        self, web, valid_layout: Layout
    ) -> None:
        """발표 영상이 없는 실행은 사선 카메라 영상을 전달한다."""
        client, _, _ = web
        run_id, tools = _finish_run(web, valid_layout)
        (tools.root / "execution/sim/present.mp4").unlink()
        (tools.root / "execution/sim/cell.mp4").write_bytes(b"cell video")
        assert "video" in client.get(f"/api/runs/{run_id}").json()["assets"]
        video = client.get(f"/api/runs/{run_id}/assets/video")
        assert video.status_code == 200 and video.content == b"cell video"

    def test_run_without_videos_has_no_video_asset(self, web, valid_layout: Layout) -> None:
        """내용이 있는 영상이 없으면 `video` 자료가 없다."""
        client, _, _ = web
        run_id, _ = _finish_run(web, valid_layout)
        assets = client.get(f"/api/runs/{run_id}").json()["assets"]
        assert "result" in assets and "video" not in assets, assets
        assert client.get(f"/api/runs/{run_id}/assets/video").status_code == 404


class TestRunImageCheck:
    """실행 조회가 `execution/image_check.json`의 카메라 판정을 전달한다.

    카메라 판정은 화면 표시에만 쓴다. 이 파일의 상태는 실행 상태와 참값 판정을 바꾸지 않는다.
    """

    def test_completed_run_returns_saved_report(self, web, valid_layout: Layout) -> None:
        """배치가 끝난 실행을 조회하면 저장한 카메라 판정을 그대로 반환한다."""
        client, _, _ = web
        run_id, tools = _finish_run(web, valid_layout)
        result = client.get(f"/api/runs/{run_id}").json()
        saved = json.loads((tools.root / "execution" / IMAGE_CHECK_FILE).read_text())
        assert result["status"] == "completed", result["error_code"]
        assert result["image_check"] == saved, f"실제: {result['image_check']}"
        assert saved["passed"] and saved["match_count"] == len(valid_layout.parts)

    def test_failed_placement_returns_mismatch(
        self, web, valid_layout: Layout, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """실행기가 부품을 옮겨 놓은 실행을 조회하면 카메라 판정의 불일치도 반환한다."""
        client, _, _ = web
        monkeypatch.setenv("FAKE_ISAAC_MODE", "miss")
        run_id, _ = _finish_run(web, valid_layout, exit_code=1)
        result = client.get(f"/api/runs/{run_id}").json()
        assert result["status"] == "aborted" and result["error_code"] == "check_failed"
        assert not result["check"]["passed"]
        image_check = result["image_check"]
        assert not image_check["passed"] and image_check["mismatch_count"] == 1, image_check
        assert image_check["parts"][0]["status"] == "mismatch"

    def test_running_run_has_no_report(self, web) -> None:
        """실행이 끝나기 전에는 카메라 판정이 없다."""
        client, _, _ = web
        view = client.post("/api/runs", json={"prompt": GOAL}).json()
        assert view["status"] == "running" and view["image_check"] is None

    def test_run_without_final_image_has_no_report(
        self, web, valid_layout: Layout, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """카메라 판정 파일이 없는 실행을 조회하면 판정 없이 200이다. 참값 판정은 그대로다."""
        client, _, _ = web
        monkeypatch.setenv("FAKE_ISAAC_MODE", "no_final_image")
        run_id, tools = _finish_run(web, valid_layout)
        assert not (tools.root / "execution" / IMAGE_CHECK_FILE).exists()
        response = client.get(f"/api/runs/{run_id}")
        assert response.status_code == 200, response.text
        result = response.json()
        assert result["image_check"] is None
        assert result["status"] == "completed" and result["check"]["passed"]

    @pytest.mark.parametrize(
        "change",
        ["not_json", "not_utf8", "unknown_field", "wrong_count", "wrong_status"],
    )
    def test_invalid_report_is_not_returned(self, web, valid_layout: Layout, change: str) -> None:
        """카메라 판정 파일의 형식이나 집계가 틀리면 판정 없이 200이다. 409가 아니다."""
        client, _, _ = web
        run_id, tools = _finish_run(web, valid_layout)
        path = tools.root / "execution" / IMAGE_CHECK_FILE
        saved = json.loads(path.read_text())
        if change == "not_json":
            path.write_text("{")
        elif change == "not_utf8":
            path.write_bytes(b"\xff\xfe")
        else:
            if change == "unknown_field":
                saved["note"] = "extra"
            elif change == "wrong_count":
                saved["match_count"] += 1
            else:
                saved["parts"][0]["status"] = "mismatch"
            path.write_text(json.dumps(saved))
        response = client.get(f"/api/runs/{run_id}")
        assert response.status_code == 200, f"{change}: {response.text}"
        result = response.json()
        assert result["image_check"] is None, f"{change}: {result['image_check']}"
        assert result["status"] == "completed" and result["check"]["passed"]


def _displaced_frame(fixture: FixtureOffset) -> BoardFrame:
    """지그 오차를 준 보드의 좌표계. 보정한 셀은 이 보드를 추정한다."""
    cell = CellConfig()
    board = fixture_scene(cell, fixture).board
    if board is not None:
        cell = cell.model_copy(
            update={
                "board_origin": board.board_origin,
                "row_direction": board.row_direction,
                "col_direction": board.col_direction,
            }
        )
    return BoardFrame(cell=cell, cell_size=Board().cell_size_mm / MM_PER_M)


FIXTURE_CORNERS = [
    FixtureOffset(x_mm=x, y_mm=y, yaw_deg=yaw)
    for x in (-FIXTURE_X_LIMIT_MM, FIXTURE_X_LIMIT_MM)
    for y in (-FIXTURE_Y_LIMIT_MM, FIXTURE_Y_LIMIT_MM)
    for yaw in (-FIXTURE_YAW_LIMIT_DEG, FIXTURE_YAW_LIMIT_DEG)
]


class TestSupplyFaults:
    """화면이 고르는 공급 결함의 부품과 색을 검사한다.

    결함은 배치 순서의 첫 부품에 준다. 첫 부품은 보정한 셀에서 정한다.
    다른 색 결함의 색은 그 부품의 계획한 색과 달라야 한다. 같으면 결함이 없는 실행이 된다.
    """

    @pytest.mark.parametrize("name", ["changeover", "kitting", "mixed", "panel"])
    def test_wrong_color_differs_from_the_first_part_at_every_fixture(self, name: str) -> None:
        """지그 오차의 범위 어디에서도 다른 색 결함의 색은 첫 부품의 계획한 색과 다르다."""
        order = parse_order((ORDERS / f"{name}.json").read_bytes())
        assert isinstance(order, BoardOrder)
        faults = supply_faults(order)
        assert faults.missing.model_dump() == {"missing": (1,), "wrong_color": ()}
        assert faults.wrong_color is not None, f"{name}: 다른 색 결함이 없다."
        (fault,) = faults.wrong_color.wrong_color
        assert fault.sequence == 1 and fault.color != Color.GRAY
        parts = order_layout(order).parts
        for fixture in (FixtureOffset(), *FIXTURE_CORNERS):
            frame = _displaced_frame(fixture)
            first = max(parts, key=lambda part: pan_distance(part, frame))
            assert first.color != fault.color, f"{name} {fixture}: {first.part_id}"

    def test_colors_of_near_parts_are_excluded(self) -> None:
        """거리 차이가 작은 두 부품은 모두 첫 부품의 후보다. 두 부품의 색을 모두 뺀다."""
        order = parse_order(
            json.dumps(
                {
                    "kind": "board",
                    "parts": [
                        {"type": "square_tile", "color": "orange", "row": 7, "col": 3},
                        {"type": "square_tile", "color": "blue", "row": 7, "col": 5},
                        {"type": "square_tile", "color": "gray", "row": 0, "col": 3},
                    ],
                }
            )
        )
        faults = supply_faults(order)
        assert faults.wrong_color is not None
        assert faults.wrong_color.wrong_color[0].color == Color.WHITE

    def test_no_wrong_color_when_candidates_use_every_color(self) -> None:
        """첫 부품의 후보가 후보 색을 모두 쓰면 다른 색 결함은 없다. 공급 누락은 있다."""
        order = parse_order(
            json.dumps(
                {
                    "kind": "board",
                    "parts": [
                        {"type": "square_tile", "color": color, "row": 7, "col": col}
                        for color, col in (("orange", 1), ("blue", 3), ("white", 5))
                    ],
                }
            )
        )
        faults = supply_faults(order)
        assert faults.wrong_color is None and faults.missing.missing == (1,)

    def test_kit_order_excludes_the_colors_of_all_items(self) -> None:
        """키트 주문에는 위치가 없다. 모든 종류의 색과 다른 색을 고른다."""
        faults = supply_faults(parse_order(json.dumps(KIT)))
        assert faults.wrong_color is not None
        assert faults.wrong_color.wrong_color[0].color == Color.ORANGE

    def test_margin_is_larger_than_the_change_of_a_distance_difference(self) -> None:
        """후보의 거리 범위는 지그 오차와 보정 오차가 두 부품의 거리 차이를 바꾸는 한도보다 크다."""
        board = Board()
        half_diagonal_mm = math.hypot(board.rows, board.cols) / 2 * board.cell_size_mm
        part_move_mm = (
            math.hypot(FIXTURE_X_LIMIT_MM, FIXTURE_Y_LIMIT_MM)
            + math.radians(FIXTURE_YAW_LIMIT_DEG) * half_diagonal_mm
            + CALIBRATION_ERROR_MM
        )
        limit_mm = 2 * part_move_mm
        assert limit_mm < FIRST_PART_MARGIN_M * MM_PER_M, f"한도: {limit_mm:.2f} mm"


class TestRunFaults:
    """실행 요청의 부품 결함이 장면과 기록으로 가는지 검사한다."""

    def test_faults_are_saved_in_the_scene_and_the_record(self, web) -> None:
        """주문과 결함으로 실행을 요청하면 장면에 결함을 쓰고 실행 기록과 목록에 표시한다."""
        client, manager, workers = web
        faults = {"missing": [2], "wrong_color": [{"sequence": 1, "color": "orange"}]}
        response = client.post(
            "/api/runs", json={"order": {"parts": [SQUARE, RECT]}, "faults": faults}
        )
        assert response.status_code == 202, response.text
        view = response.json()
        assert view["faults"] == faults and view["disposition"] is None
        assert client.get("/api/runs").json()[0]["faults"] == faults
        directory = manager.root / view["id"]
        scene = json.loads((directory / "scene.json").read_text())
        assert scene["faults"] == {
            "missing": [2],
            "wrong_color": [
                {"sequence": 1, "color": "orange", "color_rgb": list(COLOR_RGB[Color.ORANGE])}
            ],
        }
        assert JobRecord.model_validate_json((directory / "job.json").read_text()).faults
        saved = RunRequest.model_validate_json((directory / "request.json").read_text())
        assert saved.faults is None, "CLI의 입력은 결함을 갖지 않는다."
        assert len(workers) == 1

    @pytest.mark.parametrize("faults", [None, {}, {"missing": [], "wrong_color": []}])
    def test_run_without_faults_uses_the_nominal_scene(self, web, faults: object) -> None:
        """결함이 없거나 빈 결함을 주면 장면과 기록에 결함이 없다."""
        client, manager, _ = web
        view = client.post(
            "/api/runs", json={"order": {"parts": [SQUARE, RECT]}, "faults": faults}
        ).json()
        assert view["faults"] is None
        scene = json.loads((manager.root / view["id"] / "scene.json").read_text())
        assert scene["faults"] is None and scene["board"] is None

    @pytest.mark.parametrize(
        "body",
        [
            pytest.param({"prompt": GOAL, "faults": MISSING_FIRST}, id="prompt"),
            pytest.param({"faults": {"missing": [3]}}, id="part_not_in_order"),
            pytest.param({"faults": {"missing": [0]}}, id="sequence_zero"),
            pytest.param({"faults": {**ORANGE_FIRST, "missing": [1]}}, id="two_faults_on_a_part"),
            pytest.param(
                {"faults": {"wrong_color": [{"sequence": 1, "color": "red"}]}}, id="unknown_color"
            ),
            pytest.param({"faults": {"missing": [1], "tilted": [2]}}, id="unknown_fault"),
        ],
    )
    def test_invalid_faults_never_start_a_worker(self, web, body: dict[str, object]) -> None:
        """결함이 주문과 맞지 않거나 계약과 다르면 422다. 실행 기록과 워커가 없다."""
        client, manager, workers = web
        request = body if "prompt" in body else {"order": {"parts": [SQUARE, RECT]}, **body}
        response = client.post("/api/runs", json=request)
        assert response.status_code == 422, response.text
        assert response.json() == {"detail": "The run input is not valid."}
        assert not workers and not [path for path in manager.root.iterdir() if path.is_dir()]


def _finish_inspected_run(
    web, layout: Layout, *, faults: dict[str, object] | None = None, reworks: int = 0
) -> tuple[str, AssemblyTools]:
    """주문과 결함으로 실행을 시작하고 처분을 켠 Fake 실행 프로그램으로 끝낸다.

    실행 프로그램은 서버가 저장한 장면을 읽는다. `reworks`는 모델이 고르는 재작업 호출의 수다.
    """
    client, _, _ = web
    response = client.post(
        "/api/runs", json={"order": {"parts": _order_parts(layout)}, "faults": faults}
    )
    assert response.status_code == 202, response.text
    return _complete_inspected_run(web, response.json()["id"], layout, reworks=reworks)


def _complete_inspected_run(
    web, run_id: str, layout: Layout, *, reworks: int = 0
) -> tuple[str, AssemblyTools]:
    """서버가 시작한 실행을 처분을 켠 Fake 실행 프로그램으로 끝낸다. 마지막 워커가 그 실행의 워커다.

    planner의 답은 `layout`이다. 실행 프로그램은 서버가 저장한 장면을 읽는다.
    """
    _, manager, workers = web
    directory = manager.root / run_id
    fake = FakeAgentModel()
    fake.dispositions = [None] * reworks
    runtime, tools, _ = make_runtime(
        directory / "run",
        layout,
        fake,
        scene=SimScene.model_validate_json((directory / "scene.json").read_text()),
        disposition=True,
    )
    report = runtime.run()
    for sdk_client in fake.clients:
        sdk_client.close()
    workers[-1].returncode = 0 if report.state == ToolState.COMPLETED else 1
    return directory.name, tools


class TestRunDisposition:
    """실행 조회가 처분과 마지막 배치의 판정을 전달하는지 검사한다.

    재작업을 한 실행은 재작업 뒤의 판정이 `check`와 `image_check`다. 첫 판정은 따로 전달한다.
    """

    def test_run_without_faults_is_released(self, web, valid_layout: Layout) -> None:
        """결함이 없는 실행은 출하다. 첫 판정과 재작업 자료는 없다."""
        client, _, _ = web
        run_id, _ = _finish_inspected_run(web, valid_layout)
        result = client.get(f"/api/runs/{run_id}").json()
        assert result["status"] == "completed" and result["disposition"] == "released"
        assert result["check"]["passed"] and result["image_check"]["passed"]
        assert result["first_check"] is None and result["first_image_check"] is None
        assert not {"rework_video", "rework_result"} & set(result["assets"])
        assert client.get(f"/api/runs/{run_id}/assets/rework_result").status_code == 404

    def test_missing_part_run_returns_both_verdicts(self, web, valid_layout: Layout) -> None:
        """공급 누락을 재작업한 실행은 재작업 뒤의 판정과 첫 판정을 함께 반환한다."""
        client, _, _ = web
        run_id, tools = _finish_inspected_run(web, valid_layout, faults=MISSING_FIRST, reworks=1)
        result = client.get(f"/api/runs/{run_id}").json()
        assert result["status"] == "completed", result["error_code"]
        assert result["disposition"] == "reworked_released" and result["report"]["reworked"]
        assert client.get("/api/runs").json()[0]["disposition"] == "reworked_released"
        parts = len(valid_layout.parts)
        assert result["check"]["passed_count"] == parts and result["image_check"]["passed"]
        assert result["first_check"]["passed_count"] == parts - 1, result["first_check"]
        first = result["first_image_check"]
        assert first["mismatch_count"] == 1 and not first["passed"], first
        assert [part["empty"] for part in first["parts"] if part["status"] == "mismatch"] == [True]
        rework = tools.root / "execution" / "rework"
        assert result["check"] == json.loads((rework / CHECK_FILE).read_text())
        assert result["image_check"] == json.loads((rework / IMAGE_CHECK_FILE).read_text())

    def test_rework_recording_is_an_asset_after_the_run(self, web, valid_layout: Layout) -> None:
        """재작업을 한 실행은 재작업의 최종 이미지와 영상을 자료로 전달한다."""
        client, _, _ = web
        run_id, tools = _finish_inspected_run(web, valid_layout, faults=MISSING_FIRST, reworks=1)
        (tools.root / "execution/rework/sim/present.mp4").write_bytes(b"rework video")
        assets = client.get(f"/api/runs/{run_id}").json()["assets"]
        assert {"result", "rework_result", "rework_video"} <= set(assets), assets
        image = client.get(f"/api/runs/{run_id}/assets/rework_result")
        assert image.status_code == 200 and image.headers["content-type"] == "image/png"
        video = client.get(f"/api/runs/{run_id}/assets/rework_video")
        assert video.status_code == 200 and video.headers["content-type"] == "video/mp4"
        assert video.content == b"rework video"

    def test_rework_without_presentation_video_serves_cell_video(
        self, web, valid_layout: Layout
    ) -> None:
        """발표 영상이 없는 재작업은 사선 카메라 영상을 `rework_video`로 전달한다."""
        client, _, _ = web
        run_id, tools = _finish_inspected_run(web, valid_layout, faults=MISSING_FIRST, reworks=1)
        (tools.root / "execution/rework/sim/present.mp4").unlink()
        (tools.root / "execution/rework/sim/cell.mp4").write_bytes(b"rework cell video")
        assert "rework_video" in client.get(f"/api/runs/{run_id}").json()["assets"]
        video = client.get(f"/api/runs/{run_id}/assets/rework_video")
        assert video.status_code == 200 and video.content == b"rework cell video"

    def test_wrong_color_run_is_rejected(self, web, valid_layout: Layout) -> None:
        """다른 색 부품의 실행은 불량이다. 참값 판정은 통과하고 카메라 판정은 실패한다."""
        client, _, _ = web
        run_id, _ = _finish_inspected_run(web, valid_layout, faults=ORANGE_FIRST)
        result = client.get(f"/api/runs/{run_id}").json()
        assert result["status"] == "aborted" and result["disposition"] == "rejected"
        assert result["error_code"] == "inspection_rejected"
        assert result["check"]["passed"] and not result["image_check"]["passed"]
        assert result["first_check"] is None and "rework_result" not in result["assets"]

    def test_missing_part_without_rework_is_rejected(self, web, valid_layout: Layout) -> None:
        """모델이 재작업을 고르지 않은 공급 누락의 실행은 불량이다. 판정은 첫 배치의 판정이다."""
        client, _, _ = web
        run_id, _ = _finish_inspected_run(web, valid_layout, faults=MISSING_FIRST)
        result = client.get(f"/api/runs/{run_id}").json()
        assert result["disposition"] == "rejected" and result["error_code"] == "check_failed"
        assert not result["check"]["passed"] and result["first_check"] is None

    def test_rework_without_a_verdict_shows_the_first_verdict(
        self, web, valid_layout: Layout
    ) -> None:
        """재작업의 판정 파일이 없으면 첫 배치의 판정을 반환한다. 첫 판정은 따로 없다."""
        client, _, _ = web
        run_id, tools = _finish_inspected_run(web, valid_layout, faults=MISSING_FIRST, reworks=1)
        (tools.root / "execution" / "rework" / CHECK_FILE).unlink()
        result = client.get(f"/api/runs/{run_id}").json()
        assert not result["check"]["passed"] and not result["image_check"]["passed"]
        assert result["first_check"] is None and result["first_image_check"] is None

    def test_running_run_has_no_disposition(self, web, valid_layout: Layout) -> None:
        """워커가 끝나기 전에는 처분이 없다. 보고서가 있어도 같다."""
        client, _, workers = web
        run_id, _ = _finish_inspected_run(web, valid_layout)
        workers[-1].returncode = None
        result = client.get(f"/api/runs/{run_id}").json()
        assert result["status"] == "running" and result["disposition"] is None


PAIR = make_layout(
    make_part("P001", PartType.SQUARE_TILE, origin=(1, 1)),
    make_part("P002", PartType.RECT_TILE, origin=(3, 5), color=Color.GRAY),
)
SHIFT_ITEM = {"name": "first", "order": {"parts": [SQUARE, RECT]}}


def _shift_item(name: str, layout: Layout, **fields: object) -> dict[str, object]:
    """레이아웃과 같은 부품의 보드 주문을 가진 교대의 주문 1건."""
    return {"name": name, "order": {"parts": _order_parts(layout)}, **fields}


def _start_shift(web, items: list[dict[str, object]]) -> dict[str, Any]:
    client, _, _ = web
    response = client.post("/api/shifts", json={"items": items})
    assert response.status_code == 202, response.text
    return response.json()


def _finish_current_order(
    web, desk: ShiftDesk, shift: dict[str, Any], layout: Layout, *, reworks: int = 0
) -> dict[str, Any]:
    """교대의 현재 주문을 Fake 실행 프로그램으로 끝내고 교대를 진행한다. 새 교대 응답을 돌려준다."""
    client, _, _ = web
    run_id = [item["run_id"] for item in shift["items"] if item["run_id"] is not None][-1]
    _complete_inspected_run(web, run_id, layout, reworks=reworks)
    desk.advance()
    return client.get(f"/api/shifts/{shift['id']}").json()


class TestShift:
    """교대 API가 주문을 1건씩 차례로 실행하고 주문별 지표와 합계를 전달하는지 검사한다.

    주문의 실행은 서버가 시작한 실행이다. 실행 기록은 Fake 실행 프로그램과 가짜 실행기가 만든다.
    """

    def test_shift_runs_its_orders_one_after_the_other(
        self, web, desk: ShiftDesk, valid_layout: Layout
    ) -> None:
        """교대를 요청하면 첫 주문만 시작한다. 그 주문이 끝난 뒤에 다음 주문을 시작한다."""
        client, manager, workers = web
        fixture = {"x_mm": 3.0, "y_mm": 5.0, "yaw_deg": 1.5}
        items = [_shift_item("first", valid_layout), _shift_item("second", PAIR, fixture=fixture)]
        shift = _start_shift(web, items)
        first, second = shift["items"]
        assert shift["status"] == "running" and shift["finished_at"] is None
        assert (first["status"], second["status"]) == ("running", "queued")
        assert first["run_id"] == client.get("/api/runs").json()[0]["id"]
        assert second["run_id"] is None and first["report"] is None
        assert (first["ordered_parts"], second["ordered_parts"]) == (4, 2)
        assert first["order_kind"] == "board" and second["fixture"] == fixture
        assert shift["totals"]["orders"] == 2 and shift["totals"]["first_pass_yield"] is None
        # 교대가 열린 동안 셀은 다른 실행과 다른 교대를 받지 않는다.
        assert client.post("/api/runs", json={"prompt": GOAL}).status_code == 409
        assert client.post("/api/shifts", json={"items": items}).status_code == 409
        desk.advance()
        assert len(workers) == 1, "첫 주문이 끝나기 전에는 다음 주문을 시작하지 않는다."

        shift = _finish_current_order(web, desk, shift, valid_layout)
        first, second = shift["items"]
        assert (first["status"], first["disposition"]) == ("completed", "released")
        assert second["status"] == "running" and len(workers) == 2
        report = first["report"]
        assert report["name"] == "first" and report["directory"] == first["run_id"]
        assert report["passed_parts"] == report["part_count"] == 4
        # 가짜 실행기의 `sim_time_s`는 1.0이다.
        assert report["robot_time_s"] == 1.0 and report["nemotron_requests"] > 0
        assert report["model_time_s"] <= report["wall_time_s"]
        scene = json.loads((manager.root / second["run_id"] / "scene.json").read_text())
        assert scene["board"] is not None, "두 번째 주문의 지그 오차가 장면에 없다."

        shift = _finish_current_order(web, desk, shift, PAIR)
        assert shift["status"] == "finished" and shift["finished_at"] is not None
        assert [item["disposition"] for item in shift["items"]] == ["released", "released"]
        totals = shift["totals"]
        assert (totals["orders"], totals["completed_orders"], totals["first_pass_orders"]) == (
            2,
            2,
            2,
        )
        assert totals["first_pass_yield"] == totals["final_yield"] == 1.0
        assert totals["changeovers"] == 1 and totals["rejected_orders"] == 0
        assert totals["wall_time_s"] == sum(
            item["report"]["wall_time_s"] for item in shift["items"]
        )
        desk.advance()
        assert len(workers) == 2, "닫힌 교대는 실행을 시작하지 않는다."
        saved = ShiftRecord.model_validate_json(
            (manager.root / SHIFTS_DIR / shift["id"] / SHIFT_FILE).read_text()
        )
        assert saved.status == "finished"
        assert [entry.run_id for entry in saved.items] == [first["run_id"], second["run_id"]]
        assert client.get("/api/shifts").json() == [shift]
        assert client.post("/api/runs", json={"prompt": GOAL}).status_code == 202, "예약이 남았다."

    def test_shift_continues_after_a_rework_and_a_rejection(
        self, web, desk: ShiftDesk, valid_layout: Layout
    ) -> None:
        """공급 누락은 재작업 뒤 출하, 다른 색은 불량이다. 불량 뒤에도 다음 주문을 실행한다."""
        shift = _start_shift(
            web,
            [
                _shift_item("missing", valid_layout, faults=MISSING_FIRST),
                _shift_item("wrong-color", valid_layout, faults=ORANGE_FIRST),
                _shift_item("clean", valid_layout, faults={}),
            ],
        )
        shift = _finish_current_order(web, desk, shift, valid_layout, reworks=1)
        shift = _finish_current_order(web, desk, shift, valid_layout)
        assert shift["status"] == "running" and shift["items"][2]["status"] == "running"
        shift = _finish_current_order(web, desk, shift, valid_layout)
        reworked, rejected, clean = shift["items"]
        assert shift["status"] == "finished"
        assert [item["disposition"] for item in shift["items"]] == [
            "reworked_released",
            "rejected",
            "released",
        ]
        assert [item["status"] for item in shift["items"]] == ["completed", "aborted", "completed"]
        assert reworked["faults"] == MISSING_FIRST and clean["faults"] is None
        assert reworked["report"]["reworked"] and reworked["report"]["robot_time_s"] == 2.0
        assert rejected["report"]["error_code"] == "inspection_rejected"
        assert rejected["report"]["passed_parts"] == rejected["report"]["part_count"]
        totals = shift["totals"]
        assert (totals["completed_orders"], totals["first_pass_orders"]) == (2, 1)
        assert totals["rejected_orders"] == 1 and totals["changeovers"] == 0
        assert totals["first_pass_yield"] == pytest.approx(1 / 3)
        assert totals["final_yield"] == pytest.approx(2 / 3)

    def test_cancel_stops_the_current_order_and_skips_the_rest(self, web, desk: ShiftDesk) -> None:
        """교대를 중단하면 현재 주문의 워커를 멈추고 남은 주문을 시작하지 않는다."""
        client, _, workers = web
        shift = _start_shift(web, [SHIFT_ITEM, {**SHIFT_ITEM, "name": "second"}])
        stopped = client.post(f"/api/shifts/{shift['id']}/cancel")
        assert stopped.status_code == 200, stopped.text
        view = stopped.json()
        assert view["status"] == "cancelled" and view["finished_at"] is not None
        assert [item["status"] for item in view["items"]] == ["cancelled", "skipped"]
        assert view["items"][0]["report"] is None and workers[0].returncode == 130
        desk.advance()
        assert len(workers) == 1
        assert client.post(f"/api/shifts/{shift['id']}/cancel").json() == view
        assert client.post("/api/runs", json={"prompt": GOAL}).status_code == 202, "예약이 남았다."
        assert client.get("/api/shifts/unknown").status_code == 404
        assert client.post("/api/shifts/unknown/cancel").status_code == 404

    def test_cancel_of_the_current_run_closes_the_shift(self, web, desk: ShiftDesk) -> None:
        """교대의 현재 실행을 중단하면 교대도 닫는다. 남은 주문을 시작하지 않는다."""
        client, _, workers = web
        shift = _start_shift(web, [SHIFT_ITEM, {**SHIFT_ITEM, "name": "second"}])
        run_id = shift["items"][0]["run_id"]
        assert client.post(f"/api/runs/{run_id}/cancel").json()["status"] == "cancelled"
        desk.advance()
        view = client.get(f"/api/shifts/{shift['id']}").json()
        assert view["status"] == "cancelled" and len(workers) == 1
        assert [item["status"] for item in view["items"]] == ["cancelled", "skipped"]

    def test_worker_without_a_report_closes_the_shift(self, web, desk: ShiftDesk) -> None:
        """워커가 실행 보고서 없이 끝나면 교대를 중단으로 닫는다. 남은 주문을 시작하지 않는다."""
        client, _, workers = web
        shift = _start_shift(web, [SHIFT_ITEM, {**SHIFT_ITEM, "name": "second"}])
        workers[0].returncode = 2
        desk.advance()
        view = client.get(f"/api/shifts/{shift['id']}").json()
        assert view["status"] == "aborted" and len(workers) == 1
        assert [item["status"] for item in view["items"]] == ["aborted", "skipped"]
        assert view["items"][0]["report"] is None and view["totals"]["final_yield"] is None

    def test_spawn_failure_closes_the_shift(self, web, monkeypatch: pytest.MonkeyPatch) -> None:
        """첫 워커를 시작하지 못하면 교대를 중단으로 닫고 셀의 예약을 남기지 않는다."""
        client, _, workers = web
        spawn = subprocess.Popen

        def fail(*args: Any, **kwargs: Any) -> None:
            raise OSError("private worker details")

        monkeypatch.setattr(subprocess, "Popen", fail)
        response = client.post("/api/shifts", json={"items": [SHIFT_ITEM]})
        assert response.status_code == 503 and "private worker details" not in response.text
        (view,) = client.get("/api/shifts").json()
        assert view["status"] == "aborted" and view["items"][0]["status"] == "skipped"
        monkeypatch.setattr(subprocess, "Popen", spawn)
        assert client.post("/api/runs", json={"prompt": GOAL}).status_code == 202
        assert len(workers) == 1

    def test_record_write_failure_starts_no_shift(
        self, web, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """교대 기록을 쓰지 못하면 교대와 워커를 만들지 않고 셀의 예약을 남기지 않는다."""
        from pixel2place.demo import shifts

        client, _, workers = web

        def fail(*args: Any, **kwargs: Any) -> None:
            raise OSError("private record details")

        with monkeypatch.context() as patch:
            patch.setattr(shifts, "write_record", fail)
            response = client.post("/api/shifts", json={"items": [SHIFT_ITEM]})
        assert response.status_code == 503 and "private record details" not in response.text
        assert not workers and client.get("/api/shifts").json() == []
        assert client.post("/api/runs", json={"prompt": GOAL}).status_code == 202, "예약이 남았다."

    @pytest.mark.parametrize("step", ["advance", "cancel"])
    def test_shift_without_its_run_id_is_closed(
        self, cell, monkeypatch: pytest.MonkeyPatch, step: str
    ) -> None:
        """시작한 실행의 ID를 기록하지 못한 교대는 다음 확인 또는 중단에서 닫힌다.

        그 실행은 교대 기록에 없다. 교대는 그 실행을 기다리지 않는다. 다른 주문을 시작하지 않는다.
        """
        from pixel2place.demo import shifts

        manager, workers, desk = cell
        save = shifts.write_record
        writes = 0

        def fail_second(*args: Any, **kwargs: Any) -> None:
            nonlocal writes
            writes += 1
            if writes == 2:
                raise OSError("record write failed")
            save(*args, **kwargs)

        monkeypatch.setattr(shifts, "write_record", fail_second)
        request = ShiftRequest.model_validate_json(
            json.dumps({"items": [SHIFT_ITEM, {**SHIFT_ITEM, "name": "second"}]})
        )
        with pytest.raises(OSError, match="record write failed"):
            desk.start(request)
        (view,) = desk.recent()
        assert view.status == "running" and view.items[0].run_id is None and len(workers) == 1
        if step == "advance":
            with pytest.raises(RunBusyError):
                desk.advance()
        else:
            assert desk.cancel(view.id).status == "cancelled"
        (view,) = desk.recent()
        assert view.status == ("aborted" if step == "advance" else "cancelled")
        assert [item.status for item in view.items] == ["skipped", "skipped"]
        assert len(workers) == 1 and workers[0].returncode is None
        workers[0].returncode = 2
        manager.start(RunRequest(prompt=GOAL))
        assert len(workers) == 2, "예약이 남았다."

    @pytest.mark.parametrize(
        "items",
        [
            pytest.param([], id="no_orders"),
            pytest.param([SHIFT_ITEM] * (MAX_SHIFT_REQUEST_ORDERS + 1), id="too_many_orders"),
            pytest.param([{**SHIFT_ITEM, "name": "Not A Name"}], id="name"),
            pytest.param([{"name": "first", "prompt": GOAL}], id="prompt"),
            pytest.param([{**SHIFT_ITEM, "faults": {"missing": [3]}}], id="fault_for_absent_part"),
            pytest.param(
                [SHIFT_ITEM, {"name": "second", "order": {"parts": [SQUARE, SQUARE]}}],
                id="second_order_not_feasible",
            ),
            pytest.param([{**SHIFT_ITEM, "fixture": {"x_mm": 3.1}}], id="fixture_out_of_range"),
            pytest.param([{**SHIFT_ITEM, "intake_id": "a" * 32}], id="unknown_field"),
        ],
    )
    def test_invalid_shift_never_starts_a_worker(self, web, items: list[dict[str, object]]) -> None:
        """교대의 입력이 계약이나 배치 규칙을 벗어나면 422다. 교대 기록, 실행 기록과 워커가 없다."""
        client, manager, workers = web
        response = client.post("/api/shifts", json={"items": items})
        assert response.status_code == 422, response.text
        assert not workers and client.get("/api/shifts").json() == []
        assert not [path for path in manager.root.iterdir() if path.is_dir()]
        assert client.post("/api/runs", json={"prompt": GOAL}).status_code == 202, "예약이 남았다."

    def test_largest_shift_fits_in_the_body_limit(self, web) -> None:
        """부품 수가 상한인 주문을 상한만큼 가진 교대는 들여쓴 JSON으로도 본문 한도 안이다."""
        client, _, workers = web
        # 화면이 보내는 부품의 형식이다. 종류와 색은 이름이 가장 긴 값이다.
        parts = [
            {"type": "square_tile", "color": "orange", "rotation_deg": 0, "row": row, "col": col}
            for row in range(0, 8, 2)
            for col in range(0, 8, 2)
        ]
        assert len(parts) == MAX_ORDER_PARTS
        items = [
            {"name": f"order-{index}", "order": {"kind": "board", "parts": parts}}
            for index in range(MAX_SHIFT_REQUEST_ORDERS)
        ]
        body = json.dumps({"items": items}, indent=2)
        assert len(body) < MAX_BODY_BYTES, f"본문: {len(body)}바이트"
        response = client.post(
            "/api/shifts", content=body, headers={"Content-Type": "application/json"}
        )
        assert response.status_code == 202, response.text
        assert len(response.json()["items"]) == MAX_SHIFT_REQUEST_ORDERS and len(workers) == 1

    def test_cross_origin_shift_requests_are_rejected(self, web) -> None:
        """다른 출처에서 교대를 요청하거나 중단하면 요청을 거부한다."""
        client, _, workers = web
        headers = {"Origin": "https://example.com"}
        response = client.post("/api/shifts", json={"items": [SHIFT_ITEM]}, headers=headers)
        assert response.status_code == 403 and not workers
        shift = _start_shift(web, [SHIFT_ITEM])
        stopped = client.post(f"/api/shifts/{shift['id']}/cancel", headers=headers)
        assert stopped.status_code == 403
        assert client.get(f"/api/shifts/{shift['id']}").json()["status"] == "running"

    def test_monitor_starts_the_next_order_without_browser_polling(
        self, web, valid_layout: Layout
    ) -> None:
        """브라우저 조회가 없어도 서버가 끝난 주문 뒤의 주문을 시작한다."""
        _, _, workers = web
        shift = _start_shift(
            web, [_shift_item("first", valid_layout), _shift_item("second", valid_layout)]
        )
        _complete_inspected_run(web, shift["items"][0]["run_id"], valid_layout)
        deadline = monotonic() + 3
        while len(workers) < 2 and monotonic() < deadline:
            sleep(0.02)
        assert len(workers) == 2

    def test_reserved_cell_starts_only_queued_runs(self, web) -> None:
        """셀을 예약하면 큐의 실행만 시작한다. 예약을 풀면 다른 실행을 받는다."""
        client, manager, workers = web
        manager.reserve()
        with pytest.raises(RunBusyError):
            manager.reserve()
        assert client.post("/api/runs", json={"prompt": GOAL}).status_code == 409
        assert not workers
        manager.release()
        assert client.post("/api/runs", json={"prompt": GOAL}).status_code == 202
        with pytest.raises(RunBusyError):
            manager.reserve()

    def test_order_report_needs_a_finished_order_run(self, web, valid_layout: Layout) -> None:
        """끝나지 않은 실행과 주문이 없는 실행에는 교대의 지표가 없다."""
        client, manager, workers = web
        run_id = client.post("/api/runs", json={"order": {"parts": [SQUARE]}}).json()["id"]
        assert manager.order_report(run_id, name="first") is None
        workers[-1].returncode = 2
        assert manager.order_report(run_id, name="first") is None, "실행 보고서가 없다."
        prompt_run, _ = _finish_run(web, valid_layout)
        assert manager.get(prompt_run).report is not None
        assert manager.order_report(prompt_run, name="first") is None, "주문이 없다."
        with pytest.raises(KeyError):
            manager.order_report("unknown", name="first")


def test_asset_symlink_cannot_escape_run_root(web, tmp_path: Path, valid_layout: Layout) -> None:
    """산출물 링크가 실행 루트를 벗어나면 파일을 전달하지 않는다."""
    client, manager, workers = web
    view = client.post("/api/runs", json={"prompt": GOAL}).json()
    fake = FakeAgentModel()
    runtime, tools, _ = make_runtime(manager.root / view["id"] / "run", valid_layout, fake)
    runtime.run()
    for sdk_client in fake.clients:
        sdk_client.close()
    workers[0].returncode = 0
    private = tmp_path / "private.png"
    private.write_bytes(b"private file")
    image = tools.root / "attempts" / "01" / "normalized_image.png"
    image.unlink()
    image.symlink_to(private)
    response = client.get(f"/api/runs/{view['id']}/assets/layout")
    assert response.status_code == 404 and "private file" not in response.text


def test_cancel_and_shutdown_stop_worker(web) -> None:
    """중단 또는 서버 종료를 요청하면 소유한 워커를 종료한다."""
    client, manager, workers = web
    run_id = client.post("/api/runs", json={"prompt": GOAL}).json()["id"]
    stopped = client.post(f"/api/runs/{run_id}/cancel")
    assert stopped.status_code == 200 and stopped.json()["status"] == "cancelled"
    assert stopped.json()["error_code"] == "user_cancelled"
    assert stopped.json()["disposition"] is None
    assert workers[0].returncode == 130
    assert client.post(f"/api/runs/{run_id}/cancel").json()["status"] == "cancelled"
    client.post("/api/runs", json={"prompt": GOAL})
    manager.close()
    assert workers[1].returncode == 130
    assert client.post("/api/runs", json={"prompt": GOAL}).status_code == 409


def test_cli_preflight_failure_is_terminal_without_fake_success(web) -> None:
    """CLI 사전 검사가 실패하면 완료나 판정을 만들지 않는다."""
    client, _, workers = web
    run_id = client.post("/api/runs", json={"prompt": GOAL}).json()["id"]
    workers[0].returncode = 2
    result = client.get(f"/api/runs/{run_id}").json()
    assert result["status"] == "aborted" and result["error_code"] == "cli_failed"
    assert result["report"] is None and result["check"] is None
    assert result["disposition"] is None, "보고서가 없는 실행에는 처분이 없다."
    assert client.get("/api/runs/unknown").status_code == 404


def test_server_lock_and_interrupted_history(tmp_path: Path) -> None:
    """같은 루트의 서버를 거부하고 종료한 실행을 자동 재개하지 않는다."""
    root = tmp_path / "runs"
    manager = JobManager(root=root, workspace=tmp_path)
    with pytest.raises(RunBusyError):
        JobManager(root=root, workspace=tmp_path)
    manager.close()
    run_id = "a" * 32
    directory = root / run_id
    directory.mkdir()
    record = JobRecord(
        id=run_id, prompt=GOAL, scene="nominal", status="running", created_at=datetime.now(UTC)
    )
    (directory / "job.json").write_text(record.model_dump_json())
    restored = JobManager(root=root, workspace=tmp_path)
    assert restored.get(run_id).status == "interrupted"
    assert restored.get(run_id).error_code == "server_interrupted"
    restored.close()


def test_server_stop_interrupts_the_shift(served, tmp_path: Path) -> None:
    """서버가 멈추면 열린 교대를 닫는다. 다시 시작한 서버는 남은 주문을 시작하지 않는다."""
    client, manager, workers, desk = served
    shift = _start_shift((client, manager, workers), [SHIFT_ITEM, {**SHIFT_ITEM, "name": "second"}])
    desk.close()
    manager.close()
    assert workers[0].returncode == 130
    restored = JobManager(root=manager.root, workspace=tmp_path)
    try:
        again = ShiftDesk(jobs=restored)
        view = again.get(shift["id"])
        assert view.status == "interrupted" and view.finished_at is not None
        assert [item.status for item in view.items] == ["cancelled", "skipped"]
        again.advance()
        assert len(workers) == 1
        restored.start(RunRequest(prompt=GOAL))
        assert len(workers) == 2, "다시 시작한 서버에 예약이 남았다."
    finally:
        restored.close()


def _open_shift_record(root: Path, *, directory_name: str | None = None) -> tuple[str, Path]:
    """실행 중인 실행 1건과 열린 교대 1개의 기록을 쓴다. 서버가 닫지 못한 기록이다."""
    run_id, shift_id = "a" * 32, "b" * 32
    (root / run_id).mkdir(parents=True)
    job = JobRecord(
        id=run_id, prompt=GOAL, scene="nominal", status="running", created_at=datetime.now(UTC)
    )
    (root / run_id / "job.json").write_text(job.model_dump_json())
    request = ShiftOrderRequest(name="first", order=parse_order(json.dumps({"parts": [SQUARE]})))
    record = ShiftRecord(
        id=shift_id,
        status="running",
        created_at=datetime.now(UTC),
        items=(ShiftEntry(request=request, run_id=run_id), ShiftEntry(request=request)),
    )
    path = root / SHIFTS_DIR / (directory_name or shift_id) / SHIFT_FILE
    path.parent.mkdir(parents=True)
    path.write_text(record.model_dump_json())
    return shift_id, path


def test_open_shift_record_is_restored_as_interrupted(tmp_path: Path) -> None:
    """열린 채로 남은 교대 기록을 읽으면 `interrupted`로 닫는다. 기록도 고친다."""
    root = tmp_path / "runs"
    shift_id, path = _open_shift_record(root)
    manager = JobManager(root=root, workspace=tmp_path)
    try:
        view = ShiftDesk(jobs=manager).get(shift_id)
        assert view.status == "interrupted" and view.finished_at is not None
        assert [item.status for item in view.items] == ["interrupted", "skipped"]
        assert ShiftRecord.model_validate_json(path.read_text()).status == "interrupted"
    finally:
        manager.close()


def test_shift_record_in_another_directory_is_rejected(tmp_path: Path) -> None:
    """교대 기록의 ID가 디렉터리 이름과 다르면 교대 창구를 만들지 않는다."""
    root = tmp_path / "runs"
    _open_shift_record(root, directory_name="c" * 32)
    manager = JobManager(root=root, workspace=tmp_path)
    try:
        with pytest.raises(ValueError, match="Shift identity"):
            ShiftDesk(jobs=manager)
    finally:
        manager.close()


def test_order_record_without_kind_is_restored_as_board_order(tmp_path: Path) -> None:
    """주문 종류가 없는 기존 주문 기록을 읽으면 보드 주문으로 표시한다."""
    root = tmp_path / "runs"
    run_id = "c" * 32
    (root / run_id).mkdir(parents=True)
    record = JobRecord(
        id=run_id,
        prompt=GOAL,
        scene="nominal",
        status="completed",
        created_at=datetime.now(UTC),
        ordered_parts=2,
    )
    (root / run_id / "job.json").write_text(record.model_dump_json(exclude={"order_kind"}))
    manager = JobManager(root=root, workspace=tmp_path)
    try:
        summary = manager.list()[0]
        assert summary.ordered_parts == 2 and summary.order_kind == "board"
        assert summary.faults is None and summary.disposition is None
    finally:
        manager.close()


def test_record_with_scene_preset_only_is_restored_with_its_fixture(tmp_path: Path) -> None:
    """오차 값이 없는 기존 기록을 읽으면 장면 프리셋의 오차를 표시한다."""
    root = tmp_path / "runs"
    for run_id, scene in (("a" * 32, "nominal"), ("b" * 32, "board_offset")):
        (root / run_id).mkdir(parents=True)
        (root / run_id / "job.json").write_text(
            json.dumps(
                {
                    "id": run_id,
                    "prompt": GOAL,
                    "scene": scene,
                    "status": "completed",
                    "created_at": "2026-09-30T12:07:21.873211Z",
                    "finished_at": "2026-09-30T12:10:48.980513Z",
                    "worker_pid": 1,
                    "exit_code": 0,
                }
            )
        )
    manager = JobManager(root=root, workspace=tmp_path)
    try:
        nominal, shifted = manager.get("a" * 32), manager.get("b" * 32)
        assert nominal.fixture.is_nominal and nominal.ordered_parts is None
        assert nominal.order_kind is None
        assert (shifted.fixture.x_mm, shifted.fixture.y_mm, shifted.fixture.yaw_deg) == (3, 5, 1.5)
        assert shifted.order is None and shifted.order_match is None
    finally:
        manager.close()


def test_real_worker_exits_on_missing_configuration(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """인증 설정이 없으면 실제 워커가 CLI 오류로 종료한다."""
    monkeypatch.delenv("NEBIUS_API_KEY", raising=False)
    manager = JobManager(root=tmp_path / "runs", workspace=tmp_path)
    try:
        view = manager.start(RunRequest(prompt="--help"))
        deadline = monotonic() + 5
        while view.status == "running" and monotonic() < deadline:
            sleep(0.05)
            view = manager.get(view.id)
        assert view.status == "aborted" and view.error_code == "cli_failed"
        assert not view.events and not view.assets
        record = JobRecord.model_validate_json((manager.root / view.id / "job.json").read_text())
        assert record.exit_code == 2
    finally:
        manager.close()


@pytest.mark.parametrize(
    ("env_host", "expected"),
    [(None, "127.0.0.1"), ("192.0.2.10", "192.0.2.10")],
    ids=["default-loopback", "env-address"],
)
def test_demo_cli_contract(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, env_host: str | None, expected: str
) -> None:
    """웹 CLI를 시작하면 실행 루트, 포트와 `DEMO_HOST` 주소를 서버에 전달한다."""
    from pixel2place.demo import server

    monkeypatch.delenv("DEMO_HOST", raising=False)
    if env_host is not None:
        monkeypatch.setenv("DEMO_HOST", env_host)
    received: list[dict[str, object]] = []
    monkeypatch.setattr(server, "serve", lambda **options: received.append(options))
    result = CliRunner().invoke(app, ["demo-serve", "--runs", str(tmp_path), "--port", "8765"])
    assert result.exit_code == 0 and received[0]["root"] == tmp_path
    assert received[0]["port"] == 8765 and received[0]["workspace"] == Path.cwd()
    assert received[0]["host"] == expected


@pytest.mark.parametrize("env_host", ["0.0.0.0", "demo.example", ""])
def test_demo_cli_rejects_invalid_host(monkeypatch: pytest.MonkeyPatch, env_host: str) -> None:
    """`DEMO_HOST`가 인터페이스 1개의 IPv4 주소가 아니면 서버를 시작하지 않고 exit 2다."""
    from pixel2place.demo import server

    monkeypatch.setenv("DEMO_HOST", env_host)
    received: list[dict[str, object]] = []
    monkeypatch.setattr(server, "serve", lambda **options: received.append(options))
    result = CliRunner().invoke(app, ["demo-serve"])
    assert result.exit_code == 2 and not received
    assert "DEMO_HOST" in result.output


def test_spawn_failure_is_saved_and_does_not_block_next_run(web, monkeypatch: pytest.MonkeyPatch):
    """워커 시작이 실패하면 오류를 저장하고 실행 잠금을 남기지 않는다."""
    client, _, workers = web
    spawn = subprocess.Popen

    def fail(*args: Any, **kwargs: Any) -> None:
        raise OSError("private worker details")

    monkeypatch.setattr(subprocess, "Popen", fail)
    response = client.post("/api/runs", json={"prompt": GOAL})
    assert response.status_code == 503 and "private worker details" not in response.text
    assert client.get("/api/runs").json()[0]["status"] == "aborted"
    monkeypatch.setattr(subprocess, "Popen", spawn)
    assert client.post("/api/runs", json={"prompt": GOAL}).status_code == 202
    assert len(workers) == 1


def test_recorded_reference_is_separate_and_rejects_changed_verdict(web, valid_layout: Layout):
    """기존 측정은 실행 없이 읽고 저장한 판정이 다르면 거부한다."""
    client, manager, workers = web
    fake = FakeAgentModel()
    runtime, tools, _ = make_runtime(manager.workspace / "reference", valid_layout, fake)
    runtime.run()
    for sdk_client in fake.clients:
        sdk_client.close()
    root = (
        manager.workspace
        / ".local-artifacts/measurements/board-calibration/look/cal/examples"
        / "003_repetitive_panel_pocket/r0"
    )
    (root / "sim").mkdir(parents=True)
    (root.parent / "layout.json").write_text(valid_layout.model_dump_json())
    check = (tools.root / "execution/check.json").read_text()
    (root / "check.json").write_text(check)
    (root / "sim/cell.mp4").write_bytes(b"recorded video")
    response = client.get("/api/reference/after")
    assert response.status_code == 200 and response.json()["check"]["passed"]
    assert not workers and client.get("/api/runs").json() == []
    assert client.get("/api/reference/after/assets/video").content == b"recorded video"
    assert client.get("/api/reference/after/assets/check.json").status_code == 422
    changed = json.loads(check)
    changed["passed_count"] += 1
    (root / "check.json").write_text(json.dumps(changed))
    assert client.get("/api/reference/after").status_code == 409
    assert client.get("/").status_code == 200
    assert client.get("/ui/app.js").status_code == 200


def _wait_file(path: Path) -> None:
    deadline = monotonic() + 8
    while not path.is_file() and monotonic() < deadline:
        sleep(0.02)
    assert path.is_file()


def _running(pid: int) -> bool:
    path = Path(f"/proc/{pid}/stat")
    try:
        return path.read_text().split(") ", 1)[1][0] != "Z"
    except FileNotFoundError:
        return False


def _sleeping_child(tmp_path: Path) -> str:
    return (
        "import signal,time\nfrom pathlib import Path\n"
        "signal.signal(signal.SIGINT,signal.SIG_IGN)\n"
        "signal.signal(signal.SIGTERM,signal.SIG_IGN)\n"
        f"Path({str(tmp_path / 'child.ready')!r}).touch()\n"
        "time.sleep(60)\n"
    )


def test_stop_kills_children_after_worker_exits(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    """워커가 먼저 종료해도 신호를 무시하는 자식까지 중단한다."""
    from pixel2place.demo import jobs

    monkeypatch.setattr(jobs, "SHUTDOWN_GRACE_S", 0.1)
    monkeypatch.setattr(jobs, "TERMINATE_GRACE_S", 0.1)
    child = _sleeping_child(tmp_path)
    script = tmp_path / "group.py"
    script.write_text(
        "import subprocess,sys,time\nfrom pathlib import Path\n"
        f"child = subprocess.Popen([sys.executable, '-c', {child!r}])\n"
        f"Path({str(tmp_path / 'child.pid')!r}).write_text(str(child.pid))\n"
        "time.sleep(60)\n"
    )
    process = subprocess.Popen(
        [sys.executable, str(script)],
        start_new_session=True,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )
    try:
        _wait_file(tmp_path / "child.pid")
        _wait_file(tmp_path / "child.ready")
        pid = int((tmp_path / "child.pid").read_text())
        JobManager._stop(process)
        assert process.poll() is not None and not _running(pid)
    finally:
        with suppress(ProcessLookupError):
            os.killpg(process.pid, signal.SIGKILL)
        process.wait(timeout=2)


def test_parent_loss_stops_worker_and_its_child(tmp_path: Path):
    """서버가 비정상 종료하면 워커와 신호를 무시하는 자식도 종료한다."""
    (tmp_path / "request.json").write_text(RunRequest(prompt=GOAL).model_dump_json())
    child = _sleeping_child(tmp_path)
    worker_script = tmp_path / "worker.py"
    worker_script.write_text(
        "import subprocess,sys,time\nfrom pathlib import Path\n"
        "from pixel2place.demo import worker\n"
        "worker.PARENT_POLL_S = 0.05\nworker.ORPHAN_GRACE_S = 0.1\n"
        "def fake_cli(*, args):\n"
        f"    child = subprocess.Popen([sys.executable, '-c', {child!r}])\n"
        f"    Path({str(tmp_path / 'child.pid')!r}).write_text(str(child.pid))\n"
        "    time.sleep(60)\n"
        "worker.app = fake_cli\nworker.main()\n"
    )
    owner_script = tmp_path / "owner.py"
    owner_script.write_text(
        "import os,subprocess,sys,time\nfrom pathlib import Path\n"
        f"arguments = [sys.executable, {str(worker_script)!r}, '--parent',\n"
        f"             str(os.getpid()), '--job', {str(tmp_path)!r}]\n"
        "child = subprocess.Popen(arguments, start_new_session=True)\n"
        f"Path({str(tmp_path / 'worker.pid')!r}).write_text(str(child.pid))\n"
        "time.sleep(60)\n"
    )
    owner = subprocess.Popen(
        [sys.executable, str(owner_script)], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL
    )
    worker_pid = None
    try:
        _wait_file(tmp_path / "worker.pid")
        worker_pid = int((tmp_path / "worker.pid").read_text())
        _wait_file(tmp_path / "child.pid")
        _wait_file(tmp_path / "child.ready")
        child_pid = int((tmp_path / "child.pid").read_text())
        owner.terminate()
        owner.wait(timeout=2)
        deadline = monotonic() + 4
        while (_running(worker_pid) or _running(child_pid)) and monotonic() < deadline:
            sleep(0.02)
        assert not _running(worker_pid) and not _running(child_pid)
    finally:
        if worker_pid is not None:
            with suppress(ProcessLookupError):
                os.killpg(worker_pid, signal.SIGKILL)
        if owner.poll() is None:
            owner.kill()
        owner.wait(timeout=2)


def test_shutdown_stops_worker_even_when_record_write_fails(web, monkeypatch: pytest.MonkeyPatch):
    """종료 상태를 저장하지 못해도 워커를 중단하고 루트 잠금을 해제한다."""
    from pixel2place.demo import jobs

    client, manager, workers = web
    client.post("/api/runs", json={"prompt": GOAL})

    def fail(*args: Any, **kwargs: Any) -> None:
        raise OSError("record write failed")

    monkeypatch.setattr(jobs, "write_record", fail)
    with pytest.raises(OSError, match="record write failed"):
        manager.close()
    assert workers[0].returncode == 130 and manager._file_lock.closed


def test_finished_worker_is_recorded_without_browser_polling(web):
    """브라우저 조회가 없어도 서버가 종료한 워커의 상태를 저장한다."""
    client, manager, workers = web
    run_id = client.post("/api/runs", json={"prompt": GOAL}).json()["id"]
    workers[0].returncode = 2
    path = manager.root / run_id / "job.json"
    deadline = monotonic() + 3
    record = JobRecord.model_validate_json(path.read_text())
    while record.status == "running" and monotonic() < deadline:
        sleep(0.02)
        record = JobRecord.model_validate_json(path.read_text())
    assert record.status == "aborted" and record.exit_code == 2
