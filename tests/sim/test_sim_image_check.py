"""Isaac Sim 실행의 최종 이미지를 조립 도구가 판독하는 smoke 테스트.

`pytest -m sim`으로만 실행한다. Isaac을 2회 실행한다. 캡처 1회와 배치 1회다. 약 40초가 걸린다.
Isaac 설정(`ISAAC_SIM_PATH`)이나 SO-101 USD가 없으면 skip한다.
Cosmos는 Fake다. 캡처, 보정, 배치와 판독은 실제 코드다.
"""

import pytest

from pixel2place.agent.artifacts import Manifest
from pixel2place.agent.contracts import (
    CalibrateBoardArguments,
    CalibrationResult,
    ExecuteCellArguments,
    ExecutionResult,
    LayoutResult,
    PrepareLayoutArguments,
)
from pixel2place.agent.tools import AssemblyTools
from pixel2place.cell.config import CellConfig
from pixel2place.cell.image_check import ImageCheckReport, PartStatus
from pixel2place.cell.scene import BoardOffset, CameraOffset, SimScene, displace_board
from pixel2place.layout.schema import Board, PartType
from pixel2place.reasoner.client import CosmosReasoner
from tests.factories import make_layout, make_part, make_planner_response_json
from tests.fakes.fake_openai import FakeOpenAI


@pytest.mark.sim
def test_execute_cell_records_matching_part(sim_settings, tmp_path):
    """보드와 카메라의 오차가 있는 실행에서 `execute_cell`이 놓은 부품을 `match`로 기록한다.

    부품의 칸 (3, 3)은 팔 그림자가 닿는 칸 밖이다.
    """
    layout = make_layout(make_part("P001", PartType.SQUARE_TILE, origin=(3, 3)))
    cell, board = CellConfig(), Board()
    scene = SimScene(
        board=displace_board(cell, board, offset=BoardOffset(x=0.003, y=-0.004, yaw=0.02)),
        camera=CameraOffset(position=(0.008, -0.006, 0.005), rotation=(0.012, -0.01, 0.015)),
    )
    client = FakeOpenAI([make_planner_response_json(layout)])
    tools = AssemblyTools(
        out_dir=tmp_path / "run",
        prompt="Place one blue square tile on the board.",
        cell=cell,
        scene=scene,
        reasoner=CosmosReasoner(client=client, model="fake"),
        sim_settings=sim_settings,
    )

    prepared = tools.prepare_layout(PrepareLayoutArguments(retry=False), tool_call_id="prepare")
    assert isinstance(prepared, LayoutResult), prepared
    layout_id = prepared.artifacts[0].id
    calibrated = tools.calibrate_board(
        CalibrateBoardArguments(layout_id=layout_id), tool_call_id="calibrate"
    )
    assert isinstance(calibrated, CalibrationResult), calibrated
    executed = tools.execute_cell(
        ExecuteCellArguments(layout_id=layout_id, calibration_id=calibrated.artifacts[0].id),
        tool_call_id="execute",
    )

    assert isinstance(executed, ExecutionResult), executed
    assert executed.summary.passed, executed.summary
    report = ImageCheckReport.model_validate_json(
        (tools.root / "execution/image_check.json").read_text(encoding="utf-8")
    )
    assert [part.status for part in report.parts] == [PartStatus.MATCH], report.parts
    assert report.passed and not report.unexpected_cells, report
    manifest = Manifest.model_validate_json(
        (tools.root / "manifest.json").read_text(encoding="utf-8")
    )
    assert manifest.artifacts[-1].kind == "image_check", manifest.artifacts
