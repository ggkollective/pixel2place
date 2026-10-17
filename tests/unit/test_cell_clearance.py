from pathlib import Path

import numpy as np
import pytest

from pixel2place.cell.clearance import placement_clearances
from pixel2place.cell.config import CellConfig
from pixel2place.cell.result import SimResult
from pixel2place.cell.scene import BoardPlacement, placement_of
from pixel2place.cell.task import Task, compile_task
from pixel2place.layout.schema import Board, Layout, PartType
from tests.factories import make_layout, make_part, make_sim_result

MM = 0.001
POCKET_LAYOUT = Path(__file__).parents[2] / "examples" / "layouts" / "repetitive_panel_pocket.json"
# 기본 셀은 보드 u축이 +y다(board_yaw 90°). 이 셀은 u축이 +x다(board_yaw 0°).
TRANSPOSED_CELL = CellConfig(col_direction=(1.0, 0.0), row_direction=(0.0, 1.0))
CELLS = [CellConfig(), TRANSPOSED_CELL]
# 반폭 h mm 정사각형을 5° 돌리면 이웃 쪽으로 h × (cos 5° + sin 5°) mm까지 나온다.
# 이웃 면은 중심에서 (20 − h) mm에 있다.
HALF_MM = 10.0 - CellConfig().part_inset / MM
TURNED_CLEARANCE_MM = (20.0 - HALF_MM) - HALF_MM * (
    np.cos(np.radians(5.0)) + np.sin(np.radians(5.0))
)
DEFAULT_GAP_MM = 2 * CellConfig().part_inset / MM


def _pair(cell: CellConfig) -> tuple[Layout, Task]:
    """맞닿은 정사각 두 개."""
    layout = make_layout(
        make_part("P001", PartType.SQUARE_TILE, origin=(3, 3)),
        make_part("P002", PartType.SQUARE_TILE, origin=(3, 4)),
    )
    return layout, compile_task(layout, cell)


def _with_placed(result: SimResult, part_id: str, **update: object) -> SimResult:
    parts = [
        pose.model_copy(update={"placed": pose.placed.model_copy(update=update)})
        if pose.part_id == part_id
        else pose
        for pose in result.parts
    ]
    return result.model_copy(update={"parts": parts})


def _board_turned_about(cell: CellConfig, center, *, angle: float) -> BoardPlacement:
    """보드를 `center`를 축으로 `angle`(rad)만큼 돌린 자세."""
    rotation = np.array([[np.cos(angle), -np.sin(angle)], [np.sin(angle), np.cos(angle)]])
    center = np.asarray(center)
    origin = center + rotation @ (np.array(cell.board_origin) - center)
    turned = cell.model_copy(
        update={
            "board_origin": tuple(origin),
            "row_direction": tuple(rotation @ np.array(cell.row_direction)),
            "col_direction": tuple(rotation @ np.array(cell.col_direction)),
        }
    )
    return placement_of(turned, Board())


def _moved_toward_next(task: Task, *, distance: float) -> SimResult:
    """먼저 놓은 부품의 놓은 직후 자세를 다음 부품 쪽으로 `distance`(m) 옮긴다."""
    first, second = task.parts
    direction = np.array(second.target_center) - np.array(first.target_center)
    x, y = np.array(first.target_center) + distance * direction / np.linalg.norm(direction)
    result = make_sim_result(task)
    z = result.parts[0].placed.position[2]
    return _with_placed(result, first.part_id, position=(float(x), float(y), z))


@pytest.mark.unit
class TestPlacementClearances:
    """놓는 시점의 이웃 틈 테스트."""

    @pytest.mark.parametrize("cell", CELLS)
    def test_target_pose_gap_is_twice_inset(self, cell):
        """목표 자세로 맞닿은 두 부품의 틈은 part_inset의 두 배다. 첫 부품은 이웃이 없다."""
        layout, task = _pair(cell)
        clearances = placement_clearances(layout, task, make_sim_result(task))
        first, second = task.parts
        assert clearances[first.part_id] is None
        assert clearances[second.part_id] == pytest.approx(2 * cell.part_inset / MM)

    def test_gap_follows_part_inset(self):
        """part_inset이 1 mm이면 틈은 2 mm다."""
        cell = CellConfig(part_inset=0.001)
        layout, task = _pair(cell)
        clearances = placement_clearances(layout, task, make_sim_result(task))
        assert clearances[task.parts[1].part_id] == pytest.approx(2.0)

    @pytest.mark.parametrize("cell", CELLS)
    def test_placed_shift_reduces_gap(self, cell):
        """먼저 놓은 부품이 다음 부품 쪽으로 1 mm 밀려 있으면 틈이 1 mm 준다."""
        layout, task = _pair(cell)
        clearances = placement_clearances(layout, task, _moved_toward_next(task, distance=MM))
        assert clearances[task.parts[1].part_id] == pytest.approx(DEFAULT_GAP_MM - 1.0)

    def test_overlap_is_zero(self):
        """먼저 놓은 부품이 다음 부품 자리에 들어와 있으면 틈은 0이다."""
        layout, task = _pair(CellConfig())
        clearances = placement_clearances(layout, task, _moved_toward_next(task, distance=4 * MM))
        assert clearances[task.parts[1].part_id] == 0.0

    @pytest.mark.parametrize("cell", CELLS)
    def test_placed_turn_reduces_gap(self, cell):
        """먼저 놓은 정사각이 5° 돌면 모서리가 이웃 쪽으로 나와서 틈이 준다."""
        layout, task = _pair(cell)
        result = make_sim_result(task)
        first = task.parts[0]
        result = _with_placed(result, first.part_id, yaw=first.target_yaw + np.radians(5.0))
        clearances = placement_clearances(layout, task, result)
        assert clearances[task.parts[1].part_id] == pytest.approx(TURNED_CLEARANCE_MM, abs=0.01)

    @pytest.mark.parametrize("cell", CELLS)
    def test_turned_board_turns_placed_part(self, cell):
        """실제 보드가 첫 부품 중심을 축으로 5° 돌면 놓인 부품도 world에서 5° 돈다.

        놓인 부품의 yaw는 실제 보드 기준으로 목표와 같다. 다음 부품은 믿는 보드의 자리로 내려간다.
        """
        layout, task = _pair(cell)
        first = task.parts[0]
        placement = _board_turned_about(cell, first.target_center, angle=np.radians(5.0))
        result = make_sim_result(task).model_copy(update={"board": placement})
        clearances = placement_clearances(layout, task, result)
        assert clearances[task.parts[1].part_id] == pytest.approx(TURNED_CLEARANCE_MM, abs=0.01)

    def test_far_part_has_no_neighbor(self):
        """반 셀보다 멀리 있는 부품은 이웃이 아니다."""
        layout = make_layout(
            make_part("P001", PartType.SQUARE_TILE, origin=(0, 0)),
            make_part("P002", PartType.SQUARE_TILE, origin=(0, 2)),
        )
        task = compile_task(layout, CellConfig())
        clearances = placement_clearances(layout, task, make_sim_result(task))
        assert clearances == {"P001": None, "P002": None}

    def test_pocket_part_uses_nearest_neighbor(self):
        """pocket 레이아웃의 L(P005)은 가장 가까운 이웃의 틈을 쓴다. 목표 자세이면 기본 틈이다."""
        layout = Layout.model_validate_json(POCKET_LAYOUT.read_text())
        task = compile_task(layout, CellConfig())
        clearances = placement_clearances(layout, task, make_sim_result(task))
        assert clearances["P005"] == pytest.approx(DEFAULT_GAP_MM)
        assert clearances[task.parts[0].part_id] is None
