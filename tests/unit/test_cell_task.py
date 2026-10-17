import json

import numpy as np
import pytest
from pydantic import ValidationError

from pixel2place.cell.config import CellConfig, Locator
from pixel2place.cell.task import BoardFrame, GripperState, Task, TaskCompileError, compile_task
from pixel2place.generation.synthetic import COLOR_RGB
from pixel2place.layout.schema import Board, Color, PartType, Rotation
from pixel2place.robot.so101 import JAW_TIP_DEPTH, forward, gripper_angle_for_gap
from tests.factories import make_layout, make_part

CELL = CellConfig()
WAYPOINT_NAMES = ["pre_pick", "pick_approach", "pick", "lift", "pre_place", "place", "retreat"]
# pick TCP를 쓰는 waypoint. 나머지는 TCP를 쓴다.
PICK_WAYPOINTS = {"pre_pick", "pick_approach"}


def _single(part_type=PartType.SQUARE_TILE, *, origin=(0, 0), rotation=Rotation.DEG_0):
    return make_layout(make_part("P001", part_type, origin=origin, rotation=rotation))


def _waypoint(part, name):
    return next(waypoint for waypoint in part.waypoints if waypoint.name == name)


@pytest.mark.unit
class TestCompileTask:
    """작업 컴파일러 테스트."""

    def test_square_at_origin_cell(self):
        """셀 (0, 0)의 중심은 보드 원점에서 row·col 방향으로 반 셀씩 간 곳이다."""
        task = compile_task(_single(), CELL)
        (part,) = task.parts
        # 기본 셀 20 mm, row는 +x, col은 +y 방향이다.
        assert part.target_center == pytest.approx((0.125, -0.07))
        assert part.target_yaw == pytest.approx(0.0)

    def test_waypoints_and_heights(self):
        """waypoint 7개의 이름, 높이, 그리퍼 상태가 순서대로 있다."""
        (part,) = compile_task(_single(), CELL).parts
        assert [waypoint.name for waypoint in part.waypoints] == WAYPOINT_NAMES
        heights = [waypoint.position[2] for waypoint in part.waypoints]
        grasp, place, approach = CELL.grasp_z, CELL.place_z, CELL.approach_z
        expected = [approach, grasp, grasp, approach, approach, place, approach]
        assert heights == pytest.approx(expected)
        grippers = [waypoint.gripper for waypoint in part.waypoints]
        open_, closed = GripperState.OPEN, GripperState.CLOSED
        released = GripperState.RELEASED
        assert grippers == [open_, open_, closed, closed, closed, released, released]

    def test_place_keeps_jaws_above_neighbors(self):
        """놓을 때 jaw 끝은 이웃 부품 윗면보다 높다. 부품은 약 7 mm 떨어진다."""
        (part,) = compile_task(_single(), CELL).parts
        place = _waypoint(part, "place")
        assert place.position[2] > CELL.table_z + CELL.part_height
        # 파지할 때 부품 바닥은 TCP에서 (부품 높이 - 파지 깊이) 아래에 있다.
        part_bottom = place.position[2] - (CELL.part_height - CELL.grasp_depth)
        assert part_bottom - CELL.table_z == pytest.approx(0.007)

    def test_tcp_is_between_jaw_tips(self):
        """TCP는 툴 끝점에서 파지 폭의 반만큼 -x, jaw 끝 깊이만큼 +z에 있다."""
        task = compile_task(_single(), CELL)
        assert task.grip_width == pytest.approx(0.017)
        assert task.tcp == pytest.approx((-0.0085, 0.0, JAW_TIP_DEPTH))

    def test_pick_tcp_moves_away_from_fixed_jaw(self):
        """pick TCP는 TCP에서 -x로 pick_clearance만큼 간 점이다."""
        task = compile_task(_single(), CELL.model_copy(update={"pick_clearance": 0.003}))
        assert task.pick_tcp == pytest.approx((-0.0115, 0.0, JAW_TIP_DEPTH))

    def test_gripper_angles_follow_grip_width(self):
        """닫기는 파지 폭보다 좁고, 놓기는 열기보다 좁다."""
        angles = compile_task(_single(), CELL).gripper_angles
        closed = angles[GripperState.CLOSED]
        released, open_ = angles[GripperState.RELEASED], angles[GripperState.OPEN]
        assert closed < gripper_angle_for_gap(0.017) < released < open_

    def test_feeder_cells_are_around_feeder_center(self):
        """공급대의 셀 중심 평균은 feeder_center다. 회전 0°의 L 부품은 세로 막대다."""
        (part,) = compile_task(_single(PartType.L_TILE, rotation=Rotation.DEG_90), CELL).parts
        cells = np.array(part.feeder_cells)
        np.testing.assert_allclose(cells.mean(axis=0), CELL.feeder_center, atol=1e-9)
        # 회전 0°의 L: row 3칸(+x 방향)과 col 2칸(+y 방향)
        assert np.ptp(cells[:, 0]) == pytest.approx(0.04)
        assert np.ptp(cells[:, 1]) == pytest.approx(0.02)
        assert part.color_rgb == COLOR_RGB[Color.BLUE]

    @pytest.mark.parametrize("part_type", list(PartType))
    @pytest.mark.parametrize(
        ("board_origin", "col_direction"),
        [((0.115, -0.08), (0.0, 1.0)), ((0.115, 0.08), (0.0, -1.0))],
        ids=["default", "mirrored"],
    )
    def test_block_cells_follow_feeder_cells(self, part_type, board_origin, col_direction):
        """block_cells는 실행기의 블록 좌표에서 feeder_cells의 셀 번호다.

        실행기는 feeder_cells를 -feeder_yaw만큼 돌려 블록 좌표를 만든다.
        같은 부품의 이웃 셀은 block_cells로 찾는다. 직사각의 두 셀은 서로 이웃이다.
        """
        cell = CELL.model_copy(
            update={"board_origin": board_origin, "col_direction": col_direction}
        )
        task = compile_task(_single(part_type), cell)
        (part,) = task.parts
        feeder = np.array(part.feeder_cells)
        angle = -task.feeder_yaw
        rotation = np.array([[np.cos(angle), -np.sin(angle)], [np.sin(angle), np.cos(angle)]])
        local = (feeder - feeder.mean(axis=0)) @ rotation.T / (task.board.cell_size_mm / 1000)
        blocks = np.array(part.block_cells, dtype=np.float64)
        np.testing.assert_allclose(local, blocks - blocks.mean(axis=0), atol=1e-9)
        neighbors = sum(
            abs(a[0] - b[0]) + abs(a[1] - b[1]) == 1
            for index, a in enumerate(part.block_cells)
            for b in part.block_cells[index + 1 :]
        )
        # 셀 수 - 1 이상의 이웃 쌍이 있으면 셀이 하나로 이어져 있다.
        assert neighbors >= len(part.block_cells) - 1

    def test_explicit_feeder_directions_match_default(self):
        """공급대 방향을 보드 방향과 같게 주면 공급대 방향을 비운 설정과 부품 작업이 같다."""
        layout = _single(PartType.L_TILE, origin=(2, 3), rotation=Rotation.DEG_90)
        explicit = CELL.model_copy(
            update={"feeder_row_direction": (1.0, 0.0), "feeder_col_direction": (0.0, 1.0)}
        )
        assert compile_task(layout, explicit).parts == compile_task(layout, CELL).parts

    def test_rotated_board_keeps_feeder(self):
        """보드를 돌려도 공급대 방향을 고정하면 공급대 쪽 작업은 그대로다. 놓는 쪽만 바뀐다."""
        layout = _single(PartType.L_TILE, origin=(2, 3))
        angle = np.radians(1.0)
        rotated = CELL.model_copy(
            update={
                "row_direction": (np.cos(angle), np.sin(angle)),
                "col_direction": (-np.sin(angle), np.cos(angle)),
                "feeder_row_direction": CELL.row_direction,
                "feeder_col_direction": CELL.col_direction,
            }
        )
        base, turned = compile_task(layout, CELL), compile_task(layout, rotated)
        assert turned.feeder_yaw == base.feeder_yaw == base.board_yaw
        assert turned.board_yaw == pytest.approx(base.board_yaw + angle)
        (base_part,), (turned_part,) = base.parts, turned.parts
        assert turned_part.feeder_cells == pytest.approx(base_part.feeder_cells)
        assert turned_part.block_cells == base_part.block_cells
        for name in ("pre_pick", "pick", "lift"):
            base_point, turned_point = _waypoint(base_part, name), _waypoint(turned_part, name)
            assert turned_point.position == pytest.approx(base_point.position)
            assert turned_point.yaw == pytest.approx(base_point.yaw)
        assert turned_part.target_center != pytest.approx(base_part.target_center)
        assert _waypoint(turned_part, "place").yaw == pytest.approx(
            _waypoint(base_part, "place").yaw + angle
        )

    def test_pick_is_at_feeder(self):
        """정사각 부품은 공급대 중심에서 집는다."""
        (part,) = compile_task(_single(), CELL).parts
        assert _waypoint(part, "pick").position[:2] == pytest.approx(CELL.feeder_center)

    def test_joints_reach_waypoints(self):
        """관절 값의 FK는 waypoint 위치에 온다. PICK_WAYPOINTS는 pick TCP, 나머지는 TCP로 잰다."""
        layout = _single(PartType.L_TILE, origin=(4, 5), rotation=Rotation.DEG_90)
        task = compile_task(layout, CELL)
        for waypoint in task.parts[0].waypoints:
            tcp = task.pick_tcp if waypoint.name in PICK_WAYPOINTS else task.tcp
            position = forward(np.array(waypoint.joints), tcp=tcp)[:3, 3]
            np.testing.assert_allclose(position, waypoint.position, atol=1e-3)

    @pytest.mark.parametrize(
        ("name", "expected_gap"), [("pick_approach", CELL.pick_clearance), ("pick", 0.0)]
    )
    def test_fixed_jaw_moves_to_part_face(self, name, expected_gap):
        """고정 jaw 안쪽 면(툴 끝점)은 pick_approach에서 부품 면과 떨어져 있고, pick에서 닿는다."""
        task = compile_task(_single(), CELL)
        waypoint = _waypoint(task.parts[0], name)
        pose = forward(np.array(waypoint.joints))
        tool_x = pose[:3, 0]
        # 정사각 부품의 파지점은 공급대 중심이다. 고정 jaw 쪽 부품 면은 툴 +x 쪽에 있다.
        grasp_point = np.array([*CELL.feeder_center, waypoint.position[2]])
        part_face = grasp_point + tool_x * task.grip_width / 2
        gap = float(np.dot(pose[:3, 3] - part_face, tool_x))
        assert gap == pytest.approx(expected_gap, abs=1e-4)

    @pytest.mark.parametrize("rotation", list(Rotation))
    def test_clearance_only_before_pick(self, rotation):
        """L 부품의 모든 회전에서 pick 전의 두 waypoint에만 고정 jaw 여유가 있다."""
        task = compile_task(_single(PartType.L_TILE, origin=(3, 3), rotation=rotation), CELL)
        for waypoint in task.parts[0].waypoints:
            pose = forward(np.array(waypoint.joints))
            # 툴 끝점에서 waypoint 위치까지의 툴 -x 방향 거리
            offset = float(np.dot(pose[:3, 3] - np.array(waypoint.position), pose[:3, 0]))
            clearance = CELL.pick_clearance if waypoint.name in PICK_WAYPOINTS else 0.0
            assert offset == pytest.approx(task.grip_width / 2 + clearance, abs=1e-4)

    def test_rotation_sets_target_yaw(self):
        """이미지 시계 방향 90도는 위에서 본 시계 방향 90도, 곧 -90도 방위각이다."""
        layout = _single(PartType.RECT_TILE, origin=(2, 2), rotation=Rotation.DEG_90)
        (part,) = compile_task(layout, CELL).parts
        assert part.target_yaw == pytest.approx(-np.pi / 2)

    @pytest.mark.parametrize("part_type", [PartType.L_TILE, PartType.T_TILE])
    @pytest.mark.parametrize("rotation", list(Rotation))
    def test_wrist_turns_by_part_rotation(self, part_type, rotation):
        """180° 대칭이 아닌 부품은 pick과 place 사이에 손목이 부품 회전만큼 정확히 돈다."""
        task = compile_task(_single(part_type, origin=(3, 3), rotation=rotation), CELL)
        (part,) = task.parts
        pick, place = _waypoint(part, "pick"), _waypoint(part, "place")

        def tool_yaw(waypoint):
            pose = forward(np.array(waypoint.joints))
            return np.arctan2(pose[1, 0], pose[0, 0])

        turn = tool_yaw(place) - tool_yaw(pick) - part.target_yaw
        assert np.degrees(abs(np.arctan2(np.sin(turn), np.cos(turn)))) < 3.0

    def test_board_is_not_mirrored(self):
        """위에서 보면 이미지 아래(+x)에서 오른쪽(+y)으로 도는 방향이 반시계다. 이미지와 같다."""
        (down_x, down_y), (right_x, right_y) = CELL.row_direction, CELL.col_direction
        # 이미지에서 아래 방향을 반시계 90° 돌리면 오른쪽 방향이 된다. 외적의 z 성분이 양수다.
        assert down_x * right_y - down_y * right_x > 0

    def test_far_parts_first(self):
        """pan 축에서 먼 부품을 먼저 놓는다."""
        layout = make_layout(
            make_part("NEAR", PartType.SQUARE_TILE, origin=(0, 3)),
            make_part("FAR", PartType.SQUARE_TILE, origin=(7, 3), color=Color.GRAY),
        )
        task = compile_task(layout, CELL)
        assert [part.part_id for part in task.parts] == ["FAR", "NEAR"]
        assert [part.sequence for part in task.parts] == [1, 2]

    def test_infeasible_layout_raises(self, overlap_layout):
        """배치 규칙을 어긴 레이아웃은 TaskCompileError다."""
        with pytest.raises(TaskCompileError, match="OVERLAP"):
            compile_task(overlap_layout, CELL)

    def test_unreachable_raises(self):
        """작업 범위 밖의 보드는 부품과 waypoint를 알려 주고 실패한다."""
        layout = _single().model_copy(update={"board": Board(cell_size_mm=60.0)})
        layout = layout.model_copy(
            update={"parts": [make_part("P001", PartType.SQUARE_TILE, origin=(7, 7))]}
        )
        with pytest.raises(TaskCompileError, match="Part P001 waypoint pre_place"):
            compile_task(layout, CELL)

    def test_round_trips_json(self):
        """Task는 JSON으로 쓰고 다시 읽을 수 있다."""
        task = compile_task(_single(), CELL)
        assert Task.model_validate_json(task.model_dump_json()) == task

    def test_task_json_has_locator(self):
        """task.json의 cell에 핀·구멍 치수와 바깥 모따기가 있다. 실행기가 이 값을 쓴다."""
        payload = json.loads(compile_task(_single(), CELL).model_dump_json())
        assert payload["cell"]["locator"] == Locator().model_dump()
        assert payload["cell"]["part_chamfer"] == pytest.approx(0.002)

    def test_flat_board_round_trips_json(self):
        """locator가 None이면 평판이다. task.json에도 None으로 들어간다."""
        task = compile_task(_single(), CellConfig(locator=None, part_chamfer=0.0))
        assert task.cell.locator is None
        assert Task.model_validate_json(task.model_dump_json()) == task

    @pytest.mark.parametrize(
        ("cell_size_mm", "locator", "fits"),
        [(18.0, Locator(), False), (19.0, Locator(), True), (18.0, None, True)],
    )
    def test_part_floor_width(self, cell_size_mm, locator, fits):
        """부품 바닥에서 구멍 입구와 바깥 모따기 사이에 1 mm 이상 평면이 남아야 한다.

        18 mm 셀의 평면 폭은 9 − 1.5(inset) − 2(모따기) − 4.7(입구 반폭) = 0.8 mm다.
        평판이면 구멍이 없어서 5.5 mm다.
        """
        layout = _single().model_copy(update={"board": Board(cell_size_mm=cell_size_mm)})
        cell = CELL.model_copy(update={"locator": locator})
        if fits:
            compile_task(layout, cell)
        else:
            with pytest.raises(TaskCompileError, match="Part floor width 0.80 mm"):
                compile_task(layout, cell)


@pytest.mark.unit
class TestBoardFrame:
    """보드 격자 좌표 변환 테스트."""

    @pytest.mark.parametrize(
        "update",
        [
            {},
            {"board_origin": (0.115, 0.08), "col_direction": (0.0, -1.0)},
            {"row_direction": (0.8, 0.6), "col_direction": (-0.6, 0.8)},
        ],
        ids=["default", "mirrored", "rotated"],
    )
    def test_grid_uv_inverts_point(self, update):
        """grid_uv는 point의 역변환이다. 점 1개와 점 여러 개를 모두 받는다."""
        frame = BoardFrame(cell=CELL.model_copy(update=update), cell_size=0.02)
        uv = np.array([[0.5, 0.5], [3.0, 1.5], [7.5, 6.5]])
        points = np.array([frame.point(point) for point in uv])
        np.testing.assert_allclose(frame.grid_uv(points), uv, atol=1e-12)
        np.testing.assert_allclose(frame.grid_uv(points[1]), uv[1], atol=1e-12)


@pytest.mark.unit
class TestCellConfig:
    """셀 설정 테스트."""

    def test_heights(self):
        """파지 높이는 부품 윗면 아래, 놓는 높이는 부품 윗면 위다."""
        cell = CellConfig(
            table_z=0.01,
            part_height=0.02,
            grasp_depth=0.005,
            release_clearance=0.002,
            approach_clearance=0.03,
        )
        assert cell.grasp_z == pytest.approx(0.025)
        assert cell.place_z == pytest.approx(0.032)
        assert cell.approach_z == pytest.approx(0.055)

    def test_approach_must_be_above_place(self):
        """접근 높이가 놓는 높이 이하이면 검증 오류다."""
        with pytest.raises(ValidationError, match="approach_clearance"):
            CellConfig(grasp_depth=0.006, release_clearance=0.004, approach_clearance=0.01)

    def test_grasp_depth_must_be_inside_part(self):
        """grasp_depth가 부품 높이 이상이면 검증 오류다."""
        with pytest.raises(ValidationError, match="grasp_depth"):
            CellConfig(part_height=0.01, grasp_depth=0.01)

    @pytest.mark.parametrize(
        ("update", "message"),
        [
            # 잡은 부품의 바닥은 7 mm 높이에서 떨어진다. 핀 끝이 그보다 낮아야 한다.
            ({"locator": Locator(pin_height=0.007, hole_depth=0.008)}, "pin_height"),
            ({"locator": Locator(hole_depth=0.015)}, "hole_depth must be less than part_height"),
            # 모따기가 파지 부위(바닥에서 9 mm 위)에 닿으면 jaw 접촉 면이 준다.
            ({"locator": None, "part_chamfer": 0.009}, "part_chamfer must be less than"),
            ({"part_chamfer": 0.007}, "part_chamfer must not be more than locator.hole_depth"),
        ],
    )
    def test_locator_and_chamfer_rules(self, update, message):
        """핀과 모따기는 잡은 부품과 jaw 접촉 면에 닿지 않는다."""
        with pytest.raises(ValidationError, match=message):
            CellConfig(**update)

    def test_feeder_directions_default_to_board(self):
        """공급대 방향을 비우면 보드 방향을 쓴다."""
        cell = CellConfig(row_direction=(0.0, 1.0), col_direction=(1.0, 0.0))
        assert cell.feeder_directions == ((0.0, 1.0), (1.0, 0.0))

    @pytest.mark.parametrize(
        ("update", "message"),
        [
            ({"row_direction": (2.0, 0.0)}, "Board directions must have unit length"),
            ({"col_direction": (0.6, 0.8)}, "Board row and col directions must be orthogonal"),
            ({"feeder_row_direction": (1.0, 0.0)}, "Give both feeder_row_direction"),
            (
                {"feeder_row_direction": (1.0, 0.0), "feeder_col_direction": (0.0, 1.1)},
                "Feeder directions must have unit length",
            ),
            # 거울상 공급대: row에서 col로 시계 방향으로 돈다.
            (
                {"feeder_row_direction": (1.0, 0.0), "feeder_col_direction": (0.0, -1.0)},
                "same handedness",
            ),
        ],
    )
    def test_direction_rules(self, update, message):
        """방향 벡터는 단위 길이이고 직교한다. 공급대는 보드와 손잡이 방향이 같다."""
        with pytest.raises(ValidationError, match=message):
            CellConfig(**update)

    def test_flat_board_without_chamfer(self):
        """locator가 None이고 모따기가 0이면 평판 위의 상자 부품이다."""
        cell = CellConfig(locator=None, part_chamfer=0.0)
        assert cell.locator is None
        assert cell.part_chamfer == 0.0


@pytest.mark.unit
class TestLocator:
    """보드 핀과 부품 구멍 치수 테스트."""

    def test_default_mouth_half_width(self):
        """구멍 입구 반폭은 구멍 반폭 2.2 mm에 입구 모따기 2.5 mm를 더한 4.7 mm다."""
        assert Locator().mouth_half_width == pytest.approx(0.0047)

    @pytest.mark.parametrize(
        ("update", "message"),
        [
            ({"hole_width": 0.004}, "hole_width must be more than pin_diameter"),
            ({"pin_height": 0.006}, "pin_height must be less than hole_depth"),
            ({"hole_chamfer": 0.006}, "hole_chamfer must be less than hole_depth"),
            ({"pin_chamfer": 0.002}, "pin_chamfer must be less than"),
        ],
    )
    def test_pin_must_fit_hole(self, update, message):
        """구멍은 핀보다 넓고 깊다. 모따기는 구멍 깊이, 핀 반지름, 핀 높이보다 작다."""
        with pytest.raises(ValidationError, match=message):
            Locator(**update)
