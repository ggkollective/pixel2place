import numpy as np
import pytest
from pydantic import ValidationError

from pixel2place.cell.config import CellConfig
from pixel2place.cell.scene import (
    ARM_JOINT_COUNT,
    JOINT_ERROR_ROWS,
    SIM_FPS,
    BoardOffset,
    BoardPlacement,
    JointErrorLevel,
    PreplacedPart,
    SceneFaults,
    SimScene,
    WrongColor,
    board_shift_mm,
    check_scene_parts,
    displace_board,
    joint_error_frames,
    placed_cell,
    placement_of,
    sample_joint_error,
    to_placed_board,
)
from pixel2place.cell.task import compile_task
from pixel2place.generation.synthetic import COLOR_RGB
from pixel2place.layout.schema import Board, Color, PartType
from tests.factories import make_layout, make_part

CELL = CellConfig()
BOARD = Board()
MIRRORED_CELL = CellConfig(board_origin=(0.115, 0.08), col_direction=(0.0, -1.0))
# 기본 보드의 중심(m)
BOARD_CENTER = (0.195, 0.0)
# 셀 중심에서 보드 중심까지 가장 먼 거리(m). 모서리 셀 중심이다.
CORNER_RADIUS = 0.07 * np.sqrt(2)
JOINT_SIGMA = np.radians(1.0)
# 통계를 재는 행 수. 표준편차와 상관의 추정 오차가 5%보다 작다.
MANY_ROWS = 60_000


@pytest.mark.unit
class TestBoardPlacement:
    """보드 자세와 보드 오차 테스트."""

    def test_placement_of_default_cell(self):
        """기본 셀의 보드 u축은 +y 방향이다."""
        placement = placement_of(CELL, BOARD)
        assert placement.board_origin == CELL.board_origin
        assert placement.board_yaw == pytest.approx(np.pi / 2)

    def test_board_yaw_matches_task(self):
        """보드 자세의 방위각은 task의 board_yaw와 같은 식으로 계산한다."""
        layout = make_layout(make_part("P001", PartType.SQUARE_TILE, origin=(0, 0)))
        for cell in (CELL, MIRRORED_CELL):
            assert placement_of(cell, BOARD).board_yaw == compile_task(layout, cell).board_yaw

    def test_zero_offset_keeps_board(self):
        """오차가 0이면 보드 자세가 바뀌지 않는다."""
        placement = displace_board(CELL, BOARD, offset=BoardOffset(x=0.0, y=0.0, yaw=0.0))
        assert placement.board_origin == pytest.approx(CELL.board_origin)
        assert placement.row_direction == pytest.approx(CELL.row_direction)
        assert placement.board_yaw == pytest.approx(np.pi / 2)

    def test_offset_turns_about_board_center(self):
        """보드 중심은 (x, y)만큼 옮겨지고 방향은 yaw만큼 돈다."""
        angle = np.radians(1.5)
        offset = BoardOffset(x=0.002, y=-0.003, yaw=angle)
        placement = displace_board(CELL, BOARD, offset=offset)
        center = to_placed_board(BOARD_CENTER, cell=CELL, placement=placement)
        np.testing.assert_allclose(center, (0.197, -0.003), atol=1e-12)
        assert placement.row_direction == pytest.approx((np.cos(angle), np.sin(angle)))
        assert placement.board_yaw == pytest.approx(np.pi / 2 + angle)

    def test_rejects_non_unit_direction(self):
        """방향 벡터가 단위 길이가 아니면 검증 오류다."""
        with pytest.raises(ValidationError, match="unit length"):
            BoardPlacement(
                board_origin=(0.1, 0.0),
                row_direction=(2.0, 0.0),
                col_direction=(0.0, 1.0),
                board_yaw=0.0,
            )

    def test_scene_round_trips_json(self):
        """scene.json은 읽고 쓴 뒤 같다. board가 없으면 None이다."""
        placement = displace_board(CELL, BOARD, offset=BoardOffset(x=0.001, y=0.0, yaw=0.01))
        scene = SimScene(board=placement)
        assert SimScene.model_validate_json(scene.model_dump_json()) == scene
        assert SimScene.model_validate_json("{}").board is None


@pytest.mark.unit
class TestToPlacedBoard:
    """믿는 보드에서 실제 보드로 점을 옮기는 테스트."""

    @pytest.mark.parametrize("cell", [CELL, MIRRORED_CELL], ids=["default", "mirrored"])
    def test_cell_center_follows_board(self, cell):
        """셀 중심은 실제 보드의 같은 셀 중심으로 간다. 거울상 보드도 같다."""
        offset = BoardOffset(x=-0.001, y=0.004, yaw=np.radians(-1.0))
        placement = displace_board(cell, BOARD, offset=offset)
        moved_cell = cell.model_copy(
            update={
                "board_origin": placement.board_origin,
                "row_direction": placement.row_direction,
                "col_direction": placement.col_direction,
            }
        )
        layout = make_layout(make_part("P001", PartType.SQUARE_TILE, origin=(6, 1)))
        (planned,) = compile_task(layout, cell).parts
        (actual,) = compile_task(layout, moved_cell).parts
        moved = to_placed_board(planned.target_center, cell=cell, placement=placement)
        np.testing.assert_allclose(moved, actual.target_center, atol=1e-12)


@pytest.mark.unit
class TestBoardShift:
    """보드 셀 중심 이동 거리 테스트."""

    def test_translation(self):
        """이동만 있으면 모든 셀이 같은 거리만큼 움직인다."""
        placement = displace_board(CELL, BOARD, offset=BoardOffset(x=0.003, y=0.004, yaw=0.0))
        assert board_shift_mm(CELL, BOARD, placement=placement) == pytest.approx(5.0)

    def test_rotation_moves_corner_cells_most(self):
        """회전만 있으면 모서리 셀 중심이 가장 많이 움직인다."""
        angle = np.radians(1.5)
        placement = displace_board(CELL, BOARD, offset=BoardOffset(x=0.0, y=0.0, yaw=angle))
        expected_mm = CORNER_RADIUS * 2 * np.sin(angle / 2) * 1000
        assert board_shift_mm(CELL, BOARD, placement=placement) == pytest.approx(expected_mm)


@pytest.mark.unit
class TestPlacedCell:
    """실제 보드 자세의 셀 설정 테스트."""

    def test_board_follows_placement(self):
        """보드 원점과 방향은 placement의 값이다. 나머지 필드는 그대로다."""
        offset = BoardOffset(x=0.003, y=-0.004, yaw=np.radians(1.0))
        placement = displace_board(CELL, BOARD, offset=offset)
        moved = placed_cell(CELL, placement=placement)
        assert moved.board_origin == placement.board_origin
        assert moved.row_direction == placement.row_direction
        assert moved.col_direction == placement.col_direction
        assert moved.feeder_center == CELL.feeder_center
        assert moved.part_height == CELL.part_height

    def test_feeder_keeps_direction(self):
        """보드가 돌아도 공급대 방향은 원래 셀 설정의 방향이다."""
        placement = displace_board(CELL, BOARD, offset=BoardOffset(x=0.0, y=0.0, yaw=0.02))
        moved = placed_cell(CELL, placement=placement)
        assert moved.feeder_directions == CELL.feeder_directions
        assert moved.feeder_directions != (moved.row_direction, moved.col_direction)


@pytest.mark.unit
class TestJointError:
    """관절 센서 오차 테스트."""

    def test_same_seed_gives_same_rows(self):
        """같은 seed는 같은 값을 준다. 다른 seed는 다른 값을 준다."""
        level = JointErrorLevel(sigma=JOINT_SIGMA, tau=1.0)
        first = sample_joint_error(np.random.default_rng([1, 0, 0, 3]), level=level)
        again = sample_joint_error(np.random.default_rng([1, 0, 0, 3]), level=level)
        other = sample_joint_error(np.random.default_rng([1, 0, 1, 3]), level=level)
        assert first == again
        assert first != other
        assert len(first) == JOINT_ERROR_ROWS
        assert all(len(row) == ARM_JOINT_COUNT for row in first)

    def test_no_tau_gives_one_row(self):
        """지속 시간이 없으면 1행이다. 실행 동안 오차가 같다."""
        level = JointErrorLevel(sigma=JOINT_SIGMA)
        rows = sample_joint_error(np.random.default_rng(0), level=level)
        assert len(rows) == 1
        assert any(value != 0.0 for value in rows[0])

    @pytest.mark.parametrize("tau", [0.1, 1.0])
    def test_rows_have_sigma_and_tau(self, tau):
        """모든 행의 표준편차는 sigma다. 이웃한 행의 상관은 exp(-1 / (fps * tau))다."""
        level = JointErrorLevel(sigma=JOINT_SIGMA, tau=tau)
        rows = np.array(sample_joint_error(np.random.default_rng(0), level=level, rows=MANY_ROWS))
        assert rows.shape == (MANY_ROWS, ARM_JOINT_COUNT)
        assert rows.std() == pytest.approx(JOINT_SIGMA, rel=0.1)
        correlation = np.mean(rows[1:] * rows[:-1]) / np.mean(rows**2)
        assert correlation == pytest.approx(np.exp(-1.0 / (SIM_FPS * tau)), abs=0.03)

    def test_zero_sigma_gives_zeros(self):
        """sigma가 0이면 모든 값이 0이다."""
        level = JointErrorLevel(sigma=0.0, tau=1.0)
        rows = sample_joint_error(np.random.default_rng(0), level=level, rows=5)
        assert rows == ((0.0,) * ARM_JOINT_COUNT,) * 5

    @pytest.mark.parametrize("fields", [{"sigma": -0.1}, {"sigma": 0.1, "tau": 0.0}])
    def test_level_rejects_bad_values(self, fields):
        """sigma는 0 이상, tau는 0보다 커야 한다."""
        with pytest.raises(ValidationError):
            JointErrorLevel(**fields)

    def test_scene_keeps_values_through_json(self):
        """scene.json으로 쓰고 다시 읽어도 값이 비트 단위로 같다."""
        level = JointErrorLevel(sigma=JOINT_SIGMA, tau=0.1)
        rows = sample_joint_error(np.random.default_rng(0), level=level, rows=20)
        scene = SimScene(joint_error=rows)
        assert SimScene.model_validate_json(scene.model_dump_json()) == scene

    @pytest.mark.parametrize(
        "rows", [(), ((0.0, 0.0, 0.0, 0.0, float("nan")),), ((0.0, 0.0, 0.0, 0.0),)]
    )
    def test_scene_rejects_bad_rows(self, rows):
        """행이 없거나, 값이 유한하지 않거나, 열 수가 팔 관절 수와 다르면 오류다."""
        with pytest.raises(ValidationError):
            SimScene(joint_error=rows)

    def test_frames_repeat_one_row(self):
        """1행의 오차는 모든 프레임에 그 행을 쓴다."""
        row = (0.01, -0.02, 0.0, 0.03, 0.0)
        frames = joint_error_frames((row,), frames=4)
        np.testing.assert_array_equal(frames, np.array([row] * 4))

    def test_frames_take_first_rows(self):
        """여러 행의 오차는 프레임 번호가 행 번호다."""
        level = JointErrorLevel(sigma=JOINT_SIGMA, tau=0.1)
        rows = sample_joint_error(np.random.default_rng(0), level=level, rows=10)
        np.testing.assert_array_equal(joint_error_frames(rows, frames=6), np.array(rows[:6]))

    def test_frames_need_enough_rows(self):
        """여러 행의 오차가 프레임 수보다 짧으면 ValueError다."""
        level = JointErrorLevel(sigma=JOINT_SIGMA, tau=0.1)
        rows = sample_joint_error(np.random.default_rng(0), level=level, rows=3)
        with pytest.raises(ValueError, match="3 rows"):
            joint_error_frames(rows, frames=4)


@pytest.mark.unit
class TestFaultsAndPreplacedParts:
    """부품의 결함과 이미 놓인 부품의 장면 테스트."""

    ORANGE = WrongColor(sequence=2, color=Color.ORANGE, color_rgb=COLOR_RGB[Color.ORANGE])
    PLACED = PreplacedPart(
        sequence=1, position=(0.15, 0.02, 0.0075), orientation=(1.0, 0.0, 0.0, 0.0)
    )

    def test_scene_keeps_values_through_json(self):
        """결함과 놓인 부품은 JSON을 거쳐도 같다."""
        scene = SimScene(
            faults=SceneFaults(missing=(3,), wrong_color=(self.ORANGE,)), preplaced=(self.PLACED,)
        )
        assert SimScene.model_validate_json(scene.model_dump_json()) == scene

    def test_default_scene_has_no_faults(self):
        """기본 장면에는 결함과 놓인 부품이 없다."""
        scene = SimScene()
        assert scene.faults is None
        assert scene.preplaced is None

    def test_rejects_two_faults_for_one_part(self):
        """부품 1개에 결함 2개를 주지 못한다."""
        with pytest.raises(ValidationError, match="more than one fault"):
            SceneFaults(missing=(2,), wrong_color=(self.ORANGE,))
        with pytest.raises(ValidationError, match="more than one fault"):
            SceneFaults(missing=(1, 1))

    def test_rejects_sequence_below_one(self):
        """배치 순서는 1부터다."""
        with pytest.raises(ValidationError):
            SceneFaults(missing=(0,))

    def test_rejects_preplaced_missing_part(self):
        """놓인 부품은 공급 누락일 수 없다."""
        with pytest.raises(ValidationError, match="cannot be a missing part"):
            SimScene(faults=SceneFaults(missing=(1,)), preplaced=(self.PLACED,))

    def test_rejects_repeated_preplaced_part(self):
        """놓인 부품을 2번 쓰지 못한다."""
        with pytest.raises(ValidationError, match="more than one time"):
            SimScene(preplaced=(self.PLACED, self.PLACED))

    def test_rejects_rgb_that_is_not_the_palette_color(self):
        """`color_rgb`는 `color`의 팔레트 값과 같아야 한다."""
        with pytest.raises(ValidationError, match="different from the palette value"):
            WrongColor(sequence=1, color=Color.ORANGE, color_rgb=COLOR_RGB[Color.BLUE])

    @pytest.mark.parametrize(
        ("position", "orientation"),
        [
            ((float("inf"), 0.0, 0.0), (1.0, 0.0, 0.0, 0.0)),
            ((0.1, 0.0, 0.0), (float("nan"), 0.0, 0.0, 0.0)),
            ((0.1, 0.0, 0.0), (0.0, 0.0, 0.0, 0.0)),
            ((0.1, 0.0, 0.0), (1.0, 1.0, 0.0, 0.0)),
        ],
    )
    def test_rejects_pose_that_is_not_usable(self, position, orientation):
        """놓인 부품의 자세는 유한한 값과 단위 quaternion이어야 한다."""
        with pytest.raises(ValidationError):
            PreplacedPart(sequence=1, position=position, orientation=orientation)


@pytest.mark.unit
class TestCheckSceneParts:
    """장면의 부품 순서를 작업과 대조하는 테스트."""

    TASK = compile_task(
        make_layout(
            make_part("P001", PartType.SQUARE_TILE, origin=(1, 1)),
            make_part("P002", PartType.SQUARE_TILE, origin=(5, 5)),
        ),
        CELL,
    )

    @staticmethod
    def _placed(sequence: int) -> PreplacedPart:
        return PreplacedPart(
            sequence=sequence, position=(0.15, 0.02, 0.0075), orientation=(1.0, 0.0, 0.0, 0.0)
        )

    def test_accepts_parts_of_the_task(self):
        """작업의 부품만 있는 장면은 통과한다."""
        scene = SimScene(faults=SceneFaults(missing=(2,)), preplaced=(self._placed(1),))
        check_scene_parts(scene, self.TASK)
        check_scene_parts(SimScene(), self.TASK, policy=True)

    @pytest.mark.parametrize(
        "scene",
        [
            SimScene(faults=SceneFaults(missing=(3,))),
            SimScene(
                faults=SceneFaults(
                    wrong_color=(
                        WrongColor(
                            sequence=9, color=Color.ORANGE, color_rgb=COLOR_RGB[Color.ORANGE]
                        ),
                    )
                )
            ),
        ],
    )
    def test_rejects_parts_not_in_task(self, scene):
        """작업에 없는 순서의 부품을 거부한다."""
        with pytest.raises(ValueError, match="does not have"):
            check_scene_parts(scene, self.TASK)

    def test_rejects_preplaced_part_not_in_task(self):
        """작업에 없는 순서의 놓인 부품을 거부한다."""
        with pytest.raises(ValueError, match="does not have"):
            check_scene_parts(SimScene(preplaced=(self._placed(3),)), self.TASK)

    def test_rejects_all_parts_preplaced(self):
        """모든 부품이 놓여 있으면 놓을 부품이 없다."""
        scene = SimScene(preplaced=(self._placed(1), self._placed(2)))
        with pytest.raises(ValueError, match="no part to place"):
            check_scene_parts(scene, self.TASK)

    def test_policy_run_rejects_faults(self):
        """제어기 실행은 결함과 놓인 부품을 받지 않는다."""
        with pytest.raises(ValueError, match="controller run"):
            check_scene_parts(SimScene(faults=SceneFaults(missing=(1,))), self.TASK, policy=True)
