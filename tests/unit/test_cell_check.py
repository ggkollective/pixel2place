import numpy as np
import pytest

from pixel2place.cell.check import (
    CheckFailure,
    CheckTolerance,
    FailureStage,
    ResultMismatchError,
    check_result,
    yaw_period_deg,
)
from pixel2place.cell.config import CellConfig
from pixel2place.cell.result import SimResult
from pixel2place.cell.scene import BoardOffset, displace_board, to_placed_board
from pixel2place.cell.task import Task, compile_task
from pixel2place.layout.schema import PartType, Rotation
from tests.factories import make_layout, make_part, make_sim_result

MM = 0.001


@pytest.fixture(scope="module")
def task() -> Task:
    """네 종류의 부품. P003 직사각은 90° 돌아 있다."""
    layout = make_layout(
        make_part("P001", PartType.L_TILE, origin=(0, 0)),
        make_part("P002", PartType.T_TILE, origin=(0, 3)),
        make_part("P003", PartType.RECT_TILE, origin=(4, 4), rotation=Rotation.DEG_90),
        make_part("P004", PartType.SQUARE_TILE, origin=(7, 7)),
    )
    return compile_task(layout, CellConfig())


def _with_pose(result: SimResult, part_id: str, **update: object) -> SimResult:
    parts = [
        pose.model_copy(update=update) if pose.part_id == part_id else pose for pose in result.parts
    ]
    return result.model_copy(update={"parts": parts})


def _shifted(task: Task, part_id: str, *, distance: float, direction_deg: float) -> SimResult:
    result = make_sim_result(task)
    (pose,) = [pose for pose in result.parts if pose.part_id == part_id]
    x, y, z = pose.position
    angle = np.radians(direction_deg)
    position = (x + distance * np.cos(angle), y + distance * np.sin(angle), z)
    return _with_pose(result, part_id, position=position)


def _raised(task: Task, part_id: str, *, height: float) -> SimResult:
    result = make_sim_result(task)
    (pose,) = [pose for pose in result.parts if pose.part_id == part_id]
    x, y, z = pose.position
    return _with_pose(result, part_id, position=(x, y, z + height))


def _turned(task: Task, part_id: str, *, delta_deg: float) -> SimResult:
    result = make_sim_result(task)
    (pose,) = [pose for pose in result.parts if pose.part_id == part_id]
    return _with_pose(result, part_id, yaw=pose.yaw + np.radians(delta_deg))


def _part(report, part_id):
    return next(part for part in report.parts if part.part_id == part_id)


def _on_displaced_board(task: Task, offset: BoardOffset, *, follow_board: bool) -> SimResult:
    """실제 보드를 `offset`만큼 옮긴 결과.

    `follow_board`가 참이면 부품이 실제 보드의 목표 셀에 있다. 거짓이면 믿는 보드의 목표에 있다.
    부품 yaw는 실제 보드 기준으로 목표와 같다.
    """
    placement = displace_board(task.cell, task.board, offset=offset)
    result = make_sim_result(task).model_copy(update={"board": placement})
    if not follow_board:
        return result
    parts = []
    for part, pose in zip(task.parts, result.parts, strict=True):
        x, y = to_placed_board(part.target_center, cell=task.cell, placement=placement)
        position = (float(x), float(y), pose.position[2])
        placed = pose.placed.model_copy(update={"position": position})
        parts.append(pose.model_copy(update={"position": position, "placed": placed}))
    return result.model_copy(update={"parts": parts})


@pytest.mark.unit
class TestCheckResult:
    """셀 실행 결과 판정 테스트."""

    def test_target_pose_passes(self, task):
        """모든 부품이 목표 자세에 있으면 성공이다. 20 mm 셀의 위치 한계는 4 mm다."""
        report = check_result(task, make_sim_result(task))
        assert report.passed
        assert report.passed_count == len(task.parts)
        assert report.position_limit_mm == pytest.approx(4.0)
        for part in report.parts:
            assert part.position_error_mm == pytest.approx(0.0, abs=1e-9)
            assert part.yaw_error_deg == pytest.approx(0.0, abs=1e-9)
            assert part.failures == []

    def test_parts_follow_task_order(self, task):
        """판정의 부품 순서와 순서 번호는 task와 같다."""
        report = check_result(task, make_sim_result(task))
        assert [(part.sequence, part.part_id) for part in report.parts] == [
            (part.sequence, part.part_id) for part in task.parts
        ]

    @pytest.mark.parametrize(
        ("part_type", "period"),
        [
            (PartType.SQUARE_TILE, 90.0),
            (PartType.RECT_TILE, 180.0),
            (PartType.L_TILE, 360.0),
            (PartType.T_TILE, 360.0),
        ],
    )
    def test_yaw_period(self, part_type, period):
        """yaw 주기는 형태가 같아지는 가장 작은 회전이다."""
        assert yaw_period_deg(part_type) == period

    @pytest.mark.parametrize(
        ("part_id", "delta_deg"), [("P004", 90.0), ("P004", -270.0), ("P003", 180.0)]
    )
    def test_symmetric_turn_passes(self, task, part_id, delta_deg):
        """정사각의 90° 회전과 직사각의 180° 회전은 같은 배치다."""
        report = check_result(task, _turned(task, part_id, delta_deg=delta_deg))
        assert _part(report, part_id).yaw_error_deg == pytest.approx(0.0, abs=1e-9)
        assert report.passed

    @pytest.mark.parametrize("part_id", ["P001", "P002"])
    def test_asymmetric_half_turn_fails(self, task, part_id):
        """L과 T는 180° 돌면 다른 배치다."""
        report = check_result(task, _turned(task, part_id, delta_deg=180.0))
        part = _part(report, part_id)
        assert abs(part.yaw_error_deg) == pytest.approx(180.0)
        assert part.failures == [CheckFailure.YAW]
        assert not report.passed
        assert report.passed_count == len(task.parts) - 1

    @pytest.mark.parametrize("delta_deg", [363.0, -3.0, 9.9])
    def test_small_turn_passes_with_sign(self, task, delta_deg):
        """한 바퀴를 넘는 차이는 접는다. 반시계 방향이 양수다."""
        report = check_result(task, _turned(task, "P001", delta_deg=delta_deg))
        part = _part(report, "P001")
        assert part.yaw_error_deg == pytest.approx((delta_deg + 180.0) % 360.0 - 180.0)
        assert part.passed

    def test_turn_over_limit_fails(self, task):
        """yaw 오차가 10°를 넘으면 실패다."""
        report = check_result(task, _turned(task, "P004", delta_deg=10.1))
        assert _part(report, "P004").failures == [CheckFailure.YAW]

    @pytest.mark.parametrize(("distance_mm", "passed"), [(3.9, True), (4.1, False)])
    def test_position_limit(self, task, distance_mm, passed):
        """위치 오차는 수평 거리다. 4 mm를 넘으면 실패다."""
        result = _shifted(task, "P002", distance=distance_mm * MM, direction_deg=135.0)
        part = _part(check_result(task, result), "P002")
        assert part.position_error_mm == pytest.approx(distance_mm)
        assert part.passed is passed
        assert part.failures == ([] if passed else [CheckFailure.POSITION])

    @pytest.mark.parametrize(
        ("height_mm", "passed"), [(0.4, True), (-0.4, True), (0.76, False), (3.24, False)]
    )
    def test_height_limit(self, task, height_mm, passed):
        """중심 높이가 앉은 높이에서 0.5 mm 넘게 다르면 실패다. 위치 오차는 높이를 쓰지 않는다.

        0.76 mm는 덜 앉은 T, 3.24 mm는 핀 끝에 수평으로 얹힌 T의 시뮬레이터 측정값이다.
        두 부품은 기울기가 5° 아래다. 그래서 높이 한계만 이 부품을 찾는다.
        """
        part = _part(check_result(task, _raised(task, "P002", height=height_mm * MM)), "P002")
        assert part.height_error_mm == pytest.approx(height_mm)
        assert part.position_error_mm == pytest.approx(0.0, abs=1e-9)
        assert part.failures == ([] if passed else [CheckFailure.HEIGHT])

    def test_seated_height_follows_table(self, task):
        """앉은 높이는 table_z + part_height / 2다.

        작업대가 10 mm 높으면 같은 부품은 앉은 높이에서 10 mm 아래다.
        """
        cell = task.cell.model_copy(update={"table_z": 0.01})
        raised_table = task.model_copy(update={"cell": cell})
        part = _part(check_result(raised_table, make_sim_result(task)), "P004")
        assert part.height_error_mm == pytest.approx(-10.0)
        assert part.failures == [CheckFailure.HEIGHT]

    def test_position_limit_scales_with_cell_size(self, task):
        """위치 한계는 셀 크기에 비례한다. 25 mm 셀에서는 5 mm다."""
        board = task.board.model_copy(update={"cell_size_mm": 25.0})
        large = task.model_copy(update={"board": board})
        result = _shifted(large, "P004", distance=4.5 * MM, direction_deg=0.0)
        report = check_result(large, result)
        assert report.position_limit_mm == pytest.approx(5.0)
        assert report.passed

    def test_tilted_part_fails(self, task):
        """기울기가 5°를 넘으면 실패다."""
        result = _with_pose(make_sim_result(task), "P003", tilt=float(np.radians(6.0)))
        part = _part(check_result(task, result), "P003")
        assert part.tilt_deg == pytest.approx(6.0)
        assert part.failures == [CheckFailure.TILT]

    def test_fallen_part_fails(self, task):
        """작업대 아래로 떨어진 부품은 실패다."""
        result = _with_pose(make_sim_result(task), "P001", fell=True)
        part = _part(check_result(task, result), "P001")
        assert part.fell
        assert part.failures == [CheckFailure.FELL]

    def test_all_failures_are_listed(self, task):
        """실패 항목을 모두 기록한다."""
        result = _shifted(task, "P001", distance=10 * MM, direction_deg=0.0)
        result = _with_pose(result, "P001", tilt=float(np.radians(30.0)))
        part = _part(check_result(task, result), "P001")
        assert part.failures == [CheckFailure.POSITION, CheckFailure.TILT]

    def test_custom_tolerance(self, task):
        """허용 오차를 바꿀 수 있다."""
        result = _shifted(task, "P004", distance=7.0 * MM, direction_deg=90.0)
        assert not check_result(task, result).passed
        report = check_result(task, result, tolerance=CheckTolerance(position_ratio=0.5))
        assert report.position_limit_mm == pytest.approx(10.0)
        assert report.passed

    def test_report_json_has_verdicts(self, task):
        """JSON에는 부품별 판정과 전체 판정이 있다."""
        result = _turned(task, "P001", delta_deg=180.0)
        payload = check_result(task, result).model_dump(mode="json")
        assert payload["passed"] is False
        assert payload["passed_count"] == len(task.parts) - 1
        by_id = {part["part_id"]: part for part in payload["parts"]}
        assert by_id["P001"]["failures"] == ["yaw"]
        assert by_id["P001"]["passed"] is False
        assert by_id["P004"]["passed"] is True


def _moved(result: SimResult, part_id: str, *, final: dict, placed: dict) -> SimResult:
    """부품의 최종 자세와 놓은 직후 자세를 바꾼다."""
    (pose,) = [pose for pose in result.parts if pose.part_id == part_id]
    return _with_pose(result, part_id, placed=pose.placed.model_copy(update=placed), **final)


@pytest.mark.unit
class TestPlacedBoard:
    """실행기가 보드를 task와 다른 자세에 만든 결과의 판정 테스트."""

    OFFSET = BoardOffset(x=0.003, y=-0.002, yaw=np.radians(1.5))

    def test_parts_on_placed_board_pass(self, task):
        """부품이 실제 보드의 목표 셀에 있으면 오차가 0이다."""
        report = check_result(task, _on_displaced_board(task, self.OFFSET, follow_board=True))
        assert report.passed
        for part in report.parts:
            assert part.position_error_mm == pytest.approx(0.0, abs=1e-9)

    @pytest.mark.parametrize(("x_mm", "passed"), [(3.0, True), (5.0, False)])
    def test_parts_on_planned_board_miss_by_board_offset(self, task, x_mm, passed):
        """부품이 믿는 보드의 목표에 있으면 보드 이동만큼 오차가 난다."""
        offset = BoardOffset(x=x_mm * MM, y=0.0, yaw=0.0)
        report = check_result(task, _on_displaced_board(task, offset, follow_board=False))
        for part in report.parts:
            assert part.position_error_mm == pytest.approx(x_mm)
        assert report.passed is passed


@pytest.mark.unit
class TestFailedStage:
    """실패 단계 분류 테스트. 놓은 직후 자세로 나눈다."""

    def test_passed_part_has_no_stage(self, task):
        """통과한 부품은 단계가 없다."""
        report = check_result(task, make_sim_result(task))
        assert all(part.failed_stage is None for part in report.parts)

    def test_part_away_from_target_after_place_is_transport(self, task):
        """놓은 직후 목표에서 셀 크기(20 mm) 이상 떨어진 부품은 transport다."""
        far = {"position": (0.14, 0.16, 0.0075)}
        result = _moved(make_sim_result(task), "P004", final=far, placed=far)
        assert _part(check_result(task, result), "P004").failed_stage == FailureStage.TRANSPORT

    def test_fallen_part_after_place_is_transport(self, task):
        """놓은 직후 작업대 아래로 떨어진 부품은 transport다."""
        result = _moved(make_sim_result(task), "P001", final={"fell": True}, placed={"fell": True})
        assert _part(check_result(task, result), "P001").failed_stage == FailureStage.TRANSPORT

    def test_tilted_part_after_place_is_place(self, task):
        """놓은 직후 이웃에 걸려 기운 부품은 place다."""
        tilt = {"tilt": float(np.radians(15.0))}
        result = _moved(make_sim_result(task), "P003", final=tilt, placed=tilt)
        part = _part(check_result(task, result), "P003")
        assert part.failures == [CheckFailure.TILT]
        assert part.failed_stage == FailureStage.PLACE

    def test_part_on_pin_after_place_is_place(self, task):
        """놓은 직후 핀 끝에 얹힌 부품은 place다."""
        result = make_sim_result(task)
        (pose,) = [pose for pose in result.parts if pose.part_id == "P002"]
        x, y, z = pose.position
        lifted = {"position": (x, y, z + 3.24 * MM)}
        result = _moved(result, "P002", final=lifted, placed=lifted)
        part = _part(check_result(task, result), "P002")
        assert part.failures == [CheckFailure.HEIGHT]
        assert part.failed_stage == FailureStage.PLACE

    def test_part_pushed_after_place_is_later(self, task):
        """놓은 직후 통과하고 마지막에 실패한 부품은 later다."""
        result = _shifted(task, "P002", distance=5 * MM, direction_deg=0.0)
        part = _part(check_result(task, result), "P002")
        assert part.failures == [CheckFailure.POSITION]
        assert part.failed_stage == FailureStage.LATER

    def test_report_json_has_stage(self, task):
        """JSON에 부품별 실패 단계가 있다."""
        result = _shifted(task, "P002", distance=5 * MM, direction_deg=0.0)
        payload = check_result(task, result).model_dump(mode="json")
        by_id = {part["part_id"]: part for part in payload["parts"]}
        assert by_id["P002"]["failed_stage"] == "later"
        assert by_id["P001"]["failed_stage"] is None


@pytest.mark.unit
class TestResultMismatch:
    """result.json과 task.json이 맞지 않는 경우 테스트."""

    def test_missing_part(self, task):
        result = make_sim_result(task)
        result = result.model_copy(update={"parts": result.parts[1:]})
        with pytest.raises(ResultMismatchError, match="Missing"):
            check_result(task, result)

    def test_extra_part(self, task):
        result = make_sim_result(task)
        extra = result.parts[0].model_copy(update={"part_id": "P999"})
        result = result.model_copy(update={"parts": [*result.parts, extra]})
        with pytest.raises(ResultMismatchError, match="P999"):
            check_result(task, result)

    def test_duplicate_part(self, task):
        result = make_sim_result(task)
        result = result.model_copy(update={"parts": [*result.parts, result.parts[0]]})
        with pytest.raises(ResultMismatchError, match="Extra"):
            check_result(task, result)

    def test_task_version(self, task):
        result = make_sim_result(task).model_copy(update={"task_version": "0.1"})
        with pytest.raises(ResultMismatchError, match="task_version"):
            check_result(task, result)
