import numpy as np
import pytest
from PIL import Image, ImageDraw

from pixel2place.cell.calibration import CalibrationError, calibrate_board, draw_overlay
from pixel2place.cell.config import CellConfig
from pixel2place.cell.scene import (
    BoardOffset,
    BoardPlacement,
    CameraOffset,
    board_shift_mm,
    displace_board,
)
from pixel2place.cell.task import BoardFrame
from pixel2place.layout.schema import Board
from tests.fakes.topdown_renderer import (
    BOARD_GRAY,
    PIN_GRAY,
    TABLE_GRAY,
    nominal_camera,
    render_topdown,
    shaken,
)

CELL = CellConfig()
BOARD = Board()
NOMINAL = nominal_camera()
# 계획한 보드 오차 한계와 카메라 흔들림 한계
OFFSET = BoardOffset(x=0.003, y=-0.005, yaw=np.radians(1.5))
SHAKE = CameraOffset(position=(0.01, -0.008, 0.006), rotation=tuple(np.radians([1.0, -0.8, 0.6])))
TRUTH = shaken(NOMINAL, SHAKE)
# 합성 이미지에서 기대하는 셀 중심 최대 오차(mm)
SYNTHETIC_TOLERANCE_MM = 0.1


@pytest.fixture(scope="module")
def placement() -> BoardPlacement:
    return displace_board(CELL, BOARD, offset=OFFSET)


@pytest.fixture(scope="module")
def image(placement) -> Image.Image:
    return render_topdown(TRUTH, cell=CELL, board=BOARD, placement=placement)


def _cover_markers(image: Image.Image, indices: list[int]) -> Image.Image:
    """표식 판을 작업대 색으로 덮는다."""
    covered = image.copy()
    draw = ImageDraw.Draw(covered)
    gray = round(TABLE_GRAY)
    for index in indices:
        u, v = TRUTH.project([[*CELL.markers[index], CELL.table_z]])[0]
        draw.rectangle([(u - 30, v - 30), (u + 30, v + 30)], fill=(gray, gray, gray))
    return covered


def _pin_pixel(placement: BoardPlacement, *, row: int, col: int, dx: float = 0.0) -> np.ndarray:
    """실제 보드의 핀 끝 화소. `dx`는 base_link x 방향 이동(m)이다."""
    frame = BoardFrame(
        cell=CELL.model_copy(
            update={
                "board_origin": placement.board_origin,
                "row_direction": placement.row_direction,
                "col_direction": placement.col_direction,
            }
        ),
        cell_size=BOARD.cell_size_mm / 1000,
    )
    x, y = frame.point(np.array([col + 0.5, row + 0.5]))
    assert CELL.locator is not None
    return TRUTH.project([[x + dx, y, CELL.table_z + CELL.locator.pin_height]])[0]


@pytest.mark.unit
class TestCalibrateBoard:
    """합성 top-down 이미지의 보드 캘리브레이션 테스트."""

    def test_recovers_board_pose(self, image, placement):
        """보드 오차와 카메라 흔들림이 한계에 있어도 셀 중심을 0.1 mm 안으로 맞춘다."""
        calibration = calibrate_board(image, NOMINAL, cell=CELL, board=BOARD)
        assert board_shift_mm(calibration.cell, BOARD, placement=placement) < (
            SYNTHETIC_TOLERANCE_MM
        )
        report = calibration.report
        assert all(marker.found_px is not None for marker in report.markers)
        assert report.pins_used == BOARD.rows * BOARD.cols
        assert report.offset.yaw == pytest.approx(OFFSET.yaw, abs=np.radians(0.02))
        assert report.board_shift_mm == pytest.approx(
            board_shift_mm(CELL, BOARD, placement=placement), abs=SYNTHETIC_TOLERANCE_MM
        )

    def test_solves_camera_pose_from_markers(self, image):
        """표식으로 푼 카메라는 흔들린 실제 카메라와 거의 같다."""
        report = calibrate_board(image, NOMINAL, cell=CELL, board=BOARD).report
        assert np.linalg.norm(report.camera.center - TRUTH.center) < 0.001
        assert report.camera_shift_mm == pytest.approx(np.linalg.norm(SHAKE.position) * 1000, abs=1)
        assert max(marker.residual_px or 0.0 for marker in report.markers) < 0.1

    def test_keeps_other_settings(self, image):
        """보드 자세만 바꾼다. 공급대 방향은 믿는 값을 명시해서 그대로 둔다."""
        calibrated = calibrate_board(image, NOMINAL, cell=CELL, board=BOARD).cell
        assert calibrated.feeder_directions == (CELL.row_direction, CELL.col_direction)
        board_fields = {
            "board_origin",
            "row_direction",
            "col_direction",
            "feeder_row_direction",
            "feeder_col_direction",
        }
        assert calibrated.model_dump(exclude=board_fields) == CELL.model_dump(exclude=board_fields)

    def test_zero_error_keeps_board(self):
        """보드와 카메라가 공칭 자세이면 보드를 옮기지 않는다."""
        image = render_topdown(NOMINAL, cell=CELL, board=BOARD)
        calibration = calibrate_board(image, NOMINAL, cell=CELL, board=BOARD)
        assert calibration.report.board_shift_mm < SYNTHETIC_TOLERANCE_MM

    def test_one_hidden_marker(self, image, placement):
        """표식 1개가 가려져도 나머지 표식으로 카메라 자세를 푼다."""
        calibration = calibrate_board(_cover_markers(image, [0]), NOMINAL, cell=CELL, board=BOARD)
        assert calibration.report.markers[0].found_px is None
        assert board_shift_mm(calibration.cell, BOARD, placement=placement) < (
            SYNTHETIC_TOLERANCE_MM
        )

    def test_too_few_markers(self, image):
        """찾은 표식이 4개보다 적으면 오류다."""
        with pytest.raises(CalibrationError, match="Found 3 markers"):
            calibrate_board(_cover_markers(image, [0, 2, 4]), NOMINAL, cell=CELL, board=BOARD)

    def test_rejects_moved_pin(self, image, placement):
        """1.8 mm 벗어나 보이는 핀은 맞춤에서 뺀다."""
        edited = image.copy()
        draw = ImageDraw.Draw(edited)
        u, v = _pin_pixel(placement, row=3, col=3)
        board, pin = round(BOARD_GRAY), round(PIN_GRAY)
        draw.ellipse([(u - 4, v - 4), (u + 4, v + 4)], fill=(board, board, board))
        u, v = _pin_pixel(placement, row=3, col=3, dx=0.0018)
        draw.ellipse([(u - 2.4, v - 2.4), (u + 2.4, v + 2.4)], fill=(pin, pin, pin))

        calibration = calibrate_board(edited, NOMINAL, cell=CELL, board=BOARD)

        (moved,) = [pin for pin in calibration.report.pins if (pin.row, pin.col) == (3, 3)]
        assert not moved.used
        assert moved.residual_mm is not None and moved.residual_mm > 1.0
        assert calibration.report.pins_used == BOARD.rows * BOARD.cols - 1
        assert board_shift_mm(calibration.cell, BOARD, placement=placement) < (
            SYNTHETIC_TOLERANCE_MM
        )

    def test_board_without_pins(self, image, placement):
        """보드에 핀이 보이지 않으면 오류다."""
        covered = image.copy()
        corners = [(-1, -1), (BOARD.cols + 1, -1), (BOARD.cols + 1, BOARD.rows + 1)]
        corners.append((-1, BOARD.rows + 1))
        frame = BoardFrame(cell=CELL, cell_size=BOARD.cell_size_mm / 1000)
        polygon = TRUTH.project([[*frame.point(np.array(uv)), CELL.table_z] for uv in corners])
        gray = round(BOARD_GRAY)
        ImageDraw.Draw(covered).polygon(
            [(float(u), float(v)) for u, v in polygon], fill=(gray, gray, gray)
        )
        with pytest.raises(CalibrationError, match="pin"):
            calibrate_board(covered, NOMINAL, cell=CELL, board=BOARD)

    def test_image_size_mismatch(self, image):
        with pytest.raises(CalibrationError, match="not the camera size"):
            calibrate_board(image.resize((320, 240)), NOMINAL, cell=CELL, board=BOARD)

    def test_cell_without_locator(self, image):
        with pytest.raises(CalibrationError, match="no board pins"):
            calibrate_board(image, NOMINAL, cell=CellConfig(locator=None), board=BOARD)

    def test_cell_with_few_markers(self, image):
        cell = CellConfig(markers=CELL.markers[:3])
        with pytest.raises(CalibrationError, match="fewer than 4 markers"):
            calibrate_board(image, NOMINAL, cell=cell, board=BOARD)

    def test_draw_overlay(self, image):
        """overlay는 입력과 같은 크기의 RGB 이미지다."""
        calibration = calibrate_board(image, NOMINAL, cell=CELL, board=BOARD)
        overlay = draw_overlay(image, calibration, board=BOARD)
        assert overlay.size == image.size
        assert overlay.mode == "RGB"
        assert overlay.tobytes() != image.tobytes()
