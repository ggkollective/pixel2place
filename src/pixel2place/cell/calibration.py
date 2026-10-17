"""top-down 이미지로 보드 자세를 재고 셀 설정을 고친다(보드 캘리브레이션).

입력:
- top-down 이미지. 보드 위에 부품이 없다.
- 카메라(`camera.json`). intrinsics와 공칭 자세다. 추정은 공칭 자세를 시작 값으로만 쓴다.
- 믿는 셀 설정. 보드 자세, 표식 자리, 핀 치수가 있다.

단계:
1. 표식: 공칭 카메라로 예측한 화소 주변 창에서 어두운 원의 중심을 찾는다.
2. 카메라 자세: 표식 평면의 homography를 intrinsics로 분해한다.
   그 다음 Levenberg-Marquardt로 표식 재투영 오차를 줄인다.
3. 대략 이동: 믿는 핀 격자를 수평으로 옮기고 돌리면서 핀 응답을 더한다.
   격자 바깥 한 줄의 응답은 뺀다. 합이 가장 큰 (x, y, yaw)를 고른다.
   핀 응답은 핀 크기의 밝은 점에서 크고 가는 격자선에서 작다.
   보드 판의 밝기로는 찾지 않는다. 팔 그림자가 판과 같은 밝기이기 때문이다.
4. 정밀: 핀 응답이 가장 큰 화소에서 핀 끝의 밝은 원 중심을 구하고 핀 끝 평면으로 되돌린다.
   믿는 핀 자리와 2D 강체 맞춤으로 보드의 (x, y, yaw) 오차를 구한다. 잔차가 큰 핀은 뺀다.
   맞춘 자세로 핀을 다시 찾고 한 번 더 맞춘다.
5. 출력: 보드 자세를 고친 셀 설정. 공급대 방향은 믿는 값을 그대로 쓴다.

numpy와 PIL만 쓴다. 코드는 시뮬레이터와 무관하다. 실물 카메라 이미지에도 같은 코드를 쓴다.
"""

import math
from collections import deque
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from numpy.typing import NDArray
from PIL import Image, ImageDraw
from pydantic import BaseModel, ConfigDict

from pixel2place.cell.camera import Array, PinholeCamera, rotation_angle, rotation_from_vector
from pixel2place.cell.config import CellConfig, Vector2
from pixel2place.cell.scene import BoardOffset, board_shift_mm, displace_board, placed_cell
from pixel2place.cell.task import MM_PER_M, BoardFrame
from pixel2place.layout.schema import Board
from pixel2place.pipeline import write_json

# 보정 셀 설정 옆에 `<이름>.report.json`과 `<이름>.overlay.png`를 쓴다.
REPORT_SUFFIX = ".report.json"
OVERLAY_SUFFIX = ".overlay.png"
MIN_MARKERS = 4
MIN_PINS = 16
# 표식을 찾는 창의 반경 여유(m). 카메라 흔들림(위치 10 mm, 각도 1°)은 표식을 약 25 mm 옮긴다.
MARKER_SEARCH_MARGIN = 0.035
# 표식 후보 성분의 넓이를 원 넓이로 나눈 값의 범위
MARKER_AREA_RANGE = (0.6, 1.4)
# 표식 창의 어두운 쪽(원)과 밝은 쪽(판) 밝기를 재는 백분위수
MARKER_LEVEL_PERCENTILES = (1, 90)
# 원 경계의 흐린 화소를 중심 계산에 넣는 폭(px)
EDGE_BAND_PX = 2.0
# 가중 중심을 구하는 횟수. 다음 번은 앞 중심 둘레에서 구한다.
CENTROID_ROUNDS = 2
# 원 안쪽과 바깥 판의 밝기를 재는 고리(px). 원 반지름에서 잰 거리다.
INNER_LEVEL_PX = -3.0
OUTER_LEVEL_PX = (3.0, 7.0)
# 표식과 핀의 최소 밝기 차(0–255). 이보다 작으면 찾지 못한 것으로 본다.
MIN_CONTRAST = 40.0
# 표식 재투영 잔차 한계(px)
MAX_MARKER_RESIDUAL_PX = 1.0
# 핀 응답은 핀 크기 상자의 평균 밝기에서 둘레 고리의 평균 밝기를 뺀 값이다. 고리의 폭(px)이다.
PIN_RING_PX = 2
# 대략 이동을 찾는 범위와 간격(m, rad). yaw 간격의 절반은 보드 모서리에서 약 0.5 mm다.
COARSE_RANGE = 0.025
COARSE_STEP = 0.001
COARSE_YAW_RANGE = math.radians(3.0)
COARSE_YAW_STEP = math.radians(0.5)
# 가장 좋은 이동에서 반 셀 이상 떨어진 후보와 가장 좋은 이동의 점수 비 한계.
# 넘으면 격자 위치가 모호하다. 한 셀 어긋난 격자는 한 줄의 핀을 잃고 바깥 한 줄에서 핀을 얻는다.
MAX_COARSE_AMBIGUITY = 0.95
# 핀을 찾는 창의 반경(m). 첫 번째는 대략 이동 뒤, 두 번째는 첫 맞춤 뒤다.
# 대략 이동 뒤에 남는 오차는 이동 간격과 yaw 간격만큼이다.
PIN_SEARCH_RADII = (0.006, 0.003)
# 핀 중심 계산에서 밝은 쪽으로 보는 밝기. 바탕에서 가장 밝은 화소까지의 비율이다.
PIN_LEVEL_RATIO = 0.5
# 핀 중심을 구하는 원의 반지름 여유(px). 핀 반지름에 더한다. 고리는 그 바깥 `PIN_RING_PX`다.
PIN_EDGE_PX = 1.0
# 이 잔차(m)보다 큰 핀은 맞춤에서 뺀다.
PIN_OUTLIER = 0.001
# 맞춤에 쓴 핀 잔차 RMS 한계(m)
MAX_PIN_RMS = 0.0005
# 핀 제외와 다시 맞춤의 최대 반복 수
MAX_FIT_ROUNDS = 3
# Levenberg-Marquardt
LM_ITERATIONS = 50
LM_INITIAL_DAMPING = 1e-3
LM_STEP_TOLERANCE = 1e-10
# 수치 미분 간격(rad, m)
JACOBIAN_STEP = 1e-7
# overlay 표시 크기(px). 표식 십자 반길이, 찾은 표식 원 반지름, 핀 점 반지름이다.
OVERLAY_CROSS_PX = 6
OVERLAY_MARKER_PX = 4
OVERLAY_PIN_PX = 1.5
OVERLAY_COLORS = {
    "predicted": (255, 200, 0),
    "found": (0, 220, 0),
    "rejected": (255, 40, 40),
    "board": (0, 200, 255),
}

Pixel = tuple[float, float]


class CalibrationError(ValueError):
    """이미지에서 보드 자세를 구하지 못했다."""


class _ReportModel(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)


class MarkerReport(_ReportModel):
    """표식 1개의 검출 결과."""

    position: Vector2
    """표식 중심(base_link, m)."""
    predicted_px: Pixel
    """공칭 카메라로 투영한 화소."""
    found_px: Pixel | None
    """찾은 원 중심. 찾지 못하면 None이다."""
    residual_px: float | None
    """찾은 중심과 추정 카메라로 재투영한 화소의 거리."""


class PinReport(_ReportModel):
    """핀 1개의 검출 결과. `row`, `col`은 핀이 있는 보드 셀이다."""

    row: int
    col: int
    found_px: Pixel | None
    """두 번째 검색에서 찾은 핀 끝 중심. 찾지 못하면 None이다."""
    residual_mm: float | None
    """맞춘 보드의 핀 자리와 찾은 핀 끝 중심의 수평 거리."""
    used: bool
    """마지막 맞춤에 썼다."""


class CalibrationReport(_ReportModel):
    """추정 과정의 값. `calibrate-board`가 보고 JSON으로 쓴다."""

    image_size: tuple[int, int]
    markers: list[MarkerReport]
    camera: PinholeCamera
    """표식으로 푼 카메라."""
    camera_shift_mm: float
    """공칭 카메라 중심에서 추정 카메라 중심까지의 거리."""
    camera_angle_deg: float
    """공칭 카메라 방향에서 추정 카메라 방향까지의 회전 각도."""
    coarse_offset: BoardOffset
    """핀 격자 투표로 구한 대략 오차."""
    coarse_ambiguity: float
    """대략 이동의 모호함. 반 셀 이상 떨어진 다음 후보와 가장 좋은 이동의 점수 비다."""
    pins: list[PinReport]
    pins_used: int
    pin_rms_mm: float
    pin_max_mm: float
    offset: BoardOffset
    """믿는 보드에서 추정 보드까지의 오차. 믿는 보드 중심을 축으로 돌리고 옮긴다."""
    board_shift_mm: float
    """보드 셀 중심이 믿는 자리에서 추정 자리까지 움직인 거리의 최댓값."""


@dataclass(frozen=True)
class BoardCalibration:
    """보드 자세를 고친 셀 설정과 보고."""

    cell: CellConfig
    report: CalibrationReport


def calibrate_board(
    image: Image.Image, camera: PinholeCamera, *, cell: CellConfig, board: Board
) -> BoardCalibration:
    """이미지로 보드 자세를 재고 보드 자세를 고친 셀 설정을 반환한다.

    아래 경우에는 `CalibrationError`를 낸다.
    - 이미지 크기가 카메라와 다르다.
    - 셀 설정에 표식이 `MIN_MARKERS`개보다 적거나 핀이 없다.
    - 찾은 표식이 `MIN_MARKERS`개보다 적거나 표식 재투영 잔차가 한계를 넘는다.
    - 대략 이동이 모호하다.
    - 맞춤에 쓴 핀이 `MIN_PINS`개보다 적거나 핀 잔차 RMS가 한계를 넘는다.
    """
    if image.size != (camera.width, camera.height):
        raise CalibrationError(
            f"Image size {image.size[0]}x{image.size[1]} is not the camera size "
            f"{camera.width}x{camera.height}."
        )
    if len(cell.markers) < MIN_MARKERS:
        raise CalibrationError(f"Cell config has fewer than {MIN_MARKERS} markers.")
    if cell.locator is None:
        raise CalibrationError("Cell config has no board pins (locator is null).")
    gray = np.asarray(image.convert("L"), dtype=np.float64)

    found = [_find_marker(gray, camera, position, cell=cell) for position in cell.markers]
    pairs = [
        (position, pixel)
        for position, pixel in zip(cell.markers, found, strict=True)
        if pixel is not None
    ]
    if len(pairs) < MIN_MARKERS:
        raise CalibrationError(f"Found {len(pairs)} markers. Need {MIN_MARKERS} or more.")
    world = np.array([[x, y, cell.table_z] for (x, y), _ in pairs])
    pixels = np.array([pixel for _, pixel in pairs])
    solved = _solve_camera(world, pixels, camera=camera)
    residuals = np.linalg.norm(solved.project(world) - pixels, axis=1)
    if residuals.max() > MAX_MARKER_RESIDUAL_PX:
        raise CalibrationError(
            f"Marker residual {residuals.max():.2f} px is more than {MAX_MARKER_RESIDUAL_PX} px."
        )
    markers = _marker_reports(cell, found, camera=camera, solved=solved)

    frame = BoardFrame(cell=cell, cell_size=board.cell_size_mm / MM_PER_M)
    tip_z = cell.table_z + cell.locator.pin_height
    grid = _pin_grid(frame, board=board)
    center = frame.point(np.array([board.cols / 2, board.rows / 2]))
    pin_radius = cell.locator.pin_diameter / 2
    scale = solved.fx / solved.depth_of([*center, tip_z])
    response = _pin_response(gray, radius=pin_radius * scale)
    coarse, ambiguity = _coarse_offset(
        response,
        solved,
        pins=grid,
        outside=_outside_grid(frame, board=board),
        center=center,
        tip_z=tip_z,
    )
    offset = coarse
    pins: list[PinReport] = []
    for search_radius in PIN_SEARCH_RADII:
        offset, pins = _fit_pins(
            gray,
            response,
            solved,
            grid=grid,
            center=center,
            cols=board.cols,
            tip_z=tip_z,
            pin_radius=pin_radius,
            offset=offset,
            search_radius=search_radius,
        )

    placement = displace_board(cell, board, offset=offset)
    calibrated = placed_cell(cell, placement=placement)
    used = [pin.residual_mm for pin in pins if pin.used and pin.residual_mm is not None]
    report = CalibrationReport(
        image_size=image.size,
        markers=markers,
        camera=solved,
        camera_shift_mm=float(np.linalg.norm(solved.center - camera.center)) * MM_PER_M,
        camera_angle_deg=math.degrees(
            rotation_angle(solved.rotation_matrix @ camera.rotation_matrix.T)
        ),
        coarse_offset=coarse,
        coarse_ambiguity=ambiguity,
        pins=pins,
        pins_used=len(used),
        pin_rms_mm=float(np.sqrt(np.mean(np.square(used)))),
        pin_max_mm=float(max(used)),
        offset=offset,
        board_shift_mm=board_shift_mm(cell, board, placement=placement),
    )
    return BoardCalibration(cell=calibrated, report=report)


def draw_overlay(image: Image.Image, calibration: BoardCalibration, *, board: Board) -> Image.Image:
    """검출과 추정 보드를 이미지 위에 그린다.

    - 표식: 예측 화소(노랑 십자), 찾은 중심(초록 원), 찾지 못한 표식(빨강 십자)
    - 보드: 추정 카메라로 투영한 추정 보드의 격자(하늘색)
    - 핀: 맞춤에 쓴 핀(초록 점), 뺀 핀(빨강 점)
    """
    report = calibration.report
    cell = calibration.cell
    overlay = image.convert("RGB")
    draw = ImageDraw.Draw(overlay)
    frame = BoardFrame(cell=cell, cell_size=board.cell_size_mm / MM_PER_M)
    for index in range(board.cols + 1):
        _draw_line(draw, report.camera, frame, (index, 0), (index, board.rows), z=cell.table_z)
    for index in range(board.rows + 1):
        _draw_line(draw, report.camera, frame, (0, index), (board.cols, index), z=cell.table_z)
    for marker in report.markers:
        u, v = marker.predicted_px
        found = marker.found_px is not None
        color = OVERLAY_COLORS["predicted"] if found else OVERLAY_COLORS["rejected"]
        draw.line([(u - OVERLAY_CROSS_PX, v), (u + OVERLAY_CROSS_PX, v)], fill=color)
        draw.line([(u, v - OVERLAY_CROSS_PX), (u, v + OVERLAY_CROSS_PX)], fill=color)
        if marker.found_px is not None:
            _draw_dot(draw, marker.found_px, radius=OVERLAY_MARKER_PX, outline=True)
    for pin in report.pins:
        if pin.found_px is None:
            continue
        _draw_dot(draw, pin.found_px, radius=OVERLAY_PIN_PX, outline=False, used=pin.used)
    return overlay


def write_calibration(
    calibration: BoardCalibration, image: Image.Image, *, output: Path, board: Board
) -> None:
    """보정 셀 설정을 `output`에 쓴다. 옆에 보고 JSON과 overlay 이미지를 쓴다."""
    write_json(output, calibration.cell)
    write_json(output.with_name(output.stem + REPORT_SUFFIX), calibration.report)
    overlay = draw_overlay(image, calibration, board=board)
    overlay.save(output.with_name(output.stem + OVERLAY_SUFFIX))


def read_report(path: Path) -> CalibrationReport:
    """`write_calibration`이 쓴 보고 JSON을 읽는다. 추정 카메라는 보고의 `camera`다.

    Raises:
        pydantic.ValidationError: 파일이 보고 형식이 아닐 때.
    """
    return CalibrationReport.model_validate_json(path.read_text(encoding="utf-8"))


def _draw_dot(
    draw: ImageDraw.ImageDraw, center: Pixel, *, radius: float, outline: bool, used: bool = True
) -> None:
    """찾은 점. 맞춤에 썼으면 초록, 뺐으면 빨강이다. `outline`이면 테두리만 그린다."""
    u, v = center
    color = OVERLAY_COLORS["found"] if used else OVERLAY_COLORS["rejected"]
    box = [(u - radius, v - radius), (u + radius, v + radius)]
    if outline:
        draw.ellipse(box, outline=color)
    else:
        draw.ellipse(box, fill=color)


def _draw_line(
    draw: ImageDraw.ImageDraw,
    camera: PinholeCamera,
    frame: BoardFrame,
    start_uv: Vector2,
    end_uv: Vector2,
    *,
    z: float,
) -> None:
    points = [[*frame.point(np.array(uv)), z] for uv in (start_uv, end_uv)]
    (u0, v0), (u1, v1) = camera.project(points)
    draw.line([(u0, v0), (u1, v1)], fill=OVERLAY_COLORS["board"])


def _marker_reports(
    cell: CellConfig,
    found: list[Pixel | None],
    *,
    camera: PinholeCamera,
    solved: PinholeCamera,
) -> list[MarkerReport]:
    reports = []
    for position, pixel in zip(cell.markers, found, strict=True):
        point = np.array([[*position, cell.table_z]])
        predicted = camera.project(point)[0]
        residual = None
        if pixel is not None:
            residual = float(np.linalg.norm(solved.project(point)[0] - np.array(pixel)))
        reports.append(
            MarkerReport(
                position=position,
                predicted_px=(float(predicted[0]), float(predicted[1])),
                found_px=pixel,
                residual_px=residual,
            )
        )
    return reports


def _find_marker(
    gray: Array, camera: PinholeCamera, position: Vector2, *, cell: CellConfig
) -> Pixel | None:
    """표식 원의 중심 화소. 원이 창 안에 온전히 보이지 않으면 None이다.

    단계:
    1. 창 안에서 어두운 쪽과 밝은 쪽의 가운데 밝기보다 어두운 연결 성분을 구한다.
    2. 넓이가 원 넓이에 맞고 창 가장자리에 닿지 않는 성분 중 예측 화소에 가장 가까운 것을 고른다.
    3. 원 경계의 흐린 화소까지 어두운 정도로 가중한 중심을 구한다.
    """
    point = np.array([[*position, cell.table_z]])
    predicted = camera.project(point)[0]
    scale = camera.fx / camera.depth_of(point[0])
    radius = cell.marker_diameter / 2 * scale
    window = _Window.around(
        gray, predicted, half=(cell.marker_diameter / 2 + MARKER_SEARCH_MARGIN) * scale
    )
    if window is None:
        return None
    dark, bright = np.percentile(window.values, MARKER_LEVEL_PERCENTILES)
    if bright - dark < MIN_CONTRAST:
        return None
    area_low, area_high = (ratio * math.pi * radius**2 for ratio in MARKER_AREA_RANGE)
    candidates = [
        component
        for component in _components(window.values < (dark + bright) / 2)
        if area_low <= len(component) <= area_high and not window.touches_edge(component)
    ]
    if not candidates:
        return None
    centers = [window.origin + component.mean(axis=0)[::-1] for component in candidates]
    center = min(centers, key=lambda pixel: float(np.linalg.norm(pixel - predicted)))
    for _ in range(CENTROID_ROUNDS):
        center = _disc_center(gray, center, radius=radius)
    return float(center[0]), float(center[1])


def _disc_center(gray: Array, center: Array, *, radius: float) -> Array:
    """어두운 원의 가중 중심. 원 안쪽과 바깥 고리의 밝기로 화소마다 어두운 비율을 구한다."""
    columns, rows, distance = _pixel_grid(gray.shape, center, reach=radius + OUTER_LEVEL_PX[1])
    values = gray[rows, columns]
    inner = np.median(values[distance <= radius + INNER_LEVEL_PX])
    ring = (distance >= radius + OUTER_LEVEL_PX[0]) & (distance <= radius + OUTER_LEVEL_PX[1])
    outer = np.median(values[ring])
    weight = np.clip((outer - values) / (outer - inner), 0.0, 1.0)
    weight[distance > radius + EDGE_BAND_PX] = 0.0
    return np.array([np.sum(weight * columns), np.sum(weight * rows)]) / np.sum(weight)


def _pixel_grid(
    shape: tuple[int, ...], center: Array, *, reach: float
) -> tuple[NDArray[np.intp], NDArray[np.intp], Array]:
    """`center`에서 `reach` 안의 화소 (열, 행, 거리). 이미지 밖 화소는 뺀다."""
    low_u, low_v = np.floor(center - reach).astype(int)
    high_u, high_v = np.ceil(center + reach).astype(int)
    rows, columns = np.mgrid[
        max(low_v, 0) : min(high_v, shape[0] - 1) + 1,
        max(low_u, 0) : min(high_u, shape[1] - 1) + 1,
    ]
    rows, columns = rows.ravel(), columns.ravel()
    distance = np.hypot(columns - center[0], rows - center[1])
    inside = distance <= reach
    return columns[inside], rows[inside], distance[inside]


@dataclass(frozen=True)
class _Window:
    """이미지에서 자른 창. `origin`은 창 왼쪽 위 화소의 (u, v)다."""

    values: Array
    origin: Array
    clipped: bool
    """창이 이미지 경계에서 잘렸다."""

    @classmethod
    def around(cls, gray: Array, center: Array, *, half: float) -> "_Window | None":
        """`center`를 중심으로 한 변이 약 `2 × half`인 창. 창이 이미지와 겹치지 않으면 None이다."""
        low = np.floor(center - half).astype(int)
        high = np.ceil(center + half).astype(int) + 1
        clipped_low = np.maximum(low, 0)
        clipped_high = np.minimum(high, [gray.shape[1], gray.shape[0]])
        if np.any(clipped_high - clipped_low <= 0):
            return None
        values = gray[clipped_low[1] : clipped_high[1], clipped_low[0] : clipped_high[0]]
        clipped = bool(np.any(low != clipped_low) or np.any(high != clipped_high))
        return cls(values=values, origin=clipped_low.astype(np.float64), clipped=clipped)

    def touches_edge(self, component: NDArray[np.intp]) -> bool:
        rows, columns = component[:, 0], component[:, 1]
        height, width = self.values.shape
        return bool(
            rows.min() == 0
            or columns.min() == 0
            or rows.max() == height - 1
            or columns.max() == width - 1
        )


def _components(mask: NDArray[np.bool_]) -> list[NDArray[np.intp]]:
    """4-이웃 연결 성분마다 (N, 2) 화소 인덱스 (행, 열)."""
    height, width = mask.shape
    remaining = mask.copy()
    components = []
    for start in zip(*np.nonzero(mask), strict=True):
        if not remaining[start]:
            continue
        remaining[start] = False
        queue = deque([start])
        pixels = []
        while queue:
            row, column = queue.popleft()
            pixels.append((row, column))
            for next_row, next_column in (
                (row + 1, column),
                (row - 1, column),
                (row, column + 1),
                (row, column - 1),
            ):
                if (
                    0 <= next_row < height
                    and 0 <= next_column < width
                    and remaining[next_row, next_column]
                ):
                    remaining[next_row, next_column] = False
                    queue.append((next_row, next_column))
        components.append(np.array(pixels, dtype=np.intp))
    return components


def _solve_camera(world: Array, pixels: Array, *, camera: PinholeCamera) -> PinholeCamera:
    """같은 높이의 표식 (N, 3)과 화소 (N, 2)로 카메라 자세를 푼다. intrinsics는 `camera`의 값이다.

    1. 정규화 화소와 표식 xy 사이의 homography H를 DLT로 구한다.
       H의 열은 [r1, r2, r3·z + t]에 비례한다(카메라 좌표 = R_w · 점 + t).
    2. 열 길이로 배율을 정하고, 표식이 카메라 앞에 있도록 부호를 정한다.
    3. [r1, r2, r1 × r2]를 SVD로 가장 가까운 회전 행렬로 바꾼다.
    4. Levenberg-Marquardt로 화소 재투영 오차를 줄인다.
    """
    z = float(world[0, 2])
    normalized = (pixels - [camera.cx, camera.cy]) / [camera.fx, camera.fy]
    homography = _homography(world[:, :2], normalized)
    scale = 2 / (np.linalg.norm(homography[:, 0]) + np.linalg.norm(homography[:, 1]))
    middle = np.array([*world[:, :2].mean(axis=0), 1.0])
    if (homography @ middle)[2] < 0:
        scale = -scale
    first, second = homography[:, 0] * scale, homography[:, 1] * scale
    left, _, right = np.linalg.svd(np.column_stack([first, second, np.cross(first, second)]))
    world_to_camera = left @ right
    translation = homography[:, 2] * scale - world_to_camera[:, 2] * z
    initial = camera.with_pose(world_to_camera.T, -world_to_camera.T @ translation)
    return _refine_pose(world, pixels, camera=initial)


def _homography(source: Array, target: Array) -> Array:
    """점 (N, 2)를 점 (N, 2)로 보내는 3×3 homography. 두 점 집합을 정규화한 DLT다."""
    source_norm, target_norm = _normalizer(source), _normalizer(target)
    src = _apply(source_norm, source)
    dst = _apply(target_norm, target)
    rows = []
    for (x, y), (u, v) in zip(src, dst, strict=True):
        rows.append([-x, -y, -1.0, 0.0, 0.0, 0.0, u * x, u * y, u])
        rows.append([0.0, 0.0, 0.0, -x, -y, -1.0, v * x, v * y, v])
    _, _, right = np.linalg.svd(np.array(rows))
    normalized = right[-1].reshape(3, 3)
    return np.linalg.inv(target_norm) @ normalized @ source_norm


def _normalizer(points: Array) -> Array:
    """점의 평균을 원점으로, 평균 거리를 √2로 보내는 3×3 행렬(Hartley 정규화)."""
    mean = points.mean(axis=0)
    spread = np.mean(np.linalg.norm(points - mean, axis=1))
    factor = math.sqrt(2) / spread
    return np.array(
        [[factor, 0.0, -factor * mean[0]], [0.0, factor, -factor * mean[1]], [0.0, 0.0, 1.0]]
    )


def _apply(matrix: Array, points: Array) -> Array:
    homogeneous = np.column_stack([points, np.ones(len(points))]) @ matrix.T
    return homogeneous[:, :2] / homogeneous[:, 2:]


def _refine_pose(world: Array, pixels: Array, *, camera: PinholeCamera) -> PinholeCamera:
    """재투영 오차를 줄이는 카메라 자세. 변수는 base_link 축 회전 벡터와 카메라 중심 이동이다."""
    rotation, center = camera.rotation_matrix, camera.center

    def pose(parameters: Array) -> PinholeCamera:
        return camera.with_pose(
            rotation_from_vector(parameters[:3]) @ rotation, center + parameters[3:]
        )

    def residual(parameters: Array) -> Array:
        return (pose(parameters).project(world) - pixels).ravel()

    parameters = np.zeros(6)
    error = residual(parameters)
    damping = LM_INITIAL_DAMPING
    for _ in range(LM_ITERATIONS):
        jacobian = np.column_stack(
            [
                (residual(parameters + step) - error) / JACOBIAN_STEP
                for step in np.eye(6) * JACOBIAN_STEP
            ]
        )
        normal = jacobian.T @ jacobian
        gradient = jacobian.T @ error
        delta = np.linalg.solve(normal + damping * np.diag(np.diag(normal)), -gradient)
        candidate_error = residual(parameters + delta)
        if candidate_error @ candidate_error < error @ error:
            parameters, error = parameters + delta, candidate_error
            damping /= 10
        else:
            damping *= 10
        if np.linalg.norm(delta) < LM_STEP_TOLERANCE:
            break
    return pose(parameters)


def _pin_grid(frame: BoardFrame, *, board: Board) -> Array:
    """믿는 보드의 핀 자리 (rows × cols, 2). 순서는 row, col 순이다."""
    return np.array(
        [
            frame.point(np.array([col + 0.5, row + 0.5]))
            for row in range(board.rows)
            for col in range(board.cols)
        ]
    )


def _outside_grid(frame: BoardFrame, *, board: Board) -> Array:
    """믿는 보드 격자 바로 바깥 한 줄의 셀 중심. 이 자리에는 핀이 없다."""
    return np.array(
        [
            frame.point(np.array([col + 0.5, row + 0.5]))
            for row in range(-1, board.rows + 1)
            for col in range(-1, board.cols + 1)
            if row in (-1, board.rows) or col in (-1, board.cols)
        ]
    )


def _pin_response(gray: Array, *, radius: float) -> Array:
    """화소마다 핀 응답. 핀 크기 상자의 평균 밝기에서 둘레 고리의 평균 밝기를 뺀 값이다."""
    inner = max(1, round(radius))
    outer = inner + PIN_RING_PX
    inner_mean, outer_mean = _box_mean(gray, half=inner), _box_mean(gray, half=outer)
    inner_count, outer_count = (2 * inner + 1) ** 2, (2 * outer + 1) ** 2
    ring_mean = (outer_mean * outer_count - inner_mean * inner_count) / (outer_count - inner_count)
    return inner_mean - ring_mean


def _box_mean(gray: Array, *, half: int) -> Array:
    """화소마다 한 변이 `2 × half + 1`인 정사각형의 평균 밝기. 이미지 밖은 가장자리 값이다."""
    size = 2 * half + 1
    padded = np.pad(gray, half, mode="edge")
    integral = np.zeros((padded.shape[0] + 1, padded.shape[1] + 1))
    integral[1:, 1:] = padded.cumsum(axis=0).cumsum(axis=1)
    height, width = gray.shape
    total = (
        integral[size : size + height, size : size + width]
        - integral[:height, size : size + width]
        - integral[size : size + height, :width]
        + integral[:height, :width]
    )
    return total / size**2


def _coarse_offset(
    response: Array,
    camera: PinholeCamera,
    *,
    pins: Array,
    outside: Array,
    center: Array,
    tip_z: float,
) -> tuple[BoardOffset, float]:
    """핀 격자의 대략 오차와 모호함.

    후보마다 점수는 옮긴 핀 자리의 응답 합에서 옮긴 바깥 줄 자리의 양의 응답 합을 뺀 값이다.
    후보의 yaw는 `center`를 축으로 돌린다.
    가장 좋은 후보에서 반 셀 이상 떨어진 후보의 점수 비가 한계를 넘으면 `CalibrationError`를 낸다.
    """
    steps = np.arange(-COARSE_RANGE, COARSE_RANGE + COARSE_STEP / 2, COARSE_STEP)
    shifts = np.stack(np.meshgrid(steps, steps, indexing="ij"), axis=-1).reshape(-1, 2)
    yaws = np.arange(-COARSE_YAW_RANGE, COARSE_YAW_RANGE + COARSE_YAW_STEP / 2, COARSE_YAW_STEP)
    scores = np.empty((len(yaws), len(shifts)))
    for index, yaw in enumerate(yaws):
        turn = BoardOffset(x=0.0, y=0.0, yaw=float(yaw))
        hits = _sample_shifted(
            response, camera, _move(pins, turn, center=center), shifts=shifts, z=tip_z
        )
        misses = _sample_shifted(
            response, camera, _move(outside, turn, center=center), shifts=shifts, z=tip_z
        )
        scores[index] = hits.sum(axis=1) - np.clip(misses, 0, None).sum(axis=1)
    yaw_index, shift_index = np.unravel_index(int(np.argmax(scores)), scores.shape)
    best = scores[yaw_index, shift_index]
    pitch = float(np.min(np.linalg.norm(pins[1:] - pins[0], axis=1)))
    far = np.linalg.norm(shifts - shifts[shift_index], axis=1) >= pitch / 2
    ambiguity = float(scores[:, far].max() / best) if best > 0 else math.inf
    if ambiguity > MAX_COARSE_AMBIGUITY:
        raise CalibrationError(
            f"Board pin grid position is ambiguous. Score ratio {ambiguity:.3f} is more than "
            f"{MAX_COARSE_AMBIGUITY}."
        )
    shift = shifts[shift_index]
    offset = BoardOffset(x=float(shift[0]), y=float(shift[1]), yaw=float(yaws[yaw_index]))
    return offset, ambiguity


def _sample_shifted(
    image: Array, camera: PinholeCamera, points: Array, *, shifts: Array, z: float
) -> Array:
    """점 (N, 2)를 이동 (S, 2)마다 옮겨 투영한 화소의 값 (S, N). 이미지 밖은 0이다."""
    moved = (points[np.newaxis, :, :] + shifts[:, np.newaxis, :]).reshape(-1, 2)
    pixels = np.rint(camera.project(np.column_stack([moved, np.full(len(moved), z)])))
    columns, rows = pixels[:, 0].astype(int), pixels[:, 1].astype(int)
    height, width = image.shape
    inside = (columns >= 0) & (columns < width) & (rows >= 0) & (rows < height)
    values = np.zeros(len(moved))
    values[inside] = image[rows[inside], columns[inside]]
    return values.reshape(len(shifts), len(points))


def _fit_pins(
    gray: Array,
    response: Array,
    camera: PinholeCamera,
    *,
    grid: Array,
    center: Array,
    cols: int,
    tip_z: float,
    pin_radius: float,
    offset: BoardOffset,
    search_radius: float,
) -> tuple[BoardOffset, list[PinReport]]:
    """`offset`만큼 옮긴 핀 자리 둘레에서 핀을 찾고 보드 오차를 맞춘다.

    `grid`는 믿는 핀 자리(m)이고 `center`는 믿는 보드 중심이다. 반지름과 창은 m다.
    맞춤은 믿는 핀 자리 p와 찾은 핀 q 사이의 q ≈ c + t + R(θ)(p − c)다.
    잔차가 `PIN_OUTLIER`보다 큰 핀을 빼고 다시 맞춘다.
    반환 값은 (맞춘 보드 오차, 핀마다 검출 결과)다.
    """
    expected = _move(grid, offset, center=center)
    predicted = camera.project(np.column_stack([expected, np.full(len(grid), tip_z)]))
    scale = camera.fx / camera.depth_of([*center, tip_z])
    found = [
        _pin_center(gray, response, pixel, search=search_radius * scale, radius=pin_radius * scale)
        for pixel in predicted
    ]
    indices = [index for index, pixel in enumerate(found) if pixel is not None]
    if len(indices) < MIN_PINS:
        raise CalibrationError(f"Found {len(indices)} pins. Need {MIN_PINS} or more.")
    source = grid[indices]
    measured = camera.back_project(np.array([found[index] for index in indices]), z=tip_z)

    def fit_pins(used: NDArray[np.bool_]) -> tuple[BoardOffset, Array]:
        if used.sum() < MIN_PINS:
            raise CalibrationError(f"Kept {used.sum()} pins. Need {MIN_PINS} or more.")
        fit = _rigid_fit(source[used], measured[used], center=center)
        return fit, np.linalg.norm(_move(source, fit, center=center) - measured, axis=1)

    used = np.ones(len(indices), dtype=bool)
    fit, residuals = fit_pins(used)
    for _ in range(MAX_FIT_ROUNDS):
        keep = residuals <= PIN_OUTLIER
        if np.array_equal(keep, used):
            break
        used = keep
        fit, residuals = fit_pins(used)
    rms = float(np.sqrt(np.mean(np.square(residuals[used]))))
    if rms > MAX_PIN_RMS:
        raise CalibrationError(
            f"Pin residual RMS {rms * MM_PER_M:.2f} mm is more than {MAX_PIN_RMS * MM_PER_M} mm."
        )
    position = {index: order for order, index in enumerate(indices)}
    pins = [
        PinReport(
            row=index // cols,
            col=index % cols,
            found_px=pixel,
            residual_mm=None
            if index not in position
            else float(residuals[position[index]] * MM_PER_M),
            used=index in position and bool(used[position[index]]),
        )
        for index, pixel in enumerate(found)
    ]
    return fit, pins


def _pin_center(
    gray: Array, response: Array, predicted: Array, *, search: float, radius: float
) -> Pixel | None:
    """`predicted`에서 `search` 안의 핀 응답 최댓값에서 시작한 핀 끝의 가중 중심.

    바탕은 핀 둘레 고리의 중앙값이다.
    바탕에서 가장 밝은 화소까지의 `PIN_LEVEL_RATIO` 위를 핀으로 본다.
    가중 중심은 두 번 구한다. 두 번째는 첫 중심 둘레에서 구한다.
    """
    columns, rows, _ = _pixel_grid(gray.shape, predicted, reach=search)
    if len(rows) == 0:
        return None
    best = int(np.argmax(response[rows, columns]))
    center = np.array([columns[best], rows[best]], dtype=np.float64)
    core = radius + PIN_EDGE_PX
    columns, rows, distance = _pixel_grid(gray.shape, center, reach=core + PIN_RING_PX)
    values = gray[rows, columns]
    background = float(np.median(values[distance > core]))
    peak = float(values[distance <= core].max())
    if peak - background < MIN_CONTRAST:
        return None
    level = background + PIN_LEVEL_RATIO * (peak - background)
    for _ in range(CENTROID_ROUNDS):
        columns, rows, _ = _pixel_grid(gray.shape, center, reach=core)
        weight = np.clip(gray[rows, columns] - level, 0.0, None)
        center = np.array([np.sum(weight * columns), np.sum(weight * rows)]) / weight.sum()
    return float(center[0]), float(center[1])


def _rigid_fit(source: Array, target: Array, *, center: Array) -> BoardOffset:
    """target ≈ center + t + R(θ)(source − center)인 (t, θ). 최소제곱(2D Kabsch)이다."""
    local_source, local_target = source - center, target - center
    source_mean, target_mean = local_source.mean(axis=0), local_target.mean(axis=0)
    a, b = local_source - source_mean, local_target - target_mean
    angle = math.atan2(
        float(np.sum(a[:, 0] * b[:, 1] - a[:, 1] * b[:, 0])),
        float(np.sum(a[:, 0] * b[:, 0] + a[:, 1] * b[:, 1])),
    )
    shift = target_mean - _rotation(angle) @ source_mean
    return BoardOffset(x=float(shift[0]), y=float(shift[1]), yaw=angle)


def _move(points: Array, offset: BoardOffset, *, center: Array) -> Array:
    return center + np.array([offset.x, offset.y]) + (points - center) @ _rotation(offset.yaw).T


def _rotation(angle: float) -> Array:
    return np.array([[math.cos(angle), -math.sin(angle)], [math.sin(angle), math.cos(angle)]])
