"""배치가 끝난 보드의 top-down 이미지에서 부품마다 칸과 색을 확인한다(카메라 판정).

입력:
- 작업(`task.json`). 부품의 종류, 회전, 목표 위치와 색이 있다.
- 최종 top-down 이미지(`topdown_final.png`).
- 카메라. 보드 캘리브레이션이 추정한 `PinholeCamera`다.
- 보드 자세(선택). 없으면 `task.cell`의 보드다.

단계:
1. 부품마다 계획한 칸을 계산한다.
2. 칸마다 핀이 보이는지 본다. 빈 칸에는 어두운 보드 위에 밝은 핀 끝이 있다.
3. 부품마다 윗면 화소의 색을 그 색의 기준 색과 비교한다.
4. 부품을 일치, 그림자, 불일치로 나눈다.
5. 계획에 없는 칸 중에서 핀이 안 보이는 칸을 모은다.

범위:
- 판독은 칸과 색을 확인한다. 부품의 종류는 확인하지 않는다.
- 기울기, 높이와 작은 yaw 오차는 판정하지 않는다. 그 판정은 `cell.check`가 맡는다.
- 기준 색은 실행기의 최종 캡처 조명에서 잰 값이다. 조명이 바뀌면 다시 잰다.

numpy와 PIL만 쓴다.
"""

from collections import Counter
from dataclasses import dataclass
from enum import StrEnum
from typing import Annotated

import numpy as np
from numpy.typing import NDArray
from PIL import Image, ImageDraw
from pydantic import BaseModel, ConfigDict, Field, model_validator

from pixel2place.cell.camera import Array, PinholeCamera
from pixel2place.cell.grasp import centroid
from pixel2place.cell.scene import BoardPlacement, placed_cell
from pixel2place.cell.task import MM_PER_M, BoardFrame, PartTask, Task
from pixel2place.generation.synthetic import RGB
from pixel2place.layout.parts import shape_of
from pixel2place.layout.schema import Board, Cell, Color, PartType

# 최종 캡처에서 잰 부품 윗면의 색. 조명은 실행기의 돔과 태양광이다. 팔레트 색과 다르다.
SUNLIT_RGB: dict[Color, RGB] = {
    Color.BLUE: (117, 172, 215),
    Color.GRAY: (194, 195, 194),
    Color.WHITE: (222, 222, 222),
    Color.ORANGE: (222, 198, 93),
}
# 팔 그림자 안에서 잰 부품 윗면의 색
SHADOW_RGB: dict[Color, RGB] = {
    Color.BLUE: (36, 69, 119),
    Color.GRAY: (93, 94, 94),
    Color.WHITE: (134, 134, 135),
    Color.ORANGE: (133, 96, 28),
}
# 색을 재는 정사각형은 칸의 각 변에서 이만큼 안에 있다(m).
# 부품 면은 칸 경계에서 `part_inset` 안에 있다. 그 면의 가장자리 화소를 피한다.
FACE_MARGIN = 0.0025
# 핀 끝 화소 둘레의 창 2개의 반폭(px). 작은 창에서 가장 밝은 화소를, 큰 창에서 바탕 밝기를 잰다.
PIN_PEAK_HALF_PX = 2
PIN_BACKGROUND_HALF_PX = 7
# 바탕 밝기의 하한. 검은 창에서 0으로 나누지 않는다.
MIN_BACKGROUND = 1.0
# 목표 위치에서 구한 칸 번호의 반올림 잔차 한계(칸)
GRID_TOLERANCE = 1e-6
# 칸 중심에서 본 정사각형 꼭짓점의 방향. 테두리를 도는 순서다.
SQUARE_CORNERS = np.array([(-1.0, -1.0), (1.0, -1.0), (1.0, 1.0), (-1.0, 1.0)])
# overlay에서 계획에 없는 점유 칸의 색과 X의 반길이(px)
UNEXPECTED_RGB: RGB = (255, 0, 255)
OVERLAY_CROSS_PX = 4

Ratio = Annotated[float, Field(ge=0, le=1)]
Positive = Annotated[float, Field(gt=0)]


class PartStatus(StrEnum):
    """이미지에서 읽은 부품 1개의 상태."""

    MATCH = "match"
    """모든 칸의 핀이 가려져 있고 윗면이 태양광 아래의 기준 색이다."""
    SHADOWED = "shadowed"
    """모든 칸의 핀이 가려져 있다. 윗면은 그림자 색과 태양광 색 사이의 색이다.

    색을 확인하지 못했다. 통과가 아니다. 팔 그림자 안의 부품과 기운 부품이 이 상태가 된다.
    """
    MISMATCH = "mismatch"
    """핀이 보이는 칸이 있거나 윗면이 기준 색이 아니다."""


OVERLAY_RGB: dict[PartStatus, RGB] = {
    PartStatus.MATCH: (0, 220, 0),
    PartStatus.SHADOWED: (255, 200, 0),
    PartStatus.MISMATCH: (255, 40, 40),
}


class ImageCheckError(ValueError):
    """이미지에서 배치를 읽지 못했다."""


class _ImageCheckModel(BaseModel):
    # NaN은 JSON에서 null이 된다. 그러면 저장한 보고서를 다시 읽지 못한다.
    model_config = ConfigDict(extra="forbid", frozen=True, allow_inf_nan=False)


class ImageCheckLimits(_ImageCheckModel):
    """판독의 기준값."""

    coverage: Ratio = 0.95
    """`coverage`와 `shadow_coverage`의 통과 기준.

    앉은 부품은 색을 재는 정사각형을 모두 덮는다.
    1칸 부품의 정사각형은 한 변이 약 19 px다. 화소 1줄이 다른 색이면 비율이 약 0.947이다.
    """
    color_distance: Positive = 20.0
    """태양광 아래의 기준 색과 같다고 보는 RGB 거리의 한계(0–255).

    수평인 윗면의 화소의 99.9%는 기준 색에서 12 안에 있다.
    핀 끝에 얹혀 기운 윗면은 20 넘게 떨어져 있다.
    """
    shadow_distance: Positive = 25.0
    """그림자 색과 태양광 색을 잇는 선분 위의 색이라고 보는 RGB 거리의 한계."""
    pin_ratio: Positive = 1.9
    """핀이 보인다고 보는 밝기 비의 하한. 빈 칸의 비는 2.0 이상이다."""


DEFAULT_LIMITS = ImageCheckLimits()


class PartImageCheck(_ImageCheckModel):
    """부품 1개의 판독."""

    sequence: int
    part_id: str
    type: PartType
    color: Color
    cells: list[Cell]
    """계획한 보드 칸 (row, col)."""
    coverage: Ratio
    """윗면 화소 중에서 태양광 아래의 기준 색인 화소의 비율."""
    shadow_coverage: Ratio
    """윗면 화소 중에서 그림자 색과 태양광 색 사이의 색인 화소의 비율."""
    pins_hidden: bool
    """모든 칸의 핀이 안 보인다."""
    empty: bool = False
    """모든 칸의 핀이 보인다. 계획한 칸에 부품이 없다. 이 값이 없는 보고서는 False로 읽는다."""
    status: PartStatus


class ImageCheckReport(_ImageCheckModel):
    """이미지 1장의 판독. 부품은 task의 순서와 같다.

    모든 값이 저장 필드다. 그래서 `model_validate_json`이 저장한 보고서를 그대로 읽는다.
    """

    image_size: tuple[int, int]
    limits: ImageCheckLimits
    parts: list[PartImageCheck]
    unexpected_cells: list[Cell]
    """계획에 없는 칸 중에서 핀이 안 보이는 칸."""
    match_count: int
    shadowed_count: int
    mismatch_count: int
    passed: bool
    """불일치 부품과 계획에 없는 점유 칸이 없다. 그림자 부품은 실패로 세지 않는다."""

    @model_validator(mode="after")
    def _counts_follow_parts(self) -> "ImageCheckReport":
        for part in self.parts:
            expected = _status(
                pins_hidden=part.pins_hidden,
                coverage=part.coverage,
                shadow_coverage=part.shadow_coverage,
                limit=self.limits.coverage,
            )
            if part.status != expected:
                raise ValueError(f"Part {part.part_id} status does not agree with its values.")
            if part.empty and part.pins_hidden:
                raise ValueError(f"Part {part.part_id} cannot be empty with hidden pins.")
        counts = Counter(part.status for part in self.parts)
        if (self.match_count, self.shadowed_count, self.mismatch_count) != (
            counts[PartStatus.MATCH],
            counts[PartStatus.SHADOWED],
            counts[PartStatus.MISMATCH],
        ):
            raise ValueError("Status counts do not agree with the part statuses.")
        if self.passed != (self.mismatch_count == 0 and not self.unexpected_cells):
            raise ValueError("passed does not agree with the mismatch count and unexpected cells.")
        return self


def planned_cells(task: Task) -> list[list[Cell]]:
    """부품마다 계획한 보드 칸. 순서는 `task.parts`와 같다.

    `PartTask`에는 칸 목록이 없다. 회전한 형태의 칸 중심 평균이 `target_center`에 오는 칸을 구한다.

    Raises:
        ImageCheckError: 목표 위치가 격자에 맞지 않거나 칸이 보드 밖일 때.
    """
    frame = BoardFrame(cell=task.cell, cell_size=task.board.cell_size_mm / MM_PER_M)
    return [_part_cells(part, frame=frame, board=task.board) for part in task.parts]


def check_image(
    task: Task,
    image: Image.Image,
    camera: PinholeCamera,
    *,
    board: BoardPlacement | None = None,
    limits: ImageCheckLimits = DEFAULT_LIMITS,
) -> ImageCheckReport:
    """최종 top-down 이미지에서 부품마다 계획한 칸과 색을 확인한다.

    `board`는 칸을 투영할 실제 보드 자세다. 평가에서만 준다. 없으면 `task.cell`의 보드에 투영한다.

    아래 경우에는 `ImageCheckError`를 낸다.
    - 이미지 크기가 카메라와 다르다.
    - 셀 설정에 핀이 없다.
    - 부품의 목표 위치가 격자에 맞지 않거나 칸이 보드 밖이다.
    - 칸이 카메라 뒤에 있거나 칸의 창이 이미지 밖이다.
    """
    view = _board_view(task, image, camera, board=board)
    rgb = np.asarray(image.convert("RGB"), dtype=np.float64)
    cells = planned_cells(task)
    hidden = _hidden_pins(rgb.mean(axis=2), view, board=task.board, limit=limits.pin_ratio)
    parts = [
        _check_part(part, part_cells, rgb=rgb, view=view, hidden=hidden, limits=limits)
        for part, part_cells in zip(task.parts, cells, strict=True)
    ]
    planned = {cell for part_cells in cells for cell in part_cells}
    unexpected = sorted(hidden - planned)
    counts = Counter(part.status for part in parts)
    return ImageCheckReport(
        image_size=image.size,
        limits=limits,
        parts=parts,
        unexpected_cells=unexpected,
        match_count=counts[PartStatus.MATCH],
        shadowed_count=counts[PartStatus.SHADOWED],
        mismatch_count=counts[PartStatus.MISMATCH],
        passed=counts[PartStatus.MISMATCH] == 0 and not unexpected,
    )


def draw_overlay(
    image: Image.Image,
    report: ImageCheckReport,
    *,
    task: Task,
    camera: PinholeCamera,
    board: BoardPlacement | None = None,
) -> Image.Image:
    """판독한 칸을 이미지 위에 그린다. `task`, `camera`, `board`는 `check_image`에 준 값이다.

    - 부품의 칸: 색을 잰 정사각형의 테두리. 일치는 초록, 그림자는 노랑, 불일치는 빨강이다.
    - 계획에 없는 점유 칸: 핀 자리의 자홍 X
    """
    view = _board_view(task, image, camera, board=board)
    overlay = image.convert("RGB")
    draw = ImageDraw.Draw(overlay)
    for part in report.parts:
        for cell in part.cells:
            corners = view.pixels(_face_corners(cell, half=view.face_half), z=view.top_z)
            outline = [(float(u), float(v)) for u, v in corners]
            draw.polygon(outline, outline=OVERLAY_RGB[part.status])
    for cell in report.unexpected_cells:
        u, v = view.pixels(_cell_center(cell)[np.newaxis], z=view.tip_z)[0]
        reach = OVERLAY_CROSS_PX
        draw.line([(u - reach, v - reach), (u + reach, v + reach)], fill=UNEXPECTED_RGB)
        draw.line([(u - reach, v + reach), (u + reach, v - reach)], fill=UNEXPECTED_RGB)
    return overlay


@dataclass(frozen=True)
class _BoardView:
    """카메라와 투영할 보드. 보드 격자 좌표를 화소로 옮긴다."""

    camera: PinholeCamera
    frame: BoardFrame
    top_z: float
    """부품 윗면의 높이(m)."""
    tip_z: float
    """핀 끝의 높이(m)."""
    face_half: float
    """색을 재는 정사각형의 반변(칸)."""

    def pixels(self, uv: Array, *, z: float) -> Array:
        """높이 `z`의 격자 점 (N, 2)의 화소 (N, 2). 카메라 뒤의 점은 `ImageCheckError`다."""
        points = [[*self.frame.point(point), z] for point in uv]
        if any(self.camera.depth_of(point) <= 0 for point in points):
            raise ImageCheckError("The board is behind the camera.")
        return self.camera.project(points)


def _board_view(
    task: Task, image: Image.Image, camera: PinholeCamera, *, board: BoardPlacement | None
) -> _BoardView:
    if image.size != (camera.width, camera.height):
        raise ImageCheckError(
            f"Image size {image.size[0]}x{image.size[1]} is not the camera size "
            f"{camera.width}x{camera.height}."
        )
    if task.cell.locator is None:
        raise ImageCheckError("Cell config has no board pins (locator is null).")
    cell = task.cell if board is None else placed_cell(task.cell, placement=board)
    cell_size = task.board.cell_size_mm / MM_PER_M
    return _BoardView(
        camera=camera,
        frame=BoardFrame(cell=cell, cell_size=cell_size),
        top_z=cell.table_z + cell.part_height,
        tip_z=cell.table_z + task.cell.locator.pin_height,
        face_half=0.5 - FACE_MARGIN / cell_size,
    )


def _part_cells(part: PartTask, *, frame: BoardFrame, board: Board) -> list[Cell]:
    shape = sorted(shape_of(part.type, part.rotation_deg))
    # 형태의 칸 번호는 0에서 시작한다. 그 형태를 목표 위치로 옮기는 칸 수를 구한다.
    offset = frame.grid_uv(np.array(part.target_center)) - centroid(shape)
    nearest = np.rint(offset)
    if np.abs(offset - nearest).max() > GRID_TOLERANCE:
        raise ImageCheckError(f"Part {part.part_id} target is not on the board grid.")
    col_offset, row_offset = int(nearest[0]), int(nearest[1])
    cells = [(row + row_offset, col + col_offset) for row, col in shape]
    if not all(0 <= row < board.rows and 0 <= col < board.cols for row, col in cells):
        raise ImageCheckError(f"Part {part.part_id} has a cell outside the board.")
    return cells


def _hidden_pins(gray: Array, view: _BoardView, *, board: Board, limit: float) -> set[Cell]:
    """핀이 안 보이는 칸."""
    hidden: set[Cell] = set()
    for row in range(board.rows):
        for col in range(board.cols):
            cell = (row, col)
            pixel = view.pixels(_cell_center(cell)[np.newaxis], z=view.tip_z)[0]
            if _pin_ratio(gray, pixel, cell=cell) < limit:
                hidden.add(cell)
    return hidden


def _pin_ratio(gray: Array, pixel: Array, *, cell: Cell) -> float:
    """핀 끝 둘레에서 가장 밝은 화소를 바탕 밝기로 나눈 값.

    빈 칸은 어두운 보드 위에 밝은 핀 끝이 있다. 그래서 비가 크다. 부품이 덮은 칸은 밝기가 고르다.
    그림자는 핀과 보드를 함께 어둡게 한다. 그래서 비는 그림자 안에서도 거의 같다.
    """
    column, row = (int(value) for value in np.rint(pixel))
    reach = PIN_BACKGROUND_HALF_PX
    height, width = gray.shape
    if row < reach or column < reach or row + reach >= height or column + reach >= width:
        raise ImageCheckError(f"The window of cell {cell} is outside the image.")
    peak = _window(gray, row=row, column=column, half=PIN_PEAK_HALF_PX).max()
    background = np.median(_window(gray, row=row, column=column, half=reach))
    return float(peak / max(background, MIN_BACKGROUND))


def _window(gray: Array, *, row: int, column: int, half: int) -> Array:
    return gray[row - half : row + half + 1, column - half : column + half + 1]


def _check_part(
    part: PartTask,
    cells: list[Cell],
    *,
    rgb: Array,
    view: _BoardView,
    hidden: set[Cell],
    limits: ImageCheckLimits,
) -> PartImageCheck:
    pixels = np.concatenate([_face_pixels(view, cell, size=rgb.shape[:2]) for cell in cells])
    if len(pixels) == 0:
        raise ImageCheckError(f"Part {part.part_id} has no pixels in the image.")
    colors = rgb[pixels[:, 1], pixels[:, 0]]
    sunlit = np.array(SUNLIT_RGB[part.color], dtype=np.float64)
    shadow = np.array(SHADOW_RGB[part.color], dtype=np.float64)
    in_sun = np.linalg.norm(colors - sunlit, axis=1) < limits.color_distance
    in_shadow = _segment_distance(colors, shadow, sunlit) < limits.shadow_distance
    coverage, shadow_coverage = float(np.mean(in_sun)), float(np.mean(in_shadow))
    pins_hidden = all(cell in hidden for cell in cells)
    return PartImageCheck(
        sequence=part.sequence,
        part_id=part.part_id,
        type=part.type,
        color=part.color,
        cells=cells,
        coverage=coverage,
        shadow_coverage=shadow_coverage,
        pins_hidden=pins_hidden,
        empty=not any(cell in hidden for cell in cells),
        status=_status(
            pins_hidden=pins_hidden,
            coverage=coverage,
            shadow_coverage=shadow_coverage,
            limit=limits.coverage,
        ),
    )


def _face_pixels(view: _BoardView, cell: Cell, *, size: tuple[int, ...]) -> NDArray[np.intp]:
    """칸의 윗면에서 색을 잴 화소의 (열, 행). 화소 중심이 줄인 정사각형 안에 있는 화소다.

    화소 중심을 윗면 높이의 평면으로 되돌려서 격자 좌표로 고른다. `size`는 이미지의 (행, 열)이다.
    """
    corners = view.pixels(_face_corners(cell, half=view.face_half), z=view.top_z)
    low = np.floor(corners.min(axis=0)).astype(int)
    high = np.ceil(corners.max(axis=0)).astype(int)
    if low.min() < 0 or high[0] >= size[1] or high[1] >= size[0]:
        raise ImageCheckError(f"The window of cell {cell} is outside the image.")
    rows, columns = np.mgrid[low[1] : high[1] + 1, low[0] : high[0] + 1]
    candidates = np.column_stack([columns.ravel(), rows.ravel()])
    uv = view.frame.grid_uv(view.camera.back_project(candidates, z=view.top_z))
    inside = np.all(np.abs(uv - _cell_center(cell)) <= view.face_half, axis=1)
    return candidates[inside]


def _cell_center(cell: Cell) -> Array:
    row, col = cell
    return np.array([col + 0.5, row + 0.5])


def _face_corners(cell: Cell, *, half: float) -> Array:
    """색을 재는 정사각형의 꼭짓점 4개의 격자 좌표."""
    return _cell_center(cell) + half * SQUARE_CORNERS


def _segment_distance(colors: Array, start: Array, end: Array) -> Array:
    """색 (N, 3)에서 `start`와 `end`를 잇는 선분까지의 RGB 거리 (N,)."""
    direction = end - start
    position = np.clip((colors - start) @ direction / (direction @ direction), 0.0, 1.0)
    return np.linalg.norm(colors - (start + position[:, np.newaxis] * direction), axis=1)


def _status(
    *, pins_hidden: bool, coverage: float, shadow_coverage: float, limit: float
) -> PartStatus:
    if not pins_hidden:
        return PartStatus.MISMATCH
    if coverage >= limit:
        return PartStatus.MATCH
    if shadow_coverage >= limit:
        return PartStatus.SHADOWED
    return PartStatus.MISMATCH
