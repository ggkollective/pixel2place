"""셀의 합성 top-down 이미지. 캘리브레이션 캡처와 배치가 끝난 보드를 그린다.

단위 테스트와 가짜 Isaac 실행기(`--capture`)가 함께 쓴다.
화소마다 광선을 쏘아 수평면과 만나는 점의 색을 칠한다. 추정 코드의 투영 함수를 쓰지 않는다.

- 핀은 핀 끝 높이의 밝은 원판이다. 옆면과 모따기는 없다.
- 부품은 윗면 높이의 색칠한 면이다. 옆면은 없다.
- 나머지(표식, 보드 판, 격자선, 공급대, 작업대)는 모두 `table_z` 평면에 있다.
- 팔은 없다. 그림자는 작업대 위의 직사각형 띠로만 그린다.
"""

from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np
from PIL import Image

from pixel2place.cell.camera import PinholeCamera, rotation_from_vector
from pixel2place.cell.config import CellConfig, Vector2
from pixel2place.cell.scene import BoardPlacement, CameraOffset
from pixel2place.generation.synthetic import RGB
from pixel2place.layout.schema import Board

# 실행기(`sim/place_task.py`)의 top-down 카메라와 같은 공칭 자세와 렌즈
TOPDOWN_EYE = (0.19, 0.04, 0.6)
FOCAL_LENGTH_MM = 24.0
HORIZONTAL_APERTURE_MM = 20.955
# 열은 카메라 x(이미지 오른쪽 = +y), y(이미지 아래쪽 = +x), z(아래를 봄)축이다.
TOPDOWN_ROTATION = ((0.0, 1.0, 0.0), (1.0, 0.0, 0.0), (0.0, 0.0, -1.0))
BOARD_MARGIN_M = 0.01
GRID_LINE_WIDTH_M = 0.001
FEEDER_PAD_SIZE_M = 0.08
MARKER_PLATE_RATIO = 2.0
TABLE_GRAY = 200.0
BOARD_GRAY = 30.0
GRID_GRAY = 70.0
PIN_GRAY = 170.0
FEEDER_GRAY = 62.0
MARKER_PLATE_GRAY = 235.0
MARKER_GRAY = 20.0
# 그림자 띠 안에서 부품 윗면이 아닌 화소의 밝기에 곱하는 값
SHADOW_SCALE = 0.33


@dataclass(frozen=True)
class PartFace:
    """부품 1개의 윗면. 모양은 실행기의 부품 블록과 같다.

    칸마다 한 변이 칸 크기인 정사각형이다. 부품의 바깥 변만 `part_inset`만큼 줄인다.
    """

    center: Vector2
    """블록 칸 중심의 평균 위치(base_link, m)."""
    yaw: float
    """블록 x축의 방위각(rad)."""
    block_cells: tuple[tuple[int, int], ...]
    """블록 좌표의 칸 번호. 규약은 `PartTask.block_cells`와 같다."""
    rgb: RGB
    shadow_rgb: RGB
    """그림자 띠 안에서의 색."""


@dataclass(frozen=True)
class ShadowBand:
    """작업대 위의 직사각형 그림자. 변은 base_link 축과 나란하다.

    띠 안에서 부품 윗면은 `PartFace.shadow_rgb`다. 나머지 화소의 밝기는 `SHADOW_SCALE`배다.
    """

    low: Vector2
    high: Vector2


def nominal_camera(*, width: int = 640, height: int = 480) -> PinholeCamera:
    """실행기의 공칭 top-down 카메라. 화소는 정사각이다."""
    focal = FOCAL_LENGTH_MM / HORIZONTAL_APERTURE_MM * width
    return PinholeCamera(
        width=width,
        height=height,
        fx=focal,
        fy=focal,
        cx=(width - 1) / 2,
        cy=(height - 1) / 2,
        rotation=TOPDOWN_ROTATION,
        position=TOPDOWN_EYE,
    )


def shaken(camera: PinholeCamera, offset: CameraOffset | None) -> PinholeCamera:
    """`offset`을 적용한 카메라. 실행기가 캡처 카메라에 적용하는 식과 같다."""
    if offset is None:
        return camera
    return camera.with_pose(
        rotation_from_vector(offset.rotation) @ camera.rotation_matrix,
        camera.center + np.array(offset.position),
    )


def render_topdown(
    camera: PinholeCamera,
    *,
    cell: CellConfig,
    board: Board,
    placement: BoardPlacement | None = None,
    supersample: int = 2,
    faces: Sequence[PartFace] = (),
    shadow: ShadowBand | None = None,
) -> Image.Image:
    """`camera`로 본 셀의 RGB 이미지. `placement`가 없으면 보드는 셀 설정의 자세에 있다.

    `faces`와 `shadow`가 없으면 이미지는 회색이다. 부품 윗면은 핀을 가린다.
    """
    offsets = (np.arange(supersample) + 0.5) / supersample - 0.5
    rows = (np.arange(camera.height)[:, None] + offsets[None, :]).ravel()
    columns = (np.arange(camera.width)[:, None] + offsets[None, :]).ravel()
    v, u = np.meshgrid(rows, columns, indexing="ij")
    rays = (
        np.stack(
            [(u - camera.cx) / camera.fx, (v - camera.cy) / camera.fy, np.ones_like(u)], axis=-1
        )
        @ camera.rotation_matrix.T
    )
    pose = cell if placement is None else placement
    board_origin = np.array(pose.board_origin)
    row_direction, col_direction = np.array(pose.row_direction), np.array(pose.col_direction)
    cell_size = board.cell_size_mm / 1000.0

    def hit(z: float) -> np.ndarray:
        scale = (z - camera.center[2]) / rays[..., 2]
        return camera.center[:2] + rays[..., :2] * scale[..., None]

    def board_uv(points: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        local = points - board_origin
        return local @ col_direction / cell_size, local @ row_direction / cell_size

    table = hit(cell.table_z)
    gray = np.full(table.shape[:2], TABLE_GRAY)
    feeder = np.max(np.abs(table - np.array(cell.feeder_center)), axis=-1) <= FEEDER_PAD_SIZE_M / 2
    gray[feeder] = FEEDER_GRAY
    u_board, v_board = board_uv(table)
    margin = BOARD_MARGIN_M / cell_size
    plate = (
        (u_board >= -margin)
        & (u_board <= board.cols + margin)
        & (v_board >= -margin)
        & (v_board <= board.rows + margin)
    )
    gray[plate] = BOARD_GRAY
    half_line = GRID_LINE_WIDTH_M / 2 / cell_size
    inside = (u_board >= 0) & (u_board <= board.cols) & (v_board >= 0) & (v_board <= board.rows)
    on_line = (np.abs(u_board - np.round(u_board)) <= half_line) | (
        np.abs(v_board - np.round(v_board)) <= half_line
    )
    gray[inside & on_line] = GRID_GRAY
    plate_half = cell.marker_diameter * MARKER_PLATE_RATIO / 2
    for marker in cell.markers:
        local = table - np.array(marker)
        gray[np.max(np.abs(local), axis=-1) <= plate_half] = MARKER_PLATE_GRAY
        gray[np.hypot(local[..., 0], local[..., 1]) <= cell.marker_diameter / 2] = MARKER_GRAY
    if cell.locator is not None:
        tip = hit(cell.table_z + cell.locator.pin_height)
        u_tip, v_tip = board_uv(tip)
        near_u = np.clip(np.floor(u_tip), 0, board.cols - 1) + 0.5
        near_v = np.clip(np.floor(v_tip), 0, board.rows - 1) + 0.5
        distance = np.hypot(u_tip - near_u, v_tip - near_v) * cell_size
        gray[distance <= cell.locator.pin_diameter / 2] = PIN_GRAY
    colors = np.repeat(gray[..., np.newaxis], 3, axis=-1)
    shaded = np.zeros(gray.shape, dtype=bool)
    if shadow is not None:
        shaded = np.all((table >= np.array(shadow.low)) & (table <= np.array(shadow.high)), axis=-1)
        colors[shaded] *= SHADOW_SCALE
    if faces:
        top = hit(cell.table_z + cell.part_height)
        for face in faces:
            covered = _face_mask(top, face, cell_size=cell_size, inset=cell.part_inset)
            colors[covered & ~shaded] = face.rgb
            colors[covered & shaded] = face.shadow_rgb
    pixels = colors.reshape(camera.height, supersample, camera.width, supersample, 3).mean(
        axis=(1, 3)
    )
    return Image.fromarray(np.round(pixels).astype(np.uint8))


def _face_mask(points: np.ndarray, face: PartFace, *, cell_size: float, inset: float) -> np.ndarray:
    """윗면 높이의 점 (..., 2)가 부품 윗면 안에 있으면 True다."""
    local = points - np.array(face.center)
    cosine, sine = np.cos(face.yaw), np.sin(face.yaw)
    # 블록 좌표(칸). 원점은 블록 칸 중심의 평균이다.
    block_x = (local[..., 0] * cosine + local[..., 1] * sine) / cell_size
    block_y = (-local[..., 0] * sine + local[..., 1] * cosine) / cell_size
    cells = set(face.block_cells)
    mean_x, mean_y = np.mean(face.block_cells, axis=0)
    outer = 0.5 - inset / cell_size
    mask = np.zeros(points.shape[:-1], dtype=bool)
    for i, j in cells:
        x, y = block_x - (i - mean_x), block_y - (j - mean_y)
        # 이웃 칸이 같은 부품이면 칸 경계까지 채운다.
        mask |= (
            (x >= -(0.5 if (i - 1, j) in cells else outer))
            & (x <= (0.5 if (i + 1, j) in cells else outer))
            & (y >= -(0.5 if (i, j - 1) in cells else outer))
            & (y <= (0.5 if (i, j + 1) in cells else outer))
        )
    return mask
