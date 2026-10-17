"""layout.json 스키마.

이 모듈은 필드의 타입과 값 범위만 검증한다.
경계·겹침·형태 같은 배치 규칙은 `pixel2place.feasibility`가 검증한다.
"""

from enum import IntEnum, StrEnum
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field

PROJECT_VERSION = "0.1"
DEFAULT_GRID_ROWS = 8
DEFAULT_GRID_COLS = 8
DEFAULT_CELL_SIZE_MM = 20.0

Cell = tuple[int, int]
"""보드 셀 좌표 `(row, col)`. 원점은 왼쪽 위다."""


class PartType(StrEnum):
    """고정 부품 어휘. 샘플마다 새 형태를 만들지 않는다."""

    SQUARE_TILE = "square_tile"
    RECT_TILE = "rect_tile"
    L_TILE = "L_tile"
    T_TILE = "T_tile"


class Color(StrEnum):
    """고정 팔레트. RGB 값은 `pixel2place.generation.synthetic`에만 있다."""

    BLUE = "blue"
    GRAY = "gray"
    WHITE = "white"
    ORANGE = "orange"


class Rotation(IntEnum):
    """시계 방향 회전 각도(도)."""

    DEG_0 = 0
    DEG_90 = 90
    DEG_180 = 180
    DEG_270 = 270


class _StrictModel(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)


class Board(_StrictModel):
    """부품을 놓는 격자 보드."""

    type: Literal["grid"] = "grid"
    rows: int = Field(default=DEFAULT_GRID_ROWS, ge=1)
    cols: int = Field(default=DEFAULT_GRID_COLS, ge=1)
    cell_size_mm: float = Field(default=DEFAULT_CELL_SIZE_MM, gt=0)


class Part(_StrictModel):
    """보드에 놓인 부품 1개."""

    part_id: str = Field(min_length=1)
    type: PartType
    color: Color
    cells: list[Cell] = Field(min_length=1)
    rotation_deg: Rotation = Rotation.DEG_0


class Layout(_StrictModel):
    """보드 1개의 목표 배치."""

    project_version: str = PROJECT_VERSION
    board: Board = Field(default_factory=Board)
    palette: list[Color] = Field(min_length=1)
    parts: list[Part]
    feasible: bool | None = None
    notes: list[str] = Field(default_factory=list)
