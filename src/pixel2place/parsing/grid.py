"""픽셀 기반 격자 판독기.

합성 렌더러 규약의 이미지를 읽는다.

- 보드가 이미지 전체를 채운다.
- 부품 셀은 팔레트 색으로 채운다.
- 서로 다른 부품의 경계에는 어두운 외곽선이 있다.

색 값은 `pixel2place.generation.synthetic`의 상수를 쓴다.
"""

import logging

from PIL import Image, ImageStat

from pixel2place.generation.synthetic import BOARD_RGB, COLOR_RGB, PART_OUTLINE_RGB, RGB
from pixel2place.layout.parts import normalize, shape_of
from pixel2place.layout.schema import Board, Cell, Color, Layout, Part, PartType, Rotation

logger = logging.getLogger(__name__)

PART_ID_FORMAT = "P{index:03d}"
# 셀 색은 셀 가운데 영역의 중앙값으로 정한다. 영역 폭은 셀 크기에 대한 비율이다.
CELL_SAMPLE_RATIO = 0.5
# 경계 판정은 공유 변을 가로지르는 띠의 중앙값으로 정한다. 띠 폭은 외곽선 폭과 같다.
EDGE_BAND_PX = 4
# 경계 띠의 중앙값과 외곽선 색의 최대 제곱 거리.
EDGE_MAX_DISTANCE_SQ = 30**2


class GridParseError(ValueError):
    """이미지에서 부품을 하나도 읽지 못했다."""


class GridImageParser:
    """셀 색과 부품 외곽선을 픽셀로 읽어 `Layout`을 만든다."""

    def __init__(self, *, board: Board | None = None) -> None:
        self._board = board or Board()

    def parse(self, image: Image.Image) -> Layout:
        rgb = image.convert("RGB")
        board = self._board
        cell_w = rgb.width / board.cols
        cell_h = rgb.height / board.rows

        colors: dict[Cell, Color] = {}
        for row in range(board.rows):
            for col in range(board.cols):
                color = _cell_color(rgb, row=row, col=col, cell_w=cell_w, cell_h=cell_h)
                if color is not None:
                    colors[(row, col)] = color

        groups = _UnionFind(colors)
        for (row, col), color in colors.items():
            for neighbor in ((row, col + 1), (row + 1, col)):
                if colors.get(neighbor) != color:
                    continue
                if not _has_edge(rgb, (row, col), neighbor, cell_w=cell_w, cell_h=cell_h):
                    groups.union((row, col), neighbor)

        parts: list[Part] = []
        notes: list[str] = []
        for cells in sorted(groups.groups(), key=min):
            match = _match_shape(cells)
            if match is None:
                cell_list = [list(cell) for cell in sorted(cells)]
                notes.append(f"Unknown shape at cells {cell_list}.")
                logger.warning("Grid parser found an unknown shape.", extra={"cells": cell_list})
                continue
            part_type, rotation = match
            parts.append(
                Part(
                    part_id=PART_ID_FORMAT.format(index=len(parts) + 1),
                    type=part_type,
                    color=colors[min(cells)],
                    cells=sorted(cells),
                    rotation_deg=rotation,
                )
            )
        if not parts:
            raise GridParseError("Image contains no known parts.")
        palette = sorted({part.color for part in parts})
        return Layout(board=board, palette=palette, parts=parts, notes=notes)


def _distance_sq(a: RGB, b: RGB) -> int:
    return sum((x - y) ** 2 for x, y in zip(a, b, strict=True))


def _median(image: Image.Image, box: tuple[float, float, float, float]) -> RGB:
    left, top, right, bottom = (round(value) for value in box)
    region = image.crop((left, top, max(right, left + 1), max(bottom, top + 1)))
    red, green, blue = (int(value) for value in ImageStat.Stat(region).median)
    return (red, green, blue)


def _cell_color(
    image: Image.Image, *, row: int, col: int, cell_w: float, cell_h: float
) -> Color | None:
    """셀 가운데 색에 가장 가까운 팔레트 색을 반환한다. 보드 색이 더 가까우면 None이다."""
    margin_x = cell_w * (1 - CELL_SAMPLE_RATIO) / 2
    margin_y = cell_h * (1 - CELL_SAMPLE_RATIO) / 2
    left, top = col * cell_w, row * cell_h
    sample = _median(
        image, (left + margin_x, top + margin_y, left + cell_w - margin_x, top + cell_h - margin_y)
    )
    candidates: dict[Color | None, RGB] = {None: BOARD_RGB, **COLOR_RGB}
    return min(candidates, key=lambda color: _distance_sq(candidates[color], sample))


def _has_edge(
    image: Image.Image, cell: Cell, neighbor: Cell, *, cell_w: float, cell_h: float
) -> bool:
    """두 셀의 공유 변에 부품 외곽선이 있으면 True를 반환한다.

    `neighbor`는 `cell`의 오른쪽이나 아래 셀이다.
    """
    row, col = cell
    half = EDGE_BAND_PX / 2
    if neighbor == (row, col + 1):
        x = (col + 1) * cell_w
        box = (x - half, row * cell_h + cell_h / 4, x + half, row * cell_h + cell_h * 3 / 4)
    else:
        y = (row + 1) * cell_h
        box = (col * cell_w + cell_w / 4, y - half, col * cell_w + cell_w * 3 / 4, y + half)
    return _distance_sq(_median(image, box), PART_OUTLINE_RGB) <= EDGE_MAX_DISTANCE_SQ


def _match_shape(cells: frozenset[Cell]) -> tuple[PartType, Rotation] | None:
    shape = normalize(cells)
    for part_type in PartType:
        for rotation in Rotation:
            if shape_of(part_type, rotation) == shape:
                return part_type, rotation
    return None


class _UnionFind:
    def __init__(self, items: dict[Cell, Color]) -> None:
        self._parent: dict[Cell, Cell] = {item: item for item in items}

    def find(self, item: Cell) -> Cell:
        while self._parent[item] != item:
            self._parent[item] = self._parent[self._parent[item]]
            item = self._parent[item]
        return item

    def union(self, a: Cell, b: Cell) -> None:
        self._parent[self.find(a)] = self.find(b)

    def groups(self) -> list[frozenset[Cell]]:
        by_root: dict[Cell, set[Cell]] = {}
        for item in self._parent:
            by_root.setdefault(self.find(item), set()).add(item)
        return [frozenset(cells) for cells in by_root.values()]
