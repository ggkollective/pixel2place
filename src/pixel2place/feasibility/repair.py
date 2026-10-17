"""배치 규칙 위반 자동 수리.

배치 계획 모델 출력에 남은 `OVERLAP`, `OUT_OF_BOUNDS`를 코드가 고친다.
부품의 종류, 색, 회전은 바꾸지 않는다. 위치만 바꾸거나 부품을 지운다.
"""

from enum import StrEnum

from pydantic import BaseModel, ConfigDict

from pixel2place.layout.parts import place
from pixel2place.layout.schema import Board, Cell, Layout, Part


class RepairAction(StrEnum):
    """수리 1건의 종류."""

    MOVED = "MOVED"
    REMOVED = "REMOVED"


class Repair(BaseModel):
    """부품 1개의 수리 기록. `to_origin`은 MOVED일 때만 있다."""

    model_config = ConfigDict(frozen=True)

    action: RepairAction
    part_id: str
    from_origin: Cell
    to_origin: Cell | None

    @property
    def message(self) -> str:
        if self.to_origin is None:
            return f"Removed part {self.part_id} at {list(self.from_origin)}. No free position."
        return f"Moved part {self.part_id} from {list(self.from_origin)} to {list(self.to_origin)}."


class RepairResult(BaseModel):
    """수리한 레이아웃과 수리 기록. 수리가 필요 없으면 `repairs`가 비어 있다."""

    model_config = ConfigDict(frozen=True)

    layout: Layout
    repairs: list[Repair]


def repair_layout(layout: Layout) -> RepairResult:
    """겹치거나 보드 밖에 있는 부품을 옮기거나 지운다.

    단계:
    1. 부품을 목록 순서대로 검사한다. 모든 셀이 보드 안에 있고 앞 부품과 겹치지 않으면
       그 자리를 확정한다.
    2. 확정하지 못한 부품을 목록 순서대로 옮긴다. 같은 종류와 회전으로 놓을 수 있는
       가장 가까운 빈 위치를 쓴다. 거리는 원래 위치와의 맨해튼 거리다.
       거리가 같으면 row, col이 작은 위치를 쓴다.
    3. 놓을 수 있는 위치가 없으면 부품을 지운다.

    1단계에서 정상 부품의 자리를 먼저 확정한다. 그래서 위반한 부품만 움직인다.
    부품 순서는 바꾸지 않는다.
    `SHAPE_MISMATCH` 같은 다른 위반은 고치지 않는다. 입력 레이아웃은 바꾸지 않는다.
    """
    occupied: set[Cell] = set()
    fixed: dict[int, Part] = {}
    for index, part in enumerate(layout.parts):
        if _fits(part.cells, board=layout.board, occupied=occupied):
            fixed[index] = part
            occupied.update(part.cells)

    repairs: list[Repair] = []
    for index, part in enumerate(layout.parts):
        if index in fixed:
            continue
        origin = _origin(part.cells)
        new_origin = _nearest_origin(part, origin=origin, board=layout.board, occupied=occupied)
        repairs.append(
            Repair(
                action=RepairAction.REMOVED if new_origin is None else RepairAction.MOVED,
                part_id=part.part_id,
                from_origin=origin,
                to_origin=new_origin,
            )
        )
        if new_origin is None:
            continue
        cells = place(part.type, part.rotation_deg, origin=new_origin)
        fixed[index] = part.model_copy(update={"cells": cells})
        occupied.update(cells)

    if not repairs:
        return RepairResult(layout=layout, repairs=[])
    kept = [fixed[index] for index in sorted(fixed)]
    # 지운 부품의 색은 팔레트에서 뺀다. 부품이 모두 없어지면 원래 팔레트를 둔다.
    palette = sorted({part.color for part in kept}) or layout.palette
    repaired = layout.model_copy(update={"parts": kept, "palette": palette})
    return RepairResult(layout=repaired, repairs=repairs)


def _origin(cells: list[Cell]) -> Cell:
    """셀을 둘러싸는 가장 작은 사각형의 왼쪽 위 셀."""
    return (min(row for row, _ in cells), min(col for _, col in cells))


def _fits(cells: list[Cell], *, board: Board, occupied: set[Cell]) -> bool:
    return all(
        0 <= row < board.rows and 0 <= col < board.cols and (row, col) not in occupied
        for row, col in cells
    )


def _nearest_origin(part: Part, *, origin: Cell, board: Board, occupied: set[Cell]) -> Cell | None:
    origin_row, origin_col = origin
    candidates = sorted(
        ((row, col) for row in range(board.rows) for col in range(board.cols)),
        key=lambda cell: (abs(cell[0] - origin_row) + abs(cell[1] - origin_col), cell),
    )
    for candidate in candidates:
        cells = place(part.type, part.rotation_deg, origin=candidate)
        if _fits(cells, board=board, occupied=occupied):
            return candidate
    return None
