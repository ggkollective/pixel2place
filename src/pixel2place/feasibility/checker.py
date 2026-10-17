"""레이아웃 배치 규칙 검사기."""

from collections import Counter
from enum import StrEnum

from pydantic import BaseModel, ConfigDict

from pixel2place.layout.parts import matches_shape
from pixel2place.layout.schema import Cell, Layout


class IssueCode(StrEnum):
    """배치 규칙 위반 종류."""

    EMPTY_LAYOUT = "EMPTY_LAYOUT"
    DUPLICATE_PART_ID = "DUPLICATE_PART_ID"
    COLOR_NOT_IN_PALETTE = "COLOR_NOT_IN_PALETTE"
    SHAPE_MISMATCH = "SHAPE_MISMATCH"
    OUT_OF_BOUNDS = "OUT_OF_BOUNDS"
    OVERLAP = "OVERLAP"


class Issue(BaseModel):
    """위반 1건. `part_id`가 None이면 레이아웃 전체의 위반이다."""

    model_config = ConfigDict(frozen=True)

    code: IssueCode
    part_id: str | None
    message: str


class FeasibilityReport(BaseModel):
    """검사 결과. 위반이 0건이면 배치할 수 있다."""

    model_config = ConfigDict(frozen=True)

    issues: list[Issue]

    @property
    def feasible(self) -> bool:
        return not self.issues


def check_layout(layout: Layout) -> FeasibilityReport:
    """레이아웃의 모든 배치 규칙을 검사한다. 입력은 바꾸지 않는다."""
    if not layout.parts:
        return FeasibilityReport(
            issues=[
                Issue(code=IssueCode.EMPTY_LAYOUT, part_id=None, message="Layout has no parts.")
            ]
        )

    issues: list[Issue] = []
    id_counts = Counter(part.part_id for part in layout.parts)
    for part_id, count in id_counts.items():
        if count > 1:
            issues.append(
                Issue(
                    code=IssueCode.DUPLICATE_PART_ID,
                    part_id=part_id,
                    message=f"Part ID {part_id} occurs {count} times.",
                )
            )

    rows, cols = layout.board.rows, layout.board.cols
    owner_by_cell: dict[Cell, str] = {}
    for part in layout.parts:
        if part.color not in layout.palette:
            issues.append(
                Issue(
                    code=IssueCode.COLOR_NOT_IN_PALETTE,
                    part_id=part.part_id,
                    message=f"Color {part.color} is not in the palette.",
                )
            )
        if not matches_shape(part):
            issues.append(
                Issue(
                    code=IssueCode.SHAPE_MISMATCH,
                    part_id=part.part_id,
                    message=(
                        f"Cells do not match type {part.type} "
                        f"at rotation {int(part.rotation_deg)} degrees."
                    ),
                )
            )
        for cell in part.cells:
            row, col = cell
            if not (0 <= row < rows and 0 <= col < cols):
                issues.append(
                    Issue(
                        code=IssueCode.OUT_OF_BOUNDS,
                        part_id=part.part_id,
                        message=f"Cell {list(cell)} is outside the {rows}x{cols} board.",
                    )
                )
                continue
            owner = owner_by_cell.setdefault(cell, part.part_id)
            if owner != part.part_id:
                issues.append(
                    Issue(
                        code=IssueCode.OVERLAP,
                        part_id=part.part_id,
                        message=f"Cell {list(cell)} is also in part {owner}.",
                    )
                )
    return FeasibilityReport(issues=issues)


def apply_report(layout: Layout, report: FeasibilityReport) -> Layout:
    """검사 결과를 `feasible`과 `notes`에 기록한 새 레이아웃을 반환한다."""
    notes = [f"{issue.code}: {issue.message}" for issue in report.issues]
    return layout.model_copy(update={"feasible": report.feasible, "notes": notes})
