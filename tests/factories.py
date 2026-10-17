"""테스트용 레이아웃·셀 결과·카메라 판정 팩토리."""

from pixel2place.cell.image_check import (
    DEFAULT_LIMITS,
    ImageCheckReport,
    PartImageCheck,
    PartStatus,
)
from pixel2place.cell.result import RESULT_VERSION, PartPose, Pose, SimResult
from pixel2place.cell.task import Task
from pixel2place.layout.parts import place
from pixel2place.layout.schema import Color, Layout, Part, PartType, Rotation
from pixel2place.reasoner.response import (
    BBOX_SCALE,
    ParserLayout,
    ParserPart,
    PlannerLayout,
    PlannerPart,
)


def make_part(
    part_id: str,
    part_type: PartType,
    *,
    origin: tuple[int, int],
    color: Color = Color.BLUE,
    rotation: Rotation = Rotation.DEG_0,
) -> Part:
    """형태가 올바른 부품을 `origin`에 만든다."""
    return Part(
        part_id=part_id,
        type=part_type,
        color=color,
        cells=place(part_type, rotation, origin=origin),
        rotation_deg=rotation,
    )


def make_layout(*parts: Part) -> Layout:
    """부품이 쓰는 색으로 팔레트를 채운 8x8 레이아웃을 만든다."""
    palette = sorted({part.color for part in parts}) or [Color.BLUE]
    return Layout(palette=palette, parts=list(parts))


def make_planner_response_json(layout: Layout) -> str:
    """레이아웃과 같은 부품을 쓰는 배치 계획 모델 응답 JSON을 만든다."""
    parts = [
        PlannerPart(
            type=part.type,
            color=part.color,
            row=min(row for row, _ in part.cells),
            col=min(col for _, col in part.cells),
            rotation_deg=part.rotation_deg,
        )
        for part in layout.parts
    ]
    return PlannerLayout(parts=parts).model_dump_json()


def make_parser_response_json(layout: Layout) -> str:
    """레이아웃과 같은 부품을 쓰는 판독기 응답 JSON을 만든다. bbox는 부품 셀을 정확히 둘러싼다."""
    rows, cols = layout.board.rows, layout.board.cols
    parts = []
    for part in layout.parts:
        part_rows = [row for row, _ in part.cells]
        part_cols = [col for _, col in part.cells]
        bbox = [
            round(min(part_cols) * BBOX_SCALE / cols),
            round(min(part_rows) * BBOX_SCALE / rows),
            round((max(part_cols) + 1) * BBOX_SCALE / cols),
            round((max(part_rows) + 1) * BBOX_SCALE / rows),
        ]
        parts.append(
            ParserPart(
                bbox_2d=bbox, type=part.type, color=part.color, rotation_deg=part.rotation_deg
            )
        )
    return ParserLayout(parts=parts).model_dump_json()


def make_sim_result(task: Task) -> SimResult:
    """모든 부품이 목표 자세에 놓인 실행 결과를 만든다. fake Isaac 실행기의 `ok` 모드와 같다.

    놓은 직후 자세도 목표 자세다.
    """
    parts = []
    for part in task.parts:
        pose = Pose(
            position=(*part.target_center, task.cell.table_z + task.cell.part_height / 2),
            yaw=part.target_yaw,
            tilt=0.0,
            fell=False,
        )
        parts.append(PartPose(part_id=part.part_id, placed=pose, **pose.model_dump()))
    return SimResult(
        result_version=RESULT_VERSION,
        task_version=task.task_version,
        sim_time_s=1.0,
        max_arm_error_rad=0.0,
        board=None,
        parts=parts,
    )


def make_part_image_check(sequence: int, *, status: PartStatus, empty: bool) -> PartImageCheck:
    """부품 1개의 판독. ID는 `P` 뒤에 3자리의 순서다. 칸은 (순서, 순서)의 1칸이다."""
    return PartImageCheck(
        sequence=sequence,
        part_id=f"P{sequence:03d}",
        type=PartType.SQUARE_TILE,
        color=Color.BLUE,
        cells=[(sequence, sequence)],
        coverage=1.0 if status is PartStatus.MATCH else 0.0,
        shadow_coverage=0.0 if status is PartStatus.MISMATCH else 1.0,
        pins_hidden=not empty,
        empty=empty,
        status=status,
    )


def make_image_check_report(
    *parts: PartImageCheck, unexpected: tuple[tuple[int, int], ...] = ()
) -> ImageCheckReport:
    """부품의 상태에서 수와 통과 여부를 계산한 카메라 판정의 보고서."""
    mismatch = sum(part.status is PartStatus.MISMATCH for part in parts)
    return ImageCheckReport(
        image_size=(640, 480),
        limits=DEFAULT_LIMITS,
        parts=list(parts),
        unexpected_cells=list(unexpected),
        match_count=sum(part.status is PartStatus.MATCH for part in parts),
        shadowed_count=sum(part.status is PartStatus.SHADOWED for part in parts),
        mismatch_count=mismatch,
        passed=mismatch == 0 and not unexpected,
    )
