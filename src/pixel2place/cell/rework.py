"""재작업 실행의 대상과 장면.

재작업은 실행기의 두 번째 실행이다. 작업은 첫 실행과 같은 `task.json`이다.
장면이 첫 실행에서 놓인 부품의 자세를 준다. 실행기는 그 부품을 건너뛰고 대상 부품만 놓는다.
그래서 재작업 뒤의 `result.json`은 작업의 모든 부품을 갖는다. 판정과 판독은 작업 전체를 본다.
"""

from collections.abc import Collection

from pydantic import ValidationError

from pixel2place.cell.image_check import ImageCheckReport, PartStatus
from pixel2place.cell.result import SimResult
from pixel2place.cell.scene import PreplacedPart, SceneFaults, SimScene
from pixel2place.cell.task import Task


class ReworkError(ValueError):
    """이 결과로는 재작업 실행의 장면을 만들지 못한다."""


def reworkable(report: ImageCheckReport) -> tuple[int, ...]:
    """카메라 판정에서 다시 놓을 수 있는 부품의 `sequence`.

    부품이 `mismatch`이고 모든 칸의 핀이 보이면 그 칸은 비어 있다. 그 부품을 다시 놓는다.
    계획에 없는 점유 칸이 있으면 보드에 다른 물체가 있다. 그때는 다시 놓지 않는다.
    """
    if report.unexpected_cells:
        return ()
    return tuple(
        part.sequence for part in report.parts if part.status is PartStatus.MISMATCH and part.empty
    )


def rework_scene(
    scene: SimScene | None, *, task: Task, result: SimResult, sequences: Collection[int]
) -> SimScene:
    """첫 실행의 결과에서 재작업 실행의 장면을 만든다.

    - `sequences`의 부품은 공급대에서 다시 놓는다. 다른 부품은 첫 실행의 최종 자세에 둔다.
    - 넘어진 부품과 기운 부품도 그 자세에 둔다. 복원은 부품의 상태를 좋게 만들지 않는다.
    - 공급 누락은 뺀다. 재작업에서는 공급대가 부품을 준다.
    - 다른 색 결함, 보드, 카메라와 관절 센서 오차는 그대로 둔다.

    아래 경우에는 `ReworkError`를 낸다.
    - 결과의 부품이 작업과 다르다.
    - 결과의 보드 자세가 장면의 보드 자세와 다르다. 부품의 자세는 그 보드에서 잰 값이다.
    - 자세의 quaternion이 없거나 값이 유한하지 않다.
    """
    targets = set(sequences)
    known = {part.sequence for part in task.parts}
    if not targets or not targets <= known:
        raise ReworkError(f"Rework parts {sorted(targets)} are not parts of the task.")
    if [part.part_id for part in result.parts] != [part.part_id for part in task.parts]:
        raise ReworkError("Result parts are different from the task parts.")
    base = scene or SimScene()
    if result.board != base.board:
        raise ReworkError("Result board is different from the scene board.")
    preplaced = []
    for part, pose in zip(task.parts, result.parts, strict=True):
        if part.sequence in targets:
            continue
        if pose.orientation is None:
            raise ReworkError(f"Result has no orientation for part {part.part_id}.")
        try:
            placed = PreplacedPart(
                sequence=part.sequence, position=pose.position, orientation=pose.orientation
            )
        except ValidationError as exc:
            raise ReworkError(f"Result pose of part {part.part_id} is not usable.") from exc
        preplaced.append(placed)
    faults = None
    if base.faults is not None and base.faults.wrong_color:
        faults = SceneFaults(wrong_color=base.faults.wrong_color)
    return base.model_copy(update={"faults": faults, "preplaced": tuple(preplaced) or None})
