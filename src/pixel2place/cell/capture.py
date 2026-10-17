"""보드 캘리브레이션 캡처 작업(`capture.json`).

실행기는 부품 없이 셀을 만들고 팔을 고정 자세에 둔다. 그 다음 top-down 이미지 1장을 찍는다.
팔 자세는 레이아웃과 무관하다. 그래서 표식 자리를 이 자세의 이미지로 정할 수 있다.

실행기(`sim/place_task.py`)는 pixel2place를 import하지 않는다. 그래서 이 모델이 형식을 정한다.
`board`, `cell`, `board_yaw`, `feeder_yaw`의 뜻은 `Task`의 같은 이름 필드와 같다.
"""

from pydantic import BaseModel, ConfigDict, Field

from pixel2place.cell.config import CellConfig
from pixel2place.cell.task import (
    MM_PER_M,
    OPEN_GAP_MARGIN,
    U_AXIS,
    BoardFrame,
    TaskCompileError,
)
from pixel2place.layout.schema import Board
from pixel2place.robot.so101 import ARM_JOINTS, grasp_tcp, gripper_angle_for_gap
from pixel2place.robot.so101_ik import TopDownTarget, UnreachableError, solve_top_down

CAPTURE_VERSION = "0.1"
# 캡처 작업 파일 이름. 실행기 출력과 같은 디렉터리에 쓴다.
CAPTURE_FILE = "capture.json"


class CaptureTask(BaseModel):
    """`capture.json` 전체."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    capture_version: str = CAPTURE_VERSION
    board: Board
    cell: CellConfig
    board_yaw: float
    feeder_yaw: float
    joints: list[float] = Field(min_length=len(ARM_JOINTS), max_length=len(ARM_JOINTS))
    """캡처할 때 팔 관절 값(rad). 순서는 `pixel2place.robot.so101.JOINT_NAMES`다."""
    gripper: float
    """캡처할 때 그리퍼 관절 값(rad). 부품을 잡기 전의 열린 각도다."""


def compile_capture(cell: CellConfig, board: Board) -> CaptureTask:
    """캡처 작업을 만든다.

    TCP는 공급대 중심 위의 접근 높이에 있다. jaw는 공급대 u축 방향으로 열려 있다.
    부품 작업의 `pre_pick`과 거의 같은 자세다. 이 자세에서는 팔이 보드를 가리지 않는다.
    IK 해가 없으면 `TaskCompileError`를 낸다.
    """
    cell_size = board.cell_size_mm / MM_PER_M
    frame = BoardFrame(cell=cell, cell_size=cell_size)
    grip_width = cell_size - 2 * cell.part_inset
    target = TopDownTarget(
        x=cell.feeder_center[0],
        y=cell.feeder_center[1],
        z=cell.approach_z,
        yaw=frame.feeder_yaw(U_AXIS),
        tcp=grasp_tcp(grip_width),
    )
    try:
        joints = solve_top_down(target)
    except UnreachableError as exc:
        raise TaskCompileError(f"Capture pose: {exc}") from exc
    return CaptureTask(
        board=board,
        cell=cell,
        board_yaw=frame.yaw(U_AXIS),
        feeder_yaw=frame.feeder_yaw(U_AXIS),
        joints=[float(value) for value in joints],
        gripper=gripper_angle_for_gap(grip_width + OPEN_GAP_MARGIN),
    )
