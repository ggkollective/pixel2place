"""Isaac 실행기가 쓰는 `result.json`의 형식.

실행기(`sim/place_task.py`)는 pixel2place를 import하지 않는다. 그래서 이 모델이 형식을 정하고,
실행기는 같은 키로 JSON을 쓴다.
"""

from pydantic import BaseModel, ConfigDict

from pixel2place.cell.scene import BoardPlacement, Quaternion

RESULT_VERSION = "0.4"


class _ResultModel(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)


class Pose(_ResultModel):
    """부품 1개의 자세."""

    position: tuple[float, float, float]
    """블록 중심(셀 중심 평균, 높이 가운데)의 base_link 좌표(m)."""
    yaw: float
    """실제 보드의 u축에서 잰 부품 회전(rad). `PartTask.target_yaw`와 같은 기준이다."""
    tilt: float
    """부품 z축과 수직 사이의 각도(rad)."""
    fell: bool
    """부품이 작업대 아래로 떨어졌다."""
    orientation: Quaternion | None = None
    """base_link 좌표의 quaternion (w, x, y, z). 재작업 실행이 이 값으로 부품을 복원한다.

    None이면 실행기가 이 값을 쓰지 않은 결과다. 그 결과로는 재작업 장면을 만들지 못한다.
    """


class PartPose(Pose):
    """실행이 끝난 뒤 부품 1개의 자세와 놓은 직후 자세."""

    part_id: str
    placed: Pose
    """부품을 놓고 팔이 다음 부품의 접근 자세로 간 뒤의 자세.

    최종 자세와 비교하면 뒤 부품이 이 부품을 밀었는지 안다.
    """


class SimResult(_ResultModel):
    """`result.json` 전체. 부품은 task의 순서와 같다."""

    result_version: str
    task_version: str
    sim_time_s: float
    max_arm_error_rad: float
    """waypoint에 도착했을 때 팔 관절 목표와 실제 값의 최대 차이(rad)."""
    board: BoardPlacement | None
    """실행기가 만든 보드의 자세(`scene.json`의 `board`). None이면 task의 셀 설정과 같다."""
    parts: list[PartPose]
