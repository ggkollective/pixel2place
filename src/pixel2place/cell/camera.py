"""핀홀 카메라 모델(`camera.json`).

좌표 규약:
- 카메라 좌표는 OpenCV 규약이다. x는 이미지 오른쪽, y는 이미지 아래쪽, z는 보는 방향이다.
- `rotation`은 카메라 좌표의 벡터를 base_link 벡터로 바꾸는 3×3 행렬이다.
  열은 카메라 x, y, z축의 base_link 방향이다.
- `position`은 base_link에서 본 카메라 중심(m)이다.
- 화소 좌표 (u, v)는 화소 중심이 정수다. (0, 0)은 왼쪽 위 화소의 중심이다.
- 렌즈 왜곡은 없다.

Isaac 실행기(`sim/place_task.py`)는 pixel2place를 import하지 않는다. 그래서 이 모델이 형식을 정한다.
"""

import numpy as np
from numpy.typing import ArrayLike, NDArray
from pydantic import BaseModel, ConfigDict, Field, model_validator

Vector3 = tuple[float, float, float]
Matrix3 = tuple[Vector3, Vector3, Vector3]
Array = NDArray[np.float64]
# 회전 행렬의 직교성을 검사하는 허용 오차
ROTATION_TOLERANCE = 1e-6


class PinholeCamera(BaseModel):
    """이미지 크기, intrinsics, 자세."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    width: int = Field(gt=0)
    height: int = Field(gt=0)
    fx: float = Field(gt=0)
    fy: float = Field(gt=0)
    cx: float
    cy: float
    rotation: Matrix3
    position: Vector3

    @model_validator(mode="after")
    def _rotation_is_proper(self) -> "PinholeCamera":
        rotation = self.rotation_matrix
        orthogonal = np.allclose(rotation.T @ rotation, np.eye(3), atol=ROTATION_TOLERANCE)
        if not orthogonal or np.linalg.det(rotation) <= 0:
            raise ValueError("Camera rotation must be a proper rotation matrix.")
        return self

    @property
    def rotation_matrix(self) -> Array:
        return np.array(self.rotation, dtype=np.float64)

    @property
    def center(self) -> Array:
        return np.array(self.position, dtype=np.float64)

    @property
    def intrinsics(self) -> Array:
        return np.array([[self.fx, 0.0, self.cx], [0.0, self.fy, self.cy], [0.0, 0.0, 1.0]])

    def with_pose(self, rotation: ArrayLike, position: ArrayLike) -> "PinholeCamera":
        """intrinsics는 같고 자세만 다른 카메라."""
        rotation_rows = np.asarray(rotation, dtype=np.float64).tolist()
        return self.model_validate(
            {
                **self.model_dump(),
                "rotation": rotation_rows,
                "position": np.asarray(position, dtype=np.float64).tolist(),
            }
        )

    def project(self, points: ArrayLike) -> Array:
        """base_link 점 (N, 3)의 화소 좌표 (N, 2)."""
        local = (np.asarray(points, dtype=np.float64) - self.center) @ self.rotation_matrix
        return np.column_stack(
            [
                self.fx * local[:, 0] / local[:, 2] + self.cx,
                self.fy * local[:, 1] / local[:, 2] + self.cy,
            ]
        )

    def back_project(self, pixels: ArrayLike, *, z: float) -> Array:
        """화소 (N, 2)를 지나는 광선이 높이 `z`의 수평면과 만나는 점의 xy (N, 2)."""
        pixels = np.asarray(pixels, dtype=np.float64)
        normalized = np.column_stack(
            [
                (pixels[:, 0] - self.cx) / self.fx,
                (pixels[:, 1] - self.cy) / self.fy,
                np.ones(len(pixels)),
            ]
        )
        rays = normalized @ self.rotation_matrix.T
        center = self.center
        scale = (z - center[2]) / rays[:, 2]
        return center[:2] + rays[:, :2] * scale[:, np.newaxis]

    def depth_of(self, point: ArrayLike) -> float:
        """점의 카메라 z 좌표(m). 화소와 길이의 비를 어림할 때 쓴다."""
        local = (np.asarray(point, dtype=np.float64) - self.center) @ self.rotation_matrix
        return float(local[2])


def rotation_from_vector(vector: ArrayLike) -> Array:
    """회전 벡터(축 × 각도, rad)의 3×3 회전 행렬(Rodrigues)."""
    vector = np.asarray(vector, dtype=np.float64)
    angle = float(np.linalg.norm(vector))
    if angle == 0.0:
        return np.eye(3)
    x, y, z = vector / angle
    cross = np.array([[0.0, -z, y], [z, 0.0, -x], [-y, x, 0.0]])
    return np.eye(3) + np.sin(angle) * cross + (1 - np.cos(angle)) * (cross @ cross)


def rotation_angle(rotation: ArrayLike) -> float:
    """회전 행렬의 회전 각도(rad)."""
    trace = float(np.trace(np.asarray(rotation, dtype=np.float64)))
    return float(np.arccos(np.clip((trace - 1) / 2, -1.0, 1.0)))
