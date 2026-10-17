import numpy as np
import pytest
from pydantic import ValidationError

from pixel2place.cell.camera import PinholeCamera, rotation_angle, rotation_from_vector
from tests.fakes.topdown_renderer import nominal_camera


@pytest.mark.unit
class TestPinholeCamera:
    """핀홀 카메라 모델 테스트."""

    def test_nominal_center_projects_to_principal_point(self):
        """카메라 바로 아래 점은 주점에 투영된다."""
        camera = nominal_camera()
        pixel = camera.project([[0.19, 0.04, 0.0]])[0]
        np.testing.assert_allclose(pixel, (camera.cx, camera.cy))

    def test_image_axes(self):
        """이미지 오른쪽은 +y, 아래쪽은 +x다."""
        camera = nominal_camera()
        center = camera.project([[0.19, 0.04, 0.0]])[0]
        right = camera.project([[0.19, 0.05, 0.0]])[0]
        down = camera.project([[0.20, 0.04, 0.0]])[0]
        assert right[0] > center[0] and right[1] == pytest.approx(center[1])
        assert down[1] > center[1] and down[0] == pytest.approx(center[0])

    def test_back_project_inverts_project(self):
        """기울어진 카메라에서도 투영한 화소를 같은 높이로 되돌리면 원래 점이다."""
        camera = nominal_camera()
        tilted = camera.with_pose(
            rotation_from_vector([0.01, -0.02, 0.03]) @ camera.rotation_matrix,
            camera.center + [0.005, -0.003, 0.002],
        )
        points = np.array([[0.1, -0.1, 0.005], [0.3, 0.2, 0.005], [0.2, 0.0, 0.005]])
        pixels = tilted.project(points)
        np.testing.assert_allclose(tilted.back_project(pixels, z=0.005), points[:, :2], atol=1e-12)

    def test_rejects_non_rotation(self):
        """회전 행렬이 아니면 검증 오류다."""
        data = nominal_camera().model_dump()
        data["rotation"] = [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, -1.0]]
        with pytest.raises(ValidationError, match="proper rotation"):
            PinholeCamera.model_validate(data)


@pytest.mark.unit
class TestRotation:
    """회전 벡터 테스트."""

    def test_zero_vector_is_identity(self):
        np.testing.assert_array_equal(rotation_from_vector([0.0, 0.0, 0.0]), np.eye(3))

    def test_turns_about_axis(self):
        """z축 회전 벡터는 x축을 반시계로 돌린다. 회전 각도는 벡터 길이다."""
        rotation = rotation_from_vector([0.0, 0.0, np.pi / 2])
        np.testing.assert_allclose(rotation @ [1.0, 0.0, 0.0], [0.0, 1.0, 0.0], atol=1e-12)
        assert rotation_angle(rotation_from_vector([0.01, 0.02, -0.02])) == pytest.approx(0.03)
