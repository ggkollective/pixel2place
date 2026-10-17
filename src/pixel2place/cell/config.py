"""SO-101 셀의 배치 설정.

좌표는 SO-101 `base_link` 기준이다. z축이 위를 향하고 단위는 m다.
기본값은 SO-101 top-down 작업 범위 측정값으로 정했다([so101-model] 문서).

보드 좌표 규약:
- `board_origin`은 셀 (0, 0)의 왼쪽 위 모서리다.
- row가 늘면 `row_direction`으로, col이 늘면 `col_direction`으로 간다.
- 기본값은 row가 로봇에서 멀어지는 방향(+x), col이 +y 방향이다.
  위에서 보면 이미지 아래쪽이 +x, 오른쪽이 +y다. 그래서 보드는 레이아웃 이미지를 돌린 모습이다.
  col을 -y로 두면 보드가 이미지의 거울상이 된다.
- 방향 벡터는 단위 길이이고 서로 직교한다.

공급대 규약:
- 공급대는 보드와 따로 놓인 고정물이다. 부품은 공급대의 row·col 방향으로 회전 0°로 놓인다.
- 공급대 방향을 주지 않으면 보드 방향과 같다.
- 보드를 보정해서 방향이 바뀌어도 공급대 방향은 그대로다.

수평 규약 (옆 접근 파지):
- 내려갈 때는 TCP를 고정 jaw 반대쪽으로 `pick_clearance`만큼 옮긴 점(pick TCP)을 파지점에 맞춘다.
  그래서 고정 jaw가 부품 윗모서리에 얹히지 않는다.
- 파지 높이에서 TCP를 파지점에 맞춘다. 고정 jaw 면이 부품 면에 닿는다. 그 다음 그리퍼를 닫는다.

높이 규약 (TCP는 jaw 끝 높이의 jaw 가운데다):
- 파지: jaw 끝이 부품 윗면에서 `grasp_depth` 아래에 있다.
- 놓기: jaw 끝이 부품 윗면 높이에서 `release_clearance` 위에 있다. 이웃 부품의 윗면보다 높다.
  그래서 jaw가 이웃 부품에 닿지 않는다. 부품은 `grasp_depth + release_clearance`만큼 떨어진다.

위치결정 규약:
- 보드 셀 중심마다 핀이 있다. 부품 바닥의 셀마다 입구를 모따기한 구멍이 있다.
- 부품이 떨어지면 입구 모따기가 핀 끝에 닿고 부품을 셀 중심으로 민다.
- 부품의 바깥 아래 모서리를 모따기한다.
  잡고 내려가는 부품이 이웃 윗모서리에 닿으면 모따기 면이 미끄러져 들어간다.
"""

from typing import Annotated

from pydantic import BaseModel, ConfigDict, Field, model_validator

Vector2 = tuple[float, float]
Length = Annotated[float, Field(gt=0)]
# 방향 벡터의 길이와 직교를 검사하는 허용 오차
AXIS_TOLERANCE = 1e-6
# 로봇 기준 표식의 중심(m). top-down 캡처에서 팔, 그림자, 보드, 공급대를 피하는 자리다.
# 보드 판과 공급대 판에서 40 mm 이상 떨어져 있다.
DEFAULT_MARKERS: tuple[Vector2, ...] = (
    (0.07, -0.18),
    (0.20, -0.18),
    (0.34, -0.18),
    (0.34, 0.0),
    (0.34, 0.26),
    (0.25, 0.26),
)


def check_axes(row_direction: Vector2, col_direction: Vector2, *, name: str) -> None:
    """두 방향 벡터가 단위 길이이고 직교하는지 확인한다. 아니면 `ValueError`를 낸다."""
    for direction in (row_direction, col_direction):
        if abs(direction[0] ** 2 + direction[1] ** 2 - 1.0) > AXIS_TOLERANCE:
            raise ValueError(f"{name} directions must have unit length.")
    dot = row_direction[0] * col_direction[0] + row_direction[1] * col_direction[1]
    if abs(dot) > AXIS_TOLERANCE:
        raise ValueError(f"{name} row and col directions must be orthogonal.")


def handedness(row_direction: Vector2, col_direction: Vector2) -> float:
    """row에서 col로 가는 회전의 부호. 위에서 보아 반시계면 1, 거울상이면 -1이다."""
    cross = row_direction[0] * col_direction[1] - row_direction[1] * col_direction[0]
    return 1.0 if cross > 0 else -1.0


class Locator(BaseModel):
    """보드 위치결정 핀과 부품 바닥 구멍의 치수. 모따기는 45°다.

    놓은 뒤 받아들이는 기하 범위는 입구 반폭에서 핀 끝 평면의 반지름을 뺀 값이다.
    기본값은 3.5 mm(4.7 − 1.2)다.
    """

    model_config = ConfigDict(extra="forbid", frozen=True)

    pin_diameter: Length = 0.004
    """보드 핀 지름. 보드 셀 중심마다 핀 1개가 있다."""
    pin_height: Length = 0.005
    """보드 윗면에서 핀 끝까지의 높이."""
    pin_chamfer: Length = 0.0008
    """핀 끝 모따기."""
    hole_width: Length = 0.0044
    """부품 바닥 구멍의 폭. 부품의 셀마다 구멍 1개가 있다."""
    hole_depth: Length = 0.006
    """부품 바닥에서 구멍 천장까지의 깊이."""
    hole_chamfer: Length = 0.0025
    """구멍 입구 모따기."""

    @model_validator(mode="after")
    def _pin_fits_hole(self) -> "Locator":
        if self.hole_width <= self.pin_diameter:
            raise ValueError("hole_width must be more than pin_diameter.")
        if self.pin_height >= self.hole_depth:
            raise ValueError("pin_height must be less than hole_depth.")
        if self.hole_chamfer >= self.hole_depth:
            raise ValueError("hole_chamfer must be less than hole_depth.")
        if self.pin_chamfer >= min(self.pin_diameter / 2, self.pin_height):
            raise ValueError("pin_chamfer must be less than the pin radius and pin_height.")
        return self

    @property
    def mouth_half_width(self) -> float:
        """구멍 입구의 반폭."""
        return self.hole_width / 2 + self.hole_chamfer


class CellConfig(BaseModel):
    """보드, 공급대, 부품 블록의 위치와 크기."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    board_origin: Vector2 = (0.115, -0.08)
    row_direction: Vector2 = (1.0, 0.0)
    col_direction: Vector2 = (0.0, 1.0)
    feeder_center: Vector2 = (0.14, 0.16)
    """공급대에 놓인 부품의 셀 중심 평균 위치. 부품은 회전 0°로 놓인다."""
    feeder_row_direction: Vector2 | None = None
    """공급대 부품의 row 방향. None이면 `row_direction`이다. `feeder_col_direction`과 함께 준다."""
    feeder_col_direction: Vector2 | None = None
    """공급대 부품의 col 방향. None이면 `col_direction`이다."""
    table_z: float = 0.0
    part_height: Length = 0.015
    part_inset: Annotated[float, Field(ge=0)] = 0.0015
    """부품 블록의 각 면을 셀 경계에서 안쪽으로 줄이는 거리. 이웃 블록 사이에 틈을 둔다.

    틈이 좁으면 놓을 때 끌린 이웃이 틈을 1 mm 아래로 줄인다.
    그러면 다음 부품이 이웃 윗모서리에 걸린다.
    틈이 넓으면 파지 폭이 준다.
    """
    grasp_depth: Length = 0.006
    """부품 윗면에서 jaw 끝까지의 깊이."""
    pick_clearance: Annotated[float, Field(ge=0)] = 0.002
    """파지 높이로 내려갈 때 고정 jaw 안쪽 면과 부품 면 사이의 거리."""
    release_clearance: Length = 0.001
    """놓을 때 jaw 끝과 부품 윗면 높이의 간격."""
    approach_clearance: Length = 0.04
    """파지 높이 위로 접근·후퇴하는 높이."""
    locator: Locator | None = Locator()
    """보드 핀과 부품 구멍. None이면 보드가 평판이고 부품에 구멍이 없다."""
    part_chamfer: Annotated[float, Field(ge=0)] = 0.002
    """부품 바깥 아래 모서리 모따기. 같은 부품의 셀 사이 면은 깎지 않는다. 0이면 모따기가 없다."""
    markers: tuple[Vector2, ...] = DEFAULT_MARKERS
    """로봇 기준 표식의 중심(m). 캘리브레이션이 이 표식으로 카메라 자세를 푼다.

    표식은 밝은 판 위의 어두운 원이다. 원의 윗면이 `table_z`다. 판의 한 변은 원 지름의 2배다.
    """
    marker_diameter: Length = 0.02
    """표식 원의 지름."""

    @model_validator(mode="after")
    def _heights_are_ordered(self) -> "CellConfig":
        if self.grasp_depth >= self.part_height:
            raise ValueError("grasp_depth must be less than part_height.")
        if self.approach_z <= self.place_z:
            raise ValueError(
                "approach_clearance must be more than grasp_depth + release_clearance."
            )
        if self.part_chamfer >= self.part_height - self.grasp_depth:
            raise ValueError("part_chamfer must be less than part_height - grasp_depth.")
        if self.locator is not None:
            self._check_locator(self.locator)
        return self

    @model_validator(mode="after")
    def _axes_are_orthonormal(self) -> "CellConfig":
        check_axes(self.row_direction, self.col_direction, name="Board")
        if (self.feeder_row_direction is None) != (self.feeder_col_direction is None):
            raise ValueError("Give both feeder_row_direction and feeder_col_direction or neither.")
        feeder_row, feeder_col = self.feeder_directions
        check_axes(feeder_row, feeder_col, name="Feeder")
        # 손잡이 방향이 다르면 공급대의 부품이 보드에서 거울상이 된다.
        if handedness(feeder_row, feeder_col) != handedness(self.row_direction, self.col_direction):
            raise ValueError("Feeder directions must have the same handedness as the board.")
        return self

    @property
    def feeder_directions(self) -> tuple[Vector2, Vector2]:
        """공급대 부품의 (row 방향, col 방향)."""
        if self.feeder_row_direction is None or self.feeder_col_direction is None:
            return self.row_direction, self.col_direction
        return self.feeder_row_direction, self.feeder_col_direction

    def _check_locator(self, locator: Locator) -> None:
        # 잡은 부품의 바닥이 핀 끝보다 높아야 한다. 그래야 놓기 전에 핀에 닿지 않는다.
        if locator.pin_height >= self.grasp_depth + self.release_clearance:
            raise ValueError(
                "locator.pin_height must be less than grasp_depth + release_clearance."
            )
        if locator.hole_depth >= self.part_height:
            raise ValueError("locator.hole_depth must be less than part_height.")
        # 실행기는 구멍 깊이까지의 아래 부분에 모따기를 만든다.
        if self.part_chamfer > locator.hole_depth:
            raise ValueError("part_chamfer must not be more than locator.hole_depth.")

    @property
    def grasp_z(self) -> float:
        """파지할 때 TCP 높이."""
        return self.table_z + self.part_height - self.grasp_depth

    @property
    def place_z(self) -> float:
        """놓을 때 TCP 높이."""
        return self.table_z + self.part_height + self.release_clearance

    @property
    def approach_z(self) -> float:
        """접근·후퇴할 때 TCP 높이. 공급대와 보드에서 같다."""
        return self.grasp_z + self.approach_clearance
