"""실행 1건에 주는 지그 오차와 부품의 결함, 그리고 그 장면.

Owns:
- 지그 오차의 범위와 계약.
- 실행의 입력으로 받는 부품 결함의 계약.
- 지그 오차와 부품의 결함을 실행의 장면으로 바꾸는 변환.

Does not own:
- 보드 자세의 계산과 장면 파일의 형식(`pixel2place.cell.scene`).
- 실행과 그 기록(`pixel2place.agent.cell`, `pixel2place.agent.shift`, `pixel2place.demo.jobs`).
"""

import math
from typing import Annotated, Self

from pydantic import Field, model_validator

from pixel2place.agent.contracts import Contract
from pixel2place.cell.config import CellConfig
from pixel2place.cell.scene import (
    BoardOffset,
    SceneFaults,
    SimScene,
    WrongColor,
    displace_board,
)
from pixel2place.cell.task import MM_PER_M
from pixel2place.generation.synthetic import COLOR_RGB
from pixel2place.layout.schema import Board, Color

# 보드 오차 벤치마크가 검증한 범위다. 이 범위 밖의 보정과 배치는 측정하지 않았다.
FIXTURE_X_LIMIT_MM = 3.0
FIXTURE_Y_LIMIT_MM = 5.0
FIXTURE_YAW_LIMIT_DEG = 1.5


class FixtureOffset(Contract):
    """지그 위 보드의 놓기 오차. 보드 중심을 축으로 돌리고 옮긴다."""

    x_mm: Annotated[float, Field(ge=-FIXTURE_X_LIMIT_MM, le=FIXTURE_X_LIMIT_MM)] = 0.0
    y_mm: Annotated[float, Field(ge=-FIXTURE_Y_LIMIT_MM, le=FIXTURE_Y_LIMIT_MM)] = 0.0
    yaw_deg: Annotated[float, Field(ge=-FIXTURE_YAW_LIMIT_DEG, le=FIXTURE_YAW_LIMIT_DEG)] = 0.0
    """위에서 보아 반시계 회전."""

    @property
    def is_nominal(self) -> bool:
        return self.x_mm == 0 and self.y_mm == 0 and self.yaw_deg == 0


class WrongColorFault(Contract):
    """다른 색으로 공급되는 부품 1개. 색의 값은 팔레트가 정한다."""

    sequence: Annotated[int, Field(ge=1)]
    color: Color


class PartFaults(Contract):
    """실행에 넣는 부품의 결함. 부품은 배치 순서로 지정한다.

    장면 파일의 `faults`와 뜻이 같다. 색은 이름만 받는다.
    """

    missing: tuple[Annotated[int, Field(ge=1)], ...] = ()
    """공급대에 오지 않는 부품."""
    wrong_color: tuple[WrongColorFault, ...] = ()
    """다른 색으로 공급되는 부품."""

    @model_validator(mode="after")
    def _is_a_scene_fault(self) -> Self:
        # 부품 1개에 결함 2개를 준 입력은 장면의 계약이 거부한다.
        self.scene_faults()
        return self

    @property
    def sequences(self) -> tuple[int, ...]:
        """결함이 있는 부품의 배치 순서."""
        return (*self.missing, *(item.sequence for item in self.wrong_color))

    def scene_faults(self) -> SceneFaults:
        """장면 파일의 결함. 색의 값은 팔레트에서 채운다."""
        return SceneFaults(
            missing=self.missing,
            wrong_color=tuple(
                WrongColor(
                    sequence=item.sequence, color=item.color, color_rgb=COLOR_RGB[item.color]
                )
                for item in self.wrong_color
            ),
        )


def fixture_scene(
    cell: CellConfig, fixture: FixtureOffset, *, faults: PartFaults | None = None
) -> SimScene:
    """지그 오차와 부품의 결함을 실행의 장면으로 바꾼다. 둘 다 없으면 공칭 장면이다."""
    scene_faults = None if faults is None else faults.scene_faults()
    if fixture.is_nominal:
        return SimScene(faults=scene_faults)
    offset = BoardOffset(
        x=fixture.x_mm / MM_PER_M,
        y=fixture.y_mm / MM_PER_M,
        yaw=math.radians(fixture.yaw_deg),
    )
    return SimScene(board=displace_board(cell, Board(), offset=offset), faults=scene_faults)
