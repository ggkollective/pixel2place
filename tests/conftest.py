"""공용 픽스처와 마커 설정."""

import pytest
from pydantic import ValidationError

from pixel2place.config import Settings
from pixel2place.layout.schema import Color, Layout, PartType, Rotation
from tests.factories import make_layout, make_part


@pytest.fixture
def valid_layout() -> Layout:
    """네 종류의 부품이 겹치지 않는 레이아웃."""
    return make_layout(
        make_part("P001", PartType.L_TILE, origin=(0, 0)),
        make_part("P002", PartType.T_TILE, origin=(0, 3), color=Color.GRAY),
        make_part("P003", PartType.RECT_TILE, origin=(4, 4), rotation=Rotation.DEG_90),
        make_part("P004", PartType.SQUARE_TILE, origin=(7, 7), color=Color.WHITE),
    )


@pytest.fixture
def overlap_layout() -> Layout:
    """두 부품이 셀 (0, 0)을 함께 쓰는 레이아웃."""
    return make_layout(
        make_part("P001", PartType.SQUARE_TILE, origin=(0, 0)),
        make_part("P002", PartType.RECT_TILE, origin=(0, 0)),
    )


def pytest_collection_modifyitems(config: pytest.Config, items: list[pytest.Item]) -> None:
    """환경 변수와 `.env`로 Settings를 만들 수 없으면 e2e 테스트를 skip한다."""
    try:
        Settings()  # pyright: ignore[reportCallIssue]
    except ValidationError as exc:
        missing = ", ".join(str(error["loc"][0]).upper() for error in exc.errors())
        skip_e2e = pytest.mark.skip(reason=f"E2E test needs {missing}.")
    else:
        return
    for item in items:
        if "e2e" in item.keywords:
            item.add_marker(skip_e2e)
