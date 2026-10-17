"""Isaac Sim 테스트 공용 픽스처."""

import pytest
from pydantic import ValidationError

from pixel2place.config import SimSettings


@pytest.fixture
def sim_settings() -> SimSettings:
    """Isaac 설정. `ISAAC_SIM_PATH`, `python.sh`, SO-101 USD 중 하나가 없으면 skip한다."""
    try:
        settings = SimSettings()  # pyright: ignore[reportCallIssue]
    except ValidationError:
        pytest.skip("Sim test needs ISAAC_SIM_PATH.")
    if not (settings.isaac_sim_path / "python.sh").is_file():
        pytest.skip(f"Isaac python.sh is not in {settings.isaac_sim_path}.")
    if not settings.so101_usd_path.is_file():
        pytest.skip(f"SO-101 USD {settings.so101_usd_path} does not exist.")
    return settings
