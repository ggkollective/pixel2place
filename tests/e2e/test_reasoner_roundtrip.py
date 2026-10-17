"""실제 Cosmos Reasoner 서버를 호출한다. `.env`나 환경 변수에 서버 주소와 모델 ID가 필요하다."""

import pytest

from pixel2place.config import Settings
from pixel2place.generation.synthetic import SyntheticRenderer
from pixel2place.pipeline import run_roundtrip
from pixel2place.reasoner.client import CosmosReasoner

E2E_PROMPT = (
    "A top-down industrial tray layout with blue, gray, and white components "
    "arranged in a symmetric pattern."
)


@pytest.mark.e2e
class TestReasonerRoundtrip:
    """Cosmos Reasoner round-trip e2e 테스트."""

    def test_roundtrip_returns_valid_layouts(self, tmp_path):
        """배치 계획 모델과 판독기가 모두 스키마에 맞는 레이아웃을 반환한다."""
        reasoner = CosmosReasoner.from_settings(Settings())  # pyright: ignore[reportCallIssue]
        result = run_roundtrip(
            E2E_PROMPT,
            out_dir=tmp_path,
            reasoner=reasoner,
            parser=reasoner,
            renderer=SyntheticRenderer(),
        )
        # 정확도 기준은 아직 정하지 않았다.
        # 배치 계획 모델 출력의 배치 규칙과 정확도 범위만 검사한다.
        assert result.plan_feasibility.issues == [], result.plan_feasibility.issues
        assert 0.0 <= result.comparison.cell_accuracy <= 1.0
        assert result.comparison.planned_part_count > 0
