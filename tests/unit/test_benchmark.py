import json

import pytest

from pixel2place.benchmark import (
    PART_COUNT_BANDS,
    SUMMARY_FILE,
    generate_layouts,
    run_parser_benchmark,
)
from pixel2place.feasibility.checker import check_layout
from pixel2place.generation.synthetic import SyntheticRenderer
from pixel2place.parsing.grid import GridImageParser
from pixel2place.reasoner.client import CosmosReasoner
from tests.factories import make_parser_response_json
from tests.fakes.fake_openai import FakeOpenAI


@pytest.mark.unit
class TestGenerateLayouts:
    """벤치마크 레이아웃 생성 테스트."""

    def test_same_seed_same_layouts(self):
        """같은 seed는 같은 레이아웃을 만든다."""
        assert generate_layouts(count=6, seed=3) == generate_layouts(count=6, seed=3)
        assert generate_layouts(count=6, seed=3) != generate_layouts(count=6, seed=4)

    def test_layouts_are_feasible_and_follow_bands(self):
        """모든 레이아웃이 배치 규칙을 지키고 부품 수가 구간 안에 있다."""
        layouts = generate_layouts(count=24, seed=0)
        assert len(layouts) == 24
        for index, layout in enumerate(layouts):
            low, high = PART_COUNT_BANDS[index % len(PART_COUNT_BANDS)]
            assert check_layout(layout).feasible, index
            assert low <= len(layout.parts) <= high, (index, len(layout.parts))


@pytest.mark.unit
class TestRunParserBenchmark:
    """판독기 벤치마크 실행 테스트."""

    def _run(self, responses, tmp_path, *, count=3):
        layouts = generate_layouts(count=count, seed=0)
        fake = FakeOpenAI(responses(layouts))
        return run_parser_benchmark(
            layouts,
            seed=0,
            parser=CosmosReasoner(client=fake, model="m"),
            renderer=SyntheticRenderer(cell_px=16),
            out_dir=tmp_path,
        )

    def test_correct_parser_scores_one(self, tmp_path):
        """판독기가 정답을 읽으면 평균 정확도와 일치율이 1.0이고 파일을 모두 쓴다."""
        summary = self._run(
            lambda ls: [make_parser_response_json(layout) for layout in ls], tmp_path
        )
        assert summary.mean_cell_accuracy == 1.0
        assert summary.exact_match_rate == 1.0
        assert summary.part_count_match_rate == 1.0
        assert [band.count for band in summary.bands] == [1, 1, 1]
        for name in ("000/truth.json", "000/raw_image.png", "000/layout.json"):
            assert (tmp_path / name).is_file(), name
        assert json.loads((tmp_path / SUMMARY_FILE).read_text())["count"] == 3

    def test_grid_parser_scores_one(self, tmp_path):
        """픽셀 판독기는 생성한 레이아웃을 모두 정확히 읽는다."""
        summary = run_parser_benchmark(
            generate_layouts(count=6, seed=0),
            seed=0,
            parser=GridImageParser(),
            renderer=SyntheticRenderer(cell_px=16),
            out_dir=tmp_path,
        )
        assert summary.parser == "GridImageParser"
        assert summary.exact_match_rate == 1.0

    def test_parser_failure_counts_as_zero(self, tmp_path):
        """판독기가 실패한 표본은 정확도 0.0으로 세고 원시 응답을 쓴다."""
        summary = self._run(lambda ls: ["bad", "bad"], tmp_path, count=1)
        assert summary.failures == 1
        assert summary.mean_cell_accuracy == 0.0
        assert summary.samples[0].error is not None
        assert (tmp_path / "000" / "parse_raw_response.txt").is_file()
