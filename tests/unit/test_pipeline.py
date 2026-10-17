import json

import pytest

from pixel2place.generation.synthetic import SyntheticRenderer
from pixel2place.layout.schema import Color, PartType
from pixel2place.parsing.grid import GridImageParser
from pixel2place.pipeline import (
    LAYOUT_FILE,
    NORMALIZED_IMAGE_FILE,
    PLAN_LAYOUT_FILE,
    RAW_IMAGE_FILE,
    RAW_RESPONSE_FILE,
    REPORT_FILE,
    compare_layouts,
    run_roundtrip,
)
from pixel2place.reasoner.client import CosmosReasoner, ReasonerOutputError
from tests.factories import (
    make_layout,
    make_part,
    make_planner_response_json,
)
from tests.fakes.fake_openai import FakeOpenAI


@pytest.mark.unit
class TestCompareLayouts:
    """레이아웃 비교 테스트."""

    def test_identical_layouts(self, valid_layout):
        """같은 레이아웃의 셀 정확도는 1.0이다."""
        assert compare_layouts(planned=valid_layout, parsed=valid_layout).cell_accuracy == 1.0

    def test_counts_color_and_occupancy_errors(self):
        """색이 다른 셀과 한쪽에만 있는 셀을 틀린 셀로 센다."""
        planned = make_layout(make_part("P1", PartType.RECT_TILE, origin=(0, 0)))
        parsed = make_layout(
            make_part("P1", PartType.SQUARE_TILE, origin=(0, 0)),
            make_part("P2", PartType.SQUARE_TILE, origin=(0, 1), color=Color.GRAY),
            make_part("P3", PartType.SQUARE_TILE, origin=(5, 5)),
        )
        comparison = compare_layouts(planned=planned, parsed=parsed)
        # 합집합 3셀 중 (0, 0)만 같다.
        assert comparison.cell_accuracy == pytest.approx(1 / 3)
        assert (comparison.planned_part_count, comparison.parsed_part_count) == (1, 3)


@pytest.mark.unit
class TestRunRoundtrip:
    """round-trip 처리 흐름 테스트."""

    def test_writes_all_outputs(self, valid_layout, tmp_path):
        """출력 파일을 모두 쓰고 report.json에 비교 결과와 판독기 이름을 기록한다."""
        fake = FakeOpenAI([make_planner_response_json(valid_layout)])
        result = run_roundtrip(
            "tray",
            out_dir=tmp_path / "out",
            reasoner=CosmosReasoner(client=fake, model="m"),
            parser=GridImageParser(),
            renderer=SyntheticRenderer(cell_px=16),
        )
        out = tmp_path / "out"
        for name in (RAW_IMAGE_FILE, NORMALIZED_IMAGE_FILE, PLAN_LAYOUT_FILE, LAYOUT_FILE):
            assert (out / name).is_file(), f"{name} 없음"
        report = json.loads((out / REPORT_FILE).read_text())
        assert report["comparison"]["cell_accuracy"] == 1.0
        assert report["parser"] == "GridImageParser"
        assert result.plan_feasibility.feasible and result.parse_feasibility.feasible
        assert json.loads((out / LAYOUT_FILE).read_text())["feasible"] is True

    def test_repairs_infeasible_plan(self, overlap_layout, tmp_path):
        """plan 위반은 고친 뒤 그린다. report에는 수리 전 위반과 수리 기록을 쓴다."""
        planner_json = make_planner_response_json(overlap_layout)
        fake = FakeOpenAI([planner_json, planner_json])
        result = run_roundtrip(
            "tray",
            out_dir=tmp_path,
            reasoner=CosmosReasoner(client=fake, model="m"),
            parser=GridImageParser(),
            renderer=SyntheticRenderer(cell_px=16),
        )
        assert not result.plan_feasibility.feasible
        assert [repair.part_id for repair in result.plan_repairs] == ["P002"]
        plan_layout = json.loads((tmp_path / PLAN_LAYOUT_FILE).read_text())
        assert plan_layout["feasible"] is True
        assert plan_layout["notes"] == ["MOVED: Moved part P002 from [0, 0] to [0, 1]."]
        report = json.loads((tmp_path / REPORT_FILE).read_text())
        assert report["plan_repairs"][0]["action"] == "MOVED"
        assert result.comparison.cell_accuracy == 1.0

    def test_writes_raw_responses_when_parse_fails(self, valid_layout, tmp_path):
        """Cosmos 판독기가 실패하면 원시 응답을 파일에 쓰고 예외를 다시 낸다."""
        fake = FakeOpenAI([make_planner_response_json(valid_layout), "bad one", "bad two"])
        reasoner = CosmosReasoner(client=fake, model="m")
        with pytest.raises(ReasonerOutputError):
            run_roundtrip(
                "tray",
                out_dir=tmp_path,
                reasoner=reasoner,
                parser=reasoner,
                renderer=SyntheticRenderer(cell_px=16),
            )
        raw = (tmp_path / RAW_RESPONSE_FILE.format(stage="parse")).read_text()
        assert "bad one" in raw and "bad two" in raw
        assert not (tmp_path / RAW_RESPONSE_FILE.format(stage="plan")).exists()
