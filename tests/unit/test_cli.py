import pytest
from typer.testing import CliRunner

from pixel2place import cli
from pixel2place.benchmark import generate_layouts
from pixel2place.cell.config import CellConfig
from pixel2place.cell.task import Task
from pixel2place.cli import EXIT_INFEASIBLE, EXIT_USAGE, app
from pixel2place.generation.synthetic import SyntheticRenderer
from pixel2place.layout.schema import Layout, PartType
from pixel2place.reasoner.client import CosmosReasoner
from tests.factories import (
    make_layout,
    make_parser_response_json,
    make_part,
    make_planner_response_json,
)
from tests.fakes.fake_openai import FakeOpenAI

runner = CliRunner()


@pytest.mark.unit
class TestCheckCommand:
    """check 명령 테스트."""

    def test_success_exit_zero(self, valid_layout, tmp_path):
        """배치할 수 있는 레이아웃이면 exit 0이다."""
        path = tmp_path / "layout.json"
        path.write_text(valid_layout.model_dump_json())
        result = runner.invoke(app, ["check", str(path)])
        assert result.exit_code == 0, result.output
        assert "feasible" in result.output

    def test_fail_exit_one(self, tmp_path):
        """배치할 수 없는 레이아웃이면 위반을 출력하고 exit 1이다."""
        layout = make_layout(make_part("P1", PartType.RECT_TILE, origin=(7, 7)))
        path = tmp_path / "layout.json"
        path.write_text(layout.model_dump_json())
        result = runner.invoke(app, ["check", str(path)])
        assert result.exit_code == EXIT_INFEASIBLE, result.output
        assert "OUT_OF_BOUNDS P1" in result.output


@pytest.mark.unit
class TestComposeOrderCommand:
    """compose-order 명령 테스트."""

    def test_success_writes_only_instruction_to_stdout(self, tmp_path):
        """배치할 수 있는 주문이면 stdout에 지시문 한 줄만 쓰고 exit 0이다."""
        path = tmp_path / "order.json"
        path.write_text('{"parts": [{"type": "square_tile", "color": "blue", "row": 2, "col": 3}]}')
        result = runner.invoke(app, ["compose-order", str(path)])
        assert result.exit_code == 0, result.output
        assert result.stdout == (
            "On an 8 by 8 board, place a blue square tile at row 2 col 3. "
            "All rotations are 0 degrees. "
            "Each row and col is the position of the part. Rows and cols start at 0.\n"
        )

    def test_fail_exit_one_without_instruction(self, tmp_path):
        """배치할 수 없는 주문이면 stderr에 위반을 쓰고 지시문 없이 exit 1이다."""
        path = tmp_path / "order.json"
        path.write_text('{"parts": [{"type": "rect_tile", "color": "gray", "row": 7, "col": 7}]}')
        result = runner.invoke(app, ["compose-order", str(path)])
        assert result.exit_code == EXIT_INFEASIBLE, result.output
        assert result.stdout == "", f"stdout: {result.stdout}"
        assert "OUT_OF_BOUNDS P001" in result.stderr

    def test_kit_order_writes_instruction_without_positions(self, tmp_path):
        """키트 주문이면 stdout에 수량과 배치 방식의 지시문 한 줄만 쓰고 exit 0이다."""
        path = tmp_path / "order.json"
        path.write_text(
            '{"kind": "kit", "arrangement": "type_rows", "items": '
            '[{"type": "square_tile", "count": 2}]}'
        )
        result = runner.invoke(app, ["compose-order", str(path)])
        assert result.exit_code == 0, result.output
        assert result.stdout == (
            "On an 8 by 8 board, place two blue square tiles. All rotations are 0 degrees. "
            "Put parts of the same type in the same row.\n"
        )

    @pytest.mark.parametrize(
        "body",
        [
            "{",
            '{"parts": [{"type": "hex_tile"}]}',
            '{"kind": "kit", "arrangement": "type_rows", "items": '
            '[{"type": "T_tile", "count": 1}]}',
        ],
    )
    def test_fail_exit_two_with_invalid_file(self, tmp_path, body):
        """주문 파일의 형식이 틀리면 지시문 없이 exit 2다."""
        path = tmp_path / "order.json"
        path.write_text(body)
        result = runner.invoke(app, ["compose-order", str(path)])
        assert result.exit_code == EXIT_USAGE, result.output
        assert result.stdout == "", f"stdout: {result.stdout}"


@pytest.mark.unit
class TestRenderCommand:
    """render 명령 테스트."""

    def test_writes_png(self, valid_layout, tmp_path):
        """레이아웃을 PNG 파일로 쓴다."""
        layout_path = tmp_path / "layout.json"
        layout_path.write_text(valid_layout.model_dump_json())
        output = tmp_path / "board.png"
        result = runner.invoke(app, ["render", str(layout_path), "-o", str(output)])
        assert result.exit_code == 0, result.output
        assert output.read_bytes().startswith(b"\x89PNG")

    def test_makes_missing_parent_dir(self, valid_layout, tmp_path):
        """출력 경로의 부모 디렉터리가 없으면 그 디렉터리를 만든다."""
        layout_path = tmp_path / "layout.json"
        layout_path.write_text(valid_layout.model_dump_json())
        output = tmp_path / "new" / "dir" / "board.png"
        result = runner.invoke(app, ["render", str(layout_path), "-o", str(output)])
        assert result.exit_code == 0, result.output
        assert output.read_bytes().startswith(b"\x89PNG")


@pytest.mark.unit
class TestBenchParserCommand:
    """bench-판독기 명령 테스트."""

    def test_prints_summary(self, monkeypatch, tmp_path):
        """--판독기 cosmos는 Cosmos 판독기로 평가하고 요약을 출력한다."""
        layouts = generate_layouts(count=2, seed=0)
        fake = FakeOpenAI([make_parser_response_json(layout) for layout in layouts])
        monkeypatch.setattr(cli, "_reasoner", lambda: CosmosReasoner(client=fake, model="m"))
        args = ["bench-parser", "--parser", "cosmos", "--count", "2", "-o", str(tmp_path)]
        result = runner.invoke(app, args)
        assert result.exit_code == 0, result.output
        assert "mean_cell_accuracy=1.000" in result.output
        assert (tmp_path / "summary.json").is_file()

    def test_grid_parser_needs_no_settings(self, monkeypatch, tmp_path):
        """기본 grid 판독기는 Cosmos 설정 없이 동작한다."""
        monkeypatch.delenv("REASONER_BASE_URL", raising=False)
        monkeypatch.chdir(tmp_path)
        result = runner.invoke(app, ["bench-parser", "--count", "3", "-o", str(tmp_path / "out")])
        assert result.exit_code == 0, result.output
        assert "mean_cell_accuracy=1.000" in result.output


@pytest.mark.unit
class TestParseCommand:
    """parse 명령 테스트."""

    def test_grid_parser_reads_rendered_image(self, valid_layout, tmp_path):
        """렌더러 이미지를 grid 판독기로 읽어 layout.json을 쓴다."""
        image_path = tmp_path / "board.png"
        SyntheticRenderer().render(valid_layout).save(image_path)
        output = tmp_path / "layout.json"
        result = runner.invoke(app, ["parse", str(image_path), "-o", str(output)])
        assert result.exit_code == 0, result.output
        parsed = Layout.model_validate_json(output.read_text())
        assert {tuple(p.cells) for p in parsed.parts} == {
            tuple(p.cells) for p in valid_layout.parts
        }

    def test_makes_missing_parent_dir(self, valid_layout, tmp_path):
        """출력 경로의 부모 디렉터리가 없으면 그 디렉터리를 만든다."""
        image_path = tmp_path / "board.png"
        SyntheticRenderer().render(valid_layout).save(image_path)
        output = tmp_path / "new" / "dir" / "layout.json"
        result = runner.invoke(app, ["parse", str(image_path), "-o", str(output)])
        assert result.exit_code == 0, result.output
        assert len(Layout.model_validate_json(output.read_text()).parts) == len(valid_layout.parts)


@pytest.mark.unit
class TestPlanCommand:
    """plan 명령 테스트."""

    def test_writes_repaired_layout(self, monkeypatch, overlap_layout, tmp_path):
        """배치 계획 모델 위반을 고친 레이아웃과 수리 기록을 쓴다."""
        planner_json = make_planner_response_json(overlap_layout)
        fake = FakeOpenAI([planner_json, planner_json])
        monkeypatch.setattr(cli, "_reasoner", lambda: CosmosReasoner(client=fake, model="m"))
        output = tmp_path / "layout.json"
        result = runner.invoke(app, ["plan", "tray", "-o", str(output)])
        assert result.exit_code == 0, result.output
        layout = Layout.model_validate_json(output.read_text())
        assert layout.feasible is True
        assert layout.notes == ["MOVED: Moved part P002 from [0, 0] to [0, 1]."]

    def test_makes_missing_parent_dir(self, monkeypatch, valid_layout, tmp_path):
        """출력 경로의 부모 디렉터리가 없으면 그 디렉터리를 만든다."""
        fake = FakeOpenAI([make_planner_response_json(valid_layout)])
        monkeypatch.setattr(cli, "_reasoner", lambda: CosmosReasoner(client=fake, model="m"))
        output = tmp_path / "new" / "dir" / "layout.json"
        result = runner.invoke(app, ["plan", "tray", "-o", str(output)])
        assert result.exit_code == 0, result.output
        assert Layout.model_validate_json(output.read_text()).feasible is True


@pytest.mark.unit
class TestCompileTaskCommand:
    """compile-task 명령 테스트."""

    def test_writes_task(self, tmp_path):
        """레이아웃을 task.json으로 쓴다."""
        layout = make_layout(make_part("P001", PartType.SQUARE_TILE, origin=(3, 3)))
        layout_path = tmp_path / "layout.json"
        layout_path.write_text(layout.model_dump_json())
        output = tmp_path / "task.json"
        result = runner.invoke(app, ["compile-task", str(layout_path), "-o", str(output)])
        assert result.exit_code == 0, result.output
        assert "parts=1" in result.output
        assert Task.model_validate_json(output.read_text()).parts[0].part_id == "P001"

    def test_makes_missing_parent_dir(self, tmp_path):
        """출력 경로의 부모 디렉터리가 없으면 그 디렉터리를 만든다."""
        layout = make_layout(make_part("P001", PartType.SQUARE_TILE, origin=(3, 3)))
        layout_path = tmp_path / "layout.json"
        layout_path.write_text(layout.model_dump_json())
        output = tmp_path / "new" / "dir" / "task.json"
        result = runner.invoke(app, ["compile-task", str(layout_path), "-o", str(output)])
        assert result.exit_code == 0, result.output
        assert Task.model_validate_json(output.read_text()).parts[0].part_id == "P001"

    def test_reads_cell_config(self, tmp_path):
        """--cell 파일의 설정을 쓴다."""
        layout = make_layout(make_part("P001", PartType.SQUARE_TILE, origin=(3, 3)))
        layout_path = tmp_path / "layout.json"
        layout_path.write_text(layout.model_dump_json())
        cell_path = tmp_path / "cell.json"
        cell_path.write_text(CellConfig(part_height=0.012).model_dump_json())
        output = tmp_path / "task.json"
        args = ["compile-task", str(layout_path), "-o", str(output), "--cell", str(cell_path)]
        result = runner.invoke(app, args)
        assert result.exit_code == 0, result.output
        assert Task.model_validate_json(output.read_text()).cell.part_height == 0.012

    def test_infeasible_layout_exit_one(self, overlap_layout, tmp_path):
        """배치 규칙을 어긴 레이아웃이면 exit 1이다."""
        layout_path = tmp_path / "layout.json"
        layout_path.write_text(overlap_layout.model_dump_json())
        result = runner.invoke(
            app, ["compile-task", str(layout_path), "-o", str(tmp_path / "t.json")]
        )
        assert result.exit_code == EXIT_INFEASIBLE
        assert "OVERLAP" in result.output
