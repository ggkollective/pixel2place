import json

import httpx2
import pytest
from PIL import Image
from typer.testing import CliRunner

from pixel2place import cli
from pixel2place.cli import EXIT_USAGE, app
from pixel2place.generation.cosmos_client import CosmosGenerator
from pixel2place.reasoner.client import CosmosReasoner
from tests.fakes.fake_generator import FAKE_MP4_BYTES, FakeGeneratorServer, image_response
from tests.fakes.fake_openai import FakeOpenAI
from tests.unit.test_expansion import IMAGE_PROMPT, VIDEO_PROMPT

runner = CliRunner()


def _no_reasoner() -> CosmosReasoner:
    raise AssertionError("--no-expand must not call the reasoner")


def _use_server(monkeypatch, server: FakeGeneratorServer) -> None:
    monkeypatch.setattr(cli, "_generator", server.generator)


@pytest.mark.unit
class TestGenerateImageCommand:
    """generate-image 명령 테스트."""

    def test_no_expand_writes_png(self, monkeypatch, tmp_path):
        """--no-expand는 지시문을 그대로 보내고 기본 생성 인자를 쓴다."""
        server = FakeGeneratorServer(image_response(Image.new("RGB", (16, 16), "blue")))
        _use_server(monkeypatch, server)
        monkeypatch.setattr(cli, "_reasoner", _no_reasoner)
        output = tmp_path / "tray.png"
        result = runner.invoke(app, ["generate-image", "a tray", "-o", str(output), "--no-expand"])
        assert result.exit_code == 0, result.output
        assert output.read_bytes().startswith(b"\x89PNG")
        body = json.loads(server.requests[0].content)
        assert body["prompt"] == "a tray"
        assert (body["size"], body["num_inference_steps"], body["guidance_scale"]) == (
            "640x640",
            50,
            6.0,
        )
        assert body["seed"] == 0
        assert not (tmp_path / "tray.png.prompt.json").exists()

    def test_expand_sends_structured_prompt_and_saves_it(self, monkeypatch, tmp_path):
        """확장한 지시문을 서버에 보내고 출력 파일 옆에 저장한다."""
        server = FakeGeneratorServer(image_response(Image.new("RGB", (16, 16))))
        _use_server(monkeypatch, server)
        fake = FakeOpenAI([json.dumps(IMAGE_PROMPT)])
        monkeypatch.setattr(cli, "_reasoner", lambda: CosmosReasoner(client=fake, model="m"))
        output = tmp_path / "tray.png"
        args = ["generate-image", "a tray", "-o", str(output), "--size", "256x128", "--seed", "7"]
        result = runner.invoke(app, args)
        assert result.exit_code == 0, result.output
        body = json.loads(server.requests[0].content)
        assert json.loads(body["prompt"]) == IMAGE_PROMPT
        assert (body["size"], body["seed"]) == ("256x128", 7)
        assert json.loads((tmp_path / "tray.png.prompt.json").read_text()) == IMAGE_PROMPT

    def test_makes_missing_parent_dir(self, monkeypatch, tmp_path):
        """출력 경로의 부모 디렉터리가 없으면 만든다. 확장한 지시문도 거기에 쓴다."""
        server = FakeGeneratorServer(image_response(Image.new("RGB", (16, 16))))
        _use_server(monkeypatch, server)
        fake = FakeOpenAI([json.dumps(IMAGE_PROMPT)])
        monkeypatch.setattr(cli, "_reasoner", lambda: CosmosReasoner(client=fake, model="m"))
        output = tmp_path / "new" / "dir" / "tray.png"
        result = runner.invoke(app, ["generate-image", "a tray", "-o", str(output)])
        assert result.exit_code == 0, result.output
        assert output.read_bytes().startswith(b"\x89PNG")
        assert json.loads(output.with_name("tray.png.prompt.json").read_text()) == IMAGE_PROMPT

    def test_bad_size_is_usage_error(self, monkeypatch, tmp_path):
        """--size 형식이 틀리면 exit 2다."""
        _use_server(monkeypatch, FakeGeneratorServer(image_response(Image.new("RGB", (1, 1)))))
        args = ["generate-image", "p", "-o", str(tmp_path / "x.png"), "--size", "640"]
        result = runner.invoke(app, args)
        assert result.exit_code == EXIT_USAGE
        assert "WIDTHxHEIGHT" in result.output

    def test_generator_reads_settings(self, monkeypatch, tmp_path):
        """환경 변수의 생성 서버 설정으로 클라이언트를 만든다."""
        monkeypatch.chdir(tmp_path)
        monkeypatch.setenv("GENERATOR_BASE_URL", "http://generator.test:8100/v1")
        monkeypatch.setenv("COSMOS_GENERATOR_MODEL", "m")
        assert isinstance(cli._generator(), CosmosGenerator)

    def test_missing_settings_exit_two(self, monkeypatch, tmp_path):
        """생성 서버 설정이 없으면 없는 변수 이름을 쓰고 exit 2다."""
        monkeypatch.chdir(tmp_path)
        monkeypatch.delenv("GENERATOR_BASE_URL", raising=False)
        monkeypatch.delenv("COSMOS_GENERATOR_MODEL", raising=False)
        result = runner.invoke(app, ["generate-image", "p", "-o", "x.png", "--no-expand"])
        assert result.exit_code == EXIT_USAGE
        assert "GENERATOR_BASE_URL, COSMOS_GENERATOR_MODEL" in result.output


@pytest.mark.unit
class TestPreviewVideoCommand:
    """preview-video 명령 테스트."""

    def test_expand_with_first_frame_and_write_mp4(self, monkeypatch, tmp_path):
        """첫 프레임으로 영상 지시문을 확장하고 MP4를 쓴다."""
        server = FakeGeneratorServer(httpx2.Response(200, content=FAKE_MP4_BYTES))
        _use_server(monkeypatch, server)
        fake = FakeOpenAI([json.dumps(VIDEO_PROMPT)])
        monkeypatch.setattr(cli, "_reasoner", lambda: CosmosReasoner(client=fake, model="m"))
        frame_path = tmp_path / "frame.png"
        Image.new("RGBA", (16, 16), "white").save(frame_path)
        output = tmp_path / "clip.mp4"
        args = ["preview-video", str(frame_path), "place a tile", "-o", str(output)]
        result = runner.invoke(app, args)
        assert result.exit_code == 0, result.output
        assert output.read_bytes() == FAKE_MP4_BYTES
        assert json.loads((tmp_path / "clip.mp4.prompt.json").read_text()) == VIDEO_PROMPT
        _, image_part = fake.calls[0]["messages"][1]["content"]
        assert image_part["image_url"]["url"].startswith("data:image/png;base64,")
        body = server.requests[0].content
        for name, value in [("num_frames", "93"), ("fps", "24"), ("num_inference_steps", "35")]:
            assert f'name="{name}"\r\n\r\n{value}\r\n'.encode() in body, name

    def test_no_expand_sends_prompt(self, monkeypatch, tmp_path):
        """--no-expand는 VLM 없이 지시문을 그대로 보낸다."""
        server = FakeGeneratorServer(httpx2.Response(200, content=FAKE_MP4_BYTES))
        _use_server(monkeypatch, server)
        monkeypatch.setattr(cli, "_reasoner", _no_reasoner)
        frame_path = tmp_path / "frame.png"
        Image.new("RGB", (16, 16)).save(frame_path)
        output = tmp_path / "clip.mp4"
        args = ["preview-video", str(frame_path), "move", "-o", str(output), "--no-expand"]
        result = runner.invoke(app, args)
        assert result.exit_code == 0, result.output
        assert b'name="prompt"\r\n\r\nmove\r\n' in server.requests[0].content

    def test_makes_missing_parent_dir(self, monkeypatch, tmp_path):
        """출력 경로의 부모 디렉터리가 없으면 그 디렉터리를 만든다."""
        server = FakeGeneratorServer(httpx2.Response(200, content=FAKE_MP4_BYTES))
        _use_server(monkeypatch, server)
        monkeypatch.setattr(cli, "_reasoner", _no_reasoner)
        frame_path = tmp_path / "frame.png"
        Image.new("RGB", (16, 16)).save(frame_path)
        output = tmp_path / "new" / "dir" / "clip.mp4"
        args = ["preview-video", str(frame_path), "move", "-o", str(output), "--no-expand"]
        result = runner.invoke(app, args)
        assert result.exit_code == 0, result.output
        assert output.read_bytes() == FAKE_MP4_BYTES
