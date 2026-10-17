from pathlib import Path

import numpy as np
import pytest
from PIL import Image
from pydantic import ValidationError
from typer.testing import CliRunner

from pixel2place.cell.calibration import read_report
from pixel2place.cell.camera import PinholeCamera
from pixel2place.cell.config import CellConfig
from pixel2place.cell.image_check import (
    OVERLAY_RGB,
    SHADOW_RGB,
    SUNLIT_RGB,
    UNEXPECTED_RGB,
    ImageCheckError,
    ImageCheckLimits,
    ImageCheckReport,
    PartStatus,
    check_image,
    draw_overlay,
    planned_cells,
)
from pixel2place.cell.scene import (
    BoardOffset,
    BoardPlacement,
    CameraOffset,
    SimScene,
    displace_board,
    placed_cell,
    to_placed_board,
)
from pixel2place.cell.task import BoardFrame, PartTask, Task, compile_task
from pixel2place.cli import EXIT_INFEASIBLE, EXIT_USAGE, app
from pixel2place.layout.schema import Board, Color, PartType, Rotation
from pixel2place.pipeline import write_json
from tests.factories import make_layout, make_part
from tests.fakes.topdown_renderer import (
    TOPDOWN_EYE,
    TOPDOWN_ROTATION,
    PartFace,
    ShadowBand,
    nominal_camera,
    render_topdown,
    shaken,
)

CELL = CellConfig()
BOARD = Board()
NOMINAL = nominal_camera()
# 거울상 보드. col이 -y 방향이다. 보드가 차지하는 자리는 기본 보드와 같다.
MIRRORED = CELL.model_copy(update={"board_origin": (0.115, 0.08), "col_direction": (0.0, -1.0)})
OFFSET = BoardOffset(x=0.003, y=-0.005, yaw=np.radians(1.5))
SHAKE = CameraOffset(position=(0.01, -0.008, 0.006), rotation=tuple(np.radians([1.0, -0.8, 0.6])))
# 색 4개, 종류 4개. L과 T는 회전했다. 칸 (6, 1)은 비어 있다.
LAYOUT = make_layout(
    make_part("P001", PartType.SQUARE_TILE, origin=(1, 1), color=Color.BLUE),
    make_part(
        "P002", PartType.RECT_TILE, origin=(1, 4), color=Color.GRAY, rotation=Rotation.DEG_90
    ),
    make_part("P003", PartType.L_TILE, origin=(3, 1), color=Color.WHITE, rotation=Rotation.DEG_90),
    make_part(
        "P004", PartType.T_TILE, origin=(5, 4), color=Color.ORANGE, rotation=Rotation.DEG_180
    ),
)
EMPTY_CELL = (6, 1)
# 칸 (0, 0)~(1, 2)를 덮는 그림자 띠. 띠의 변은 핀에서 5 mm 이상 떨어져 있다.
SHADOW = ShadowBand(low=(0.10, -0.10), high=(0.16, -0.015))
SHADOWED_PART = "P001"
# 저장한 Isaac 실행 3건. 통과, 실패, 팔 그림자 안의 부품이다.
DATA_DIR = Path(__file__).parents[1] / "data" / "image_check"
STORED_CASES = ["pass", "fail", "shadow"]
# 저장한 보고서와 다시 잰 비율의 허용 차이
STORED_RATIO_TOLERANCE = 0.01

runner = CliRunner()


def _face(
    task: Task,
    part: PartTask,
    *,
    placement: BoardPlacement | None = None,
    shift: tuple[float, float] = (0.0, 0.0),
    color: Color | None = None,
) -> PartFace:
    """부품 1개의 윗면. 기본은 계획한 자리와 계획한 색이다.

    `placement`는 실제 보드 자세다. `shift`는 base_link 방향의 이동(m)이다.
    칸은 `block_cells`로 그린다. 판독기의 `planned_cells`를 쓰지 않는다.
    """
    center, board_yaw = np.array(part.target_center), task.board_yaw
    if placement is not None:
        center = to_placed_board(part.target_center, cell=task.cell, placement=placement)
        board_yaw = placement.board_yaw
    drawn = part.color if color is None else color
    return PartFace(
        center=(float(center[0] + shift[0]), float(center[1] + shift[1])),
        yaw=board_yaw + part.target_yaw,
        block_cells=tuple(part.block_cells),
        rgb=SUNLIT_RGB[drawn],
        shadow_rgb=SHADOW_RGB[drawn],
    )


def _faces(task: Task, *, placement: BoardPlacement | None = None) -> dict[str, PartFace]:
    return {part.part_id: _face(task, part, placement=placement) for part in task.parts}


def _part(task: Task, part_id: str) -> PartTask:
    return next(part for part in task.parts if part.part_id == part_id)


def _statuses(report: ImageCheckReport) -> dict[str, PartStatus]:
    return {part.part_id: part.status for part in report.parts}


def _colors(image: Image.Image) -> set[tuple[int, ...]]:
    """이미지에 있는 색."""
    pixels = np.asarray(image).reshape(-1, 3)
    return {tuple(color) for color in np.unique(pixels, axis=0).tolist()}


def _check_stored(case: str) -> ImageCheckReport:
    """저장한 실행 1건을 판독한다.

    `fail`은 보정 없는 실행이다. 공칭 카메라와 `scene.json`의 실제 보드 자세를 쓴다.
    나머지는 캘리브레이션 보고서의 카메라와 `task.cell`의 보드를 쓴다.
    """
    folder = DATA_DIR / case
    stored_task = Task.model_validate_json((folder / "task.json").read_text(encoding="utf-8"))
    with Image.open(folder / "topdown_final.png") as opened:
        shot = opened.convert("RGB")
    if case == "fail":
        camera = PinholeCamera.model_validate_json(
            (folder / "camera.json").read_text(encoding="utf-8")
        )
        scene = SimScene.model_validate_json((folder / "scene.json").read_text(encoding="utf-8"))
        return check_image(stored_task, shot, camera, board=scene.board)
    return check_image(stored_task, shot, read_report(folder / "cell.report.json").camera)


def _stored_args(case: str) -> list[str]:
    """저장한 실행 1건의 `image-check` 인자."""
    folder = DATA_DIR / case
    args = ["image-check", str(folder / "task.json"), str(folder / "topdown_final.png")]
    if case == "fail":
        return [
            *args,
            "--camera",
            str(folder / "camera.json"),
            "--scene",
            str(folder / "scene.json"),
        ]
    return [*args, "--calibration", str(folder / "cell.report.json")]


@pytest.fixture(scope="module")
def task() -> Task:
    return compile_task(LAYOUT, CELL)


@pytest.fixture(scope="module")
def image(task) -> Image.Image:
    """모든 부품이 계획한 자리에 있는 이미지."""
    return render_topdown(NOMINAL, cell=CELL, board=BOARD, faces=list(_faces(task).values()))


@pytest.fixture(scope="module")
def report(task, image) -> ImageCheckReport:
    return check_image(task, image, NOMINAL)


@pytest.mark.unit
class TestPlannedCells:
    """작업에서 부품의 보드 칸을 다시 구하는 테스트."""

    @pytest.mark.parametrize("cell", [CELL, MIRRORED], ids=["default", "mirrored"])
    def test_cells_equal_layout_cells(self, cell):
        """회전한 L과 T의 칸은 레이아웃의 칸과 같다. 거울상 보드에서도 같다."""
        compiled = compile_task(LAYOUT, cell)
        layout_cells = {part.part_id: sorted(part.cells) for part in LAYOUT.parts}
        planned = dict(
            zip((part.part_id for part in compiled.parts), planned_cells(compiled), strict=True)
        )
        assert planned == layout_cells

    def test_fail_with_target_off_the_grid(self, task):
        """목표 위치가 칸 중심의 평균이 아니면 ImageCheckError를 낸다."""
        part = task.parts[0]
        x, y = part.target_center
        moved = part.model_copy(update={"target_center": (x + 0.004, y)})
        with pytest.raises(ImageCheckError, match="not on the board grid"):
            planned_cells(task.model_copy(update={"parts": [moved]}))

    def test_fail_with_cell_outside_the_board(self, task):
        """목표 위치가 보드 밖의 칸이면 ImageCheckError를 낸다."""
        part = task.parts[0]
        x, y = part.target_center
        board_width = BOARD.cols * BOARD.cell_size_mm / 1000
        moved = part.model_copy(update={"target_center": (x, y + board_width)})
        with pytest.raises(ImageCheckError, match="outside the board"):
            planned_cells(task.model_copy(update={"parts": [moved]}))


@pytest.mark.unit
class TestCheckImage:
    """합성 top-down 이미지의 카메라 판정 테스트."""

    def test_all_parts_in_place_match(self, task, report):
        """모든 부품이 계획한 자리에 있으면 모두 match이고 판정은 통과다."""
        assert set(_statuses(report).values()) == {PartStatus.MATCH}
        assert all(part.pins_hidden and part.coverage == 1.0 for part in report.parts)
        assert report.unexpected_cells == []
        assert (report.match_count, report.shadowed_count, report.mismatch_count) == (4, 0, 0)
        assert report.passed
        assert [part.sequence for part in report.parts] == [part.sequence for part in task.parts]

    def test_match_with_camera_error_and_calibrated_board(self):
        """카메라와 보드가 공칭 자세가 아니어도 추정 카메라와 보정한 셀 설정으로 모두 match다."""
        placement = displace_board(CELL, BOARD, offset=OFFSET)
        calibrated = compile_task(LAYOUT, placed_cell(CELL, placement=placement))
        truth = shaken(NOMINAL, SHAKE)
        shot = render_topdown(
            truth, cell=calibrated.cell, board=BOARD, faces=list(_faces(calibrated).values())
        )
        result = check_image(calibrated, shot, truth)
        assert set(_statuses(result).values()) == {PartStatus.MATCH}
        assert result.passed

    def test_actual_board_pose_replaces_task_board(self, task):
        """실제 보드 자세를 주면 그 보드에 투영한다. 주지 않으면 옮긴 보드의 부품은 mismatch다."""
        placement = displace_board(CELL, BOARD, offset=OFFSET)
        faces = list(_faces(task, placement=placement).values())
        shot = render_topdown(NOMINAL, cell=CELL, board=BOARD, placement=placement, faces=faces)
        assert check_image(task, shot, NOMINAL, board=placement).passed
        assert PartStatus.MISMATCH in _statuses(check_image(task, shot, NOMINAL)).values()

    def test_match_on_mirrored_board(self):
        """거울상 보드에서도 블록 칸으로 그린 부품이 계획한 칸에 있다."""
        mirrored = compile_task(LAYOUT, MIRRORED)
        shot = render_topdown(
            NOMINAL, cell=MIRRORED, board=BOARD, faces=list(_faces(mirrored).values())
        )
        result = check_image(mirrored, shot, NOMINAL)
        assert set(_statuses(result).values()) == {PartStatus.MATCH}
        assert result.unexpected_cells == []

    def test_moved_part_is_mismatch(self, task):
        """부품 1개가 5 mm 옮겨져 있으면 그 부품만 mismatch다."""
        faces = _faces(task)
        faces["P001"] = _face(task, _part(task, "P001"), shift=(0.0, 0.005))
        shot = render_topdown(NOMINAL, cell=CELL, board=BOARD, faces=list(faces.values()))
        result = check_image(task, shot, NOMINAL)
        statuses = _statuses(result)
        assert statuses.pop("P001") == PartStatus.MISMATCH
        assert set(statuses.values()) == {PartStatus.MATCH}
        assert not result.passed

    def test_coverage_limit_is_applied(self, task):
        """2 mm 옮긴 부품은 기본 기준에서 mismatch이고 더 낮은 기준에서 match다.

        윗면의 정사각형에서 화소 1줄이 보드 색이다. 덮인 비율은 약 0.947이다.
        """
        faces = _faces(task)
        faces["P001"] = _face(task, _part(task, "P001"), shift=(0.0, 0.002))
        shot = render_topdown(NOMINAL, cell=CELL, board=BOARD, faces=list(faces.values()))
        assert _statuses(check_image(task, shot, NOMINAL))["P001"] == PartStatus.MISMATCH
        loose = ImageCheckLimits(coverage=0.9)
        result = check_image(task, shot, NOMINAL, limits=loose)
        assert _statuses(result)["P001"] == PartStatus.MATCH
        assert result.limits == loose

    def test_missing_part_is_mismatch(self, task):
        """부품 1개가 없으면 그 칸의 핀이 보이고 그 부품은 mismatch다."""
        faces = _faces(task)
        del faces["P003"]
        shot = render_topdown(NOMINAL, cell=CELL, board=BOARD, faces=list(faces.values()))
        result = check_image(task, shot, NOMINAL)
        missing = next(part for part in result.parts if part.part_id == "P003")
        assert missing.status == PartStatus.MISMATCH
        assert not missing.pins_hidden
        assert missing.coverage == 0.0
        assert result.mismatch_count == 1

    def test_only_missing_part_is_empty(self, task):
        """없는 부품만 빈 칸으로 읽는다. 5 mm 옮겨진 부품은 칸을 덮고 있다."""
        faces = _faces(task)
        del faces["P003"]
        faces["P001"] = _face(task, _part(task, "P001"), shift=(0.0, 0.005))
        shot = render_topdown(NOMINAL, cell=CELL, board=BOARD, faces=list(faces.values()))
        result = check_image(task, shot, NOMINAL)
        assert {part.part_id for part in result.parts if part.empty} == {"P003"}
        assert _statuses(result)["P001"] == PartStatus.MISMATCH

    def test_wrong_color_is_mismatch(self, task):
        """파랑 자리의 주황 부품과 주황 자리의 파랑 부품은 mismatch다. 핀은 가려져 있다."""
        faces = _faces(task)
        faces["P001"] = _face(task, _part(task, "P001"), color=Color.ORANGE)
        faces["P004"] = _face(task, _part(task, "P004"), color=Color.BLUE)
        shot = render_topdown(NOMINAL, cell=CELL, board=BOARD, faces=list(faces.values()))
        result = check_image(task, shot, NOMINAL)
        statuses = _statuses(result)
        assert statuses["P001"] == statuses["P004"] == PartStatus.MISMATCH
        assert statuses["P002"] == statuses["P003"] == PartStatus.MATCH
        assert all(part.pins_hidden for part in result.parts)

    def test_white_part_on_gray_target_is_mismatch(self, task):
        """회색 자리의 흰색 부품은 mismatch다. 흰색은 회색의 그림자 선분 밖에 있다."""
        faces = _faces(task)
        faces["P002"] = _face(task, _part(task, "P002"), color=Color.WHITE)
        shot = render_topdown(NOMINAL, cell=CELL, board=BOARD, faces=list(faces.values()))
        assert _statuses(check_image(task, shot, NOMINAL))["P002"] == PartStatus.MISMATCH

    def test_gray_part_on_white_target_reads_as_shadowed(self, task):
        """흰색 자리의 회색 부품은 shadowed로 읽는다. 판독의 한계다.

        태양광 아래의 회색은 흰색의 그림자 선분 위에 있다. 색만으로 두 경우를 나누지 못한다.
        """
        faces = _faces(task)
        faces["P003"] = _face(task, _part(task, "P003"), color=Color.GRAY)
        shot = render_topdown(NOMINAL, cell=CELL, board=BOARD, faces=list(faces.values()))
        result = check_image(task, shot, NOMINAL)
        assert _statuses(result)["P003"] == PartStatus.SHADOWED
        assert result.passed

    @pytest.mark.parametrize("planned", list(Color))
    @pytest.mark.parametrize("supplied", list(Color))
    def test_color_pairs(self, planned, supplied):
        """계획한 색과 다른 색의 부품은 mismatch다. 흰색 자리의 회색만 shadowed다.

        다른 색 부품의 결함은 mismatch로 읽는 쌍만 쓴다.
        """
        layout = make_layout(make_part("P001", PartType.SQUARE_TILE, origin=(3, 3), color=planned))
        task = compile_task(layout, CELL)
        face = _face(task, task.parts[0], color=supplied)
        shot = render_topdown(NOMINAL, cell=CELL, board=BOARD, faces=[face])
        part = check_image(task, shot, NOMINAL).parts[0]
        expected = PartStatus.MISMATCH
        if planned == supplied:
            expected = PartStatus.MATCH
        elif (planned, supplied) == (Color.WHITE, Color.GRAY):
            expected = PartStatus.SHADOWED
        assert part.status == expected
        assert not part.empty

    def test_part_in_shadow_band_is_shadowed(self, task):
        """그림자 띠 안의 부품은 shadowed다. 띠 안의 빈 칸은 점유로 읽지 않는다."""
        shot = render_topdown(
            NOMINAL, cell=CELL, board=BOARD, faces=list(_faces(task).values()), shadow=SHADOW
        )
        result = check_image(task, shot, NOMINAL)
        statuses = _statuses(result)
        assert statuses.pop(SHADOWED_PART) == PartStatus.SHADOWED
        assert set(statuses.values()) == {PartStatus.MATCH}
        assert result.unexpected_cells == []
        assert (result.match_count, result.shadowed_count, result.mismatch_count) == (3, 1, 0)
        # shadowed는 실패로 세지 않는다. 일치로도 세지 않는다.
        assert result.passed

    def test_part_on_unplanned_cell_is_unexpected(self, task):
        """계획에 없는 칸에 부품이 있으면 그 칸이 unexpected_cells에 있고 판정은 실패다."""
        row, col = EMPTY_CELL
        frame = BoardFrame(cell=CELL, cell_size=BOARD.cell_size_mm / 1000)
        x, y = frame.point(np.array([col + 0.5, row + 0.5]))
        extra = PartFace(
            center=(float(x), float(y)),
            yaw=task.board_yaw,
            block_cells=((0, 0),),
            rgb=SUNLIT_RGB[Color.BLUE],
            shadow_rgb=SHADOW_RGB[Color.BLUE],
        )
        faces = [*_faces(task).values(), extra]
        shot = render_topdown(NOMINAL, cell=CELL, board=BOARD, faces=faces)
        result = check_image(task, shot, NOMINAL)
        assert result.unexpected_cells == [EMPTY_CELL]
        assert result.mismatch_count == 0
        assert not result.passed

    def test_fail_with_wrong_image_size(self, task, image):
        """이미지 크기가 카메라와 다르면 ImageCheckError를 낸다."""
        with pytest.raises(ImageCheckError, match="is not the camera size"):
            check_image(task, image.resize((320, 240)), NOMINAL)

    def test_fail_without_board_pins(self):
        """셀 설정에 핀이 없으면 ImageCheckError를 낸다."""
        flat = CELL.model_copy(update={"locator": None})
        flat_task = compile_task(LAYOUT, flat)
        shot = render_topdown(NOMINAL, cell=flat, board=BOARD)
        with pytest.raises(ImageCheckError, match="no board pins"):
            check_image(flat_task, shot, NOMINAL)

    def test_fail_with_cell_outside_the_image(self, task, image):
        """카메라가 보드의 일부만 보면 ImageCheckError를 낸다."""
        x, y, z = NOMINAL.position
        aside = NOMINAL.with_pose(NOMINAL.rotation_matrix, (x, y + 0.2, z))
        with pytest.raises(ImageCheckError, match="outside the image"):
            check_image(task, image, aside)

    def test_fail_with_part_face_outside_the_image(self, task, image):
        """핀의 창은 이미지 안이고 부품 윗면의 정사각형만 이미지 밖이어도 ImageCheckError를 낸다.

        col 0의 핀 끝은 이미지 왼쪽 변에서 8 px에 있다. 핀의 창은 반폭이 7 px다.
        윗면의 정사각형은 반변이 약 9 px다.
        """
        x, y, z = NOMINAL.position
        aside = NOMINAL.with_pose(NOMINAL.rotation_matrix, (x, y + 0.1428, z))
        on_edge = compile_task(
            make_layout(make_part("P001", PartType.SQUARE_TILE, origin=(3, 0))), CELL
        )
        # col 0에 부품이 없는 작업은 같은 카메라에서 읽힌다.
        check_image(task, image, aside)
        with pytest.raises(ImageCheckError, match="outside the image"):
            check_image(on_edge, image, aside)

    def test_fail_with_board_behind_the_camera(self, task, image):
        """카메라가 위를 보면 ImageCheckError를 낸다."""
        upward = NOMINAL.with_pose(
            [[0.0, -1.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, 1.0]], NOMINAL.position
        )
        with pytest.raises(ImageCheckError, match="behind the camera"):
            check_image(task, image, upward)

    def test_fail_with_no_pixels_in_a_part(self):
        """칸이 화소 1개보다 작아서 부품의 화소가 없으면 ImageCheckError를 낸다."""
        far = PinholeCamera(
            width=64,
            height=48,
            fx=5.0,
            fy=5.0,
            cx=31.5,
            cy=23.5,
            rotation=TOPDOWN_ROTATION,
            position=TOPDOWN_EYE,
        )
        single = compile_task(
            make_layout(make_part("P001", PartType.SQUARE_TILE, origin=(1, 1))), CELL
        )
        with pytest.raises(ImageCheckError, match="no pixels"):
            check_image(single, Image.new("RGB", (far.width, far.height)), far)


@pytest.mark.unit
class TestImageCheckReport:
    """보고서 모델의 저장과 일관성 검사 테스트."""

    def test_json_round_trip(self, report, tmp_path):
        """보고서를 JSON으로 쓰고 다시 읽으면 같은 보고서다."""
        path = tmp_path / "image_check.json"
        write_json(path, report)
        assert ImageCheckReport.model_validate_json(path.read_text(encoding="utf-8")) == report

    @pytest.mark.parametrize(
        ("field", "value", "message"),
        [
            ("match_count", 3, "Status counts"),
            ("passed", False, "passed does not agree"),
            ("unexpected_cells", [EMPTY_CELL], "passed does not agree"),
        ],
    )
    def test_fail_with_inconsistent_summary(self, report, field, value, message):
        """집계나 passed가 부품 상태와 다르면 보고서를 읽지 않는다."""
        payload = {**report.model_dump(mode="json"), field: value}
        with pytest.raises(ValidationError, match=message):
            ImageCheckReport.model_validate(payload)

    def test_fail_with_status_that_disagrees_with_values(self, report):
        """부품의 상태가 그 부품의 측정값과 다르면 보고서를 읽지 않는다."""
        payload = report.model_dump(mode="json")
        payload["parts"][0]["status"] = PartStatus.SHADOWED
        with pytest.raises(ValidationError, match="status does not agree"):
            ImageCheckReport.model_validate(payload)


@pytest.mark.unit
class TestDrawOverlay:
    """판독 overlay 테스트."""

    def test_overlay_marks_statuses(self, task):
        """overlay는 입력과 크기가 같다. 상태의 색과 계획에 없는 점유 칸의 색이 있다."""
        row, col = EMPTY_CELL
        frame = BoardFrame(cell=CELL, cell_size=BOARD.cell_size_mm / 1000)
        x, y = frame.point(np.array([col + 0.5, row + 0.5]))
        extra = PartFace(
            center=(float(x), float(y)),
            yaw=task.board_yaw,
            block_cells=((0, 0),),
            rgb=SUNLIT_RGB[Color.BLUE],
            shadow_rgb=SHADOW_RGB[Color.BLUE],
        )
        faces = _faces(task)
        del faces["P003"]
        shot = render_topdown(
            NOMINAL, cell=CELL, board=BOARD, faces=[*faces.values(), extra], shadow=SHADOW
        )
        result = check_image(task, shot, NOMINAL)
        overlay = draw_overlay(shot, result, task=task, camera=NOMINAL)
        assert overlay.size == shot.size
        assert overlay.mode == "RGB"
        assert set(OVERLAY_RGB.values()) | {UNEXPECTED_RGB} <= _colors(overlay)
        # 입력 이미지는 바꾸지 않는다.
        assert UNEXPECTED_RGB not in _colors(shot)


@pytest.mark.unit
class TestStoredRuns:
    """저장한 Isaac 실행의 최종 이미지 테스트. 조명, 팔 그림자와 부품의 옆면이 있다."""

    @pytest.mark.parametrize("case", STORED_CASES)
    def test_report_equals_stored_report(self, case):
        """저장한 실행을 판독하면 함께 저장한 보고서와 같다."""
        expected = ImageCheckReport.model_validate_json(
            (DATA_DIR / case / "image_check.json").read_text(encoding="utf-8")
        )
        result = _check_stored(case)
        ratios = {"coverage", "shadow_coverage"}
        assert result.model_dump(exclude={"parts"}) == expected.model_dump(exclude={"parts"})
        for part, stored in zip(result.parts, expected.parts, strict=True):
            assert part.model_dump(exclude=ratios) == stored.model_dump(exclude=ratios)
            assert part.coverage == pytest.approx(stored.coverage, abs=STORED_RATIO_TOLERANCE)
            assert part.shadow_coverage == pytest.approx(
                stored.shadow_coverage, abs=STORED_RATIO_TOLERANCE
            )

    def test_passed_run_matches(self):
        """모든 부품이 통과한 보정 실행은 모든 부품이 match다."""
        result = _check_stored("pass")
        assert set(_statuses(result).values()) == {PartStatus.MATCH}
        assert result.unexpected_cells == []
        assert result.passed

    def test_failed_run_has_mismatches(self):
        """부품이 넘어진 실행은 mismatch와 계획에 없는 점유 칸이 있다."""
        result = _check_stored("fail")
        assert result.mismatch_count == 7
        assert result.unexpected_cells == [(6, 0)]
        assert not result.passed

    def test_part_in_arm_shadow_is_shadowed(self):
        """팔 그림자 안의 칸 (0, 7)의 부품은 shadowed다. 다른 부품은 match다."""
        result = _check_stored("shadow")
        shadowed = [part for part in result.parts if part.status == PartStatus.SHADOWED]
        assert [part.cells for part in shadowed] == [[(0, 7)]]
        assert result.match_count == len(result.parts) - 1
        assert result.unexpected_cells == []
        assert result.passed


@pytest.mark.unit
class TestImageCheckCommand:
    """image-check 명령 테스트. 입력은 저장한 Isaac 실행이다."""

    def test_success_with_calibration_report(self, tmp_path):
        """통과한 실행은 부품마다 1줄과 집계를 출력하고 exit 0이다. 보고서와 overlay를 쓴다."""
        output, overlay = tmp_path / "out" / "image_check.json", tmp_path / "out" / "overlay.png"
        args = [*_stored_args("pass"), "-o", str(output), "--overlay", str(overlay)]
        result = runner.invoke(app, args)
        assert result.exit_code == 0, result.output
        lines = result.output.splitlines()
        assert lines[0] == (
            "P003 seq=1 square_tile white cells=(6,1) coverage=1.00 shadow_coverage=1.00 "
            "pins=hidden match"
        )
        assert lines[-1] == "match=7 shadowed=0 mismatch=0 unexpected=0 success"
        report = ImageCheckReport.model_validate_json(output.read_text(encoding="utf-8"))
        assert report.passed
        assert len(lines) == len(report.parts) + 1
        with Image.open(overlay) as drawn:
            assert drawn.size == report.image_size

    def test_fail_exit_one_with_mismatch(self):
        """불일치와 계획에 없는 점유가 있으면 그 칸을 출력하고 exit 1이다."""
        result = runner.invoke(app, _stored_args("fail"))
        assert result.exit_code == EXIT_INFEASIBLE, result.output
        lines = result.output.splitlines()
        assert lines[-2] == "unexpected_cells=(6,0)"
        assert lines[-1] == "match=1 shadowed=2 mismatch=7 unexpected=1 failure"
        assert "pins=hidden mismatch" in result.output

    def test_shadowed_part_keeps_exit_zero(self):
        """그림자 부품은 집계에 따로 나온다. 불일치가 없으면 exit 0이다."""
        result = runner.invoke(app, _stored_args("shadow"))
        assert result.exit_code == 0, result.output
        assert "cells=(0,7) coverage=0.00 shadow_coverage=1.00 pins=hidden shadowed" in (
            result.output
        )
        assert result.output.splitlines()[-1] == (
            "match=6 shadowed=1 mismatch=0 unexpected=0 success"
        )

    @pytest.mark.parametrize("both", [False, True], ids=["no_camera", "two_cameras"])
    def test_fail_without_exactly_one_camera(self, both):
        """--camera와 --calibration을 모두 주거나 모두 안 주면 exit 2다."""
        fail, shadow = DATA_DIR / "fail", DATA_DIR / "shadow"
        args = ["image-check", str(fail / "task.json"), str(fail / "topdown_final.png")]
        if both:
            args += ["--camera", str(fail / "camera.json")]
            args += ["--calibration", str(shadow / "cell.report.json")]
        result = runner.invoke(app, args)
        assert result.exit_code == EXIT_USAGE, result.output
        assert "Give --camera or --calibration" in result.output

    def test_fail_with_report_that_is_not_valid(self):
        """--calibration 파일이 보고 형식이 아니면 exit 2다."""
        folder = DATA_DIR / "fail"
        args = ["image-check", str(folder / "task.json"), str(folder / "topdown_final.png")]
        result = runner.invoke(app, [*args, "--calibration", str(folder / "camera.json")])
        assert result.exit_code == EXIT_USAGE, result.output
        assert "is not valid" in result.output

    def test_fail_with_file_that_is_not_an_image(self):
        """이미지 인자가 이미지 파일이 아니면 exit 2다."""
        folder = DATA_DIR / "fail"
        args = ["image-check", str(folder / "task.json"), str(folder / "scene.json")]
        result = runner.invoke(app, [*args, "--camera", str(folder / "camera.json")])
        assert result.exit_code == EXIT_USAGE, result.output
        assert "is not valid" in result.output

    def test_fail_with_image_of_another_size(self, tmp_path):
        """이미지 크기가 카메라와 다르면 exit 2다."""
        folder = DATA_DIR / "fail"
        small = tmp_path / "small.png"
        with Image.open(folder / "topdown_final.png") as opened:
            opened.resize((320, 240)).save(small)
        args = ["image-check", str(folder / "task.json"), str(small)]
        result = runner.invoke(app, [*args, "--camera", str(folder / "camera.json")])
        assert result.exit_code == EXIT_USAGE, result.output
        assert "is not the camera size" in result.output
