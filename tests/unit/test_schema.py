import pytest
from pydantic import ValidationError

from pixel2place.layout.schema import Layout, Rotation

SPEC_EXAMPLE = {
    "project_version": "0.1",
    "board": {"type": "grid", "rows": 8, "cols": 8, "cell_size_mm": 25},
    "palette": ["blue", "gray", "white"],
    "parts": [
        {
            "part_id": "P001",
            "type": "L_tile",
            "color": "blue",
            "cells": [[1, 2], [2, 2], [3, 2], [3, 3]],
            "rotation_deg": 0,
        }
    ],
    "feasible": True,
    "notes": [],
}


@pytest.mark.interface
class TestLayoutSchema:
    """layout.json 스키마 계약 테스트."""

    def test_success_with_spec_example(self):
        """스펙 문서의 JSON 예시를 검증 오류 없이 읽는다."""
        layout = Layout.model_validate(SPEC_EXAMPLE)
        assert layout.parts[0].cells[0] == (1, 2)
        assert layout.parts[0].rotation_deg is Rotation.DEG_0

    def test_json_round_trip_is_stable(self):
        """JSON으로 쓰고 다시 읽으면 같은 레이아웃이 된다."""
        layout = Layout.model_validate(SPEC_EXAMPLE)
        assert Layout.model_validate_json(layout.model_dump_json()) == layout

    @pytest.mark.parametrize(
        ("path", "value"),
        [
            (("parts", 0, "type"), "hex_tile"),
            (("parts", 0, "color"), "purple"),
            (("parts", 0, "rotation_deg"), 45),
            (("board", "rows"), 0),
        ],
    )
    def test_fail_with_value_out_of_vocabulary(self, path, value):
        """어휘나 범위 밖의 값이 있으면 ValidationError를 낸다."""
        data = Layout.model_validate(SPEC_EXAMPLE).model_dump(mode="json")
        target = data
        for key in path[:-1]:
            target = target[key]
        target[path[-1]] = value
        with pytest.raises(ValidationError):
            Layout.model_validate(data)

    def test_fail_with_unknown_field(self):
        """정의하지 않은 필드가 있으면 ValidationError를 낸다."""
        with pytest.raises(ValidationError):
            Layout.model_validate({**SPEC_EXAMPLE, "extra": 1})
