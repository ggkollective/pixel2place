"""strict 도구 스키마의 평탄화와 기존 도구 정의의 유지를 검사한다."""

import json
from enum import StrEnum
from typing import Annotated

import pytest
from pydantic import BaseModel, Field

from pixel2place.agent.contracts import (
    CalibrateBoardArguments,
    Contract,
    ExecuteCellArguments,
    PrepareLayoutArguments,
)
from pixel2place.agent.schema import strict_parameters

pytestmark = pytest.mark.unit


class _Shade(StrEnum):
    """Docstring that must not reach the model."""

    DARK = "dark"
    LIGHT = "light"


class _Item(Contract):
    """Nested docstring that must not reach the model."""

    shade: _Shade
    count: Annotated[int, Field(ge=1, le=3)] = 1


class _Arguments(Contract):
    """Top docstring that must not reach the model."""

    items: tuple[_Item, ...]
    description: str
    default: bool = False
    limit: _Shade | None = None


class TestStrictParameters:
    """중첩 모델과 enum의 평탄화, 그리고 평평한 기존 모델의 불변을 검사한다."""

    def test_references_are_inlined_and_docstrings_are_removed(self) -> None:
        """`$ref`를 정의로 바꾼다. docstring과 기본값은 스키마에 없다."""
        schema = strict_parameters(_Arguments)
        text = json.dumps(schema)
        assert "$ref" not in text and "$defs" not in text and "must not reach" not in text
        properties = schema["properties"]
        assert isinstance(properties, dict)
        item = properties["items"]["items"]
        assert item["properties"]["shade"]["enum"] == ["dark", "light"]
        assert item["properties"]["count"] == {
            "maximum": 3,
            "minimum": 1,
            "title": "Count",
            "type": "integer",
        }
        assert properties["limit"]["anyOf"] == [
            {"enum": ["dark", "light"], "title": "_Shade", "type": "string"},
            {"type": "null"},
        ]

    def test_every_property_is_required_even_with_a_default(self) -> None:
        """기본값이 있는 속성도 필수다. 속성 이름이 스키마 키와 같아도 남는다."""
        schema = strict_parameters(_Arguments)
        properties = schema["properties"]
        assert isinstance(properties, dict)
        assert list(properties) == ["items", "description", "default", "limit"]
        assert schema["required"] == ["items", "description", "default", "limit"]
        assert properties["items"]["items"]["required"] == ["shade", "count"]

    @pytest.mark.parametrize(
        "model", [PrepareLayoutArguments, CalibrateBoardArguments, ExecuteCellArguments]
    )
    def test_flat_models_keep_the_schema_that_the_runtime_sends(
        self, model: type[BaseModel]
    ) -> None:
        """평평한 인자 모델의 스키마는 pydantic 스키마에서 최상위 설명만 뺀 값이다."""
        expected = model.model_json_schema()
        expected.pop("description", None)
        assert json.dumps(strict_parameters(model)) == json.dumps(expected)
