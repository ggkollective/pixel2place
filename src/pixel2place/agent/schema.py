"""strict 함수 호출에 보내는 인자 스키마."""

from pydantic import BaseModel

# 모델에 보내지 않는 키다. docstring과 기본값은 인자 계약이 아니다.
_DROPPED_KEYS = frozenset(("description", "default", "$defs"))
_SCHEMA_KEYS = frozenset(("items", "additionalProperties"))
_SCHEMA_LIST_KEYS = frozenset(("anyOf", "oneOf", "allOf", "prefixItems"))
_REF_PREFIX = "#/$defs/"


def strict_parameters(model: type[BaseModel]) -> dict[str, object]:
    """인자 모델의 JSON 스키마를 `$ref`가 없는 한 객체로 만든다.

    설명과 기본값을 지운다. 모든 속성을 필수로 둔다.
    키 순서는 pydantic의 순서를 유지한다. 같은 모델은 같은 JSON이 된다.
    """
    schema = model.model_json_schema()
    definitions = schema.get("$defs", {})
    return _flatten(schema, definitions=definitions)


def _flatten(node: dict[str, object], *, definitions: dict[str, object]) -> dict[str, object]:
    reference = node.get("$ref")
    if isinstance(reference, str):
        target = definitions[reference.removeprefix(_REF_PREFIX)]
        assert isinstance(target, dict)
        node = {**target, **{key: value for key, value in node.items() if key != "$ref"}}
    flat: dict[str, object] = {}
    for key, value in node.items():
        if key in _DROPPED_KEYS:
            continue
        if key == "properties" and isinstance(value, dict):
            # 속성 이름은 스키마 키가 아니다. 이름은 그대로 두고 값만 평평하게 만든다.
            flat[key] = {
                name: _flatten(item, definitions=definitions) for name, item in value.items()
            }
        elif key in _SCHEMA_KEYS and isinstance(value, dict):
            flat[key] = _flatten(value, definitions=definitions)
        elif key in _SCHEMA_LIST_KEYS and isinstance(value, list):
            flat[key] = [_flatten(item, definitions=definitions) for item in value]
        else:
            flat[key] = value
    properties = flat.get("properties")
    if isinstance(properties, dict):
        flat["required"] = list(properties)
    return flat
