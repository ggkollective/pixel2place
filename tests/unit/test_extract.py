import pytest

from pixel2place.reasoner.extract import JsonExtractionError, extract_json_object


@pytest.mark.unit
class TestExtractJsonObject:
    """응답 텍스트의 JSON 추출 테스트."""

    def test_plain_json(self):
        """JSON만 있는 응답을 그대로 읽는다."""
        assert extract_json_object('{"a": 1}') == {"a": 1}

    def test_ignores_think_block_with_braces(self):
        """<think> 블록 안의 중괄호는 무시한다."""
        text = '<think>try {"a": 0} first</think>\nAnswer: {"a": 2}'
        assert extract_json_object(text) == {"a": 2}

    def test_reads_code_fence(self):
        """코드 펜스 안의 JSON을 읽는다."""
        assert extract_json_object('Here:\n```json\n{"a": [1, 2]}\n```\nDone.') == {"a": [1, 2]}

    def test_skips_invalid_brace(self):
        """디코드할 수 없는 `{`는 건너뛰고 다음 객체를 읽는다."""
        assert extract_json_object('use {braces} then {"a": 3}') == {"a": 3}

    def test_fail_without_object(self):
        """JSON 객체가 없으면 JsonExtractionError를 낸다."""
        with pytest.raises(JsonExtractionError):
            extract_json_object("[1, 2, 3] no object")
