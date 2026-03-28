import pydantic
import pytest

from checkllm.deterministic import DeterministicChecks
from checkllm.models import CheckResult


class TestContains:
    def test_passes_when_substring_present(self):
        dc = DeterministicChecks()
        result = dc.contains("The weather is sunny today", "sunny")
        assert result.passed is True
        assert result.score == 1.0
        assert result.cost == 0.0

    def test_fails_when_substring_absent(self):
        dc = DeterministicChecks()
        result = dc.contains("The weather is sunny today", "rainy")
        assert result.passed is False
        assert result.score == 0.0


class TestNotContains:
    def test_passes_when_substring_absent(self):
        dc = DeterministicChecks()
        result = dc.not_contains("The weather is sunny", "rainy")
        assert result.passed is True

    def test_fails_when_substring_present(self):
        dc = DeterministicChecks()
        result = dc.not_contains("The weather is sunny", "sunny")
        assert result.passed is False


class TestMaxTokens:
    def test_passes_under_limit(self):
        dc = DeterministicChecks()
        result = dc.max_tokens("hello world", limit=100)
        assert result.passed is True

    def test_fails_over_limit(self):
        dc = DeterministicChecks()
        long_text = "word " * 200
        result = dc.max_tokens(long_text, limit=10)
        assert result.passed is False


class TestLatency:
    def test_passes_under_limit(self):
        dc = DeterministicChecks()
        result = dc.latency(500, max_ms=1000)
        assert result.passed is True

    def test_fails_over_limit(self):
        dc = DeterministicChecks()
        result = dc.latency(1500, max_ms=1000)
        assert result.passed is False


class TestCost:
    def test_passes_under_limit(self):
        dc = DeterministicChecks()
        result = dc.cost(0.03, max_usd=0.05)
        assert result.passed is True

    def test_fails_over_limit(self):
        dc = DeterministicChecks()
        result = dc.cost(0.10, max_usd=0.05)
        assert result.passed is False


class TestJsonSchema:
    def test_passes_valid_json(self):
        class MyModel(pydantic.BaseModel):
            name: str
            age: int

        dc = DeterministicChecks()
        result = dc.json_schema('{"name": "Alice", "age": 30}', schema=MyModel)
        assert result.passed is True

    def test_fails_invalid_json(self):
        class MyModel(pydantic.BaseModel):
            name: str
            age: int

        dc = DeterministicChecks()
        result = dc.json_schema('{"name": "Alice"}', schema=MyModel)
        assert result.passed is False

    def test_fails_malformed_json(self):
        class MyModel(pydantic.BaseModel):
            name: str

        dc = DeterministicChecks()
        result = dc.json_schema("not json at all", schema=MyModel)
        assert result.passed is False


class TestRegex:
    def test_passes_when_pattern_matches(self):
        dc = DeterministicChecks()
        result = dc.regex("Call 555-1234 for info", pattern=r"\d{3}-\d{4}")
        assert result.passed is True

    def test_fails_when_pattern_does_not_match(self):
        dc = DeterministicChecks()
        result = dc.regex("No phone number here", pattern=r"\d{3}-\d{4}")
        assert result.passed is False


class TestAllReturnCheckResult:
    def test_all_methods_return_check_result(self):
        dc = DeterministicChecks()
        results = [
            dc.contains("hello", "hello"),
            dc.not_contains("hello", "bye"),
            dc.max_tokens("hello", limit=100),
            dc.latency(100, max_ms=200),
            dc.cost(0.01, max_usd=0.05),
            dc.regex("abc123", pattern=r"\d+"),
        ]
        for r in results:
            assert isinstance(r, CheckResult)
            assert r.cost == 0.0
            assert r.latency_ms == 0
