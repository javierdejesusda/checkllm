from unittest.mock import AsyncMock

import pytest

from checkllm.check import CheckCollector
from checkllm.config import CheckllmConfig
from checkllm.models import CheckFailedError, CheckResult, JudgeResponse


class TestCheckCollectorDeterministic:
    def test_contains_pass(self):
        collector = CheckCollector(config=CheckllmConfig())
        collector.contains("hello world", "hello")
        assert len(collector.results) == 1
        assert collector.results[0].passed is True

    def test_not_contains_pass(self):
        collector = CheckCollector(config=CheckllmConfig())
        collector.not_contains("hello world", "bye")
        assert collector.results[0].passed is True

    def test_max_tokens(self):
        collector = CheckCollector(config=CheckllmConfig())
        collector.max_tokens("short text", limit=100)
        assert collector.results[0].passed is True

    def test_latency(self):
        collector = CheckCollector(config=CheckllmConfig())
        collector.latency(100, max_ms=200)
        assert collector.results[0].passed is True

    def test_cost(self):
        collector = CheckCollector(config=CheckllmConfig())
        collector.cost(0.01, max_usd=0.05)
        assert collector.results[0].passed is True

    def test_json_schema(self):
        from pydantic import BaseModel

        class M(BaseModel):
            name: str

        collector = CheckCollector(config=CheckllmConfig())
        collector.json_schema('{"name": "test"}', schema=M)
        assert collector.results[0].passed is True

    def test_regex(self):
        collector = CheckCollector(config=CheckllmConfig())
        collector.regex("abc123", pattern=r"\d+")
        assert collector.results[0].passed is True


class TestCheckCollectorTeardown:
    def test_teardown_passes_when_all_checks_pass(self):
        collector = CheckCollector(config=CheckllmConfig())
        collector.contains("hello", "hello")
        collector.not_contains("hello", "bye")
        collector.teardown()  # Should not raise

    def test_teardown_raises_when_check_fails(self):
        collector = CheckCollector(config=CheckllmConfig())
        collector.contains("hello", "bye")  # This will fail
        with pytest.raises(CheckFailedError) as exc_info:
            collector.teardown()
        assert len(exc_info.value.failed_results) == 1

    def test_teardown_reports_all_failures(self):
        collector = CheckCollector(config=CheckllmConfig())
        collector.contains("hello", "bye")
        collector.not_contains("hello", "hello")
        with pytest.raises(CheckFailedError) as exc_info:
            collector.teardown()
        assert len(exc_info.value.failed_results) == 2


class TestCheckCollectorMultipleChecks:
    def test_collects_multiple_results(self):
        collector = CheckCollector(config=CheckllmConfig())
        collector.contains("hello world", "hello")
        collector.not_contains("hello world", "bye")
        collector.regex("abc123", pattern=r"\d+")
        assert len(collector.results) == 3
        assert all(r.passed for r in collector.results)
