"""Tests for soft assertions (check.expect)."""
import pytest

from checkllm.check import CheckCollector
from checkllm.config import CheckllmConfig
from checkllm.models import CheckFailedError


class TestSoftAssertions:
    def test_expect_doesnt_fail_test(self):
        collector = CheckCollector(config=CheckllmConfig(cache_enabled=False))
        # Hard check passes
        collector.contains("hello world", "hello")
        # Soft check would fail but shouldn't cause test failure
        collector.expect.contains("hello world", "missing_substring")
        # Teardown should NOT raise — soft check was softened
        collector.teardown()

    def test_expect_preserves_score(self):
        collector = CheckCollector(config=CheckllmConfig(cache_enabled=False))
        result = collector.expect.contains("hello world", "missing")
        # Score should be 0.0 (it failed) but passed should be True (softened)
        assert result.passed is True
        assert result.score == 0.0

    def test_expect_marks_reasoning(self):
        collector = CheckCollector(config=CheckllmConfig(cache_enabled=False))
        result = collector.expect.contains("hello", "bye")
        assert "[soft]" in result.reasoning

    def test_expect_passing_check_unchanged(self):
        collector = CheckCollector(config=CheckllmConfig(cache_enabled=False))
        result = collector.expect.contains("hello world", "hello")
        assert result.passed is True
        assert result.score == 1.0
        assert "[soft]" not in result.reasoning  # Passing checks stay as-is

    def test_expect_results_recorded(self):
        collector = CheckCollector(config=CheckllmConfig(cache_enabled=False))
        collector.expect.contains("hello", "bye")
        collector.expect.regex("abc123", pattern=r"\d+")
        assert len(collector.results) == 2

    def test_mixed_hard_and_soft(self):
        collector = CheckCollector(config=CheckllmConfig(cache_enabled=False))
        # Hard checks
        collector.contains("hello world", "hello")
        # Soft checks (would fail)
        collector.expect.contains("hello world", "missing")
        collector.expect.word_count("hello world", min_words=100)
        # Teardown should succeed — only hard checks matter
        collector.teardown()

    def test_hard_fails_still_raise(self):
        collector = CheckCollector(config=CheckllmConfig(cache_enabled=False))
        # Hard check fails
        collector.contains("hello", "bye")
        # Soft check also "fails"
        collector.expect.contains("hello", "missing")
        # Teardown SHOULD raise due to the hard failure
        with pytest.raises(CheckFailedError):
            collector.teardown()

    def test_expect_all_of(self):
        collector = CheckCollector(config=CheckllmConfig(cache_enabled=False))
        result = collector.expect.all_of("hello", ["hello", "missing"])
        assert result.passed is True  # softened
        assert "[soft]" in result.reasoning

    def test_expect_is_json(self):
        collector = CheckCollector(config=CheckllmConfig(cache_enabled=False))
        result = collector.expect.is_json("not json")
        assert result.passed is True  # softened

    def test_expect_is_valid_python(self):
        collector = CheckCollector(config=CheckllmConfig(cache_enabled=False))
        result = collector.expect.is_valid_python("def broken(")
        assert result.passed is True  # softened
