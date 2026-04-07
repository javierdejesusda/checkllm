"""Verify deterministic-only usage does not require openai."""
import pytest

from checkllm.config import CheckllmConfig
from checkllm.check import CheckCollector


class TestDeterministicOnly:
    def test_contains_works_without_judge(self):
        config = CheckllmConfig()
        collector = CheckCollector(config=config)
        result = collector.contains("hello world", "hello")
        assert result.passed

    def test_regex_works_without_judge(self):
        config = CheckllmConfig()
        collector = CheckCollector(config=config)
        result = collector.regex("abc123", r"\d+")
        assert result.passed

    def test_is_json_works_without_judge(self):
        config = CheckllmConfig()
        collector = CheckCollector(config=config)
        result = collector.is_json('{"key": "value"}')
        assert result.passed

    def test_bleu_works_without_judge(self):
        config = CheckllmConfig()
        collector = CheckCollector(config=config)
        result = collector.bleu("the cat sat on the mat", "the cat sat on the mat")
        assert result.passed
