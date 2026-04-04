"""Tests for auto-detection integration in CheckCollector."""
from unittest.mock import patch, MagicMock
import pytest
from checkllm.check import CheckCollector
from checkllm.config import CheckllmConfig
from checkllm.judge import JudgeConfigError


def test_auto_detect_creates_openai_judge():
    config = CheckllmConfig(judge_backend="auto")
    collector = CheckCollector(config=config)
    with patch.dict("os.environ", {"OPENAI_API_KEY": "sk-test"}, clear=False):
        judge = collector._get_judge()
        from checkllm.judge import OpenAIJudge
        assert isinstance(judge, OpenAIJudge)


def test_auto_detect_creates_anthropic_judge():
    config = CheckllmConfig(judge_backend="auto")
    collector = CheckCollector(config=config)
    with patch.dict("os.environ", {"ANTHROPIC_API_KEY": "sk-ant-test"}, clear=True):
        judge = collector._get_judge()
        from checkllm.judge import AnthropicJudge
        assert isinstance(judge, AnthropicJudge)


def test_auto_detect_raises_helpful_error_when_nothing_found():
    config = CheckllmConfig(judge_backend="auto")
    collector = CheckCollector(config=config)
    with patch.dict("os.environ", {}, clear=True):
        with patch("checkllm.discovery._ollama_is_running", return_value=False):
            with pytest.raises(JudgeConfigError, match="No LLM judge backend found"):
                collector._get_judge()


def test_explicit_backend_skips_auto_detect():
    """When user sets backend explicitly, auto-detect is not used."""
    config = CheckllmConfig(judge_backend="openai", judge_model="gpt-4o-mini")
    collector = CheckCollector(config=config)
    with patch.dict("os.environ", {"OPENAI_API_KEY": "sk-test"}, clear=False):
        judge = collector._get_judge()
        from checkllm.judge import OpenAIJudge
        assert isinstance(judge, OpenAIJudge)
        assert judge.model == "gpt-4o-mini"
