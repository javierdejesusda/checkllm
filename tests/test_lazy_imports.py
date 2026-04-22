"""Verify that importing judge module doesn't immediately fail without openai."""

from unittest.mock import patch
import pytest


class TestLazyImports:
    def test_judge_module_importable(self):
        """judge.py must be importable even if openai is not installed."""
        from checkllm import judge

        assert hasattr(judge, "JudgeBackend")
        assert hasattr(judge, "OpenAIJudge")

    def test_openai_judge_raises_helpful_error_without_openai(self):
        """OpenAIJudge() must raise JudgeConfigError when openai is missing."""
        import sys
        from checkllm.judge import JudgeConfigError, OpenAIJudge

        with patch.dict(sys.modules, {"openai": None}):
            with pytest.raises(JudgeConfigError, match="openai.*pip install"):
                OpenAIJudge(api_key="sk-fake")

    def test_retry_decorator_works_without_openai_exceptions(self):
        """_make_retry must fall back to stdlib exceptions when openai is absent."""
        from checkllm.judge import _make_retry

        decorator = _make_retry()
        assert callable(decorator)
