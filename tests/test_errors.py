"""Tests for user-friendly error messages."""

from checkllm.errors import (
    format_budget_error,
    format_missing_dependency_error,
)


def test_budget_error_includes_amounts():
    msg = format_budget_error(budget=5.0, spent=5.01, completed=14, total=20)
    assert "$5.00" in msg
    assert "14" in msg
    assert "20" in msg
    assert "--budget" in msg


def test_budget_error_suggests_alternatives():
    msg = format_budget_error(budget=5.0, spent=5.01, completed=14, total=20)
    assert "deterministic" in msg.lower() or "--budget" in msg


def test_missing_dependency_error():
    msg = format_missing_dependency_error("anthropic", "AnthropicJudge")
    assert "pip install checkllm[anthropic]" in msg
    assert "AnthropicJudge" in msg


def test_missing_dependency_for_gemini():
    msg = format_missing_dependency_error("gemini", "GeminiJudge")
    assert "pip install checkllm[gemini]" in msg


def test_missing_dependency_for_litellm():
    msg = format_missing_dependency_error("litellm", "LiteLLMJudge")
    assert "pip install checkllm[litellm]" in msg


def test_missing_dependency_for_embeddings():
    msg = format_missing_dependency_error("embeddings", "SentenceTransformerEmbeddings")
    assert "pip install checkllm[embeddings]" in msg
