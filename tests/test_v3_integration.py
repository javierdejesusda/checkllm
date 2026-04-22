"""Integration tests for v3.0-alpha features."""

from checkllm.check import CheckCollector
from checkllm.config import CheckllmConfig
from checkllm.testing import MockJudge
from checkllm.discovery import format_no_judge_error
from checkllm.errors import format_budget_error, format_missing_dependency_error
from checkllm.estimator import estimate_check_cost


def test_full_fluent_chain_with_mock_judge():
    """End-to-end: fluent chain with deterministic + mock judge checks."""
    config = CheckllmConfig(judge_backend="auto")
    judge = MockJudge(default_score=0.95)
    collector = CheckCollector(config=config, judge=judge)

    output = "Python is a high-level programming language created by Guido van Rossum."

    collector.that(output).contains("Python").not_contains("JavaScript").max_tokens(50).has_no_pii()

    # Also use traditional API
    collector.relevance(output, query="What is Python?")

    assert len(collector.results) == 5
    assert all(r.passed for r in collector.results)


def test_auto_detect_with_mock_judge_override():
    """Auto-detect is skipped when a judge is explicitly provided."""
    config = CheckllmConfig(judge_backend="auto")
    judge = MockJudge(default_score=0.9)
    collector = CheckCollector(config=config, judge=judge)
    # Should use the mock judge, not try to auto-detect
    result = collector.hallucination("output", context="context")
    assert result.passed


def test_cost_estimate_for_mixed_checks():
    """Estimate correctly separates deterministic from judge checks."""
    det = estimate_check_cost("contains")
    judge = estimate_check_cost("hallucination", model="gpt-4o-mini")
    assert det.cost == 0.0
    assert judge.cost > 0.0


def test_error_messages_are_actionable():
    msg = format_no_judge_error()
    assert "OPENAI_API_KEY" in msg
    assert "Ollama" in msg
    assert "checkllm init" in msg

    msg = format_budget_error(5.0, 5.01, 14, 20)
    assert "--budget" in msg

    msg = format_missing_dependency_error("anthropic", "AnthropicJudge")
    assert "pip install" in msg
