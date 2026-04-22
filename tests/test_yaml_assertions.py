"""Tests for promptfoo-style declarative YAML assertions."""

from __future__ import annotations

import pytest

from checkllm.models import JudgeResponse
from checkllm.yaml_assertions import (
    Assertion,
    AssertionResults,
    SUPPORTED_TYPES,
    evaluate_assertions,
    parse_assertions,
)


class _StubJudge:
    """Minimal JudgeBackend-compatible stub for tests."""

    def __init__(self, score: float = 0.9, cost: float = 0.001) -> None:
        self.score = score
        self.last_cost = cost
        self.total_cost = 0.0
        self._fixed_cost = cost

    async def evaluate(self, prompt: str, system_prompt: str | None = None) -> JudgeResponse:
        self.last_cost = self._fixed_cost
        self.total_cost += self._fixed_cost
        return JudgeResponse(
            score=self.score,
            reasoning="stub",
            raw_output='{"score": %.2f, "reasoning": "stub"}' % self.score,
            cost=self._fixed_cost,
        )


def test_supported_types_includes_promptfoo_basics() -> None:
    for t in {
        "contains",
        "not-contains",
        "regex",
        "equals",
        "model",
        "llm-rubric",
        "similarity",
        "model-graded-relevance",
        "model-graded-faithfulness",
        "cost",
        "latency",
    }:
        assert t in SUPPORTED_TYPES


def test_parse_contains_assertion() -> None:
    [a] = parse_assertions([{"type": "contains", "value": "hello"}])
    assert a.type == "contains"
    assert a.value == "hello"


def test_parse_normalises_underscores_to_dashes() -> None:
    [a] = parse_assertions([{"type": "not_contains", "value": "bad"}])
    assert a.type == "not-contains"


def test_parse_unknown_type_raises() -> None:
    with pytest.raises(ValueError):
        parse_assertions([{"type": "frobnicate", "value": 1}])


def test_parse_llm_rubric_requires_rubric() -> None:
    with pytest.raises(ValueError):
        parse_assertions([{"type": "llm-rubric"}])


def test_parse_model_requires_prompt() -> None:
    with pytest.raises(ValueError):
        parse_assertions([{"type": "model"}])


def test_parse_model_graded_relevance_requires_query() -> None:
    with pytest.raises(ValueError):
        parse_assertions([{"type": "model-graded-relevance"}])


def test_parse_model_graded_faithfulness_requires_context() -> None:
    with pytest.raises(ValueError):
        parse_assertions([{"type": "model-graded-faithfulness"}])


def test_parse_cost_requires_numeric_value() -> None:
    with pytest.raises(ValueError):
        parse_assertions([{"type": "cost"}])


def test_parse_all_types_smoke() -> None:
    parsed = parse_assertions(
        [
            {"type": "contains", "value": "x"},
            {"type": "not-contains", "value": "y"},
            {"type": "regex", "value": "^a"},
            {"type": "equals", "value": "ab"},
            {"type": "model", "prompt": "judge me", "threshold": 0.5},
            {"type": "llm-rubric", "rubric": "Must be concise.", "threshold": 0.7},
            {"type": "similarity", "reference": "ab", "threshold": 0.6},
            {"type": "model-graded-relevance", "query": "q", "threshold": 0.8},
            {"type": "model-graded-faithfulness", "context": "ctx", "threshold": 0.8},
            {"type": "cost", "value": 0.02},
            {"type": "latency", "value": 1500},
        ]
    )
    assert len(parsed) == 11
    assert all(isinstance(a, Assertion) for a in parsed)


async def test_evaluate_contains_pass() -> None:
    judge = _StubJudge()
    [a] = parse_assertions([{"type": "contains", "value": "hello"}])
    res = await evaluate_assertions("hello world", [a], judge=judge)
    assert res.passed is True
    assert res.individual[0].passed is True


async def test_evaluate_contains_fail() -> None:
    judge = _StubJudge()
    [a] = parse_assertions([{"type": "contains", "value": "missing"}])
    res = await evaluate_assertions("hello world", [a], judge=judge)
    assert res.passed is False


async def test_evaluate_regex_invalid_pattern_does_not_raise() -> None:
    judge = _StubJudge()
    [a] = parse_assertions([{"type": "regex", "value": "(unclosed"}])
    res = await evaluate_assertions("x", [a], judge=judge)
    assert res.passed is False


async def test_evaluate_llm_rubric_uses_judge_score() -> None:
    judge = _StubJudge(score=0.95)
    assertions = parse_assertions([{"type": "llm-rubric", "rubric": "Good?", "threshold": 0.7}])
    res = await evaluate_assertions("candidate output", assertions, judge=judge)
    assert res.passed is True
    assert res.individual[0].score == pytest.approx(0.95)


async def test_evaluate_llm_rubric_below_threshold_fails() -> None:
    judge = _StubJudge(score=0.2)
    assertions = parse_assertions([{"type": "llm-rubric", "rubric": "Good?", "threshold": 0.7}])
    res = await evaluate_assertions("candidate output", assertions, judge=judge)
    assert res.passed is False


async def test_evaluate_cost_assertion_passes_when_within_budget() -> None:
    judge = _StubJudge(cost=0.0001)
    # Prime judge.last_cost via a preceding rubric call.
    assertions = parse_assertions(
        [
            {"type": "llm-rubric", "rubric": "x", "threshold": 0.5},
            {"type": "cost", "value": 0.01},
        ]
    )
    res = await evaluate_assertions("x", assertions, judge=judge)
    assert res.individual[1].passed is True


async def test_evaluate_cost_assertion_fails_when_over_budget() -> None:
    judge = _StubJudge(cost=0.5)
    assertions = parse_assertions(
        [
            {"type": "llm-rubric", "rubric": "x", "threshold": 0.5},
            {"type": "cost", "value": 0.01},
        ]
    )
    res = await evaluate_assertions("x", assertions, judge=judge)
    assert res.individual[1].passed is False


async def test_evaluate_latency_assertion_reads_context() -> None:
    judge = _StubJudge()
    [a] = parse_assertions([{"type": "latency", "value": 1000}])
    res_ok = await evaluate_assertions("x", [a], judge=judge, context={"latency_ms": 500})
    assert res_ok.passed is True
    res_bad = await evaluate_assertions("x", [a], judge=judge, context={"latency_ms": 5000})
    assert res_bad.passed is False


async def test_evaluate_model_uses_bare_prompt() -> None:
    judge = _StubJudge(score=0.85)
    [a] = parse_assertions([{"type": "model", "prompt": "Is this polite?", "threshold": 0.8}])
    res = await evaluate_assertions("pls and ty", [a], judge=judge)
    assert res.passed is True


async def test_evaluate_similarity_uses_reference() -> None:
    judge = _StubJudge()
    [a] = parse_assertions([{"type": "similarity", "reference": "hello world", "threshold": 0.5}])
    res = await evaluate_assertions("hello world!", [a], judge=judge)
    assert res.passed is True


def test_assertion_results_roundtrip_fields() -> None:
    r = AssertionResults(passed=True, individual=[])
    assert r.passed is True
    assert r.individual == []
