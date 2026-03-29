"""Tests for the programmatic API module."""
from __future__ import annotations

import pytest

from checkllm.api import Evaluator, check_output, evaluate, parse_check_shorthand
from checkllm.config import CheckllmConfig
from checkllm.guardrails import ValidationResult
from checkllm.testing import MockJudge


# ---------------------------------------------------------------------------
# parse_check_shorthand
# ---------------------------------------------------------------------------


class TestParseCheckShorthand:
    def test_simple_check(self):
        result = parse_check_shorthand("no_pii")
        assert result == {"type": "no_pii", "params": {}}

    def test_max_tokens_with_int(self):
        result = parse_check_shorthand("max_tokens:200")
        assert result == {"type": "max_tokens", "params": {"limit": 200}}
        assert isinstance(result["params"]["limit"], int)

    def test_min_tokens_with_int(self):
        result = parse_check_shorthand("min_tokens:50")
        assert result == {"type": "min_tokens", "params": {"minimum": 50}}

    def test_contains_with_string(self):
        result = parse_check_shorthand("contains:hello")
        assert result == {"type": "contains", "params": {"substring": "hello"}}

    def test_not_contains_with_string(self):
        result = parse_check_shorthand("not_contains:goodbye")
        assert result == {"type": "not_contains", "params": {"substring": "goodbye"}}

    def test_starts_with(self):
        result = parse_check_shorthand("starts_with:Hello")
        assert result == {"type": "starts_with", "params": {"prefix": "Hello"}}

    def test_ends_with(self):
        result = parse_check_shorthand("ends_with:.")
        assert result == {"type": "ends_with", "params": {"suffix": "."}}

    def test_regex(self):
        result = parse_check_shorthand(r"regex:\d+")
        assert result == {"type": "regex", "params": {"pattern": r"\d+"}}

    def test_float_value(self):
        result = parse_check_shorthand("greater_than:3.14")
        assert result == {"type": "greater_than", "params": {"value": 3.14}}
        assert isinstance(result["params"]["value"], float)

    def test_unknown_type_uses_value_key(self):
        result = parse_check_shorthand("custom_check:foo")
        assert result == {"type": "custom_check", "params": {"value": "foo"}}

    def test_language(self):
        result = parse_check_shorthand("language:en")
        assert result == {"type": "language", "params": {"expected": "en"}}

    def test_colon_in_value(self):
        # Only the first colon splits
        result = parse_check_shorthand("regex:https://example.com")
        assert result["type"] == "regex"
        assert result["params"]["pattern"] == "https://example.com"


# ---------------------------------------------------------------------------
# evaluate() - async
# ---------------------------------------------------------------------------


class TestEvaluateAsync:
    @pytest.mark.asyncio
    async def test_evaluate_deterministic_pass(self):
        result = await evaluate(
            output="Hello world, the answer is 42.",
            checks=[
                {"type": "contains", "params": {"substring": "42"}},
                {"type": "no_pii"},
            ],
        )
        assert isinstance(result, ValidationResult)
        assert result.valid is True
        assert len(result.results) == 2

    @pytest.mark.asyncio
    async def test_evaluate_deterministic_fail(self):
        result = await evaluate(
            output="Contact user@example.com",
            checks=["no_pii"],
        )
        assert result.valid is False

    @pytest.mark.asyncio
    async def test_evaluate_shorthand_strings(self):
        result = await evaluate(
            output="Hello world",
            checks=["no_pii", "max_tokens:1000"],
        )
        assert result.valid is True

    @pytest.mark.asyncio
    async def test_evaluate_with_mock_judge(self):
        judge = MockJudge(default_score=0.95)
        result = await evaluate(
            output="This is a test output.",
            checks=[
                {"type": "toxicity"},
                {"type": "no_pii"},
            ],
            judge=judge,
        )
        assert result.valid is True
        assert len(result.results) == 2
        judge.assert_called()

    @pytest.mark.asyncio
    async def test_evaluate_mixed_checks(self):
        judge = MockJudge(default_score=0.9)
        result = await evaluate(
            output="The answer is 42.",
            checks=[
                {"type": "contains", "params": {"substring": "42"}},
                "no_pii",
                {"type": "toxicity"},
            ],
            judge=judge,
            threshold=0.8,
        )
        assert result.valid is True
        assert len(result.results) == 3


# ---------------------------------------------------------------------------
# check_output() - sync
# ---------------------------------------------------------------------------


class TestCheckOutput:
    def test_sync_deterministic_pass(self):
        result = check_output(
            "Hello world",
            checks=["no_pii", "max_tokens:1000"],
        )
        assert result.valid is True

    def test_sync_deterministic_fail(self):
        result = check_output(
            "Email: user@example.com",
            checks=["no_pii"],
        )
        assert result.valid is False

    def test_sync_with_dict_checks(self):
        result = check_output(
            "Hello world!",
            checks=[
                {"type": "contains", "params": {"substring": "Hello"}},
                {"type": "max_tokens", "params": {"limit": 100}},
            ],
        )
        assert result.valid is True

    def test_sync_with_mock_judge(self):
        judge = MockJudge(default_score=0.9)
        result = check_output(
            "Clean output text.",
            checks=["no_pii", {"type": "toxicity"}],
            judge=judge,
        )
        assert result.valid is True
        judge.assert_called()


# ---------------------------------------------------------------------------
# Evaluator builder
# ---------------------------------------------------------------------------


class TestEvaluatorBuilder:
    def test_basic_builder(self):
        judge = MockJudge(default_score=0.9)
        evaluator = (
            Evaluator()
            .with_judge(judge)
            .add_check("no_pii")
            .add_check("contains", substring="hello")
        )
        result = evaluator.run("hello world")
        assert result.valid is True

    def test_builder_chaining_returns_self(self):
        evaluator = Evaluator()
        assert evaluator.with_judge("openai") is evaluator
        assert evaluator.with_threshold(0.9) is evaluator
        assert evaluator.with_budget(5.0) is evaluator
        assert evaluator.add_check("no_pii") is evaluator
        assert evaluator.with_config(CheckllmConfig()) is evaluator

    def test_with_threshold(self):
        judge = MockJudge(default_score=0.85)
        # 0.85 passes at 0.8 threshold
        evaluator = (
            Evaluator()
            .with_judge(judge)
            .with_threshold(0.8)
            .add_check("toxicity")
        )
        result = evaluator.run("some text")
        assert result.valid is True

    def test_with_threshold_too_high(self):
        judge = MockJudge(default_score=0.85)
        # 0.85 fails at 0.9 threshold
        evaluator = (
            Evaluator()
            .with_judge(judge)
            .with_threshold(0.9)
            .add_check("toxicity")
        )
        result = evaluator.run("some text")
        assert result.valid is False

    def test_with_budget(self):
        evaluator = (
            Evaluator()
            .with_budget(5.0)
            .add_check("no_pii")
        )
        result = evaluator.run("hello")
        assert result.valid is True

    def test_with_config(self):
        cfg = CheckllmConfig(default_threshold=0.5)
        judge = MockJudge(default_score=0.6)
        evaluator = (
            Evaluator()
            .with_config(cfg)
            .with_judge(judge)
            .add_check("toxicity")
        )
        # 0.6 passes at 0.5 threshold from config
        result = evaluator.run("some text")
        assert result.valid is True

    def test_with_judge_string_backend(self):
        # When using a string backend, judge is lazily initialised by Guard.
        # We only test that deterministic checks work without an API key.
        evaluator = (
            Evaluator()
            .with_judge("openai", model="gpt-4o-mini")
            .add_check("no_pii")
        )
        result = evaluator.run("hello world")
        assert result.valid is True

    def test_add_check_with_params(self):
        evaluator = (
            Evaluator()
            .add_check("contains", substring="hello")
            .add_check("max_tokens", limit=100)
        )
        result = evaluator.run("hello there")
        assert result.valid is True

    def test_multiple_checks(self):
        judge = MockJudge(default_score=0.95)
        evaluator = (
            Evaluator()
            .with_judge(judge)
            .add_check("no_pii")
            .add_check("contains", substring="world")
            .add_check("max_tokens", limit=1000)
            .add_check("toxicity")
        )
        result = evaluator.run("hello world")
        assert result.valid is True
        assert len(result.results) == 4


# ---------------------------------------------------------------------------
# Evaluator async
# ---------------------------------------------------------------------------


class TestEvaluatorAsync:
    @pytest.mark.asyncio
    async def test_arun(self):
        judge = MockJudge(default_score=0.9)
        evaluator = (
            Evaluator()
            .with_judge(judge)
            .add_check("no_pii")
            .add_check("toxicity")
        )
        result = await evaluator.arun("hello world")
        assert result.valid is True
        assert len(result.results) == 2

    @pytest.mark.asyncio
    async def test_arun_failure(self):
        evaluator = (
            Evaluator()
            .add_check("contains", substring="missing_word")
        )
        result = await evaluator.arun("hello world")
        assert result.valid is False


# ---------------------------------------------------------------------------
# Evaluator batch_run
# ---------------------------------------------------------------------------


class TestEvaluatorBatchRun:
    def test_batch_run(self):
        judge = MockJudge(default_score=0.9)
        evaluator = (
            Evaluator()
            .with_judge(judge)
            .add_check("no_pii")
        )
        results = evaluator.batch_run([
            "Hello world",
            "Another safe output",
            "Third output here",
        ])
        assert len(results) == 3
        assert all(r.valid for r in results)
        assert all(isinstance(r, ValidationResult) for r in results)

    def test_batch_run_mixed_results(self):
        evaluator = (
            Evaluator()
            .add_check("no_pii")
        )
        results = evaluator.batch_run([
            "Hello world",
            "Email: user@example.com",
            "Safe text",
        ])
        assert len(results) == 3
        assert results[0].valid is True
        assert results[1].valid is False
        assert results[2].valid is True

    def test_batch_run_empty_list(self):
        evaluator = (
            Evaluator()
            .add_check("no_pii")
        )
        results = evaluator.batch_run([])
        assert results == []

    def test_batch_run_single_item(self):
        evaluator = (
            Evaluator()
            .add_check("contains", substring="yes")
        )
        results = evaluator.batch_run(["yes indeed"])
        assert len(results) == 1
        assert results[0].valid is True
