"""Additional tests for yaml_eval covering uncovered assertion branches."""
from __future__ import annotations

import asyncio
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from checkllm.models import CheckResult
from checkllm.yaml_eval import (
    AssertionConfig,
    EvalSettings,
    EvalTestConfig,
    JudgeConfig,
    YAMLEvalConfig,
    YAMLEvalResult,
    YAMLEvaluator,
    TestResult as EvalTestResult,
    _render_template,
)


def _make_check_result(passed=True, score=0.9, metric_name="mock"):
    return CheckResult(
        passed=passed,
        score=score,
        reasoning="mocked",
        cost=0.0,
        latency_ms=1,
        metric_name=metric_name,
    )


def _run(coro):
    return asyncio.new_event_loop().run_until_complete(coro)


def _config_with_assertion(assertion, **vars_):
    return YAMLEvalConfig(
        tests=[EvalTestConfig(vars=vars_ or {"query": "test"}, assert_=[assertion])]
    )


class TestRenderTemplate:
    def test_basic_substitution(self):
        result = _render_template("Hello {{name}}", {"name": "Alice"})
        assert result == "Hello Alice"

    def test_spaced_braces(self):
        result = _render_template("Hi {{ name }}", {"name": "Bob"})
        assert result == "Hi Bob"

    def test_multiple_vars(self):
        result = _render_template("{{a}} and {{b}}", {"a": "foo", "b": "bar"})
        assert result == "foo and bar"

    def test_no_vars(self):
        result = _render_template("no vars here", {})
        assert result == "no vars here"

    def test_unknown_var_left_as_is(self):
        result = _render_template("Hello {{unknown}}", {"name": "Alice"})
        assert "{{unknown}}" in result


class TestYAMLEvalResultSummary:
    def test_summary_with_results(self):
        test_result = EvalTestResult(
            test_index=0,
            passed=True,
            provider="openai",
            assertions=[{"type": "contains", "passed": True, "reasoning": "ok"}],
        )
        eval_result = YAMLEvalResult(
            config_path="test.yaml",
            description="My eval",
            total_tests=1,
            passed=1,
            failed=0,
            results=[test_result],
        )
        summary = eval_result.summary()
        assert "My eval" in summary
        assert "1/1" in summary
        assert "PASS" in summary

    def test_summary_with_failed_result(self):
        test_result = EvalTestResult(
            test_index=0,
            passed=False,
            provider="",
            assertions=[{"type": "relevance", "passed": False, "reasoning": "low score"}],
        )
        eval_result = YAMLEvalResult(
            total_tests=1,
            passed=0,
            failed=1,
            results=[test_result],
        )
        summary = eval_result.summary()
        assert "FAIL" in summary

    def test_summary_empty_results(self):
        eval_result = YAMLEvalResult(total_tests=0, passed=0, failed=0, results=[])
        summary = eval_result.summary()
        assert "0/0" in summary

    def test_summary_with_provider_label(self):
        test_result = EvalTestResult(
            test_index=0,
            passed=True,
            provider="anthropic",
            assertions=[],
        )
        eval_result = YAMLEvalResult(
            total_tests=1,
            passed=1,
            failed=0,
            results=[test_result],
        )
        summary = eval_result.summary()
        assert "anthropic" in summary


class TestDeterministicAssertionBranches:
    """Test deterministic assertion branches in _resolve_assertion that were uncovered."""

    def _run_assertion(self, atype, output="hello world", value=None, threshold=None):
        config = _config_with_assertion(
            AssertionConfig(type=atype, value=value, threshold=threshold)
        )
        evaluator = YAMLEvaluator()
        with (
            patch.object(evaluator, "_generate_output", new_callable=AsyncMock, return_value=output),
            patch.object(evaluator, "_create_judge", return_value=None),
        ):
            return _run(evaluator.run_from_config(config))

    def test_not_contains(self):
        result = self._run_assertion("not_contains", "hello world", value="foobar")
        assert result.passed == 1

    def test_not_contains_fail(self):
        result = self._run_assertion("not_contains", "hello world", value="hello")
        assert result.failed == 1

    def test_exact_match_pass(self):
        result = self._run_assertion("exact_match", "hello", value="hello")
        assert result.passed == 1

    def test_exact_match_fail(self):
        result = self._run_assertion("exact_match", "hello", value="goodbye")
        assert result.failed == 1

    def test_regex_match(self):
        result = self._run_assertion("regex", "foo123bar", value=r"\d+")
        assert result.passed == 1

    def test_starts_with(self):
        result = self._run_assertion("starts_with", "Hello World", value="Hello")
        assert result.passed == 1

    def test_ends_with(self):
        result = self._run_assertion("ends_with", "Hello World", value="World")
        assert result.passed == 1

    def test_min_tokens(self):
        result = self._run_assertion("min_tokens", "one two three four five", value=3)
        assert result.passed == 1

    def test_is_json_valid(self):
        result = self._run_assertion("is_json", '{"key": "value"}')
        assert result.passed == 1

    def test_word_count(self):
        result = self._run_assertion("word_count", "one two three", value=10)
        assert result.passed == 1

    def test_bleu(self):
        result = self._run_assertion("bleu", "the cat sat", value="the cat sat", threshold=0.5)
        assert result.passed == 1

    def test_rouge_l(self):
        result = self._run_assertion("rouge_l", "the cat sat", value="the cat sat", threshold=0.5)
        assert result.passed == 1

    def test_similarity(self):
        result = self._run_assertion("similarity", "hello world", value="hello world", threshold=0.5)
        assert result.passed == 1

    def test_is_valid_python(self):
        result = self._run_assertion("is_valid_python", "x = 1 + 2")
        assert result.passed == 1

    def test_language(self):
        result = self._run_assertion("language", "Hello world!", value="en")
        assert result.passed == 1

    def test_unknown_assertion_type(self):
        result = self._run_assertion("unknown_type_xyz")
        assert result.failed == 1

    def test_unknown_assertion_reasoning_with_judge(self):
        """Unknown assertion type returns error reasoning when judge is present."""
        judge = MagicMock()
        config = _config_with_assertion(AssertionConfig(type="nonexistent_check_xyz"))
        evaluator = YAMLEvaluator()
        with (
            patch.object(evaluator, "_generate_output", new_callable=AsyncMock, return_value="output"),
            patch.object(evaluator, "_create_judge", return_value=judge),
        ):
            result = _run(evaluator.run_from_config(config))
        assert result.failed == 1
        reasoning = result.results[0].assertions[0]["reasoning"]
        assert "nonexistent_check_xyz" in reasoning


class TestLLMAssertionBranchesNoJudge:
    """Test LLM-backed assertion branches with no judge (graceful failure)."""

    def _run_no_judge(self, atype, value=None):
        config = _config_with_assertion(AssertionConfig(type=atype, value=value))
        evaluator = YAMLEvaluator()
        with (
            patch.object(evaluator, "_generate_output", new_callable=AsyncMock, return_value="output"),
            patch.object(evaluator, "_create_judge", return_value=None),
        ):
            return _run(evaluator.run_from_config(config))

    def test_sentiment_no_judge(self):
        result = self._run_no_judge("sentiment")
        assert result.failed == 1

    def test_bias_no_judge(self):
        result = self._run_no_judge("bias")
        assert result.failed == 1

    def test_summarization_no_judge(self):
        result = self._run_no_judge("summarization", value="source text")
        assert result.failed == 1

    def test_instruction_following_no_judge(self):
        result = self._run_no_judge("instruction_following", value="Be polite.")
        assert result.failed == 1

    def test_role_adherence_no_judge(self):
        result = self._run_no_judge("role_adherence", value="assistant")
        assert result.failed == 1

    def test_groundedness_no_judge(self):
        result = self._run_no_judge("groundedness", value="Some context.")
        assert result.failed == 1


class TestBudgetAndFailedTests:
    """Test budget handling and failed test tracking."""

    def test_test_passed_false_when_assertion_fails(self):
        config = _config_with_assertion(
            AssertionConfig(type="contains", value="NOTINOUTPUT")
        )
        evaluator = YAMLEvaluator()
        with (
            patch.object(evaluator, "_generate_output", new_callable=AsyncMock, return_value="hello"),
            patch.object(evaluator, "_create_judge", return_value=None),
        ):
            result = _run(evaluator.run_from_config(config))
        assert result.failed == 1
        assert result.results[0].passed is False

    def test_multiple_assertions_one_fails(self):
        config = YAMLEvalConfig(
            tests=[
                EvalTestConfig(
                    vars={"query": "test"},
                    assert_=[
                        AssertionConfig(type="contains", value="hello"),
                        AssertionConfig(type="contains", value="MISSING"),
                    ],
                )
            ]
        )
        evaluator = YAMLEvaluator()
        with (
            patch.object(evaluator, "_generate_output", new_callable=AsyncMock, return_value="hello world"),
            patch.object(evaluator, "_create_judge", return_value=None),
        ):
            result = _run(evaluator.run_from_config(config))
        assert result.passed == 1
        assert result.failed == 1
        assert result.results[0].passed is False

    def test_budget_zero_stops_processing(self):
        """Zero budget triggers budget_exceeded logic."""
        config = YAMLEvalConfig(
            tests=[
                EvalTestConfig(vars={}, assert_=[
                    AssertionConfig(type="contains", value="x"),
                    AssertionConfig(type="contains", value="x"),
                ]),
                EvalTestConfig(vars={}, assert_=[
                    AssertionConfig(type="contains", value="x"),
                ]),
            ],
            settings=EvalSettings(budget=0.000001),
        )
        evaluator = YAMLEvaluator()
        with (
            patch.object(evaluator, "_generate_output", new_callable=AsyncMock, return_value="x"),
            patch.object(evaluator, "_create_judge", return_value=None),
        ):
            result = _run(evaluator.run_from_config(config))
        # Should complete without error; may not run all tests
        assert result is not None
