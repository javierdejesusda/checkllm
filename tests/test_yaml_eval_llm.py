"""Tests for LLM-backed assertion branches and run_from_config in YAMLEvaluator."""

from __future__ import annotations

import asyncio
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
import yaml

from checkllm.models import CheckResult
from checkllm.yaml_eval import (
    AssertionConfig,
    EvalSettings,
    EvalTestConfig,
    JudgeConfig,
    YAMLEvalConfig,
    YAMLEvalResult,
    YAMLEvaluator,
)


def _make_check_result(passed: bool = True, score: float = 0.9) -> CheckResult:
    """Return a minimal passing CheckResult for use in mocks."""
    return CheckResult(
        passed=passed,
        score=score,
        reasoning="mocked",
        cost=0.001,
        latency_ms=10,
        metric_name="mocked",
    )


def _run(coro):
    """Run a coroutine synchronously using a fresh event loop."""
    return asyncio.new_event_loop().run_until_complete(coro)


def _make_judge() -> MagicMock:
    """Return a MagicMock suitable for use as a judge backend."""
    return MagicMock()


def _config_with_assertion(assertion: AssertionConfig, **vars_) -> YAMLEvalConfig:
    """Build a minimal YAMLEvalConfig with one test and one assertion."""
    return YAMLEvalConfig(
        tests=[
            EvalTestConfig(
                vars=vars_ or {"query": "test query"},
                assert_=[assertion],
            )
        ]
    )


class TestResolveAssertionNoJudge:
    """LLM assertions fail gracefully when no judge is configured."""

    def test_relevance_no_judge(self):
        config = _config_with_assertion(AssertionConfig(type="relevance"))
        evaluator = YAMLEvaluator()
        with (
            patch.object(evaluator, "_generate_output", new_callable=AsyncMock, return_value="out"),
            patch.object(evaluator, "_create_judge", return_value=None),
        ):
            result = _run(evaluator.run_from_config(config))
        assert result.failed == 1
        assert "judge required" in result.results[0].assertions[0]["reasoning"].lower()

    def test_hallucination_no_judge(self):
        config = _config_with_assertion(AssertionConfig(type="hallucination"))
        evaluator = YAMLEvaluator()
        with (
            patch.object(evaluator, "_generate_output", new_callable=AsyncMock, return_value="out"),
            patch.object(evaluator, "_create_judge", return_value=None),
        ):
            result = _run(evaluator.run_from_config(config))
        assert result.failed == 1

    def test_toxicity_no_judge(self):
        config = _config_with_assertion(AssertionConfig(type="toxicity"))
        evaluator = YAMLEvaluator()
        with (
            patch.object(evaluator, "_generate_output", new_callable=AsyncMock, return_value="out"),
            patch.object(evaluator, "_create_judge", return_value=None),
        ):
            result = _run(evaluator.run_from_config(config))
        assert result.failed == 1

    def test_rubric_no_judge(self):
        config = _config_with_assertion(AssertionConfig(type="rubric", value="Be helpful."))
        evaluator = YAMLEvaluator()
        with (
            patch.object(evaluator, "_generate_output", new_callable=AsyncMock, return_value="out"),
            patch.object(evaluator, "_create_judge", return_value=None),
        ):
            result = _run(evaluator.run_from_config(config))
        assert result.failed == 1

    def test_unknown_type_no_judge(self):
        config = _config_with_assertion(AssertionConfig(type="fancy_new_metric"))
        evaluator = YAMLEvaluator()
        with (
            patch.object(evaluator, "_generate_output", new_callable=AsyncMock, return_value="out"),
            patch.object(evaluator, "_create_judge", return_value=None),
        ):
            result = _run(evaluator.run_from_config(config))
        assert result.failed == 1
        assert "judge required" in result.results[0].assertions[0]["reasoning"].lower()


class TestResolveAssertionLLMBranches:
    """Each LLM-backed branch routes to the correct metric and passes the result through."""

    def _run_llm_assertion(
        self,
        assertion: AssertionConfig,
        mock_result: CheckResult,
        metric_path: str,
        vars_: dict | None = None,
    ) -> YAMLEvalResult:
        """Helper: run one assertion with the metric.evaluate patched."""
        config = _config_with_assertion(assertion, **(vars_ or {"query": "test query"}))
        evaluator = YAMLEvaluator()
        with (
            patch.object(evaluator, "_generate_output", new_callable=AsyncMock, return_value="test output"),
            patch.object(evaluator, "_create_judge", return_value=_make_judge()),
            patch(metric_path, new_callable=AsyncMock, return_value=mock_result),
        ):
            return _run(evaluator.run_from_config(config))

    def test_relevance_passes(self):
        result = self._run_llm_assertion(
            AssertionConfig(type="relevance", threshold=0.8),
            _make_check_result(passed=True, score=0.9),
            "checkllm.metrics.relevance.RelevanceMetric.evaluate",
        )
        assert result.passed == 1
        assert result.failed == 0

    def test_relevance_fails(self):
        result = self._run_llm_assertion(
            AssertionConfig(type="relevance", threshold=0.8),
            _make_check_result(passed=False, score=0.3),
            "checkllm.metrics.relevance.RelevanceMetric.evaluate",
        )
        assert result.failed == 1
        assert result.passed == 0

    def test_relevance_uses_query_var(self):
        """Verify that the 'query' variable is extracted and passed to the metric."""
        config = YAMLEvalConfig(
            tests=[
                EvalTestConfig(
                    vars={"query": "what is AI?"},
                    assert_=[AssertionConfig(type="relevance")],
                )
            ]
        )
        evaluator = YAMLEvaluator()
        captured_kwargs: list[dict] = []

        async def capture_evaluate(**kwargs):
            captured_kwargs.append(kwargs)
            return _make_check_result()

        with (
            patch.object(evaluator, "_generate_output", new_callable=AsyncMock, return_value="output"),
            patch.object(evaluator, "_create_judge", return_value=_make_judge()),
            patch(
                "checkllm.metrics.relevance.RelevanceMetric.evaluate",
                side_effect=capture_evaluate,
            ),
        ):
            _run(evaluator.run_from_config(config))

        assert len(captured_kwargs) == 1
        assert captured_kwargs[0]["query"] == "what is AI?"

    def test_hallucination_passes(self):
        result = self._run_llm_assertion(
            AssertionConfig(type="hallucination"),
            _make_check_result(passed=True, score=0.95),
            "checkllm.metrics.hallucination.HallucinationMetric.evaluate",
            vars_={"query": "q", "context": "some context"},
        )
        assert result.passed == 1

    def test_hallucination_uses_context_var(self):
        """Verify that 'context' variable feeds into HallucinationMetric."""
        config = YAMLEvalConfig(
            tests=[
                EvalTestConfig(
                    vars={"context": "The sky is blue."},
                    assert_=[AssertionConfig(type="hallucination")],
                )
            ]
        )
        evaluator = YAMLEvaluator()
        captured_kwargs: list[dict] = []

        async def capture_evaluate(**kwargs):
            captured_kwargs.append(kwargs)
            return _make_check_result()

        with (
            patch.object(evaluator, "_generate_output", new_callable=AsyncMock, return_value="output"),
            patch.object(evaluator, "_create_judge", return_value=_make_judge()),
            patch(
                "checkllm.metrics.hallucination.HallucinationMetric.evaluate",
                side_effect=capture_evaluate,
            ),
        ):
            _run(evaluator.run_from_config(config))

        assert captured_kwargs[0]["context"] == "The sky is blue."

    def test_toxicity_passes(self):
        result = self._run_llm_assertion(
            AssertionConfig(type="toxicity"),
            _make_check_result(passed=True, score=0.99),
            "checkllm.metrics.toxicity.ToxicityMetric.evaluate",
        )
        assert result.passed == 1

    def test_toxicity_fails(self):
        result = self._run_llm_assertion(
            AssertionConfig(type="toxicity"),
            _make_check_result(passed=False, score=0.1),
            "checkllm.metrics.toxicity.ToxicityMetric.evaluate",
        )
        assert result.failed == 1

    def test_fluency_passes(self):
        result = self._run_llm_assertion(
            AssertionConfig(type="fluency", threshold=0.7),
            _make_check_result(passed=True, score=0.85),
            "checkllm.metrics.fluency.FluencyMetric.evaluate",
        )
        assert result.passed == 1

    def test_coherence_passes(self):
        result = self._run_llm_assertion(
            AssertionConfig(type="coherence"),
            _make_check_result(passed=True, score=0.9),
            "checkllm.metrics.coherence.CoherenceMetric.evaluate",
        )
        assert result.passed == 1

    def test_correctness_uses_value(self):
        """Correctness should use assertion.value as the expected answer."""
        config = YAMLEvalConfig(
            tests=[
                EvalTestConfig(
                    vars={"query": "what is 2+2?"},
                    assert_=[AssertionConfig(type="correctness", value="4")],
                )
            ]
        )
        evaluator = YAMLEvaluator()
        captured_kwargs: list[dict] = []

        async def capture_evaluate(**kwargs):
            captured_kwargs.append(kwargs)
            return _make_check_result()

        with (
            patch.object(evaluator, "_generate_output", new_callable=AsyncMock, return_value="4"),
            patch.object(evaluator, "_create_judge", return_value=_make_judge()),
            patch(
                "checkllm.metrics.correctness.CorrectnessMetric.evaluate",
                side_effect=capture_evaluate,
            ),
        ):
            _run(evaluator.run_from_config(config))

        assert captured_kwargs[0]["expected"] == "4"

    def test_correctness_falls_back_to_expected_var(self):
        """When no assertion value, correctness uses the 'expected' var."""
        config = YAMLEvalConfig(
            tests=[
                EvalTestConfig(
                    vars={"expected": "forty-two"},
                    assert_=[AssertionConfig(type="correctness")],
                )
            ]
        )
        evaluator = YAMLEvaluator()
        captured_kwargs: list[dict] = []

        async def capture_evaluate(**kwargs):
            captured_kwargs.append(kwargs)
            return _make_check_result()

        with (
            patch.object(evaluator, "_generate_output", new_callable=AsyncMock, return_value="42"),
            patch.object(evaluator, "_create_judge", return_value=_make_judge()),
            patch(
                "checkllm.metrics.correctness.CorrectnessMetric.evaluate",
                side_effect=capture_evaluate,
            ),
        ):
            _run(evaluator.run_from_config(config))

        assert captured_kwargs[0]["expected"] == "forty-two"

    def test_faithfulness_passes(self):
        result = self._run_llm_assertion(
            AssertionConfig(type="faithfulness"),
            _make_check_result(passed=True, score=0.92),
            "checkllm.metrics.faithfulness.FaithfulnessMetric.evaluate",
            vars_={"context": "ctx", "query": "q"},
        )
        assert result.passed == 1

    def test_faithfulness_uses_context_and_query(self):
        config = YAMLEvalConfig(
            tests=[
                EvalTestConfig(
                    vars={"context": "doc text", "query": "my query"},
                    assert_=[AssertionConfig(type="faithfulness")],
                )
            ]
        )
        evaluator = YAMLEvaluator()
        captured_kwargs: list[dict] = []

        async def capture_evaluate(**kwargs):
            captured_kwargs.append(kwargs)
            return _make_check_result()

        with (
            patch.object(evaluator, "_generate_output", new_callable=AsyncMock, return_value="answer"),
            patch.object(evaluator, "_create_judge", return_value=_make_judge()),
            patch(
                "checkllm.metrics.faithfulness.FaithfulnessMetric.evaluate",
                side_effect=capture_evaluate,
            ),
        ):
            _run(evaluator.run_from_config(config))

        assert captured_kwargs[0]["context"] == "doc text"
        assert captured_kwargs[0]["query"] == "my query"

    def test_rubric_uses_assertion_value_as_criteria(self):
        config = YAMLEvalConfig(
            tests=[
                EvalTestConfig(
                    vars={"query": "q"},
                    assert_=[AssertionConfig(type="rubric", value="Must be concise.")],
                )
            ]
        )
        evaluator = YAMLEvaluator()
        captured_kwargs: list[dict] = []

        async def capture_evaluate(**kwargs):
            captured_kwargs.append(kwargs)
            return _make_check_result()

        with (
            patch.object(evaluator, "_generate_output", new_callable=AsyncMock, return_value="short"),
            patch.object(evaluator, "_create_judge", return_value=_make_judge()),
            patch(
                "checkllm.metrics.rubric.RubricMetric.evaluate",
                side_effect=capture_evaluate,
            ),
        ):
            _run(evaluator.run_from_config(config))

        assert captured_kwargs[0]["criteria"] == "Must be concise."

    def test_rubric_default_criteria_when_no_value(self):
        """When rubric has no value, a default criteria string is used."""
        config = YAMLEvalConfig(
            tests=[
                EvalTestConfig(
                    vars={"query": "q"},
                    assert_=[AssertionConfig(type="rubric")],
                )
            ]
        )
        evaluator = YAMLEvaluator()
        captured_kwargs: list[dict] = []

        async def capture_evaluate(**kwargs):
            captured_kwargs.append(kwargs)
            return _make_check_result()

        with (
            patch.object(evaluator, "_generate_output", new_callable=AsyncMock, return_value="answer"),
            patch.object(evaluator, "_create_judge", return_value=_make_judge()),
            patch(
                "checkllm.metrics.rubric.RubricMetric.evaluate",
                side_effect=capture_evaluate,
            ),
        ):
            _run(evaluator.run_from_config(config))

        assert len(captured_kwargs[0]["criteria"]) > 0

    def test_role_adherence_passes(self):
        result = self._run_llm_assertion(
            AssertionConfig(type="role_adherence", value="friendly assistant"),
            _make_check_result(passed=True, score=0.88),
            "checkllm.metrics.role_adherence.RoleAdherenceMetric.evaluate",
            vars_={"query": "q", "role": "friendly assistant"},
        )
        assert result.passed == 1

    def test_groundedness_passes(self):
        result = self._run_llm_assertion(
            AssertionConfig(type="groundedness", value="source text"),
            _make_check_result(passed=True, score=0.91),
            "checkllm.metrics.groundedness.GroundednessMetric.evaluate",
        )
        assert result.passed == 1

    def test_groundedness_uses_context_when_no_value(self):
        """groundedness falls back to context var when assertion.value is None."""
        config = YAMLEvalConfig(
            tests=[
                EvalTestConfig(
                    vars={"context": "background doc"},
                    assert_=[AssertionConfig(type="groundedness")],
                )
            ]
        )
        evaluator = YAMLEvaluator()
        captured_kwargs: list[dict] = []

        async def capture_evaluate(**kwargs):
            captured_kwargs.append(kwargs)
            return _make_check_result()

        with (
            patch.object(evaluator, "_generate_output", new_callable=AsyncMock, return_value="output"),
            patch.object(evaluator, "_create_judge", return_value=_make_judge()),
            patch(
                "checkllm.metrics.groundedness.GroundednessMetric.evaluate",
                side_effect=capture_evaluate,
            ),
        ):
            _run(evaluator.run_from_config(config))

        assert captured_kwargs[0]["sources"] == ["background doc"]

    def test_unknown_type_with_judge_fails(self):
        """Unknown assertion types fall through to the catch-all and report failure."""
        config = _config_with_assertion(AssertionConfig(type="totally_unknown_type"))
        evaluator = YAMLEvaluator()
        with (
            patch.object(evaluator, "_generate_output", new_callable=AsyncMock, return_value="out"),
            patch.object(evaluator, "_create_judge", return_value=_make_judge()),
        ):
            result = _run(evaluator.run_from_config(config))
        assert result.failed == 1
        assert "unknown" in result.results[0].assertions[0]["reasoning"].lower()

    def test_threshold_from_assertion_overrides_settings(self):
        """Assertion-level threshold overrides the settings default threshold."""
        config = YAMLEvalConfig(
            settings=EvalSettings(threshold=0.5),
            tests=[
                EvalTestConfig(
                    vars={"query": "q"},
                    assert_=[AssertionConfig(type="relevance", threshold=0.95)],
                )
            ],
        )
        evaluator = YAMLEvaluator()
        captured_init_args: list[dict] = []

        original_init = __import__(
            "checkllm.metrics.relevance", fromlist=["RelevanceMetric"]
        ).RelevanceMetric.__init__

        def mock_init(self, judge, threshold=0.8):
            captured_init_args.append({"threshold": threshold})
            self.judge = judge
            self.threshold = threshold
            self.system_prompt = ""

        with (
            patch.object(evaluator, "_generate_output", new_callable=AsyncMock, return_value="output"),
            patch.object(evaluator, "_create_judge", return_value=_make_judge()),
            patch("checkllm.metrics.relevance.RelevanceMetric.__init__", mock_init),
            patch(
                "checkllm.metrics.relevance.RelevanceMetric.evaluate",
                new_callable=AsyncMock,
                return_value=_make_check_result(),
            ),
        ):
            _run(evaluator.run_from_config(config))

        assert len(captured_init_args) == 1
        assert captured_init_args[0]["threshold"] == 0.95

    def test_threshold_falls_back_to_settings(self):
        """When assertion has no threshold, settings.threshold is used."""
        config = YAMLEvalConfig(
            settings=EvalSettings(threshold=0.6),
            tests=[
                EvalTestConfig(
                    vars={"query": "q"},
                    assert_=[AssertionConfig(type="fluency")],
                )
            ],
        )
        evaluator = YAMLEvaluator()
        captured_init_args: list[dict] = []

        def mock_init(self, judge, threshold=0.8):
            captured_init_args.append({"threshold": threshold})
            self.judge = judge
            self.threshold = threshold
            self.system_prompt = ""

        with (
            patch.object(evaluator, "_generate_output", new_callable=AsyncMock, return_value="output"),
            patch.object(evaluator, "_create_judge", return_value=_make_judge()),
            patch("checkllm.metrics.fluency.FluencyMetric.__init__", mock_init),
            patch(
                "checkllm.metrics.fluency.FluencyMetric.evaluate",
                new_callable=AsyncMock,
                return_value=_make_check_result(),
            ),
        ):
            _run(evaluator.run_from_config(config))

        assert captured_init_args[0]["threshold"] == 0.6

    def test_cost_accumulation_across_llm_assertions(self):
        """Total cost should sum costs returned from metric.evaluate calls."""
        mock_result = CheckResult(
            passed=True,
            score=0.9,
            reasoning="good",
            cost=0.005,
            latency_ms=10,
            metric_name="relevance",
        )
        config = YAMLEvalConfig(
            tests=[
                EvalTestConfig(
                    vars={"query": "q"},
                    assert_=[
                        AssertionConfig(type="relevance"),
                        AssertionConfig(type="fluency"),
                    ],
                )
            ]
        )
        evaluator = YAMLEvaluator()
        with (
            patch.object(evaluator, "_generate_output", new_callable=AsyncMock, return_value="output"),
            patch.object(evaluator, "_create_judge", return_value=_make_judge()),
            patch(
                "checkllm.metrics.relevance.RelevanceMetric.evaluate",
                new_callable=AsyncMock,
                return_value=mock_result,
            ),
            patch(
                "checkllm.metrics.fluency.FluencyMetric.evaluate",
                new_callable=AsyncMock,
                return_value=mock_result,
            ),
        ):
            result = _run(evaluator.run_from_config(config))

        assert abs(result.cost - 0.010) < 1e-9


class TestRunFromConfig:
    """Integration tests for run_from_config result structure."""

    def test_multiple_providers_multiple_tests(self):
        """Two providers × two tests = four test results."""
        config = YAMLEvalConfig(
            description="multi-provider run",
            prompts=["Answer: {{query}}"],
            providers=["openai:gpt-4o", "anthropic:claude-sonnet-4-6"],
            tests=[
                EvalTestConfig(
                    vars={"query": "q1"},
                    assert_=[AssertionConfig(type="contains", value="hello")],
                ),
                EvalTestConfig(
                    vars={"query": "q2"},
                    assert_=[AssertionConfig(type="contains", value="world")],
                ),
            ],
        )
        evaluator = YAMLEvaluator()

        async def mock_generate(prompt, provider_str, judge_config):
            if "q1" in prompt:
                return "hello there"
            return "world peace"

        with patch.object(evaluator, "_generate_output", side_effect=mock_generate):
            result = _run(evaluator.run_from_config(config))

        assert len(result.results) == 4
        assert result.total_tests == 4
        assert result.passed == 4
        assert result.failed == 0

    def test_result_has_in_memory_config_path(self):
        config = YAMLEvalConfig(tests=[])
        evaluator = YAMLEvaluator()
        with patch.object(evaluator, "_generate_output", new_callable=AsyncMock, return_value=""):
            result = _run(evaluator.run_from_config(config))
        assert result.config_path == "<in-memory>"

    def test_result_carries_description(self):
        config = YAMLEvalConfig(description="my eval suite", tests=[])
        evaluator = YAMLEvaluator()
        with patch.object(evaluator, "_generate_output", new_callable=AsyncMock, return_value=""):
            result = _run(evaluator.run_from_config(config))
        assert result.description == "my eval suite"

    def test_budget_stops_evaluation(self):
        """When cost exceeds budget, remaining assertions are skipped."""
        expensive_result = CheckResult(
            passed=True,
            score=0.9,
            reasoning="ok",
            cost=3.0,
            latency_ms=5,
            metric_name="relevance",
        )
        config = YAMLEvalConfig(
            settings=EvalSettings(budget=2.0),
            tests=[
                EvalTestConfig(
                    vars={"query": "q"},
                    assert_=[
                        AssertionConfig(type="relevance"),
                        AssertionConfig(type="relevance"),
                    ],
                ),
                EvalTestConfig(
                    vars={"query": "q2"},
                    assert_=[AssertionConfig(type="relevance")],
                ),
            ],
        )
        evaluator = YAMLEvaluator()
        with (
            patch.object(evaluator, "_generate_output", new_callable=AsyncMock, return_value="output"),
            patch.object(evaluator, "_create_judge", return_value=_make_judge()),
            patch(
                "checkllm.metrics.relevance.RelevanceMetric.evaluate",
                new_callable=AsyncMock,
                return_value=expensive_result,
            ),
        ):
            result = _run(evaluator.run_from_config(config))

        assert result.total_tests < 3

    def test_empty_tests_returns_zero_counts(self):
        config = YAMLEvalConfig(tests=[])
        evaluator = YAMLEvaluator()
        with patch.object(evaluator, "_generate_output", new_callable=AsyncMock, return_value=""):
            result = _run(evaluator.run_from_config(config))
        assert result.total_tests == 0
        assert result.passed == 0
        assert result.failed == 0
        assert result.results == []

    def test_per_test_result_fields_populated(self):
        config = YAMLEvalConfig(
            prompts=["Say: {{query}}"],
            providers=["openai:gpt-4o"],
            tests=[
                EvalTestConfig(
                    vars={"query": "hello"},
                    assert_=[AssertionConfig(type="contains", value="hello")],
                    description="greeting check",
                ),
            ],
        )
        evaluator = YAMLEvaluator()
        with patch.object(evaluator, "_generate_output", new_callable=AsyncMock, return_value="hello world"):
            result = _run(evaluator.run_from_config(config))

        tr = result.results[0]
        assert tr.test_index == 0
        assert tr.provider == "openai:gpt-4o"
        assert tr.prompt == "Say: hello"
        assert tr.output == "hello world"
        assert tr.vars == {"query": "hello"}
        assert tr.passed is True
        assert len(tr.assertions) == 1
        assert tr.assertions[0]["type"] == "contains"
        assert tr.assertions[0]["passed"] is True

    def test_duration_ms_populated(self):
        config = YAMLEvalConfig(tests=[])
        evaluator = YAMLEvaluator()
        with patch.object(evaluator, "_generate_output", new_callable=AsyncMock, return_value=""):
            result = _run(evaluator.run_from_config(config))
        assert result.duration_ms >= 0


class TestRunFileLoading:
    """Tests for run() which loads config from a YAML file."""

    def test_run_loads_yaml_and_executes(self, tmp_path):
        config_data = {
            "description": "file-based eval",
            "prompts": ["{{query}}"],
            "providers": ["default"],
            "tests": [
                {
                    "vars": {"query": "What is 2+2?"},
                    "assert": [
                        {"type": "contains", "value": "4"},
                    ],
                }
            ],
        }
        yaml_file = tmp_path / "eval.yaml"
        yaml_file.write_text(yaml.dump(config_data), encoding="utf-8")

        evaluator = YAMLEvaluator()
        with patch.object(evaluator, "_generate_output", new_callable=AsyncMock, return_value="The answer is 4."):
            result = _run(evaluator.run(str(yaml_file)))

        assert isinstance(result, YAMLEvalResult)
        assert result.description == "file-based eval"
        assert result.config_path == str(yaml_file)
        assert result.passed == 1
        assert result.failed == 0

    def test_run_raises_on_missing_file(self, tmp_path):
        evaluator = YAMLEvaluator()
        with pytest.raises(FileNotFoundError):
            _run(evaluator.run(str(tmp_path / "missing.yaml")))

    def test_run_multiple_assertions_from_file(self, tmp_path):
        config_data = {
            "tests": [
                {
                    "vars": {"query": "test"},
                    "assert": [
                        {"type": "contains", "value": "hello"},
                        {"type": "no_pii"},
                        {"type": "max_tokens", "value": 500},
                    ],
                }
            ],
        }
        yaml_file = tmp_path / "multi.yaml"
        yaml_file.write_text(yaml.dump(config_data), encoding="utf-8")

        evaluator = YAMLEvaluator()
        with patch.object(evaluator, "_generate_output", new_callable=AsyncMock, return_value="hello world"):
            result = _run(evaluator.run(str(yaml_file)))

        assert result.total_tests == 3
        assert result.passed == 3

    def test_run_with_llm_assertion_patched(self, tmp_path):
        config_data = {
            "judge": {"backend": "openai", "model": "gpt-4o"},
            "tests": [
                {
                    "vars": {"query": "Is this helpful?"},
                    "assert": [{"type": "relevance", "threshold": 0.7}],
                }
            ],
        }
        yaml_file = tmp_path / "llm_assert.yaml"
        yaml_file.write_text(yaml.dump(config_data), encoding="utf-8")

        evaluator = YAMLEvaluator()
        with (
            patch.object(evaluator, "_generate_output", new_callable=AsyncMock, return_value="yes it is"),
            patch.object(evaluator, "_create_judge", return_value=_make_judge()),
            patch(
                "checkllm.metrics.relevance.RelevanceMetric.evaluate",
                new_callable=AsyncMock,
                return_value=_make_check_result(passed=True, score=0.85),
            ),
        ):
            result = _run(evaluator.run(str(yaml_file)))

        assert result.passed == 1
        assert result.config_path == str(yaml_file)
