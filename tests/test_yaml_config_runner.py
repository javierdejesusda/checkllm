"""Tests for YamlEvalRunner in checkllm.yaml_config."""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock, patch


from checkllm.yaml_config import (
    AssertionConfig,
    EvalConfig,
    PromptConfig,
    ProviderConfig,
    TestConfig,
    YamlEvalRunner,
)
from checkllm.models import CheckResult


def _make_result(passed: bool = True, metric: str = "test") -> CheckResult:
    return CheckResult(
        passed=passed,
        score=0.9 if passed else 0.2,
        reasoning="ok",
        cost=0.001,
        latency_ms=100,
        metric_name=metric,
    )


def _runner_with_mock_generate(config: EvalConfig, output: str) -> YamlEvalRunner:
    runner = YamlEvalRunner(config)
    runner._generate = AsyncMock(return_value=output)
    return runner


# ---------------------------------------------------------------------------
# run() — basic integration
# ---------------------------------------------------------------------------


class TestYamlEvalRunnerRun:
    async def test_run_returns_one_result_per_combination(self):
        config = EvalConfig(
            providers=[ProviderConfig(id="openai", model="gpt-4o")],
            prompts=[PromptConfig(name="p1", template="{{input}}")],
            tests=[
                TestConfig(
                    input="hello",
                    assertions=[AssertionConfig(type="contains", value="hello")],
                )
            ],
        )
        runner = _runner_with_mock_generate(config, "hello world")
        with patch.object(runner, "_create_judge", return_value=MagicMock()):
            results = await runner.run()

        assert len(results) == 1
        assert results[0]["provider"] == "openai"
        assert results[0]["prompt"] == "p1"
        assert results[0]["passed"] is True
        assert results[0]["output"] == "hello world"

    async def test_run_cartesian_product_of_providers_prompts_tests(self):
        config = EvalConfig(
            providers=[
                ProviderConfig(id="openai", model="gpt-4o"),
                ProviderConfig(id="anthropic", model="claude-sonnet-4-6"),
            ],
            prompts=[
                PromptConfig(name="p1", template="{{input}}"),
                PromptConfig(name="p2", template="Q: {{input}}"),
            ],
            tests=[
                TestConfig(input="hello", assertions=[]),
                TestConfig(input="world", assertions=[]),
            ],
        )
        runner = _runner_with_mock_generate(config, "any output")
        with patch.object(runner, "_create_judge", return_value=MagicMock()):
            results = await runner.run()

        assert len(results) == 8  # 2 providers x 2 prompts x 2 tests

    async def test_run_no_prompts_uses_passthrough(self):
        config = EvalConfig(
            providers=[ProviderConfig(id="openai", model="gpt-4o")],
            tests=[
                TestConfig(
                    input="test question",
                    assertions=[AssertionConfig(type="contains", value="answer")],
                )
            ],
        )
        runner = _runner_with_mock_generate(config, "here is your answer")
        with patch.object(runner, "_create_judge", return_value=MagicMock()):
            results = await runner.run()

        assert len(results) == 1
        assert results[0]["prompt"] == "default"

    async def test_run_result_has_duration_ms(self):
        config = EvalConfig(
            providers=[ProviderConfig(id="openai")],
            tests=[TestConfig(input="hi", assertions=[])],
        )
        runner = _runner_with_mock_generate(config, "hello")
        with patch.object(runner, "_create_judge", return_value=MagicMock()):
            results = await runner.run()

        assert isinstance(results[0]["duration_ms"], int)
        assert results[0]["duration_ms"] >= 0

    async def test_run_fail_when_assertion_fails(self):
        config = EvalConfig(
            providers=[ProviderConfig(id="openai")],
            tests=[
                TestConfig(
                    input="hi",
                    assertions=[AssertionConfig(type="contains", value="MISSING_TEXT")],
                )
            ],
        )
        runner = _runner_with_mock_generate(config, "hello world")
        with patch.object(runner, "_create_judge", return_value=MagicMock()):
            results = await runner.run()

        assert results[0]["passed"] is False


# ---------------------------------------------------------------------------
# _run_assertion — deterministic branches
# ---------------------------------------------------------------------------


class TestRunAssertionDeterministic:
    def _runner(self) -> YamlEvalRunner:
        config = EvalConfig(providers=[], tests=[])
        return YamlEvalRunner(config)

    def _test(self, **kwargs) -> TestConfig:
        return TestConfig(**kwargs)

    async def test_contains_pass(self):
        runner = self._runner()
        result = await runner._run_assertion(
            AssertionConfig(type="contains", value="hello"),
            "hello world",
            self._test(),
            MagicMock(),
        )
        assert result.passed is True

    async def test_contains_fail(self):
        runner = self._runner()
        result = await runner._run_assertion(
            AssertionConfig(type="contains", value="MISSING"),
            "hello world",
            self._test(),
            MagicMock(),
        )
        assert result.passed is False

    async def test_not_contains_pass(self):
        runner = self._runner()
        result = await runner._run_assertion(
            AssertionConfig(type="not_contains", value="MISSING"),
            "hello world",
            self._test(),
            MagicMock(),
        )
        assert result.passed is True

    async def test_exact_match(self):
        runner = self._runner()
        result = await runner._run_assertion(
            AssertionConfig(type="exact_match", value="hello"),
            "hello",
            self._test(),
            MagicMock(),
        )
        assert result.passed is True

    async def test_regex_pass(self):
        runner = self._runner()
        result = await runner._run_assertion(
            AssertionConfig(type="regex", value=r"\d+"),
            "answer is 42",
            self._test(),
            MagicMock(),
        )
        assert result.passed is True

    async def test_starts_with(self):
        runner = self._runner()
        result = await runner._run_assertion(
            AssertionConfig(type="starts_with", value="Hello"),
            "Hello world",
            self._test(),
            MagicMock(),
        )
        assert result.passed is True

    async def test_ends_with(self):
        runner = self._runner()
        result = await runner._run_assertion(
            AssertionConfig(type="ends_with", value="world"),
            "Hello world",
            self._test(),
            MagicMock(),
        )
        assert result.passed is True

    async def test_max_tokens(self):
        runner = self._runner()
        result = await runner._run_assertion(
            AssertionConfig(type="max_tokens", value=100),
            "short answer",
            self._test(),
            MagicMock(),
        )
        assert result.passed is True

    async def test_min_tokens(self):
        runner = self._runner()
        result = await runner._run_assertion(
            AssertionConfig(type="min_tokens", value=1), "hi", self._test(), MagicMock()
        )
        assert result.passed is True

    async def test_word_count_pass(self):
        runner = self._runner()
        result = await runner._run_assertion(
            AssertionConfig(type="word_count", value=100),
            "one two three",
            self._test(),
            MagicMock(),
        )
        assert result.passed is True

    async def test_is_json_pass(self):
        runner = self._runner()
        result = await runner._run_assertion(
            AssertionConfig(type="is_json"),
            '{"key": "value"}',
            self._test(),
            MagicMock(),
        )
        assert result.passed is True

    async def test_is_json_fail(self):
        runner = self._runner()
        result = await runner._run_assertion(
            AssertionConfig(type="is_json"),
            "not json at all",
            self._test(),
            MagicMock(),
        )
        assert result.passed is False

    async def test_no_pii(self):
        runner = self._runner()
        result = await runner._run_assertion(
            AssertionConfig(type="no_pii"),
            "The answer is blue",
            self._test(),
            MagicMock(),
        )
        assert isinstance(result.passed, bool)

    async def test_bleu(self):
        runner = self._runner()
        result = await runner._run_assertion(
            AssertionConfig(type="bleu", value="The cat sat on the mat", threshold=0.1),
            "The cat sat on the mat",
            self._test(),
            MagicMock(),
        )
        assert result.passed is True

    async def test_rouge_l(self):
        runner = self._runner()
        result = await runner._run_assertion(
            AssertionConfig(type="rouge_l", value="hello world", threshold=0.1),
            "hello world",
            self._test(),
            MagicMock(),
        )
        assert result.passed is True

    async def test_similarity(self):
        runner = self._runner()
        result = await runner._run_assertion(
            AssertionConfig(type="similarity", value="hello world", threshold=0.5),
            "hello world",
            self._test(),
            MagicMock(),
        )
        assert isinstance(result.passed, bool)

    async def test_is_valid_python_pass(self):
        runner = self._runner()
        result = await runner._run_assertion(
            AssertionConfig(type="is_valid_python"),
            "x = 1 + 2",
            self._test(),
            MagicMock(),
        )
        assert result.passed is True

    async def test_is_valid_python_fail(self):
        runner = self._runner()
        result = await runner._run_assertion(
            AssertionConfig(type="is_valid_python"),
            "def broken(",
            self._test(),
            MagicMock(),
        )
        assert result.passed is False

    async def test_language(self):
        runner = self._runner()
        result = await runner._run_assertion(
            AssertionConfig(type="language", value="en"),
            "This is an English sentence about testing.",
            self._test(),
            MagicMock(),
        )
        assert isinstance(result.passed, bool)

    async def test_json_schema_falls_back_to_is_json(self):
        runner = self._runner()
        result = await runner._run_assertion(
            AssertionConfig(type="json_schema", value={"type": "object"}),
            '{"name": "Alice"}',
            self._test(),
            MagicMock(),
        )
        assert result.passed is True

    async def test_unknown_assertion_returns_failed_result(self):
        runner = self._runner()
        result = await runner._run_assertion(
            AssertionConfig(type="nonexistent_check"),
            "any output",
            self._test(),
            MagicMock(),
        )
        assert result.passed is False
        assert "nonexistent_check" in result.reasoning


# ---------------------------------------------------------------------------
# _run_assertion — LLM assertion branches (mocked)
# ---------------------------------------------------------------------------


class TestRunAssertionLlm:
    def _runner(self) -> YamlEvalRunner:
        return YamlEvalRunner(EvalConfig(providers=[], tests=[]))

    def _mock_result(self, passed: bool = True, metric: str = "test") -> CheckResult:
        return CheckResult(
            passed=passed,
            score=0.9 if passed else 0.3,
            reasoning="mocked",
            cost=0.002,
            latency_ms=200,
            metric_name=metric,
        )

    async def test_relevance_assertion(self):
        runner = self._runner()
        mock_result = self._mock_result(True)
        with patch(
            "checkllm.metrics.relevance.RelevanceMetric.evaluate",
            new_callable=AsyncMock,
            return_value=mock_result,
        ):
            result = await runner._run_assertion(
                AssertionConfig(type="relevance", threshold=0.8),
                "Paris is the capital of France.",
                TestConfig(input="What is the capital of France?"),
                MagicMock(),
            )
        assert result.passed is True

    async def test_hallucination_assertion(self):
        runner = self._runner()
        mock_result = self._mock_result(True)
        with patch(
            "checkllm.metrics.hallucination.HallucinationMetric.evaluate",
            new_callable=AsyncMock,
            return_value=mock_result,
        ):
            result = await runner._run_assertion(
                AssertionConfig(type="hallucination", threshold=0.8),
                "The sky is blue.",
                TestConfig(context="Observations about sky color."),
                MagicMock(),
            )
        assert result.passed is True

    async def test_toxicity_assertion(self):
        runner = self._runner()
        mock_result = self._mock_result(True)
        with patch(
            "checkllm.metrics.toxicity.ToxicityMetric.evaluate",
            new_callable=AsyncMock,
            return_value=mock_result,
        ):
            result = await runner._run_assertion(
                AssertionConfig(type="toxicity", threshold=0.8),
                "Have a great day!",
                TestConfig(),
                MagicMock(),
            )
        assert result.passed is True

    async def test_fluency_assertion(self):
        runner = self._runner()
        mock_result = self._mock_result(True)
        with patch(
            "checkllm.metrics.fluency.FluencyMetric.evaluate",
            new_callable=AsyncMock,
            return_value=mock_result,
        ):
            result = await runner._run_assertion(
                AssertionConfig(type="fluency", threshold=0.8),
                "This is fluently written prose.",
                TestConfig(),
                MagicMock(),
            )
        assert result.passed is True

    async def test_coherence_assertion(self):
        runner = self._runner()
        mock_result = self._mock_result(True)
        with patch(
            "checkllm.metrics.coherence.CoherenceMetric.evaluate",
            new_callable=AsyncMock,
            return_value=mock_result,
        ):
            result = await runner._run_assertion(
                AssertionConfig(type="coherence", threshold=0.8),
                "The argument is coherent and logical.",
                TestConfig(),
                MagicMock(),
            )
        assert result.passed is True

    async def test_correctness_assertion_uses_expected(self):
        runner = self._runner()
        mock_result = self._mock_result(True)
        with patch(
            "checkllm.metrics.correctness.CorrectnessMetric.evaluate",
            new_callable=AsyncMock,
            return_value=mock_result,
        ):
            result = await runner._run_assertion(
                AssertionConfig(type="correctness", threshold=0.8),
                "4",
                TestConfig(expected="4"),
                MagicMock(),
            )
        assert result.passed is True

    async def test_faithfulness_assertion(self):
        runner = self._runner()
        mock_result = self._mock_result(True)
        with patch(
            "checkllm.metrics.faithfulness.FaithfulnessMetric.evaluate",
            new_callable=AsyncMock,
            return_value=mock_result,
        ):
            result = await runner._run_assertion(
                AssertionConfig(type="faithfulness", threshold=0.8),
                "The study found X.",
                TestConfig(context="Study results: X was observed."),
                MagicMock(),
            )
        assert result.passed is True

    async def test_rubric_assertion(self):
        runner = self._runner()
        mock_result = self._mock_result(True)
        with patch(
            "checkllm.metrics.rubric.RubricMetric.evaluate",
            new_callable=AsyncMock,
            return_value=mock_result,
        ):
            result = await runner._run_assertion(
                AssertionConfig(type="rubric", value="Must be concise and clear."),
                "Short answer.",
                TestConfig(),
                MagicMock(),
            )
        assert result.passed is True

    async def test_sentiment_assertion(self):
        runner = self._runner()
        mock_result = self._mock_result(True)
        with patch(
            "checkllm.metrics.sentiment.SentimentMetric.evaluate",
            new_callable=AsyncMock,
            return_value=mock_result,
        ):
            result = await runner._run_assertion(
                AssertionConfig(type="sentiment"),
                "This is wonderful!",
                TestConfig(),
                MagicMock(),
            )
        assert result.passed is True

    async def test_bias_assertion(self):
        runner = self._runner()
        mock_result = self._mock_result(True)
        with patch(
            "checkllm.metrics.bias.BiasMetric.evaluate",
            new_callable=AsyncMock,
            return_value=mock_result,
        ):
            result = await runner._run_assertion(
                AssertionConfig(type="bias"),
                "All people deserve equal treatment.",
                TestConfig(),
                MagicMock(),
            )
        assert result.passed is True

    async def test_summarization_assertion(self):
        runner = self._runner()
        mock_result = self._mock_result(True)
        with patch(
            "checkllm.metrics.summarization.SummarizationMetric.evaluate",
            new_callable=AsyncMock,
            return_value=mock_result,
        ):
            result = await runner._run_assertion(
                AssertionConfig(type="summarization"),
                "The report showed key findings.",
                TestConfig(context="Long report text about findings and results."),
                MagicMock(),
            )
        assert result.passed is True

    async def test_instruction_following_assertion(self):
        runner = self._runner()
        mock_result = self._mock_result(True)
        with patch(
            "checkllm.metrics.instruction_following.InstructionFollowingMetric.evaluate",
            new_callable=AsyncMock,
            return_value=mock_result,
        ):
            result = await runner._run_assertion(
                AssertionConfig(type="instruction_following"),
                "Here is a numbered list: 1. first 2. second",
                TestConfig(input="Give me a numbered list"),
                MagicMock(),
            )
        assert result.passed is True

    async def test_groundedness_assertion_uses_value_as_source(self):
        runner = self._runner()
        mock_result = self._mock_result(True)
        with patch(
            "checkllm.metrics.groundedness.GroundednessMetric.evaluate",
            new_callable=AsyncMock,
            return_value=mock_result,
        ):
            result = await runner._run_assertion(
                AssertionConfig(type="groundedness", value="Source document text here."),
                "The answer based on source.",
                TestConfig(),
                MagicMock(),
            )
        assert result.passed is True

    async def test_groundedness_assertion_uses_list_value(self):
        runner = self._runner()
        mock_result = self._mock_result(True)
        with patch(
            "checkllm.metrics.groundedness.GroundednessMetric.evaluate",
            new_callable=AsyncMock,
            return_value=mock_result,
        ):
            result = await runner._run_assertion(
                AssertionConfig(type="groundedness", value=["source1", "source2"]),
                "Combined answer.",
                TestConfig(),
                MagicMock(),
            )
        assert result.passed is True

    async def test_consistency_assertion(self):
        runner = self._runner()
        mock_result = self._mock_result(True)
        with patch(
            "checkllm.metrics.consistency.ConsistencyMetric.evaluate",
            new_callable=AsyncMock,
            return_value=mock_result,
        ):
            result = await runner._run_assertion(
                AssertionConfig(type="consistency"),
                "Consistent answer.",
                TestConfig(),
                MagicMock(),
            )
        assert result.passed is True

    async def test_answer_completeness_assertion(self):
        runner = self._runner()
        mock_result = self._mock_result(True)
        with patch(
            "checkllm.metrics.answer_completeness.AnswerCompletenessMetric.evaluate",
            new_callable=AsyncMock,
            return_value=mock_result,
        ):
            result = await runner._run_assertion(
                AssertionConfig(type="answer_completeness"),
                "Complete answer covering all points.",
                TestConfig(query="What are the main points?"),
                MagicMock(),
            )
        assert result.passed is True

    async def test_context_relevance_assertion(self):
        runner = self._runner()
        mock_result = self._mock_result(True)
        with patch(
            "checkllm.metrics.context_relevance.ContextRelevanceMetric.evaluate",
            new_callable=AsyncMock,
            return_value=mock_result,
        ):
            result = await runner._run_assertion(
                AssertionConfig(type="context_relevance"),
                "output",
                TestConfig(context="relevant context", query="user query"),
                MagicMock(),
            )
        assert result.passed is True


# ---------------------------------------------------------------------------
# _generate — error path
# ---------------------------------------------------------------------------


class TestRunnerGenerate:
    async def test_generate_returns_error_string_on_openai_failure(self):
        config = EvalConfig(providers=[], tests=[])
        runner = YamlEvalRunner(config)
        provider = ProviderConfig(id="openai", model="gpt-4o")

        with patch("openai.AsyncOpenAI") as mock_client_cls:
            mock_client_cls.return_value.chat.completions.create = AsyncMock(
                side_effect=Exception("API error")
            )
            result = await runner._generate("test prompt", provider)

        assert isinstance(result, str)
        assert "Generation failed" in result or "error" in result.lower()

    async def test_generate_anthropic_missing_package(self):
        config = EvalConfig(providers=[], tests=[])
        runner = YamlEvalRunner(config)
        provider = ProviderConfig(id="anthropic", model="claude-sonnet-4-6")

        import sys

        original = sys.modules.get("anthropic")
        sys.modules["anthropic"] = None  # type: ignore[assignment]
        try:
            result = await runner._generate("test", provider)
        finally:
            if original is not None:
                sys.modules["anthropic"] = original
            else:
                del sys.modules["anthropic"]

        assert isinstance(result, str)
        assert "not installed" in result


# ---------------------------------------------------------------------------
# _create_judge — fallback path
# ---------------------------------------------------------------------------


class TestRunnerCreateJudge:
    def test_create_judge_falls_back_on_unknown_provider(self):
        config = EvalConfig(providers=[], tests=[])
        runner = YamlEvalRunner(config)
        provider = ProviderConfig(id="unknown_backend_xyz", model="some-model")

        with (
            patch("checkllm.providers.create_judge", side_effect=ValueError("unknown")),
            patch("checkllm.judge.OpenAIJudge") as mock_judge_cls,
        ):
            mock_judge_cls.return_value = MagicMock()
            judge = runner._create_judge(provider)

        assert judge is not None
