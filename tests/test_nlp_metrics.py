"""Tests for NLP metrics and new deterministic checks."""

from __future__ import annotations

import asyncio

import pytest

from checkllm.deterministic import DeterministicChecks


@pytest.fixture
def det() -> DeterministicChecks:
    return DeterministicChecks()


class TestMeteorScore:
    """Tests for METEOR score computation."""

    def test_identical_text(self, det: DeterministicChecks) -> None:
        result = det.meteor("the cat sat on the mat", "the cat sat on the mat")
        assert result.passed is True
        assert 0.0 <= result.score <= 1.0
        assert result.score > 0.5

    def test_different_text(self, det: DeterministicChecks) -> None:
        result = det.meteor("completely unrelated words here", "the cat sat on the mat")
        assert 0.0 <= result.score <= 1.0

    def test_empty_input(self, det: DeterministicChecks) -> None:
        result = det.meteor("", "the cat sat on the mat")
        assert result.passed is False
        assert result.score == 0.0

    def test_partial_overlap(self, det: DeterministicChecks) -> None:
        result = det.meteor("the cat sat on the mat", "the cat was on a mat")
        assert 0.0 <= result.score <= 1.0
        assert result.metric_name == "meteor"

    def test_threshold(self, det: DeterministicChecks) -> None:
        result = det.meteor("hello world", "hello world", threshold=0.9)
        assert result.score > 0.0


class TestGleuScore:
    """Tests for GLEU score computation."""

    def test_identical_text(self, det: DeterministicChecks) -> None:
        result = det.gleu("the quick brown fox", "the quick brown fox")
        assert result.passed is True
        assert 0.0 <= result.score <= 1.0
        assert result.score > 0.5

    def test_different_text(self, det: DeterministicChecks) -> None:
        result = det.gleu("completely unrelated", "the quick brown fox")
        assert 0.0 <= result.score <= 1.0
        assert result.score < 0.5

    def test_empty_input(self, det: DeterministicChecks) -> None:
        result = det.gleu("", "reference text")
        assert result.passed is False
        assert result.score == 0.0

    def test_partial_overlap(self, det: DeterministicChecks) -> None:
        result = det.gleu("the quick brown fox jumps", "the quick brown fox runs")
        assert 0.0 <= result.score <= 1.0
        assert result.metric_name == "gleu"


class TestChrfScore:
    """Tests for ChrF score computation."""

    def test_identical_text(self, det: DeterministicChecks) -> None:
        result = det.chrf("the quick brown fox", "the quick brown fox")
        assert result.passed is True
        assert 0.0 <= result.score <= 1.0
        assert result.score > 0.8

    def test_different_text(self, det: DeterministicChecks) -> None:
        result = det.chrf("xyz abc", "the quick brown fox")
        assert 0.0 <= result.score <= 1.0

    def test_empty_input(self, det: DeterministicChecks) -> None:
        result = det.chrf("", "reference text")
        assert result.passed is False
        assert result.score == 0.0

    def test_similar_characters(self, det: DeterministicChecks) -> None:
        result = det.chrf("hello world!", "hello world?")
        assert result.score > 0.7
        assert result.metric_name == "chrf"


class TestLatencyCheck:
    """Tests for latency check."""

    def test_within_limit(self, det: DeterministicChecks) -> None:
        start = 100.0
        end = 101.0
        result = det.latency_check(start, end, max_ms=2000.0)
        assert result.passed is True
        assert result.metric_name == "latency_check"

    def test_exceeds_limit(self, det: DeterministicChecks) -> None:
        start = 100.0
        end = 110.0
        result = det.latency_check(start, end, max_ms=5000.0)
        assert result.passed is False

    def test_exact_threshold(self, det: DeterministicChecks) -> None:
        start = 0.0
        end = 5.0
        result = det.latency_check(start, end, max_ms=5000.0)
        assert result.passed is True


class TestCostCheck:
    """Tests for cost check."""

    def test_low_cost_passes(self, det: DeterministicChecks) -> None:
        result = det.cost_check(100, 50, "gpt-4o-mini", max_cost=1.0)
        assert result.passed is True
        assert result.metric_name == "cost_check"

    def test_high_cost_fails(self, det: DeterministicChecks) -> None:
        result = det.cost_check(1_000_000, 500_000, "gpt-4", max_cost=0.01)
        assert result.passed is False

    def test_unknown_model_uses_default(self, det: DeterministicChecks) -> None:
        result = det.cost_check(100, 50, "unknown-model-xyz", max_cost=1.0)
        assert result.passed is True
        assert result.score > 0.0


class TestStringDistance:
    """Tests for multi-method string distance."""

    def test_levenshtein(self, det: DeterministicChecks) -> None:
        result = det.string_distance("hello", "hello", method="levenshtein")
        assert result.passed is True
        assert result.score == 1.0

    def test_hamming(self, det: DeterministicChecks) -> None:
        result = det.string_distance("hello", "hallo", method="hamming")
        assert result.passed is True
        assert 0.0 <= result.score <= 1.0

    def test_jaro(self, det: DeterministicChecks) -> None:
        result = det.string_distance("martha", "marhta", method="jaro")
        assert result.passed is True
        assert result.score > 0.9

    def test_jaro_winkler(self, det: DeterministicChecks) -> None:
        result = det.string_distance("martha", "marhta", method="jaro_winkler")
        assert result.passed is True
        assert result.score > 0.9

    def test_unknown_method(self, det: DeterministicChecks) -> None:
        result = det.string_distance("hello", "hello", method="unknown")
        assert result.passed is False

    def test_different_strings(self, det: DeterministicChecks) -> None:
        result = det.string_distance("abc", "xyz", method="levenshtein", threshold=0.9)
        assert result.passed is False
        assert result.metric_name == "string_distance"


class TestExactMatchStrict:
    """Tests for strict exact match with options."""

    def test_exact_match(self, det: DeterministicChecks) -> None:
        result = det.exact_match_strict("hello world", "hello world")
        assert result.passed is True

    def test_case_mismatch_fails(self, det: DeterministicChecks) -> None:
        result = det.exact_match_strict("Hello World", "hello world")
        assert result.passed is False

    def test_ignore_case(self, det: DeterministicChecks) -> None:
        result = det.exact_match_strict("Hello World", "hello world", ignore_case=True)
        assert result.passed is True

    def test_whitespace_mismatch_fails(self, det: DeterministicChecks) -> None:
        result = det.exact_match_strict("hello  world", "hello world")
        assert result.passed is False

    def test_ignore_whitespace(self, det: DeterministicChecks) -> None:
        result = det.exact_match_strict("hello  world", "hello world", ignore_whitespace=True)
        assert result.passed is True

    def test_both_options(self, det: DeterministicChecks) -> None:
        result = det.exact_match_strict(
            "Hello  World", "hello world", ignore_case=True, ignore_whitespace=True
        )
        assert result.passed is True
        assert result.metric_name == "exact_match_strict"


class TestNonLLMContextPrecision:
    """Tests for NonLLM context precision metric."""

    def test_all_relevant(self) -> None:
        from checkllm.metrics.nonllm_context_precision import (
            NonLLMContextPrecisionMetric,
        )

        metric = NonLLMContextPrecisionMetric(threshold=0.5)
        contexts = [
            "Python is a programming language created by Guido van Rossum.",
            "Python supports object-oriented programming.",
        ]
        reference = "Python is a programming language that supports object-oriented programming."
        result = asyncio.new_event_loop().run_until_complete(metric.evaluate(contexts, reference))
        assert 0.0 <= result.score <= 1.0
        assert result.score > 0.0
        assert result.metric_name == "nonllm_context_precision"

    def test_no_relevant(self) -> None:
        from checkllm.metrics.nonllm_context_precision import (
            NonLLMContextPrecisionMetric,
        )

        metric = NonLLMContextPrecisionMetric(threshold=0.5)
        contexts = [
            "The weather forecast for today is sunny.",
            "Basketball was invented in 1891.",
        ]
        reference = "Python is a programming language."
        result = asyncio.new_event_loop().run_until_complete(metric.evaluate(contexts, reference))
        assert result.score == 0.0

    def test_empty_contexts(self) -> None:
        from checkllm.metrics.nonllm_context_precision import (
            NonLLMContextPrecisionMetric,
        )

        metric = NonLLMContextPrecisionMetric()
        result = asyncio.new_event_loop().run_until_complete(metric.evaluate([], "some reference"))
        assert result.passed is False
        assert result.score == 0.0


class TestNonLLMContextRecall:
    """Tests for NonLLM context recall metric."""

    def test_full_recall(self) -> None:
        from checkllm.metrics.nonllm_context_recall import NonLLMContextRecallMetric

        metric = NonLLMContextRecallMetric(threshold=0.5, similarity_threshold=0.3)
        contexts = [
            "Python is a high-level programming language known for readability.",
            "It was created by Guido van Rossum and released in 1991.",
        ]
        reference = (
            "Python is a high-level programming language. It was created by Guido van Rossum."
        )
        result = asyncio.new_event_loop().run_until_complete(metric.evaluate(contexts, reference))
        assert 0.0 <= result.score <= 1.0
        assert result.score > 0.0
        assert result.metric_name == "nonllm_context_recall"

    def test_no_recall(self) -> None:
        from checkllm.metrics.nonllm_context_recall import NonLLMContextRecallMetric

        metric = NonLLMContextRecallMetric(threshold=0.5)
        contexts = [
            "The weather is sunny today.",
        ]
        reference = "Python is a programming language. It supports multiple paradigms."
        result = asyncio.new_event_loop().run_until_complete(metric.evaluate(contexts, reference))
        assert result.score < 0.5

    def test_empty_reference(self) -> None:
        from checkllm.metrics.nonllm_context_recall import NonLLMContextRecallMetric

        metric = NonLLMContextRecallMetric()
        result = asyncio.new_event_loop().run_until_complete(metric.evaluate(["some context"], ""))
        assert result.passed is False


class TestFaithfulnessHHEM:
    """Tests for FaithfulnessHHEM metric."""

    def test_faithful_response(self) -> None:
        from checkllm.metrics.faithfulness_hhem import FaithfulnessHHEMMetric

        metric = FaithfulnessHHEMMetric(threshold=0.5)
        context = (
            "Python is a high-level programming language created by Guido van Rossum. "
            "It was first released in 1991. Python supports multiple programming paradigms."
        )
        output = (
            "Python is a programming language created by Guido van Rossum. It was released in 1991."
        )
        result = asyncio.new_event_loop().run_until_complete(metric.evaluate(output, context))
        assert 0.0 <= result.score <= 1.0
        assert result.score > 0.0
        assert result.metric_name == "faithfulness_hhem"
        assert result.cost == 0.0

    def test_unfaithful_response(self) -> None:
        from checkllm.metrics.faithfulness_hhem import FaithfulnessHHEMMetric

        metric = FaithfulnessHHEMMetric(threshold=0.8)
        context = "The Earth orbits the Sun."
        output = (
            "Jupiter is the largest planet in the solar system. "
            "Mars has two moons called Phobos and Deimos."
        )
        result = asyncio.new_event_loop().run_until_complete(metric.evaluate(output, context))
        assert result.score < 0.8

    def test_empty_output(self) -> None:
        from checkllm.metrics.faithfulness_hhem import FaithfulnessHHEMMetric

        metric = FaithfulnessHHEMMetric()
        result = asyncio.new_event_loop().run_until_complete(metric.evaluate("", "some context"))
        assert result.passed is False
        assert result.score == 0.0


class TestQuotedSpansAlignment:
    """Tests for QuotedSpansAlignment metric."""

    def test_all_quotes_match(self) -> None:
        from checkllm.metrics.quoted_spans import QuotedSpansAlignmentMetric

        metric = QuotedSpansAlignmentMetric(threshold=0.8)
        contexts = [
            "The report states that climate change is accelerating rapidly.",
            "Scientists warn that sea levels could rise by 2 meters.",
        ]
        output = 'The report says "climate change is accelerating rapidly" and "sea levels could rise by 2 meters".'
        result = asyncio.new_event_loop().run_until_complete(metric.evaluate(output, contexts))
        assert result.score == 1.0
        assert result.passed is True
        assert result.metric_name == "quoted_spans_alignment"

    def test_no_quotes_in_output(self) -> None:
        from checkllm.metrics.quoted_spans import QuotedSpansAlignmentMetric

        metric = QuotedSpansAlignmentMetric()
        result = asyncio.new_event_loop().run_until_complete(
            metric.evaluate("No quotes here.", ["some context"])
        )
        assert result.passed is True
        assert result.score == 1.0

    def test_unmatched_quotes(self) -> None:
        from checkllm.metrics.quoted_spans import QuotedSpansAlignmentMetric

        metric = QuotedSpansAlignmentMetric(threshold=0.8)
        contexts = ["The sky is blue."]
        output = 'The source says "the grass is green" which is important.'
        result = asyncio.new_event_loop().run_until_complete(metric.evaluate(output, contexts))
        assert result.score == 0.0
        assert result.passed is False

    def test_empty_contexts(self) -> None:
        from checkllm.metrics.quoted_spans import QuotedSpansAlignmentMetric

        metric = QuotedSpansAlignmentMetric()
        result = asyncio.new_event_loop().run_until_complete(
            metric.evaluate('She said "hello world" today.', [])
        )
        assert result.passed is False


class TestDataCompyMetric:
    """Tests for DataCompy metric."""

    def test_identical_csv(self) -> None:
        from checkllm.metrics.datacompy_score import DataCompyMetric

        metric = DataCompyMetric(threshold=0.7)
        csv_data = "name,age\nAlice,30\nBob,25"
        result = asyncio.new_event_loop().run_until_complete(metric.evaluate(csv_data, csv_data))
        assert result.score == 1.0
        assert result.passed is True
        assert result.metric_name == "datacompy_score"

    def test_identical_json(self) -> None:
        from checkllm.metrics.datacompy_score import DataCompyMetric

        metric = DataCompyMetric(threshold=0.7)
        json_data = '[{"name": "Alice", "age": "30"}, {"name": "Bob", "age": "25"}]'
        result = asyncio.new_event_loop().run_until_complete(metric.evaluate(json_data, json_data))
        assert result.score == 1.0
        assert result.passed is True

    def test_mismatched_data(self) -> None:
        from checkllm.metrics.datacompy_score import DataCompyMetric

        metric = DataCompyMetric(threshold=0.9)
        output = "name,age\nAlice,30\nCharlie,40"
        reference = "name,age\nAlice,30\nBob,25"
        result = asyncio.new_event_loop().run_until_complete(metric.evaluate(output, reference))
        assert result.score < 1.0

    def test_invalid_format(self) -> None:
        from checkllm.metrics.datacompy_score import DataCompyMetric

        metric = DataCompyMetric()
        result = asyncio.new_event_loop().run_until_complete(
            metric.evaluate("[invalid json", "name,age\nAlice,30")
        )
        assert result.passed is False
        assert result.score == 0.0


class TestPerplexityCheck:
    """Tests for perplexity check."""

    def test_normal_text(self, det: DeterministicChecks) -> None:
        text = "The quick brown fox jumps over the lazy dog near the river bank."
        result = det.perplexity_check(text, max_perplexity=100.0)
        assert 0.0 <= result.score <= 1.0
        assert result.metric_name == "perplexity_check"

    def test_repetitive_text(self, det: DeterministicChecks) -> None:
        text = "the the the the the the the the the the"
        result = det.perplexity_check(text, max_perplexity=5.0)
        assert result.passed is False

    def test_empty_text(self, det: DeterministicChecks) -> None:
        result = det.perplexity_check("", max_perplexity=50.0)
        assert result.passed is True
