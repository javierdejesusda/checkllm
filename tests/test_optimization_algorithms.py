"""Tests for MIPROv2, COPRO, SIMBA optimizers and create_optimizer factory."""

from __future__ import annotations

from typing import Any

import pytest

from checkllm.optimize import (
    COPROOptimizer,
    MIPROv2Optimizer,
    OptimizationResult,
    PromptOptimizer,
    SIMBAOptimizer,
    create_optimizer,
)
from checkllm.testing import MockJudge


async def _dummy_metric(prompt: str, case: dict[str, Any]) -> float:
    """Simple metric that returns a fixed score based on prompt length."""
    return min(1.0, len(prompt) / 100.0)


async def _perfect_metric(prompt: str, case: dict[str, Any]) -> float:
    """Metric that always returns a perfect score."""
    return 1.0


async def _failing_metric(prompt: str, case: dict[str, Any]) -> float:
    """Metric that always returns a low score."""
    return 0.2


SAMPLE_TEST_CASES: list[dict[str, Any]] = [
    {"input": "What is Python?", "expected": "A programming language."},
    {"input": "Explain AI", "expected": "Artificial intelligence is..."},
    {"input": "Define LLM", "expected": "Large language model..."},
]


class TestMIPROv2Optimizer:
    """Tests for the MIPROv2 multi-prompt instruction proposal optimizer."""

    @pytest.fixture
    def judge(self):
        return MockJudge(default_score=0.85, default_reasoning="Improved prompt text")

    @pytest.fixture
    def optimizer(self, judge):
        return MIPROv2Optimizer(judge=judge)

    @pytest.mark.asyncio
    async def test_generates_candidate_instructions(self, optimizer):
        candidates = await optimizer._generate_candidates(
            "Summarize this document.", num_candidates=5
        )
        assert len(candidates) == 5
        for c in candidates:
            assert isinstance(c, str)
            assert len(c) > 0

    @pytest.mark.asyncio
    async def test_optimize_returns_valid_result(self, optimizer):
        result = await optimizer.optimize(
            prompt="Summarize this.",
            test_cases=SAMPLE_TEST_CASES,
            metric_fn=_dummy_metric,
            num_candidates=3,
        )
        assert isinstance(result, OptimizationResult)
        assert result.best_prompt
        assert 0.0 <= result.best_score <= 1.0
        assert result.variants_tested > 0
        assert len(result.history) > 0

    @pytest.mark.asyncio
    async def test_optimize_with_demonstrations(self, optimizer):
        result = await optimizer.optimize(
            prompt="Test prompt",
            test_cases=SAMPLE_TEST_CASES,
            metric_fn=_dummy_metric,
            num_candidates=2,
            max_demos=2,
        )
        assert isinstance(result, OptimizationResult)
        assert result.generations == 2

    @pytest.mark.asyncio
    async def test_optimize_no_demos(self, optimizer):
        result = await optimizer.optimize(
            prompt="Test prompt",
            test_cases=SAMPLE_TEST_CASES,
            metric_fn=_dummy_metric,
            num_candidates=2,
            max_demos=0,
        )
        assert isinstance(result, OptimizationResult)
        demo_variants = [v for v in result.history if v.generation == 2]
        assert len(demo_variants) == 0

    @pytest.mark.asyncio
    async def test_tracks_improvement(self, optimizer):
        result = await optimizer.optimize(
            prompt="x",
            test_cases=SAMPLE_TEST_CASES,
            metric_fn=_dummy_metric,
            num_candidates=3,
        )
        assert result.initial_score == pytest.approx(
            result.best_score - result.improvement, abs=1e-9
        )


class TestCOPROOptimizer:
    """Tests for the coordinated prompt optimizer."""

    @pytest.fixture
    def judge(self):
        return MockJudge(
            default_score=0.8,
            default_reasoning="Failure pattern: prompt lacks specificity",
        )

    @pytest.fixture
    def optimizer(self, judge):
        return COPROOptimizer(judge=judge)

    @pytest.mark.asyncio
    async def test_identifies_failure_patterns(self, optimizer):
        failures = [
            ({"input": "What is X?", "expected": "Y"}, 0.3),
            ({"input": "Explain Z", "expected": "W"}, 0.2),
        ]
        patterns = await optimizer._identify_failure_patterns("Test prompt", failures)
        assert isinstance(patterns, str)
        assert len(patterns) > 0

    @pytest.mark.asyncio
    async def test_optimize_returns_valid_result(self, optimizer):
        result = await optimizer.optimize(
            prompt="Evaluate this.",
            test_cases=SAMPLE_TEST_CASES,
            metric_fn=_dummy_metric,
            max_iterations=2,
        )
        assert isinstance(result, OptimizationResult)
        assert result.best_prompt
        assert result.variants_tested > 0
        assert result.generations == 2

    @pytest.mark.asyncio
    async def test_stops_when_no_failures(self, optimizer):
        result = await optimizer.optimize(
            prompt="Perfect prompt",
            test_cases=SAMPLE_TEST_CASES,
            metric_fn=_perfect_metric,
            max_iterations=5,
            failure_threshold=0.5,
        )
        assert isinstance(result, OptimizationResult)
        assert result.best_score == 1.0

    @pytest.mark.asyncio
    async def test_history_contains_metadata(self, optimizer):
        result = await optimizer.optimize(
            prompt="Short",
            test_cases=SAMPLE_TEST_CASES,
            metric_fn=_failing_metric,
            max_iterations=2,
        )
        improved_variants = [v for v in result.history if v.generation > 0]
        for v in improved_variants:
            assert "failure_patterns" in v.metadata

    @pytest.mark.asyncio
    async def test_improvement_field(self, optimizer):
        result = await optimizer.optimize(
            prompt="Test",
            test_cases=SAMPLE_TEST_CASES,
            metric_fn=_dummy_metric,
            max_iterations=2,
        )
        assert abs(result.improvement - (result.best_score - result.initial_score)) < 1e-9


class TestSIMBAOptimizer:
    """Tests for the similarity-based prompt adaptation optimizer."""

    @pytest.fixture
    def judge(self):
        return MockJudge(default_score=0.88, default_reasoning="Adapted prompt text")

    @pytest.fixture
    def optimizer(self, judge):
        return SIMBAOptimizer(judge=judge)

    def test_find_most_similar(self):
        pool = [
            "Summarize the document briefly",
            "Translate this text to Spanish",
            "Evaluate the factual accuracy of the response",
        ]
        result = SIMBAOptimizer._find_most_similar("Summarize the document", pool)
        assert result == "Summarize the document briefly"

    def test_find_most_similar_exact_match(self):
        pool = ["Hello world", "Goodbye world", "Hello world"]
        result = SIMBAOptimizer._find_most_similar("Hello world", pool)
        assert result == "Hello world"

    @pytest.mark.asyncio
    async def test_optimize_without_pool(self, optimizer):
        result = await optimizer.optimize(
            prompt="Evaluate quality",
            test_cases=SAMPLE_TEST_CASES,
            metric_fn=_dummy_metric,
            max_iterations=2,
        )
        assert isinstance(result, OptimizationResult)
        assert result.best_prompt
        assert result.variants_tested > 0

    @pytest.mark.asyncio
    async def test_optimize_with_pool(self, optimizer):
        pool = [
            "Rate the quality of this response on a scale of 1-10",
            "Assess the faithfulness of the answer to the context",
            "Determine if the output contains hallucinations",
        ]
        result = await optimizer.optimize(
            prompt="Evaluate quality",
            test_cases=SAMPLE_TEST_CASES,
            metric_fn=_dummy_metric,
            prompt_pool=pool,
            max_iterations=2,
        )
        assert isinstance(result, OptimizationResult)
        assert result.variants_tested >= 3

    @pytest.mark.asyncio
    async def test_optimize_pool_selects_best_match(self, optimizer):
        pool = [
            "x" * 200,
            "Evaluate the quality of a short response",
        ]
        result = await optimizer.optimize(
            prompt="short",
            test_cases=SAMPLE_TEST_CASES,
            metric_fn=_dummy_metric,
            prompt_pool=pool,
            max_iterations=1,
        )
        assert isinstance(result, OptimizationResult)
        assert result.best_score >= result.initial_score

    @pytest.mark.asyncio
    async def test_returns_optimization_result_type(self, optimizer):
        result = await optimizer.optimize(
            prompt="Test",
            test_cases=SAMPLE_TEST_CASES,
            metric_fn=_dummy_metric,
            max_iterations=1,
        )
        assert hasattr(result, "best_prompt")
        assert hasattr(result, "best_score")
        assert hasattr(result, "improvement")
        assert hasattr(result, "history")
        assert hasattr(result, "summary")


class TestCreateOptimizer:
    """Tests for the create_optimizer factory function."""

    @pytest.fixture
    def judge(self):
        return MockJudge()

    def test_create_genetic(self, judge):
        opt = create_optimizer("genetic", judge)
        assert isinstance(opt, PromptOptimizer)

    def test_create_mipro(self, judge):
        opt = create_optimizer("mipro", judge)
        assert isinstance(opt, MIPROv2Optimizer)

    def test_create_miprov2(self, judge):
        opt = create_optimizer("miprov2", judge)
        assert isinstance(opt, MIPROv2Optimizer)

    def test_create_copro(self, judge):
        opt = create_optimizer("copro", judge)
        assert isinstance(opt, COPROOptimizer)

    def test_create_simba(self, judge):
        opt = create_optimizer("simba", judge)
        assert isinstance(opt, SIMBAOptimizer)

    def test_case_insensitive(self, judge):
        assert isinstance(create_optimizer("GENETIC", judge), PromptOptimizer)
        assert isinstance(create_optimizer("MiPrO", judge), MIPROv2Optimizer)
        assert isinstance(create_optimizer("COPRO", judge), COPROOptimizer)
        assert isinstance(create_optimizer("Simba", judge), SIMBAOptimizer)

    def test_unknown_strategy_raises(self, judge):
        with pytest.raises(ValueError, match="Unknown optimization strategy"):
            create_optimizer("unknown_strategy", judge)

    def test_all_return_valid_types(self, judge):
        for name in ("genetic", "mipro", "copro", "simba"):
            opt = create_optimizer(name, judge)
            assert opt is not None
