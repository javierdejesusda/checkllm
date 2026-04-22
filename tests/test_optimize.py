from __future__ import annotations

from unittest.mock import AsyncMock

import pytest

from checkllm.optimize import OptimizationResult, PromptOptimizer, PromptVariant
from checkllm.models import JudgeResponse


class TestPromptOptimizer:
    def test_variant_model(self):
        v = PromptVariant(prompt="test", score=0.8, generation=1)
        assert v.prompt == "test"

    def test_result_model(self):
        r = OptimizationResult(
            best_prompt="improved",
            best_score=0.9,
            initial_score=0.5,
            improvement=0.4,
            generations=3,
            variants_tested=10,
        )
        assert r.improvement == 0.4
        text = r.summary()
        assert "0.90" in text
        assert "+0.40" in text

    @pytest.mark.asyncio
    async def test_optimize_improves_score(self):
        judge = AsyncMock()
        call_count = 0

        async def mock_evaluate(prompt, system_prompt=None):
            nonlocal call_count
            call_count += 1
            if "improved" in prompt.lower() or call_count > 1:
                return JudgeResponse(score=0.85, reasoning="Improved version", cost=0.001)
            return JudgeResponse(score=0.7, reasoning="Original", cost=0.001)

        judge.evaluate = mock_evaluate
        optimizer = PromptOptimizer(judge=judge)
        result = await optimizer.aoptimize(
            prompt="Write a poem",
            generations=2,
            population_size=2,
        )
        assert isinstance(result, OptimizationResult)
        assert result.best_score >= result.initial_score
        assert result.variants_tested >= 3

    @pytest.mark.asyncio
    async def test_custom_metric_fn(self):
        judge = AsyncMock()
        judge.evaluate.return_value = JudgeResponse(score=0.8, reasoning="Mutation", cost=0.001)

        async def custom_metric(prompt: str) -> float:
            return 0.9 if len(prompt) > 20 else 0.5

        optimizer = PromptOptimizer(judge=judge)
        result = await optimizer.aoptimize(
            prompt="Short",
            generations=1,
            population_size=2,
            metric_fn=custom_metric,
        )
        assert result.variants_tested >= 2

    def test_sync_run(self):
        judge = AsyncMock()
        judge.evaluate.return_value = JudgeResponse(score=0.8, reasoning="Test", cost=0.001)
        optimizer = PromptOptimizer(judge=judge)
        result = optimizer.optimize(
            prompt="Test prompt",
            generations=1,
            population_size=2,
        )
        assert isinstance(result, OptimizationResult)
