"""Prompt optimization engine using evolutionary/genetic search.

Uses an LLM to generate prompt variants and scores them against
configurable metric functions, selecting the best across generations.

Usage::

    from checkllm.optimize import PromptOptimizer, OptimizationResult
    from checkllm.judge import OpenAIJudge

    optimizer = PromptOptimizer(judge=OpenAIJudge())
    result = optimizer.optimize(
        prompt="Summarize this document.",
        objective="Improve clarity and completeness",
        generations=5,
        population_size=4,
    )
    print(result.summary())
"""

from __future__ import annotations

import asyncio
import concurrent.futures
import logging
from typing import Any, Callable, Awaitable

from pydantic import BaseModel, Field

from checkllm.judge import JudgeBackend

logger = logging.getLogger("checkllm.optimize")


class PromptVariant(BaseModel):
    """A prompt variant with its evaluation score."""

    prompt: str
    score: float = 0.0
    scores: dict[str, float] = Field(default_factory=dict)
    generation: int = 0
    metadata: dict[str, Any] = Field(default_factory=dict)


class OptimizationResult(BaseModel):
    """Result of a prompt optimization run."""

    best_prompt: str
    best_score: float
    initial_score: float
    improvement: float
    generations: int
    variants_tested: int
    history: list[PromptVariant] = Field(default_factory=list)

    def summary(self) -> str:
        """Return a human-readable summary of the optimization result."""
        lines = [
            "Prompt Optimization Result",
            f"  Initial score: {self.initial_score:.2f}",
            f"  Best score:    {self.best_score:.2f}",
            f"  Improvement:   {self.improvement:+.2f}",
            f"  Generations:   {self.generations}",
            f"  Variants tested: {self.variants_tested}",
        ]
        return "\n".join(lines)


class PromptOptimizer:
    """Genetic/evolutionary prompt optimizer.

    Generates prompt variants by asking an LLM to mutate the prompt,
    scores them against configurable metrics, and selects the best.

    Args:
        judge: The LLM judge backend used for mutation and scoring.
        metric_fn: Optional default scoring function. If None, uses the
            judge to score prompt quality directly.
    """

    def __init__(
        self,
        judge: JudgeBackend,
        metric_fn: Callable[[str], Awaitable[float]] | None = None,
    ) -> None:
        self.judge = judge
        self.metric_fn = metric_fn

    def optimize(
        self,
        prompt: str,
        objective: str = "Improve clarity, specificity, and effectiveness",
        generations: int = 5,
        population_size: int = 4,
        metric_fn: Callable[[str], Awaitable[float]] | None = None,
    ) -> OptimizationResult:
        """Run optimization synchronously.

        Args:
            prompt: The initial prompt to optimize.
            objective: Description of the optimization goal.
            generations: Number of evolutionary generations to run.
            population_size: Number of variants to keep each generation.
            metric_fn: Optional scoring function; overrides instance default.

        Returns:
            An OptimizationResult with the best prompt and score history.
        """
        try:
            loop = asyncio.get_running_loop()
        except RuntimeError:
            loop = None
        if loop and loop.is_running():
            with concurrent.futures.ThreadPoolExecutor() as pool:
                return pool.submit(
                    asyncio.run,
                    self.aoptimize(prompt, objective, generations, population_size, metric_fn),
                ).result()
        return asyncio.run(
            self.aoptimize(prompt, objective, generations, population_size, metric_fn)
        )

    async def aoptimize(
        self,
        prompt: str,
        objective: str = "Improve clarity, specificity, and effectiveness",
        generations: int = 5,
        population_size: int = 4,
        metric_fn: Callable[[str], Awaitable[float]] | None = None,
    ) -> OptimizationResult:
        """Run evolutionary prompt optimization asynchronously.

        Args:
            prompt: The initial prompt to optimize.
            objective: Description of the optimization goal.
            generations: Number of evolutionary generations to run.
            population_size: Number of variants to keep each generation.
            metric_fn: Optional scoring function; overrides instance default.

        Returns:
            An OptimizationResult with the best prompt and score history.
        """
        fn = metric_fn or self.metric_fn
        history: list[PromptVariant] = []

        initial_score = await self._score_prompt(prompt, fn)
        best = PromptVariant(prompt=prompt, score=initial_score, generation=0)
        history.append(best)

        population = [best]
        for gen in range(1, generations + 1):
            new_variants: list[PromptVariant] = []
            for parent in population[:max(1, population_size // 2)]:
                for _ in range(2):
                    mutated = await self._mutate_prompt(parent.prompt, objective)
                    score = await self._score_prompt(mutated, fn)
                    variant = PromptVariant(prompt=mutated, score=score, generation=gen)
                    new_variants.append(variant)
                    history.append(variant)

            combined = population + new_variants
            combined.sort(key=lambda v: v.score, reverse=True)
            population = combined[:population_size]

            if population[0].score > best.score:
                best = population[0]
                logger.info("Gen %d: new best score %.2f", gen, best.score)

        return OptimizationResult(
            best_prompt=best.prompt,
            best_score=best.score,
            initial_score=initial_score,
            improvement=best.score - initial_score,
            generations=generations,
            variants_tested=len(history),
            history=history,
        )

    async def _mutate_prompt(self, prompt: str, objective: str) -> str:
        """Ask the LLM to create a variant of the prompt.

        Args:
            prompt: The prompt to mutate.
            objective: The improvement objective to guide mutation.

        Returns:
            A mutated version of the prompt, or the original on failure.
        """
        mutation_prompt = (
            f"You are a prompt engineer. Your job is to improve prompts.\n\n"
            f"Objective: {objective}\n\n"
            f"Original prompt:\n{prompt}\n\n"
            f"Create an improved version of this prompt. "
            f"Make meaningful changes while preserving the core intent. "
            f"Output ONLY the improved prompt, nothing else."
        )
        response = await self.judge.evaluate(prompt=mutation_prompt)
        return response.reasoning.strip() or prompt

    async def _score_prompt(
        self,
        prompt: str,
        metric_fn: Callable[[str], Awaitable[float]] | None = None,
    ) -> float:
        """Score a prompt using the metric function or a default quality judge.

        Args:
            prompt: The prompt to score.
            metric_fn: Optional callable returning a float score in [0, 1].

        Returns:
            A float score in [0, 1] representing prompt quality.
        """
        if metric_fn:
            return await metric_fn(prompt)

        scoring_prompt = (
            f"Rate the quality of this prompt on a 0.0-1.0 scale.\n"
            f"Consider: clarity, specificity, effectiveness, completeness.\n\n"
            f"Prompt to evaluate:\n{prompt}\n\n"
            f'Respond with JSON: {{"score": <float>, "reasoning": "<explanation>"}}'
        )
        response = await self.judge.evaluate(prompt=scoring_prompt)
        return response.score
