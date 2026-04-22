"""Prompt optimization engine with multiple strategies.

Provides genetic/evolutionary search (``PromptOptimizer``), multi-prompt
instruction proposal optimization (``MIPROv2Optimizer``), coordinated
prompt optimization (``COPROOptimizer``), and similarity-based prompt
adaptation (``SIMBAOptimizer``).

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

    # Or use the factory:
    from checkllm.optimize import create_optimizer
    opt = create_optimizer("mipro", judge=OpenAIJudge())
"""

from __future__ import annotations

import asyncio
import concurrent.futures
import json
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
            for parent in population[: max(1, population_size // 2)]:
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


class MIPROv2Optimizer:
    """Multi-prompt Instruction PRoposal Optimizer v2.

    Two-phase optimization:

    1. **Instruction optimization** -- Generate *N* candidate instructions,
       evaluate each against the test cases, and select the best.
    2. **Demonstration optimization** -- Select the best few-shot examples
       from a pool using score-based ranking.

    Args:
        judge: The LLM judge backend used for generating candidates and
            scoring.

    Usage::

        optimizer = MIPROv2Optimizer(judge=judge)
        result = await optimizer.optimize(
            prompt="Summarize this document.",
            test_cases=cases,
            metric_fn=my_metric,
            num_candidates=10,
        )
    """

    def __init__(self, judge: JudgeBackend) -> None:
        self.judge = judge

    async def optimize(
        self,
        prompt: str,
        test_cases: list[dict[str, Any]],
        metric_fn: Callable[[str, dict[str, Any]], Awaitable[float]],
        num_candidates: int = 10,
        max_demos: int = 3,
    ) -> OptimizationResult:
        """Run two-phase instruction + demonstration optimization.

        Args:
            prompt: The initial prompt/instruction to optimize.
            test_cases: List of test case dicts used for evaluation.
            metric_fn: Async callable ``(prompt, case) -> float`` that
                scores a prompt against a single test case.
            num_candidates: Number of candidate instructions to generate
                in phase 1.
            max_demos: Maximum number of few-shot demonstrations to
                select in phase 2.

        Returns:
            An OptimizationResult with the best prompt found.
        """
        history: list[PromptVariant] = []

        initial_score = await self._evaluate_prompt(prompt, test_cases, metric_fn)
        history.append(PromptVariant(prompt=prompt, score=initial_score, generation=0))

        candidates = await self._generate_candidates(prompt, num_candidates)

        best_prompt = prompt
        best_score = initial_score

        for idx, candidate in enumerate(candidates, start=1):
            score = await self._evaluate_prompt(candidate, test_cases, metric_fn)
            variant = PromptVariant(prompt=candidate, score=score, generation=1)
            history.append(variant)
            if score > best_score:
                best_score = score
                best_prompt = candidate
                logger.info("MIPROv2 candidate %d: new best score %.2f", idx, best_score)

        if max_demos > 0 and test_cases:
            demo_prompt = await self._optimize_demonstrations(
                best_prompt, test_cases, metric_fn, max_demos
            )
            demo_score = await self._evaluate_prompt(demo_prompt, test_cases, metric_fn)
            history.append(PromptVariant(prompt=demo_prompt, score=demo_score, generation=2))
            if demo_score > best_score:
                best_score = demo_score
                best_prompt = demo_prompt
                logger.info("MIPROv2 demo phase: new best score %.2f", best_score)

        return OptimizationResult(
            best_prompt=best_prompt,
            best_score=best_score,
            initial_score=initial_score,
            improvement=best_score - initial_score,
            generations=2,
            variants_tested=len(history),
            history=history,
        )

    async def _generate_candidates(self, prompt: str, num_candidates: int) -> list[str]:
        """Generate candidate instruction variants via the LLM.

        Args:
            prompt: The seed prompt.
            num_candidates: How many candidates to produce.

        Returns:
            List of candidate prompt strings.
        """
        candidates: list[str] = []
        for i in range(num_candidates):
            gen_prompt = (
                f"You are a prompt engineering expert. Generate a new, improved "
                f"version of the following instruction. Variation #{i + 1} of "
                f"{num_candidates}. Make it meaningfully different while "
                f"preserving the core intent.\n\n"
                f"Original instruction:\n{prompt}\n\n"
                f"Output ONLY the new instruction, nothing else."
            )
            response = await self.judge.evaluate(prompt=gen_prompt)
            result = response.reasoning.strip()
            if result:
                candidates.append(result)
            else:
                candidates.append(prompt)
        return candidates

    async def _evaluate_prompt(
        self,
        prompt: str,
        test_cases: list[dict[str, Any]],
        metric_fn: Callable[[str, dict[str, Any]], Awaitable[float]],
    ) -> float:
        """Score a prompt against all test cases and return the mean.

        Args:
            prompt: The prompt to evaluate.
            test_cases: List of test case dicts.
            metric_fn: Scoring function.

        Returns:
            Mean score across all test cases.
        """
        if not test_cases:
            return 0.0
        scores = [await metric_fn(prompt, tc) for tc in test_cases]
        return sum(scores) / len(scores)

    async def _optimize_demonstrations(
        self,
        prompt: str,
        test_cases: list[dict[str, Any]],
        metric_fn: Callable[[str, dict[str, Any]], Awaitable[float]],
        max_demos: int,
    ) -> str:
        """Append the best-scoring examples as few-shot demonstrations.

        Args:
            prompt: The current best instruction.
            test_cases: Pool of examples to choose from.
            metric_fn: Scoring function.
            max_demos: Maximum demos to append.

        Returns:
            Prompt augmented with selected demonstrations.
        """
        scored: list[tuple[float, dict[str, Any]]] = []
        for tc in test_cases:
            score = await metric_fn(prompt, tc)
            scored.append((score, tc))
        scored.sort(key=lambda x: x[0], reverse=True)

        demos = scored[:max_demos]
        if not demos:
            return prompt

        demo_text_parts: list[str] = []
        for i, (score, tc) in enumerate(demos, start=1):
            tc_str = json.dumps(tc, ensure_ascii=False, default=str)
            demo_text_parts.append(f"Example {i}: {tc_str}")

        demo_section = "\n".join(demo_text_parts)
        return f"{prompt}\n\nExamples:\n{demo_section}"


class COPROOptimizer:
    """Coordinated PRompt Optimization.

    Iteratively refines a prompt by:

    1. Running the prompt on test cases.
    2. Identifying failure patterns from low-scoring cases.
    3. Generating targeted improvements for those failures.
    4. Validating improvements do not regress passing cases.

    Args:
        judge: The LLM judge backend used for analysis and generation.

    Usage::

        optimizer = COPROOptimizer(judge=judge)
        result = await optimizer.optimize(
            prompt="...",
            test_cases=cases,
            metric_fn=my_metric,
            max_iterations=5,
        )
    """

    def __init__(self, judge: JudgeBackend) -> None:
        self.judge = judge

    async def optimize(
        self,
        prompt: str,
        test_cases: list[dict[str, Any]],
        metric_fn: Callable[[str, dict[str, Any]], Awaitable[float]],
        max_iterations: int = 5,
        failure_threshold: float = 0.5,
    ) -> OptimizationResult:
        """Run iterative failure-driven prompt optimization.

        Args:
            prompt: The initial prompt to optimize.
            test_cases: List of test case dicts.
            metric_fn: Async callable ``(prompt, case) -> float``.
            max_iterations: Maximum optimization iterations.
            failure_threshold: Score below which a case is considered a
                failure.

        Returns:
            An OptimizationResult with the best prompt found.
        """
        history: list[PromptVariant] = []

        initial_score = await self._mean_score(prompt, test_cases, metric_fn)
        history.append(PromptVariant(prompt=prompt, score=initial_score, generation=0))

        best_prompt = prompt
        best_score = initial_score

        for iteration in range(1, max_iterations + 1):
            case_scores = await self._score_all(best_prompt, test_cases, metric_fn)

            failures = [
                (tc, sc) for tc, sc in zip(test_cases, case_scores) if sc < failure_threshold
            ]

            if not failures:
                logger.info("COPRO iteration %d: no failures, stopping", iteration)
                break

            failure_patterns = await self._identify_failure_patterns(best_prompt, failures)

            improved = await self._generate_improvement(best_prompt, failure_patterns)
            improved_score = await self._mean_score(improved, test_cases, metric_fn)
            history.append(
                PromptVariant(
                    prompt=improved,
                    score=improved_score,
                    generation=iteration,
                    metadata={"failure_patterns": failure_patterns},
                )
            )

            if improved_score > best_score:
                passing_before = [
                    (tc, sc) for tc, sc in zip(test_cases, case_scores) if sc >= failure_threshold
                ]
                regression = False
                if passing_before:
                    new_passing_scores = await self._score_all(
                        improved,
                        [tc for tc, _ in passing_before],
                        metric_fn,
                    )
                    old_mean = sum(sc for _, sc in passing_before) / len(passing_before)
                    new_mean = sum(new_passing_scores) / len(new_passing_scores)
                    if new_mean < old_mean - 0.05:
                        regression = True
                        logger.info(
                            "COPRO iteration %d: regression detected "
                            "(%.2f -> %.2f on passing cases)",
                            iteration,
                            old_mean,
                            new_mean,
                        )

                if not regression:
                    best_score = improved_score
                    best_prompt = improved
                    logger.info(
                        "COPRO iteration %d: new best score %.2f",
                        iteration,
                        best_score,
                    )

        return OptimizationResult(
            best_prompt=best_prompt,
            best_score=best_score,
            initial_score=initial_score,
            improvement=best_score - initial_score,
            generations=max_iterations,
            variants_tested=len(history),
            history=history,
        )

    async def _score_all(
        self,
        prompt: str,
        test_cases: list[dict[str, Any]],
        metric_fn: Callable[[str, dict[str, Any]], Awaitable[float]],
    ) -> list[float]:
        """Score a prompt against every test case.

        Args:
            prompt: The prompt to evaluate.
            test_cases: List of test case dicts.
            metric_fn: Scoring function.

        Returns:
            List of scores, one per test case.
        """
        return [await metric_fn(prompt, tc) for tc in test_cases]

    async def _mean_score(
        self,
        prompt: str,
        test_cases: list[dict[str, Any]],
        metric_fn: Callable[[str, dict[str, Any]], Awaitable[float]],
    ) -> float:
        """Compute the mean score across all test cases.

        Args:
            prompt: The prompt to evaluate.
            test_cases: List of test case dicts.
            metric_fn: Scoring function.

        Returns:
            Mean score.
        """
        scores = await self._score_all(prompt, test_cases, metric_fn)
        return sum(scores) / len(scores) if scores else 0.0

    async def _identify_failure_patterns(
        self,
        prompt: str,
        failures: list[tuple[dict[str, Any], float]],
    ) -> str:
        """Ask the LLM to identify common patterns among failures.

        Args:
            prompt: The current prompt.
            failures: List of ``(test_case, score)`` tuples for failed cases.

        Returns:
            A text description of observed failure patterns.
        """
        failure_descriptions: list[str] = []
        for tc, score in failures[:10]:
            tc_str = json.dumps(tc, ensure_ascii=False, default=str)[:300]
            failure_descriptions.append(f"  Score {score:.2f}: {tc_str}")

        failures_text = "\n".join(failure_descriptions)

        analysis_prompt = (
            f"Analyze why the following prompt fails on certain test cases.\n\n"
            f"Prompt:\n{prompt}\n\n"
            f"Failed test cases (with scores):\n{failures_text}\n\n"
            f"Identify the common failure patterns. Be specific about "
            f"what the prompt is missing or getting wrong. "
            f"Output ONLY the analysis."
        )
        response = await self.judge.evaluate(prompt=analysis_prompt)
        return response.reasoning.strip()

    async def _generate_improvement(self, prompt: str, failure_patterns: str) -> str:
        """Generate an improved prompt addressing failure patterns.

        Args:
            prompt: The current prompt.
            failure_patterns: Description of what went wrong.

        Returns:
            An improved prompt string.
        """
        improvement_prompt = (
            f"You are a prompt engineer. Improve the following prompt to "
            f"address the identified failure patterns.\n\n"
            f"Current prompt:\n{prompt}\n\n"
            f"Failure patterns:\n{failure_patterns}\n\n"
            f"Generate an improved version that fixes these issues while "
            f"preserving what works. Output ONLY the improved prompt."
        )
        response = await self.judge.evaluate(prompt=improvement_prompt)
        result = response.reasoning.strip()
        return result if result else prompt


class SIMBAOptimizer:
    """SIMilarity-BAsed prompt adaptation.

    Uses a pool of known-good prompts and adapts the closest match to the
    current task:

    1. Compute a lightweight text-overlap similarity between the task
       description and each prompt in the pool.
    2. Select the most similar prompt.
    3. Adapt it using LLM-guided mutation.
    4. Iterate with feedback from metric scores.

    Args:
        judge: The LLM judge backend used for adaptation and scoring.

    Usage::

        optimizer = SIMBAOptimizer(judge=judge)
        result = await optimizer.optimize(
            prompt="...",
            test_cases=cases,
            metric_fn=my_metric,
            prompt_pool=existing_prompts,
        )
    """

    def __init__(self, judge: JudgeBackend) -> None:
        self.judge = judge

    async def optimize(
        self,
        prompt: str,
        test_cases: list[dict[str, Any]],
        metric_fn: Callable[[str, dict[str, Any]], Awaitable[float]],
        prompt_pool: list[str] | None = None,
        max_iterations: int = 5,
    ) -> OptimizationResult:
        """Run similarity-based prompt adaptation.

        Args:
            prompt: The initial prompt / task description.
            test_cases: List of test case dicts.
            metric_fn: Async callable ``(prompt, case) -> float``.
            prompt_pool: Optional list of known-good prompts.  If ``None``,
                falls back to LLM-generated candidates.
            max_iterations: Maximum adaptation iterations.

        Returns:
            An OptimizationResult with the best prompt found.
        """
        history: list[PromptVariant] = []

        initial_score = await self._mean_score(prompt, test_cases, metric_fn)
        history.append(PromptVariant(prompt=prompt, score=initial_score, generation=0))

        best_prompt = prompt
        best_score = initial_score

        if prompt_pool:
            best_match = self._find_most_similar(prompt, prompt_pool)
            match_score = await self._mean_score(best_match, test_cases, metric_fn)
            history.append(PromptVariant(prompt=best_match, score=match_score, generation=0))
            if match_score > best_score:
                best_score = match_score
                best_prompt = best_match
                logger.info("SIMBA: pool match score %.2f", best_score)

        for iteration in range(1, max_iterations + 1):
            adapted = await self._adapt_prompt(best_prompt, prompt, test_cases, iteration)
            adapted_score = await self._mean_score(adapted, test_cases, metric_fn)
            history.append(
                PromptVariant(
                    prompt=adapted,
                    score=adapted_score,
                    generation=iteration,
                )
            )
            if adapted_score > best_score:
                best_score = adapted_score
                best_prompt = adapted
                logger.info(
                    "SIMBA iteration %d: new best score %.2f",
                    iteration,
                    best_score,
                )

        return OptimizationResult(
            best_prompt=best_prompt,
            best_score=best_score,
            initial_score=initial_score,
            improvement=best_score - initial_score,
            generations=max_iterations,
            variants_tested=len(history),
            history=history,
        )

    @staticmethod
    def _find_most_similar(query: str, pool: list[str]) -> str:
        """Find the most similar prompt using word-overlap (Jaccard).

        Args:
            query: The target task description.
            pool: List of candidate prompts.

        Returns:
            The most similar prompt from the pool.
        """
        query_words = set(query.lower().split())
        best_similarity = -1.0
        best_match = pool[0]

        for candidate in pool:
            candidate_words = set(candidate.lower().split())
            union = query_words | candidate_words
            if not union:
                continue
            intersection = query_words & candidate_words
            similarity = len(intersection) / len(union)
            if similarity > best_similarity:
                best_similarity = similarity
                best_match = candidate

        return best_match

    async def _mean_score(
        self,
        prompt: str,
        test_cases: list[dict[str, Any]],
        metric_fn: Callable[[str, dict[str, Any]], Awaitable[float]],
    ) -> float:
        """Compute the mean score across all test cases.

        Args:
            prompt: The prompt to evaluate.
            test_cases: List of test case dicts.
            metric_fn: Scoring function.

        Returns:
            Mean score.
        """
        if not test_cases:
            return 0.0
        scores = [await metric_fn(prompt, tc) for tc in test_cases]
        return sum(scores) / len(scores)

    async def _adapt_prompt(
        self,
        base_prompt: str,
        task_description: str,
        test_cases: list[dict[str, Any]],
        iteration: int,
    ) -> str:
        """Adapt a prompt using the LLM, guided by the task description.

        Args:
            base_prompt: The prompt to adapt.
            task_description: The original task description for context.
            test_cases: Test cases for context.
            iteration: Current iteration number.

        Returns:
            An adapted prompt string.
        """
        sample_cases = test_cases[:3]
        cases_str = json.dumps(sample_cases, ensure_ascii=False, default=str)[:500]

        adapt_prompt = (
            f"You are a prompt adaptation expert. Adapt the following prompt "
            f"to better suit the target task. This is iteration {iteration}.\n\n"
            f"Base prompt:\n{base_prompt}\n\n"
            f"Target task:\n{task_description}\n\n"
            f"Sample test cases:\n{cases_str}\n\n"
            f"Create an improved version that better matches the task. "
            f"Output ONLY the adapted prompt, nothing else."
        )
        response = await self.judge.evaluate(prompt=adapt_prompt)
        result = response.reasoning.strip()
        return result if result else base_prompt


def create_optimizer(
    strategy: str,
    judge: JudgeBackend,
) -> PromptOptimizer | MIPROv2Optimizer | COPROOptimizer | SIMBAOptimizer:
    """Create an optimizer by strategy name.

    Args:
        strategy: One of ``"genetic"``, ``"mipro"``, ``"copro"``, or
            ``"simba"`` (case-insensitive).
        judge: The LLM judge backend to use.

    Returns:
        An optimizer instance matching the requested strategy.

    Raises:
        ValueError: If the strategy name is not recognized.
    """
    strategy_lower = strategy.strip().lower()
    factories: dict[
        str,
        Callable[..., PromptOptimizer | MIPROv2Optimizer | COPROOptimizer | SIMBAOptimizer],
    ] = {
        "genetic": lambda: PromptOptimizer(judge=judge),
        "mipro": lambda: MIPROv2Optimizer(judge=judge),
        "miprov2": lambda: MIPROv2Optimizer(judge=judge),
        "copro": lambda: COPROOptimizer(judge=judge),
        "simba": lambda: SIMBAOptimizer(judge=judge),
    }

    factory = factories.get(strategy_lower)
    if factory is None:
        valid = ", ".join(sorted(factories.keys()))
        raise ValueError(f"Unknown optimization strategy {strategy!r}. Valid strategies: {valid}")
    return factory()
