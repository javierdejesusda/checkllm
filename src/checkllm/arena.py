"""Arena A/B testing for comparing prompts or models with statistical significance.

Runs two candidate prompts against a shared set of test inputs, scores each via
an LLM judge, and applies Welch's t-test to determine whether the performance
difference is statistically significant.

Usage::

    from checkllm.arena import Arena

    arena = Arena(judge=my_judge)
    result = arena.compare(
        candidate_a=("prompt-v1", "You are a helpful assistant. {input}"),
        candidate_b=("prompt-v2", "You are a concise assistant. {input}"),
        test_inputs=["Summarise the French Revolution.", "What is 2+2?"],
    )
    print(result.summary())
"""

from __future__ import annotations

import asyncio
import statistics

from pydantic import BaseModel, Field
from scipy import stats as scipy_stats

from checkllm.judge import JudgeBackend
from checkllm.models import JudgeResponse


class ArenaCandidate(BaseModel):
    """A prompt/model candidate in an arena comparison."""

    name: str
    prompt: str
    scores: list[float] = Field(default_factory=list)
    avg_score: float = 0.0
    cost: float = 0.0


class ArenaResult(BaseModel):
    """Result of an arena comparison between two candidates."""

    candidate_a: ArenaCandidate
    candidate_b: ArenaCandidate
    winner: str = ""  # name of winner, or "tie"
    p_value: float = 1.0
    statistically_significant: bool = False
    confidence_level: float = 0.95
    effect_size: float = 0.0
    num_trials: int = 0

    def summary(self) -> str:
        """Return a human-readable summary of the arena comparison.

        Returns:
            A multi-line string showing per-candidate scores, winner, p-value,
            and effect size.
        """
        lines = [
            "Arena Result",
            f"  {self.candidate_a.name}: {self.candidate_a.avg_score:.3f} (n={len(self.candidate_a.scores)})",
            f"  {self.candidate_b.name}: {self.candidate_b.avg_score:.3f} (n={len(self.candidate_b.scores)})",
            f"  Winner: {self.winner}",
            f"  p-value: {self.p_value:.4f} {'(significant)' if self.statistically_significant else '(not significant)'}",
            f"  Effect size: {self.effect_size:.3f}",
        ]
        return "\n".join(lines)


class Arena:
    """A/B testing arena for comparing prompts or models.

    Uses an LLM judge to score each candidate on every test input, then applies
    Welch's t-test to determine statistical significance of the score difference.
    """

    def __init__(self, judge: JudgeBackend) -> None:
        self.judge = judge

    def compare(
        self,
        candidate_a: tuple[str, str],
        candidate_b: tuple[str, str],
        test_inputs: list[str],
        confidence_level: float = 0.95,
    ) -> ArenaResult:
        """Compare two candidates synchronously.

        Args:
            candidate_a: A (name, prompt) tuple for the first candidate.
            candidate_b: A (name, prompt) tuple for the second candidate.
            test_inputs: List of inputs to evaluate both candidates against.
            confidence_level: Confidence threshold for significance testing.

        Returns:
            An ArenaResult with scores, winner, and statistical test results.
        """
        try:
            loop = asyncio.get_running_loop()
        except RuntimeError:
            loop = None
        if loop and loop.is_running():
            import concurrent.futures

            with concurrent.futures.ThreadPoolExecutor() as pool:
                return pool.submit(
                    asyncio.run,
                    self.acompare(candidate_a, candidate_b, test_inputs, confidence_level),
                ).result()
        return asyncio.run(
            self.acompare(candidate_a, candidate_b, test_inputs, confidence_level)
        )

    async def acompare(
        self,
        candidate_a: tuple[str, str],
        candidate_b: tuple[str, str],
        test_inputs: list[str],
        confidence_level: float = 0.95,
    ) -> ArenaResult:
        """Compare two candidates asynchronously.

        Args:
            candidate_a: A (name, prompt) tuple for the first candidate.
            candidate_b: A (name, prompt) tuple for the second candidate.
            test_inputs: List of inputs to evaluate both candidates against.
            confidence_level: Confidence threshold for significance testing.

        Returns:
            An ArenaResult with scores, winner, and statistical test results.
        """
        a = ArenaCandidate(name=candidate_a[0], prompt=candidate_a[1])
        b = ArenaCandidate(name=candidate_b[0], prompt=candidate_b[1])

        for test_input in test_inputs:
            score_a = await self._score(a.prompt, test_input)
            a.scores.append(score_a.score)
            a.cost += score_a.cost

            score_b = await self._score(b.prompt, test_input)
            b.scores.append(score_b.score)
            b.cost += score_b.cost

        a.avg_score = statistics.mean(a.scores) if a.scores else 0.0
        b.avg_score = statistics.mean(b.scores) if b.scores else 0.0

        p_value = 1.0
        significant = False
        effect_size = 0.0
        if len(a.scores) >= 2 and len(b.scores) >= 2:
            _t_stat, p_value = scipy_stats.ttest_ind(a.scores, b.scores, equal_var=False)
            significant = p_value < (1.0 - confidence_level)
            combined = a.scores + b.scores
            pooled_std = statistics.stdev(combined) if len(combined) > 1 else 1.0
            if pooled_std > 0:
                effect_size = abs(a.avg_score - b.avg_score) / pooled_std

        winner = "tie"
        if significant:
            winner = a.name if a.avg_score > b.avg_score else b.name

        return ArenaResult(
            candidate_a=a,
            candidate_b=b,
            winner=winner,
            p_value=p_value,
            statistically_significant=significant,
            confidence_level=confidence_level,
            effect_size=effect_size,
            num_trials=len(test_inputs),
        )

    async def _score(self, prompt: str, test_input: str) -> JudgeResponse:
        """Score a prompt/input pair using the judge.

        Args:
            prompt: The candidate prompt template.
            test_input: The test input to evaluate.

        Returns:
            A JudgeResponse containing the score and reasoning.
        """
        scoring = (
            f"Evaluate this prompt's response quality on 0.0-1.0.\n\n"
            f"Prompt: {prompt}\n\nInput: {test_input}\n\n"
            f'Respond with JSON: {{"score": <float>, "reasoning": "<explanation>"}}'
        )
        return await self.judge.evaluate(prompt=scoring)
