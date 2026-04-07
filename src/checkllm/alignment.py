from __future__ import annotations

import json
import random
from pathlib import Path
from typing import Any

from pydantic import BaseModel, Field

from checkllm.judge import JudgeBackend
from checkllm.models import JudgeResponse


class HumanLabel(BaseModel):
    """A human-labeled example for metric alignment.

    Args:
        output: The LLM output that was evaluated.
        context: The context or reference material.
        query: The original user query.
        human_score: The human-assigned score (0.0 to 1.0).
        reference: Optional reference/gold answer.
        metadata: Optional extra metadata for this example.
    """

    output: str
    context: str
    query: str
    human_score: float = Field(ge=0.0, le=1.0)
    reference: str | None = None
    metadata: dict[str, Any] | None = None


class AlignmentResult(BaseModel):
    """Result of aligning a metric's judge prompts to human labels.

    Args:
        metric_name: Name of the metric that was aligned.
        aligned_prompt: The optimized system prompt.
        original_prompt: The original system prompt before alignment.
        correlation_before: Pearson correlation before alignment.
        correlation_after: Pearson correlation after alignment.
        improvement: Percentage improvement in correlation.
        few_shot_examples: Selected few-shot examples (if strategy includes few_shot).
        iterations_run: Number of optimization iterations executed.
    """

    metric_name: str
    aligned_prompt: str
    original_prompt: str
    correlation_before: float
    correlation_after: float
    improvement: float
    few_shot_examples: list[dict[str, Any]] = Field(default_factory=list)
    iterations_run: int


def _pearson_correlation(xs: list[float], ys: list[float]) -> float:
    """Compute Pearson correlation coefficient between two sequences.

    Args:
        xs: First sequence of values.
        ys: Second sequence of values.

    Returns:
        Pearson r in [-1, 1], or 0.0 if computation is degenerate.
    """
    n = len(xs)
    if n < 2:
        return 0.0

    mean_x = sum(xs) / n
    mean_y = sum(ys) / n

    cov = sum((x - mean_x) * (y - mean_y) for x, y in zip(xs, ys))
    var_x = sum((x - mean_x) ** 2 for x in xs)
    var_y = sum((y - mean_y) ** 2 for y in ys)

    denom = (var_x * var_y) ** 0.5
    if denom == 0.0:
        return 0.0

    return cov / denom


_REWRITE_SYSTEM_PROMPT = (
    "You are an expert at writing LLM evaluation prompts. Your task is to "
    "rewrite a system prompt so that the LLM judge's scores more closely "
    "match human judgments.\n\n"
    "Respond with JSON: {\"prompt\": \"<rewritten system prompt>\"}"
)


class MetricAligner:
    """Aligns LLM judge prompts to match human labels using gradient-free optimization.

    Uses iterative prompt rewriting and few-shot selection to improve the
    Pearson correlation between judge scores and human-labeled scores.

    Args:
        judge: The judge backend to use for evaluation.
        seed: Random seed for reproducibility.
    """

    def __init__(self, judge: JudgeBackend, seed: int = 42) -> None:
        self.judge = judge
        self._rng = random.Random(seed)

    async def align(
        self,
        metric_name: str,
        labels: list[HumanLabel],
        iterations: int = 5,
        strategy: str = "both",
        base_prompt: str | None = None,
    ) -> AlignmentResult:
        """Align a metric's judge prompt to match human labels.

        Args:
            metric_name: Name of the metric being aligned.
            labels: Human-labeled examples to align against.
            iterations: Number of optimization rounds.
            strategy: One of "few_shot", "instruction_rewrite", or "both".
            base_prompt: Starting system prompt. If None, uses a generic one.

        Returns:
            AlignmentResult with the optimized prompt and correlation stats.

        Raises:
            ValueError: If fewer than 2 labels are provided or strategy is invalid.
        """
        if len(labels) < 2:
            raise ValueError("At least 2 labels are required for alignment.")
        valid_strategies = {"few_shot", "instruction_rewrite", "both"}
        if strategy not in valid_strategies:
            raise ValueError(
                f"strategy must be one of {valid_strategies}, got {strategy!r}"
            )

        current_prompt = base_prompt or (
            f"You are an expert {metric_name} evaluator. Score from 0.0 to 1.0.\n"
            "Respond with JSON: {\"score\": <float>, \"reasoning\": \"<explanation>\"}"
        )
        original_prompt = current_prompt

        correlation_before = await self._evaluate_correlation(current_prompt, labels)
        best_correlation = correlation_before
        best_prompt = current_prompt
        few_shot_examples: list[dict[str, Any]] = []

        for _ in range(iterations):
            candidate_prompt = current_prompt

            if strategy in ("few_shot", "both"):
                selected = self._select_few_shot(labels, n=3)
                examples_block = self._format_few_shot(selected)
                candidate_prompt = candidate_prompt + "\n\n" + examples_block
                few_shot_examples = [
                    {
                        "output": ex.output[:200],
                        "query": ex.query[:200],
                        "human_score": ex.human_score,
                    }
                    for ex in selected
                ]

            if strategy in ("instruction_rewrite", "both"):
                candidate_prompt = await self._rewrite_instructions(
                    candidate_prompt, labels, best_correlation
                )

            new_correlation = await self._evaluate_correlation(
                candidate_prompt, labels
            )

            if new_correlation > best_correlation:
                best_correlation = new_correlation
                best_prompt = candidate_prompt
                current_prompt = candidate_prompt

        improvement = (
            ((best_correlation - correlation_before) / abs(correlation_before) * 100)
            if correlation_before != 0.0
            else (best_correlation * 100 if best_correlation != 0.0 else 0.0)
        )

        return AlignmentResult(
            metric_name=metric_name,
            aligned_prompt=best_prompt,
            original_prompt=original_prompt,
            correlation_before=correlation_before,
            correlation_after=best_correlation,
            improvement=improvement,
            few_shot_examples=few_shot_examples,
            iterations_run=iterations,
        )

    async def _evaluate_correlation(
        self, prompt: str, labels: list[HumanLabel]
    ) -> float:
        """Run the judge with the given prompt on all labels and compute Pearson r.

        Args:
            prompt: System prompt to use for the judge.
            labels: Human-labeled examples to evaluate against.

        Returns:
            Pearson correlation between judge scores and human scores.
        """
        judge_scores: list[float] = []
        human_scores: list[float] = []

        for label in labels:
            eval_prompt = (
                f"Query: {label.query}\n"
                f"Context: {label.context}\n"
                f"Output: {label.output}\n\n"
                "Score this output."
            )
            response: JudgeResponse = await self.judge.evaluate(
                prompt=eval_prompt, system_prompt=prompt
            )
            judge_scores.append(response.score)
            human_scores.append(label.human_score)

        return _pearson_correlation(human_scores, judge_scores)

    def _select_few_shot(
        self, labels: list[HumanLabel], n: int = 3
    ) -> list[HumanLabel]:
        """Select diverse few-shot examples that maximize score-range coverage.

        Picks examples spread across the score range to give the judge a
        well-calibrated set of reference points.

        Args:
            labels: Pool of human-labeled examples to select from.
            n: Number of examples to select.

        Returns:
            Selected subset of labels with diverse scores.
        """
        if len(labels) <= n:
            return list(labels)

        sorted_labels = sorted(labels, key=lambda lb: lb.human_score)

        if n == 1:
            mid = len(sorted_labels) // 2
            return [sorted_labels[mid]]

        selected: list[HumanLabel] = [sorted_labels[0], sorted_labels[-1]]

        remaining = [lb for lb in sorted_labels if lb not in selected]
        while len(selected) < n and remaining:
            best_candidate = None
            best_min_dist = -1.0
            for candidate in remaining:
                min_dist = min(
                    abs(candidate.human_score - s.human_score) for s in selected
                )
                if min_dist > best_min_dist:
                    best_min_dist = min_dist
                    best_candidate = candidate
            if best_candidate is not None:
                selected.append(best_candidate)
                remaining.remove(best_candidate)

        return selected

    def _format_few_shot(self, examples: list[HumanLabel]) -> str:
        """Format few-shot examples into a prompt block.

        Args:
            examples: The labeled examples to format.

        Returns:
            Formatted string block with examples.
        """
        lines = ["Here are reference examples with known scores:"]
        for i, ex in enumerate(examples, 1):
            lines.append(
                f"\nExample {i}:\n"
                f"  Query: {ex.query}\n"
                f"  Output: {ex.output}\n"
                f"  Expected score: {ex.human_score}"
            )
        return "\n".join(lines)

    async def _rewrite_instructions(
        self,
        prompt: str,
        labels: list[HumanLabel],
        current_correlation: float,
    ) -> str:
        """Ask the judge to rewrite the system prompt to better match human labels.

        Args:
            prompt: The current system prompt.
            labels: Human-labeled examples for reference.
            current_correlation: Current Pearson correlation for context.

        Returns:
            The rewritten system prompt string.
        """
        mismatch_examples = self._rng.sample(labels, min(5, len(labels)))
        examples_text = "\n".join(
            f"- Query: {lb.query[:100]}, Human score: {lb.human_score}"
            for lb in mismatch_examples
        )

        rewrite_prompt = (
            f"Current system prompt:\n{prompt}\n\n"
            f"Current correlation with human judgments: {current_correlation:.3f}\n\n"
            f"Example human ratings:\n{examples_text}\n\n"
            "Rewrite the system prompt to better match these human ratings. "
            "Keep the JSON output format instruction."
        )

        response = await self.judge.evaluate(
            prompt=rewrite_prompt, system_prompt=_REWRITE_SYSTEM_PROMPT
        )

        if response.raw_output:
            try:
                parsed = json.loads(response.raw_output)
                rewritten = parsed.get("prompt")
                if rewritten and isinstance(rewritten, str):
                    return rewritten
            except (json.JSONDecodeError, ValueError):
                pass

        return prompt

    def apply(self, metric: Any, result: AlignmentResult) -> None:
        """Apply an alignment result to a metric instance.

        Sets the metric's system_prompt attribute to the aligned prompt.

        Args:
            metric: A metric instance with a system_prompt attribute.
            result: The alignment result to apply.

        Raises:
            AttributeError: If the metric has no system_prompt attribute.
        """
        if not hasattr(metric, "system_prompt"):
            raise AttributeError(
                f"{type(metric).__name__} has no 'system_prompt' attribute."
            )
        metric.system_prompt = result.aligned_prompt

    def save(self, result: AlignmentResult, path: str | Path) -> None:
        """Serialize an alignment result to a JSON file.

        Args:
            result: The alignment result to save.
            path: File path to write the JSON to.
        """
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(result.model_dump_json(indent=2), encoding="utf-8")

    def load(self, path: str | Path) -> AlignmentResult:
        """Deserialize an alignment result from a JSON file.

        Args:
            path: File path to read the JSON from.

        Returns:
            The deserialized AlignmentResult.
        """
        path = Path(path)
        data = path.read_text(encoding="utf-8")
        return AlignmentResult.model_validate_json(data)
