from __future__ import annotations

import time

from checkllm.models import CheckResult


class ToolCallF1Metric:
    """Computes F1 score for tool call predictions.

    A purely deterministic metric that compares predicted tool names against
    expected tool names and computes precision, recall, and F1. No LLM judge
    is required.
    """

    def __init__(self, threshold: float = 0.7) -> None:
        self.threshold = threshold

    async def evaluate(
        self,
        predicted_tools: list[str],
        expected_tools: list[str],
    ) -> CheckResult:
        """Compute precision, recall, and F1 for tool call predictions.

        Args:
            predicted_tools: The list of tool names that were predicted/called.
            expected_tools: The list of tool names that were expected.

        Returns:
            A CheckResult with F1 score and precision/recall details.
        """
        start = time.perf_counter_ns()

        predicted_set = set(predicted_tools)
        expected_set = set(expected_tools)

        true_positives = len(predicted_set & expected_set)

        if len(predicted_set) > 0:
            precision = true_positives / len(predicted_set)
        else:
            precision = 0.0

        if len(expected_set) > 0:
            recall = true_positives / len(expected_set)
        else:
            recall = 1.0 if len(predicted_set) == 0 else 0.0

        if precision + recall > 0:
            f1 = 2 * precision * recall / (precision + recall)
        else:
            f1 = 0.0

        elapsed_ms = (time.perf_counter_ns() - start) // 1_000_000

        reasoning = (
            f"Precision: {precision:.2f} ({true_positives}/{len(predicted_set) or 0}), "
            f"Recall: {recall:.2f} ({true_positives}/{len(expected_set) or 0}), "
            f"F1: {f1:.2f}. "
            f"Correct: {sorted(predicted_set & expected_set)}. "
            f"Missing: {sorted(expected_set - predicted_set)}. "
            f"Extra: {sorted(predicted_set - expected_set)}."
        )

        return CheckResult(
            passed=f1 >= self.threshold,
            score=f1,
            reasoning=reasoning,
            cost=0.0,
            latency_ms=int(elapsed_ms),
            metric_name="tool_call_f1",
        )
