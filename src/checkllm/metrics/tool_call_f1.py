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

        from collections import Counter

        predicted_counts = Counter(predicted_tools)
        expected_counts = Counter(expected_tools)

        true_positives = sum(
            min(predicted_counts[k], expected_counts[k])
            for k in predicted_counts
            if k in expected_counts
        )
        total_predicted = len(predicted_tools)
        total_expected = len(expected_tools)

        if total_predicted > 0:
            precision = true_positives / total_predicted
        else:
            precision = 0.0

        if total_expected > 0:
            recall = true_positives / total_expected
        else:
            recall = 1.0 if total_predicted == 0 else 0.0

        if precision + recall > 0:
            f1 = 2 * precision * recall / (precision + recall)
        else:
            f1 = 0.0

        elapsed_ms = (time.perf_counter_ns() - start) // 1_000_000

        predicted_set = set(predicted_tools)
        expected_set = set(expected_tools)

        reasoning = (
            f"Precision: {precision:.2f} ({true_positives}/{total_predicted}), "
            f"Recall: {recall:.2f} ({true_positives}/{total_expected}), "
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
