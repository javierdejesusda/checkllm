"""Chart value extraction metric.

Scores whether extracted numerical values from a chart match expected values
within a numeric tolerance. Supports both deterministic comparison (when the
caller supplies extracted values directly) and vision-judge extraction (when
only the image and the expected values are available).
"""

from __future__ import annotations

import json
import time
from typing import Iterable, Mapping

from checkllm.judge import JudgeBackend
from checkllm.metrics.image_text_alignment import _ensure_payloads
from checkllm.models import CheckResult
from checkllm.multimodal import ImageSource, call_vision_judge

CHART_VALUE_EXTRACTION_SYSTEM_PROMPT = """You are an expert at reading numerical values from charts. You will be shown a chart image. Return the requested numerical values as JSON.

For each label the user asks about, read the value from the chart. If a value cannot be determined from the chart, return null for that label.

Respond with JSON only of the form:
{"values": {"<label>": <number or null>, ...}, "reasoning": "<brief explanation>"}"""


def _rel_error(actual: float, expected: float) -> float:
    """Relative error between two numbers, with a small floor on the denominator."""
    denom = max(abs(expected), 1e-9)
    return abs(actual - expected) / denom


class ChartValueExtractionMetric:
    """Extracts and compares numerical values from charts."""

    def __init__(
        self,
        judge: JudgeBackend,
        threshold: float = 0.8,
        tolerance: float = 0.05,
    ) -> None:
        """Initialize the metric.

        Args:
            judge: Vision-capable judge used for extraction.
            threshold: Minimum fraction of values within tolerance to pass.
            tolerance: Maximum relative error (e.g., ``0.05`` for 5%) for a
                value to count as correct.
        """
        self.judge = judge
        self.threshold = threshold
        self.tolerance = tolerance
        self.system_prompt: str = CHART_VALUE_EXTRACTION_SYSTEM_PROMPT

    async def evaluate(
        self,
        expected_values: Mapping[str, float],
        image: ImageSource | Iterable[ImageSource] | None = None,
        extracted_values: Mapping[str, float] | None = None,
    ) -> CheckResult:
        """Evaluate extracted chart values against expected ones.

        Args:
            expected_values: Mapping of label -> expected numeric value.
            image: Chart image source(s). Required when ``extracted_values``
                is not supplied.
            extracted_values: Pre-extracted values to compare deterministically.

        Returns:
            A ``CheckResult`` scored as the fraction of labels within tolerance.
        """
        start = time.perf_counter_ns()
        cost = 0.0
        reasoning_extra = ""

        resolved_values: Mapping[str, float | None]
        if extracted_values is None:
            if image is None:
                raise ValueError("chart_value_extraction requires either image or extracted_values")
            payloads = _ensure_payloads(image)
            labels = list(expected_values.keys())
            prompt = "Read the following values from the chart:\n" + "\n".join(
                f"- {label}" for label in labels
            )
            response = await call_vision_judge(
                self.judge,
                prompt=prompt,
                images=payloads,
                system_prompt=self.system_prompt,
            )
            cost = response.cost
            try:
                parsed = json.loads(response.raw_output or "{}")
                raw_values = parsed.get("values", {})
            except (json.JSONDecodeError, ValueError, TypeError):
                raw_values = {}
            resolved_values = {label: _coerce_number(raw_values.get(label)) for label in labels}
            reasoning_extra = response.reasoning
        else:
            resolved_values = extracted_values

        correct = 0
        total = 0
        per_label: list[str] = []
        for label, expected in expected_values.items():
            total += 1
            actual = resolved_values.get(label)
            if actual is None:
                per_label.append(f"{label}: missing (expected {expected})")
                continue
            err = _rel_error(actual, expected)
            ok = err <= self.tolerance
            if ok:
                correct += 1
            per_label.append(
                f"{label}: got {actual}, expected {expected} "
                f"(err {err:.2%} {'OK' if ok else 'FAIL'})"
            )

        score = correct / total if total else 0.0
        elapsed_ms = (time.perf_counter_ns() - start) // 1_000_000
        reasoning = "; ".join(per_label)
        if reasoning_extra:
            reasoning = f"{reasoning} | {reasoning_extra}"

        return CheckResult(
            passed=score >= self.threshold,
            score=score,
            reasoning=reasoning,
            cost=cost,
            latency_ms=int(elapsed_ms),
            metric_name="chart_value_extraction",
        )


def _coerce_number(value: object) -> float | None:
    """Coerce a JSON value to float, returning None on failure."""
    if value is None:
        return None
    try:
        return float(value)  # type: ignore[arg-type]  # value is unknown JSON shape
    except (TypeError, ValueError):
        return None
