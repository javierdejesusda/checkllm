"""Diagram / chart comprehension metric.

Tests whether a vision LLM correctly answered a question about a diagram or
chart. The judge compares the answer against an expected answer grounded in
the image.
"""

from __future__ import annotations

import time
from typing import Iterable

from checkllm.judge import JudgeBackend
from checkllm.metrics.image_text_alignment import _ensure_payloads
from checkllm.models import CheckResult
from checkllm.multimodal import ImageSource, call_vision_judge

DIAGRAM_COMPREHENSION_SYSTEM_PROMPT = """You are an expert evaluator of diagram and chart comprehension. You will be shown an image (a diagram, flowchart, chart, or plot), a question about it, an expected answer, and a candidate answer.

Score from 0.0 to 1.0 based on how correctly the candidate answers the question with respect to the image and the expected answer:
- 1.0 = Candidate matches the expected answer and is consistent with the image. Minor wording differences are fine.
- 0.7 = Mostly correct but missing a small detail or slightly imprecise.
- 0.5 = Partially correct (e.g., right trend but wrong magnitude, or right node but wrong connection).
- 0.2 = Largely incorrect.
- 0.0 = Completely wrong, unrelated, or contradicts the image.

Verify the candidate against the image, not just against the expected answer — if the expected answer is stated incorrectly the image takes precedence.

Respond with JSON: {"score": <float>, "reasoning": "<explanation>"}"""


class DiagramComprehensionMetric:
    """Scores diagram / chart question-answering using a vision judge."""

    def __init__(self, judge: JudgeBackend, threshold: float = 0.7) -> None:
        self.judge = judge
        self.threshold = threshold
        self.system_prompt: str = DIAGRAM_COMPREHENSION_SYSTEM_PROMPT

    async def evaluate(
        self,
        image: ImageSource | Iterable[ImageSource],
        question: str,
        expected_answer: str,
        candidate_answer: str,
    ) -> CheckResult:
        """Evaluate a candidate answer to a question about a diagram.

        Args:
            image: The diagram or chart image source(s).
            question: The question posed about the image.
            expected_answer: The ground-truth answer.
            candidate_answer: The model's answer to evaluate.

        Returns:
            A ``CheckResult`` with the comprehension score.
        """
        payloads = _ensure_payloads(image)
        prompt = (
            f"Question:\n{question}\n\n"
            f"Expected answer:\n{expected_answer}\n\n"
            f"Candidate answer:\n{candidate_answer}\n\n"
            "Score the candidate answer against the image and the expected answer."
        )
        start = time.perf_counter_ns()
        response = await call_vision_judge(
            self.judge, prompt=prompt, images=payloads, system_prompt=self.system_prompt
        )
        elapsed_ms = (time.perf_counter_ns() - start) // 1_000_000

        return CheckResult(
            passed=response.score >= self.threshold,
            score=response.score,
            reasoning=response.reasoning,
            cost=response.cost,
            latency_ms=int(elapsed_ms),
            metric_name="diagram_comprehension",
        )
