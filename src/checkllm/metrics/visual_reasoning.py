"""Visual reasoning metric.

Scores multi-step visual reasoning (counting, spatial relationships,
compositional queries). Uses chain-of-thought prompting: the judge is asked
to outline the reasoning steps before scoring correctness against an
expected answer.
"""
from __future__ import annotations

import time
from typing import Iterable

from checkllm.judge import JudgeBackend
from checkllm.metrics.image_text_alignment import _ensure_payloads
from checkllm.models import CheckResult
from checkllm.multimodal import ImageSource, call_vision_judge

VISUAL_REASONING_SYSTEM_PROMPT = """You are an expert evaluator of visual reasoning. You will be shown an image, a reasoning question, an expected answer, and a candidate answer. Assess whether the candidate answer is correct for the question given the image.

Use step-by-step reasoning in your reasoning field:
1. Identify the objects, colors, and spatial relationships in the image that the question depends on.
2. Derive the correct answer from the image.
3. Compare the candidate answer to both the image-derived answer and the expected answer.

Score from 0.0 to 1.0:
- 1.0 = Candidate answer is correct (semantically equivalent to the image-derived / expected answer).
- 0.7 = Mostly correct but imprecise (e.g., off by one in a count, approximate position).
- 0.5 = Partially correct; gets the type of thing but wrong specifics.
- 0.2 = Mostly incorrect.
- 0.0 = Wrong or unrelated.

The image is the ground truth. If the expected answer conflicts with the image, trust the image.

Respond with JSON: {"score": <float>, "reasoning": "<step-by-step explanation>"}"""


class VisualReasoningMetric:
    """Scores multi-step visual reasoning with chain-of-thought prompting."""

    def __init__(self, judge: JudgeBackend, threshold: float = 0.7) -> None:
        self.judge = judge
        self.threshold = threshold
        self.system_prompt: str = VISUAL_REASONING_SYSTEM_PROMPT

    async def evaluate(
        self,
        image: ImageSource | Iterable[ImageSource],
        question: str,
        expected_answer: str,
        candidate_answer: str,
    ) -> CheckResult:
        """Evaluate a visual reasoning answer.

        Args:
            image: Image source(s) the question refers to.
            question: The reasoning question.
            expected_answer: The ground-truth answer.
            candidate_answer: The model's answer.

        Returns:
            A ``CheckResult`` with the reasoning score.
        """
        payloads = _ensure_payloads(image)
        prompt = (
            f"Question:\n{question}\n\n"
            f"Expected answer:\n{expected_answer}\n\n"
            f"Candidate answer:\n{candidate_answer}\n\n"
            "Walk through the visual reasoning step by step, then score the candidate answer."
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
            metric_name="visual_reasoning",
        )
