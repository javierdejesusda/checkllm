from __future__ import annotations

import time

from checkllm.judge import JudgeBackend
from checkllm.models import CheckResult

SUMMARIZATION_SYSTEM_PROMPT = """You are an expert summarization quality evaluator. Your job is to assess the quality of a summary by evaluating its accuracy, conciseness, and retention of key information from the source material.

Score from 0.0 to 1.0:
- 1.0 = The summary is excellent: it accurately captures all key information from the source, is appropriately concise, introduces no errors, and maintains the correct emphasis and nuance.
- 0.8 = The summary is good: it captures the main points accurately with only minor omissions of less important details. It is reasonably concise.
- 0.5 = The summary is adequate: it captures some key points but misses others, may include minor inaccuracies, or is either too verbose or too terse.
- 0.3 = The summary is poor: it misses major points, contains inaccuracies, significantly distorts the source meaning, or is inappropriately long/short.
- 0.0 = The summary is completely inaccurate, irrelevant, or fails to summarize the source in any meaningful way.

Key evaluation criteria:
1. Accuracy: Does the summary faithfully represent the source without introducing errors or distortions?
2. Coverage: Are the most important points and key information from the source retained?
3. Conciseness: Is the summary appropriately condensed without being too terse or too verbose?
4. Coherence: Does the summary read as a well-organized, coherent piece of text?
5. Emphasis: Does the summary preserve the relative importance of topics from the source?
6. No hallucination: Does the summary avoid adding information not present in the source?

Respond with JSON: {"score": <float>, "reasoning": "<explanation>"}"""


class SummarizationMetric:
    """Evaluates summary quality: accuracy, conciseness, and key information retention."""

    def __init__(self, judge: JudgeBackend, threshold: float = 0.8) -> None:
        self.judge = judge
        self.threshold = threshold
        self.system_prompt: str = SUMMARIZATION_SYSTEM_PROMPT

    async def evaluate(self, output: str, source: str) -> CheckResult:
        prompt = (
            f"Source Material:\n{source}\n\n"
            f"Summary to evaluate:\n{output}\n\n"
            "Evaluate the quality of this summary. Is it accurate, concise, "
            "and does it retain the key information? Score it."
        )

        start = time.perf_counter_ns()
        response = await self.judge.evaluate(
            prompt=prompt, system_prompt=self.system_prompt
        )
        elapsed_ms = (time.perf_counter_ns() - start) // 1_000_000

        return CheckResult(
            passed=response.score >= self.threshold,
            score=response.score,
            reasoning=response.reasoning,
            cost=response.cost,
            latency_ms=int(elapsed_ms),
            metric_name="summarization",
        )
