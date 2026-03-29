from __future__ import annotations

import time

from checkllm.judge import JudgeBackend
from checkllm.models import CheckResult

CONSISTENCY_SYSTEM_PROMPT = """You are an expert consistency evaluator. Your job is to assess whether multiple outputs are consistent with each other — that is, whether they convey the same information without contradictions.

Score from 0.0 to 1.0:
- 1.0 = All outputs are fully consistent. They convey the same core information and do not contradict each other in any way, even if they use different wording.
- 0.8 = The outputs are mostly consistent with only trivial differences in emphasis or phrasing that do not constitute contradictions.
- 0.5 = The outputs are partially consistent. They agree on some points but contradict each other on others, or some outputs include information that conflicts with others.
- 0.3 = The outputs are largely inconsistent. There are significant contradictions on key points, even if they share some common ground.
- 0.0 = The outputs are completely contradictory. They make opposing claims about the same facts or arrive at opposite conclusions.

Key evaluation criteria:
1. Factual consistency: Do the outputs agree on facts, numbers, dates, and names?
2. Logical consistency: Are the conclusions and reasoning compatible across outputs?
3. Tonal consistency: Do the outputs convey a similar sentiment or stance?
4. Completeness consistency: Do the outputs cover similar ground, or does one omit critical information the others include?
5. Contradiction identification: Are there any direct contradictions between outputs?

Respond with JSON: {"score": <float>, "reasoning": "<explanation>"}"""


class ConsistencyMetric:
    """Compares multiple outputs for consistency with each other.

    Score interpretation: 1.0 = fully consistent, 0.0 = contradictory.
    """

    def __init__(self, judge: JudgeBackend, threshold: float = 0.8) -> None:
        self.judge = judge
        self.threshold = threshold
        self.system_prompt: str = CONSISTENCY_SYSTEM_PROMPT

    async def evaluate(self, outputs: list[str]) -> CheckResult:
        numbered = "\n\n".join(
            f"Output {i + 1}:\n{text}" for i, text in enumerate(outputs)
        )
        prompt = (
            f"The following are multiple outputs to compare for consistency:\n\n"
            f"{numbered}\n\n"
            "Are these outputs consistent with each other? "
            "Identify any contradictions or discrepancies. Score it."
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
            metric_name="consistency",
        )
