from __future__ import annotations

import time

from checkllm.judge import JudgeBackend
from checkllm.models import CheckResult

NOISE_SENSITIVITY_SYSTEM_PROMPT = """You are an expert evaluator of LLM robustness to noisy context. Your task is to assess whether a response was inappropriately influenced by irrelevant or noisy information injected into the context.

In real-world RAG pipelines, retrieved context often includes irrelevant passages. A robust model should produce the same correct answer regardless of noisy context. This metric measures how much the response changed or degraded due to noise.

Evaluation process:
1. Compare the response produced with the noisy context against the clean context.
2. Identify any information in the response that appears to come from the noisy context.
3. Assess whether the noise caused the response to become incorrect, misleading, or less precise.
4. Evaluate whether the core answer remains intact despite the noise.

Score from 0.0 to 1.0:
- 1.0 = The response is completely robust; noise had zero effect on the answer quality.
- 0.8 = The response is largely unaffected; minor stylistic differences but no factual impact.
- 0.5 = The response shows moderate influence from noise; some incorrect or irrelevant details crept in.
- 0.3 = The response is significantly degraded by noise; core claims are altered.
- 0.0 = The response is completely corrupted by noise; the answer is wrong because of noisy context.

Key evaluation criteria:
1. Does the response incorporate claims from the noisy context that do not belong?
2. Has the core answer changed compared to what clean context would support?
3. Are there hedging statements or confusion introduced by the noise?
4. Did the model confuse noisy entities or facts with real ones?

Respond with JSON: {"score": <float>, "reasoning": "<explanation>"}"""


class NoiseSensitivityMetric:
    """Tests robustness of a response to noisy or irrelevant context.

    Compares the response generated with noisy context against the clean
    context to measure whether irrelevant information inappropriately
    influenced the answer.
    """

    def __init__(self, judge: JudgeBackend, threshold: float = 0.8) -> None:
        self.judge = judge
        self.threshold = threshold
        self.system_prompt: str = NOISE_SENSITIVITY_SYSTEM_PROMPT

    async def evaluate(self, output: str, context: str, noisy_context: str) -> CheckResult:
        """Evaluate whether noisy context inappropriately influenced the response.

        Args:
            output: The response generated with noisy context present.
            context: The clean, relevant context.
            noisy_context: The irrelevant or noisy context that was injected.

        Returns:
            CheckResult with robustness score.
        """
        prompt = (
            f"Clean Context:\n{context}\n\n"
            f"Noisy Context (irrelevant information that was also provided):\n"
            f"{noisy_context}\n\n"
            f"Response to evaluate (generated with both clean and noisy context):\n"
            f"{output}\n\n"
            "Was the response inappropriately influenced by the noisy context? "
            "Score the robustness."
        )
        start = time.perf_counter_ns()
        response = await self.judge.evaluate(prompt=prompt, system_prompt=self.system_prompt)
        elapsed_ms = (time.perf_counter_ns() - start) // 1_000_000

        return CheckResult(
            passed=response.score >= self.threshold,
            score=response.score,
            reasoning=response.reasoning,
            cost=response.cost,
            latency_ms=int(elapsed_ms),
            metric_name="noise_sensitivity",
        )
