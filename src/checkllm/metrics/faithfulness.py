from __future__ import annotations

import time

from checkllm.judge import JudgeBackend
from checkllm.models import CheckResult

FAITHFULNESS_SYSTEM_PROMPT = """You are an expert faithfulness evaluator for Retrieval-Augmented Generation (RAG) systems. Your job is to assess whether the answer is faithful to the retrieved context — meaning it does not introduce any information, claims, or inferences that are not directly supported by the provided context.

This is different from hallucination detection: faithfulness focuses specifically on whether the answer adds unsupported claims beyond what the context provides, even if those claims might be true in general.

Score from 0.0 to 1.0:
- 1.0 = Every statement in the answer is directly supported by or logically derivable from the context. No external information is introduced.
- 0.8 = The answer is mostly faithful with only trivial additions (e.g., common knowledge connectors) that don't change meaning.
- 0.5 = The answer mixes supported claims with unsupported ones. Some statements go beyond the context.
- 0.3 = The answer introduces significant information not found in the context, even if it may be factually correct.
- 0.0 = The answer is largely fabricated or draws heavily from knowledge outside the context.

Key evaluation criteria:
1. Does every factual claim in the answer trace back to the context?
2. Are there inferences made that the context does not support?
3. Does the answer introduce external knowledge not present in the context?
4. Are numerical values, dates, or names consistent with the context?

Respond with JSON: {"score": <float>, "reasoning": "<explanation>"}"""


class FaithfulnessMetric:
    """Checks whether an answer is faithful to retrieved context in a RAG pipeline.

    Unlike hallucination detection which checks general grounding, faithfulness
    specifically evaluates whether the answer introduces unsupported claims
    beyond what the provided context contains.
    """

    def __init__(self, judge: JudgeBackend, threshold: float = 0.8) -> None:
        self.judge = judge
        self.threshold = threshold
        self.system_prompt: str = FAITHFULNESS_SYSTEM_PROMPT

    async def evaluate(self, output: str, context: str, query: str | None = None) -> CheckResult:
        parts = [f"Retrieved Context:\n{context}\n"]
        if query:
            parts.append(f"Original Query:\n{query}\n")
        parts.append(f"Answer to evaluate:\n{output}\n")
        parts.append(
            "Is the answer faithful to the retrieved context? "
            "Does it introduce any unsupported claims? Score it."
        )
        prompt = "\n".join(parts)

        start = time.perf_counter_ns()
        response = await self.judge.evaluate(prompt=prompt, system_prompt=self.system_prompt)
        elapsed_ms = (time.perf_counter_ns() - start) // 1_000_000

        return CheckResult(
            passed=response.score >= self.threshold,
            score=response.score,
            reasoning=response.reasoning,
            cost=response.cost,
            latency_ms=int(elapsed_ms),
            metric_name="faithfulness",
        )
