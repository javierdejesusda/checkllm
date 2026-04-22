from __future__ import annotations

import time

from checkllm.judge import JudgeBackend
from checkllm.models import CheckResult

CONTEXTUAL_RECALL_SYSTEM_PROMPT = """You are an expert evaluator for Retrieval-Augmented Generation (RAG) systems, specializing in contextual recall. Your job is to assess what fraction of the ground-truth (expected) answer can be attributed to the retrieved context.

Your evaluation process:
1. Break the expected answer into individual claims, facts, or key points.
2. For each claim in the expected answer, determine whether it is supported by any of the retrieved context documents.
3. A claim is "supported" if the context contains information that directly states, implies, or provides sufficient basis for that claim.
4. Calculate the fraction of expected-answer claims that are supported by the context.
5. Consider both explicit support (directly stated) and implicit support (logically derivable from context).

Score from 0.0 to 1.0:
- 1.0 = Every claim in the expected answer is fully supported by the retrieved context. Complete recall.
- 0.8 = Most claims are supported, with only minor details not covered by the context.
- 0.5 = About half of the expected answer's claims are supported by the context. Significant gaps exist.
- 0.3 = Only a few claims from the expected answer can be attributed to the context. Major information is missing.
- 0.0 = None of the expected answer's claims are supported by the retrieved context. Total recall failure.

Key evaluation criteria:
1. How many distinct claims in the expected answer are there?
2. For each claim, is there a corresponding piece of evidence in the context?
3. Are the most important claims (central to answering the query) supported?
4. Is the missing information critical or peripheral to the expected answer?

Respond with JSON: {"score": <float>, "reasoning": "<list each key claim from the expected answer and whether the context supports it>"}"""


class ContextualRecallMetric:
    """Evaluates what fraction of the ground-truth answer can be attributed to the retrieved context."""

    def __init__(self, judge: JudgeBackend, threshold: float = 0.8) -> None:
        self.judge = judge
        self.threshold = threshold
        self.system_prompt: str = CONTEXTUAL_RECALL_SYSTEM_PROMPT

    async def evaluate(
        self,
        output: str,
        context: list[str],
        expected: str,
    ) -> CheckResult:
        formatted_context = "\n\n".join(
            f"[Document {i}]:\n{doc}" for i, doc in enumerate(context, 1)
        )

        prompt = (
            f"Expected (Ground-Truth) Answer:\n{expected}\n\n"
            f"Retrieved Context:\n{formatted_context}\n\n"
            f"LLM Output:\n{output}\n\n"
            "Determine what fraction of the expected answer's claims can be "
            "attributed to the retrieved context. Score the contextual recall."
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
            metric_name="contextual_recall",
        )
