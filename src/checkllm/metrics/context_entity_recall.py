from __future__ import annotations

import time

from checkllm.judge import JudgeBackend
from checkllm.models import CheckResult

CONTEXT_ENTITY_RECALL_SYSTEM_PROMPT = """You are an expert entity recall evaluator. Your task is to extract named entities from the reference answer and evaluate what proportion of those entities are present in the provided context.

This metric measures the quality of retrieved context by checking whether it contains the entities needed to produce the reference answer. High entity recall means the retrieval step found context that covers the key facts.

Evaluation process:
1. Extract all named entities from the reference (people, places, organizations, dates, numerical values, technical terms, product names, etc.).
2. For each entity in the reference, check if it appears in the context (allowing minor variations like abbreviations or synonyms).
3. Compute entity recall as the proportion of reference entities found in the context.

Score from 0.0 to 1.0:
- 1.0 = Every entity in the reference is present in the context.
- 0.8 = Most entities are covered; only minor peripheral entities are missing.
- 0.5 = About half the reference entities appear in the context.
- 0.3 = Most reference entities are missing from the context.
- 0.0 = None of the reference entities are found in the context.

Key evaluation criteria:
1. Are the primary entities (the most important ones for answering) present?
2. Are secondary entities (supporting details) covered?
3. Do entity variants (abbreviations, aliases) count as matches?
4. Are numerical values and dates matched?

Respond with JSON: {"score": <float>, "reasoning": "<explanation>"}"""


class ContextEntityRecallMetric:
    """Evaluates entity-level recall between retrieved context and reference.

    Extracts named entities from the reference answer and measures what
    proportion appear in the provided context. This assesses retrieval
    quality by checking if the context contains the entities necessary
    to produce the correct answer.
    """

    def __init__(self, judge: JudgeBackend, threshold: float = 0.8) -> None:
        self.judge = judge
        self.threshold = threshold
        self.system_prompt: str = CONTEXT_ENTITY_RECALL_SYSTEM_PROMPT

    async def evaluate(self, context: str, reference: str) -> CheckResult:
        """Evaluate entity recall of context against a reference answer.

        Args:
            context: The retrieved context to evaluate.
            reference: The known-correct reference answer containing target entities.

        Returns:
            CheckResult with entity recall score.
        """
        prompt = (
            f"Reference Answer (source of expected entities):\n{reference}\n\n"
            f"Retrieved Context:\n{context}\n\n"
            "Extract entities from the reference and check which ones "
            "appear in the context. Score the entity recall."
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
            metric_name="context_entity_recall",
        )
