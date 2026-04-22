from __future__ import annotations

import time

from checkllm.judge import JudgeBackend
from checkllm.models import CheckResult

HALLUCINATION_SYSTEM_PROMPT = """You are an expert hallucination evaluator. Your job is to assess whether an LLM output is consistent with the provided context, treating the output as an answer to the user's query.

Score from 0.0 to 1.0:
- 1.0 = The output is fully consistent with the context and does not contradict or fabricate any facts. Direct short answers that correctly reflect the context earn 1.0.
- 0.5 = The output mixes supported and unsupported claims, or paraphrases the context with minor drift.
- 0.0 = The output contradicts the context, invents facts not present in it, or otherwise makes claims the context cannot back up.

Important guidance for short answers:
- Treat the output as a direct answer to the query. A short answer such as "yes", "no", a single number, a date, or a single named entity is NOT automatically unsupported.
- Score a short answer 1.0 when it is consistent with what the context implies about the query.
- Score a short answer 0.0 when it contradicts the context or picks a wrong entity/number/choice.
- Do not penalise short answers for "lack of claims to evaluate" — the claim is the answer itself, resolved against the query.

If the output is empty, nonsensical, or completely off-topic from the query, score 0.0.

Respond with JSON: {"score": <float>, "reasoning": "<explanation>"}"""


class HallucinationMetric:
    """Checks whether LLM output is grounded in the provided context."""

    def __init__(self, judge: JudgeBackend, threshold: float = 0.8) -> None:
        self.judge = judge
        self.threshold = threshold
        self.system_prompt: str = HALLUCINATION_SYSTEM_PROMPT

    async def evaluate(
        self,
        output: str,
        context: str,
        query: str | None = None,
    ) -> CheckResult:
        """Evaluate whether ``output`` is faithfully grounded in ``context``.

        Args:
            output: The LLM response to evaluate.
            context: The reference text the output should be grounded in.
            query: Optional user query the output is answering. When
                provided, the judge is told to treat ``output`` as a direct
                answer to ``query``, which meaningfully improves short-answer
                grading (numbers, named entities, yes/no) on datasets such as
                HaluBench's HaluEval subset.

        Returns:
            A CheckResult with the judged score, reasoning, and latency.
        """
        if query:
            prompt = (
                f"Query:\n{query}\n\n"
                f"Context:\n{context}\n\n"
                f"Output to evaluate:\n{output}\n\n"
                "Is the output a faithful answer to the query given the context? Score it."
            )
        else:
            prompt = (
                f"Context:\n{context}\n\n"
                f"Output to evaluate:\n{output}\n\n"
                "Is the output grounded in the context? Score it."
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
            metric_name="hallucination",
        )
