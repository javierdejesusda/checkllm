from __future__ import annotations

import time

from checkllm.judge import JudgeBackend
from checkllm.models import CheckResult

NV_CONTEXT_RELEVANCE_PROMPT_A = """You are an expert evaluator for RAG context relevance. Rate how relevant the retrieved context is for answering the given query.

Score from 0.0 to 1.0:
- 1.0 = The context is perfectly relevant and contains all information needed to answer the query.
- 0.7 = The context is mostly relevant with minor gaps.
- 0.5 = The context is partially relevant. Some useful information but significant gaps.
- 0.3 = The context has low relevance to the query.
- 0.0 = The context is completely irrelevant to the query.

Respond with JSON: {"score": <float>, "reasoning": "<explanation>"}"""

NV_CONTEXT_RELEVANCE_PROMPT_B = """You are an expert evaluator for RAG context quality. Rate how much irrelevant information is present in the retrieved context relative to the query.

Score from 0.0 to 1.0:
- 1.0 = The context contains NO irrelevant information. Every piece is useful for the query.
- 0.7 = The context contains mostly relevant information with minor irrelevant portions.
- 0.5 = The context contains a roughly equal mix of relevant and irrelevant information.
- 0.3 = The context contains mostly irrelevant information with only small relevant portions.
- 0.0 = The context is entirely irrelevant noise with no useful information for the query.

Respond with JSON: {"score": <float>, "reasoning": "<explanation>"}"""

NV_ANSWER_ACCURACY_PROMPT_A = """You are an expert evaluator for answer accuracy. Rate how accurate the response is compared to the reference answer.

Score from 0.0 to 1.0:
- 1.0 = The response is perfectly accurate. All facts match the reference answer.
- 0.7 = The response is mostly accurate with minor discrepancies.
- 0.5 = The response is partially accurate. Some facts match but others don't.
- 0.3 = The response has significant inaccuracies compared to the reference.
- 0.0 = The response is completely inaccurate.

Respond with JSON: {"score": <float>, "reasoning": "<explanation>"}"""

NV_ANSWER_ACCURACY_PROMPT_B = """You are an expert evaluator for answer completeness. Rate how accurately the reference answer represents the content in the response — checking if the response contains valid information beyond or different from the reference.

Score from 0.0 to 1.0:
- 1.0 = The reference fully captures the response content. They are semantically equivalent.
- 0.7 = The reference mostly captures the response with minor additional valid points in the response.
- 0.5 = The reference partially captures the response. Significant valid content in the response is not in the reference.
- 0.3 = The reference captures little of the response content.
- 0.0 = The reference and response are completely different.

Respond with JSON: {"score": <float>, "reasoning": "<explanation>"}"""

NV_GROUNDEDNESS_PROMPT_A = """You are an expert evaluator for response groundedness. Rate whether the response is grounded in the provided contexts — meaning every claim can be traced back to the contexts.

Score from 0.0 to 1.0:
- 1.0 = The response is fully grounded. Every statement is supported by the contexts.
- 0.7 = The response is mostly grounded with minor unsupported additions.
- 0.5 = The response is partially grounded. Some claims are supported, others are not.
- 0.3 = The response is poorly grounded. Most claims lack support from the contexts.
- 0.0 = The response has no grounding in the contexts.

Respond with JSON: {"score": <float>, "reasoning": "<explanation>"}"""

NV_GROUNDEDNESS_PROMPT_B = """You are an expert evaluator for information addition. Rate how much information the response adds that is NOT present in the provided contexts.

Score from 0.0 to 1.0:
- 1.0 = The response adds NO information beyond the contexts. Fully contained.
- 0.7 = The response adds minimal information beyond the contexts.
- 0.5 = The response adds a moderate amount of information not in the contexts.
- 0.3 = The response adds significant information not in the contexts.
- 0.0 = The response is almost entirely composed of information not in the contexts.

Respond with JSON: {"score": <float>, "reasoning": "<explanation>"}"""


class NVContextRelevanceMetric:
    """Dual-judge context relevance metric using NVIDIA's approach.

    Uses two different evaluation prompts and averages their scores for
    more robust assessment. Prompt A evaluates relevance directly while
    Prompt B evaluates the amount of irrelevant information (inverted).
    """

    def __init__(self, judge: JudgeBackend, threshold: float = 0.7) -> None:
        self.judge = judge
        self.threshold = threshold
        self.system_prompt_a: str = NV_CONTEXT_RELEVANCE_PROMPT_A
        self.system_prompt_b: str = NV_CONTEXT_RELEVANCE_PROMPT_B

    async def evaluate(
        self,
        context: str,
        query: str,
    ) -> CheckResult:
        """Evaluate context relevance using dual-judge approach.

        Args:
            context: The retrieved context to evaluate.
            query: The query the context should be relevant to.

        Returns:
            A CheckResult with averaged dual-judge score and reasoning.
        """
        prompt_a = (
            f"Query:\n{query}\n\n"
            f"Retrieved Context:\n{context}\n\n"
            "Rate how relevant the context is for answering the query."
        )
        prompt_b = (
            f"Query:\n{query}\n\n"
            f"Retrieved Context:\n{context}\n\n"
            "Rate how much of the context is relevant versus irrelevant "
            "for the given query."
        )

        start = time.perf_counter_ns()
        response_a = await self.judge.evaluate(prompt=prompt_a, system_prompt=self.system_prompt_a)
        response_b = await self.judge.evaluate(prompt=prompt_b, system_prompt=self.system_prompt_b)
        elapsed_ms = (time.perf_counter_ns() - start) // 1_000_000

        avg_score = (response_a.score + response_b.score) / 2.0
        total_cost = response_a.cost + response_b.cost
        reasoning = (
            f"Judge A (relevance): {response_a.score:.2f} - {response_a.reasoning} | "
            f"Judge B (irrelevance check): {response_b.score:.2f} - {response_b.reasoning}"
        )

        return CheckResult(
            passed=avg_score >= self.threshold,
            score=avg_score,
            reasoning=reasoning,
            cost=total_cost,
            latency_ms=int(elapsed_ms),
            metric_name="nv_context_relevance",
        )


class NVAnswerAccuracyMetric:
    """Dual-judge answer accuracy metric using NVIDIA's approach.

    Uses two prompts with swapped roles: one checks response against reference,
    the other checks reference against response. Scores are averaged.
    """

    def __init__(self, judge: JudgeBackend, threshold: float = 0.7) -> None:
        self.judge = judge
        self.threshold = threshold
        self.system_prompt_a: str = NV_ANSWER_ACCURACY_PROMPT_A
        self.system_prompt_b: str = NV_ANSWER_ACCURACY_PROMPT_B

    async def evaluate(
        self,
        response_text: str,
        reference: str,
        query: str | None = None,
    ) -> CheckResult:
        """Evaluate answer accuracy using dual-judge approach.

        Args:
            response_text: The response to evaluate.
            reference: The reference answer to compare against.
            query: Optional query that produced the response.

        Returns:
            A CheckResult with averaged dual-judge score and reasoning.
        """
        query_part = f"Query:\n{query}\n\n" if query else ""
        prompt_a = (
            f"{query_part}"
            f"Reference Answer:\n{reference}\n\n"
            f"Response to Evaluate:\n{response_text}\n\n"
            "How accurate is the response compared to the reference answer?"
        )
        prompt_b = (
            f"{query_part}"
            f"Response:\n{response_text}\n\n"
            f"Reference Answer:\n{reference}\n\n"
            "How accurately does the reference represent the response content?"
        )

        start = time.perf_counter_ns()
        response_a = await self.judge.evaluate(prompt=prompt_a, system_prompt=self.system_prompt_a)
        response_b = await self.judge.evaluate(prompt=prompt_b, system_prompt=self.system_prompt_b)
        elapsed_ms = (time.perf_counter_ns() - start) // 1_000_000

        avg_score = (response_a.score + response_b.score) / 2.0
        total_cost = response_a.cost + response_b.cost
        reasoning = (
            f"Judge A (response vs reference): {response_a.score:.2f} - "
            f"{response_a.reasoning} | "
            f"Judge B (reference vs response): {response_b.score:.2f} - "
            f"{response_b.reasoning}"
        )

        return CheckResult(
            passed=avg_score >= self.threshold,
            score=avg_score,
            reasoning=reasoning,
            cost=total_cost,
            latency_ms=int(elapsed_ms),
            metric_name="nv_answer_accuracy",
        )


class NVResponseGroundednessMetric:
    """Dual-judge groundedness metric using NVIDIA's approach.

    Uses two prompts: one checks if the response is grounded in contexts,
    the other checks if the response adds information not in contexts (inverted).
    Scores are averaged.
    """

    def __init__(self, judge: JudgeBackend, threshold: float = 0.7) -> None:
        self.judge = judge
        self.threshold = threshold
        self.system_prompt_a: str = NV_GROUNDEDNESS_PROMPT_A
        self.system_prompt_b: str = NV_GROUNDEDNESS_PROMPT_B

    async def evaluate(
        self,
        response_text: str,
        contexts: list[str],
    ) -> CheckResult:
        """Evaluate response groundedness using dual-judge approach.

        Args:
            response_text: The response to evaluate for groundedness.
            contexts: A list of context strings the response should be grounded in.

        Returns:
            A CheckResult with averaged dual-judge score and reasoning.
        """
        formatted_contexts = "\n\n".join(
            f"Context {i + 1}:\n{ctx}" for i, ctx in enumerate(contexts)
        )
        prompt_a = (
            f"Contexts:\n{formatted_contexts}\n\n"
            f"Response:\n{response_text}\n\n"
            "Is the response grounded in the provided contexts?"
        )
        prompt_b = (
            f"Contexts:\n{formatted_contexts}\n\n"
            f"Response:\n{response_text}\n\n"
            "Does the response add information not present in the contexts?"
        )

        start = time.perf_counter_ns()
        response_a = await self.judge.evaluate(prompt=prompt_a, system_prompt=self.system_prompt_a)
        response_b = await self.judge.evaluate(prompt=prompt_b, system_prompt=self.system_prompt_b)
        elapsed_ms = (time.perf_counter_ns() - start) // 1_000_000

        avg_score = (response_a.score + response_b.score) / 2.0
        total_cost = response_a.cost + response_b.cost
        reasoning = (
            f"Judge A (groundedness): {response_a.score:.2f} - "
            f"{response_a.reasoning} | "
            f"Judge B (info addition): {response_b.score:.2f} - "
            f"{response_b.reasoning}"
        )

        return CheckResult(
            passed=avg_score >= self.threshold,
            score=avg_score,
            reasoning=reasoning,
            cost=total_cost,
            latency_ms=int(elapsed_ms),
            metric_name="nv_response_groundedness",
        )
