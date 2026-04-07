from __future__ import annotations

import time

from checkllm.judge import JudgeBackend
from checkllm.models import CheckResult

RESPONSE_COMPLETENESS_SYSTEM_PROMPT = """You are an expert completeness evaluator. Your task is to assess whether a response fully addresses ALL parts of a question, including multi-part questions, compound queries, and questions with implicit sub-questions.

Many LLM responses partially answer a question by addressing the most obvious part while ignoring subtler sub-questions. This metric penalizes incomplete responses that leave parts of the question unanswered.

Evaluation process:
1. Parse the query to identify all distinct questions, sub-questions, and implicit information requests.
2. For each identified part, check whether the response provides a substantive answer.
3. Distinguish between parts that are merely acknowledged and parts that are fully answered.
4. Compute a score based on the proportion of fully addressed parts.

Score from 0.0 to 1.0:
- 1.0 = Every part of the question is fully and substantively addressed.
- 0.8 = Most parts are addressed; one minor sub-question may be only partially covered.
- 0.5 = About half the question parts are fully addressed; others are missing or superficial.
- 0.3 = Only one part of a multi-part question is addressed.
- 0.0 = The response does not address any part of the question.

Key evaluation criteria:
1. How many distinct parts does the question have?
2. Is each part answered with substance, not just acknowledged?
3. Are implicit sub-questions (e.g., "why" implied by "how") addressed?
4. Does the response provide depth proportional to each part's complexity?

Respond with JSON: {"score": <float>, "reasoning": "<explanation>"}"""


class ResponseCompletenessMetric:
    """Evaluates whether a response fully addresses all parts of a query.

    Parses multi-part questions to identify all sub-questions and checks
    that each one receives a substantive answer. Penalizes partial
    responses that ignore parts of the question.
    """

    def __init__(self, judge: JudgeBackend, threshold: float = 0.8) -> None:
        self.judge = judge
        self.threshold = threshold
        self.system_prompt: str = RESPONSE_COMPLETENESS_SYSTEM_PROMPT

    async def evaluate(self, output: str, query: str) -> CheckResult:
        """Evaluate whether the response addresses all parts of the query.

        Args:
            output: The response to evaluate.
            query: The original question (potentially multi-part).

        Returns:
            CheckResult with completeness score.
        """
        prompt = (
            f"Original Query:\n{query}\n\n"
            f"Response to evaluate:\n{output}\n\n"
            "Identify all parts of the query and assess whether each is "
            "fully addressed in the response. Score the completeness."
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
            metric_name="response_completeness",
        )
