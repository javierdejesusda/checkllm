from __future__ import annotations

import time

from checkllm.judge import JudgeBackend
from checkllm.models import CheckResult

ANSWER_COMPLETENESS_SYSTEM_PROMPT = """You are an expert answer completeness evaluator. Your job is to assess whether the answer fully addresses all parts and aspects of the question. A complete answer covers every sub-question, constraint, and requirement expressed in the query.

Score from 0.0 to 1.0:
- 1.0 = The answer comprehensively addresses every part of the question. All sub-questions are answered, all requested details are provided, and nothing is left unanswered.
- 0.8 = The answer addresses the main aspects of the question with only minor omissions that don't significantly impact usefulness.
- 0.5 = The answer partially addresses the question. Some important parts are covered but others are missing or only superficially addressed.
- 0.3 = The answer addresses only a small portion of the question. Major aspects are missing or ignored.
- 0.0 = The answer does not address the question at all, or provides completely irrelevant content.

Key evaluation criteria:
1. Does the answer address every distinct sub-question or component of the query?
2. Are all explicitly requested details, examples, or explanations provided?
3. Does the answer satisfy any constraints mentioned in the query (e.g., "list three", "explain why")?
4. Is the depth of the answer appropriate for the complexity of the question?
5. Are there any obvious gaps or aspects the answer should have covered but didn't?

Respond with JSON: {"score": <float>, "reasoning": "<explanation>"}"""


class AnswerCompletenessMetric:
    """Checks whether an answer fully addresses all parts of the question."""

    def __init__(self, judge: JudgeBackend, threshold: float = 0.8) -> None:
        self.judge = judge
        self.threshold = threshold
        self.system_prompt: str = ANSWER_COMPLETENESS_SYSTEM_PROMPT

    async def evaluate(self, output: str, query: str) -> CheckResult:
        prompt = (
            f"Question:\n{query}\n\n"
            f"Answer to evaluate:\n{output}\n\n"
            "Does the answer fully address all parts of the question? "
            "Are there any gaps or missing aspects? Score it."
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
            metric_name="answer_completeness",
        )
