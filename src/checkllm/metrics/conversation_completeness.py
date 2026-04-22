from __future__ import annotations

import time

from checkllm.conversation import ConversationalTestCase
from checkllm.judge import JudgeBackend
from checkllm.models import CheckResult

CONVERSATION_COMPLETENESS_SYSTEM_PROMPT = """You are an expert evaluator of conversation completeness. Your job is to assess whether all of the user's requests and questions across the full multi-turn exchange were adequately addressed by the assistant.

Score from 0.0 to 1.0:
- 1.0 = Every user request, question, and sub-question across all turns was fully addressed. Nothing was left unanswered or incomplete.
- 0.8 = Nearly all user requests were addressed. Only minor or tangential aspects were missed, and the overall conversation goal was achieved.
- 0.5 = About half of the user's requests were addressed. Some important questions or tasks were completed, but others were missed or only partially handled.
- 0.3 = Only a small portion of the user's requests were addressed. Most questions or tasks remain unanswered or incomplete.
- 0.0 = None of the user's requests were addressed. The assistant failed to engage with the user's needs entirely.

Evaluation steps:
1. Identify all distinct user requests, questions, and tasks across every user turn in the conversation.
2. For each identified request, check whether the assistant provided an adequate response in any subsequent turn.
3. Score based on the completeness ratio — the fraction of user requests that were satisfactorily addressed.
4. Consider follow-up requests that refine or extend earlier ones as separate items to be addressed.
5. Weight requests by importance — a missed core request should penalize more than a missed minor clarification.

Respond with JSON: {"score": <float>, "reasoning": "<explanation>"}"""


class ConversationCompletenessMetric:
    """Evaluates whether all user requests across a multi-turn conversation
    were addressed by the assistant.

    Score interpretation: 1.0 = fully complete, 0.0 = nothing addressed.
    """

    def __init__(self, judge: JudgeBackend, threshold: float = 0.8) -> None:
        self.judge = judge
        self.threshold = threshold
        self.system_prompt: str = CONVERSATION_COMPLETENESS_SYSTEM_PROMPT

    async def evaluate(self, conversation: ConversationalTestCase) -> CheckResult:
        transcript = conversation.format_transcript()
        prompt = (
            f"Conversation transcript:\n{transcript}\n\n"
            "Identify every distinct user request, question, and task across "
            "the conversation. For each, determine whether the assistant "
            "adequately addressed it. Score based on the overall completeness "
            "ratio."
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
            metric_name="conversation_completeness",
        )
