from __future__ import annotations

import time

from checkllm.conversation import ConversationalTestCase
from checkllm.judge import JudgeBackend
from checkllm.models import CheckResult

KNOWLEDGE_RETENTION_SYSTEM_PROMPT = """You are an expert evaluator of knowledge retention in multi-turn conversations. Your job is to assess whether the assistant remembered and correctly used information from earlier turns in the conversation.

Score from 0.0 to 1.0:
- 1.0 = The assistant perfectly retained all facts, preferences, and context introduced earlier in the conversation. Every reference to prior information is accurate and consistent.
- 0.8 = The assistant retained most earlier information with only minor lapses that do not significantly affect the quality of its responses.
- 0.5 = The assistant partially retained earlier information. Some facts are remembered and used correctly, but others are forgotten, ignored, or used inaccurately.
- 0.3 = The assistant retained very little earlier information. Most facts introduced earlier are forgotten or contradicted in later turns.
- 0.0 = The assistant showed no retention of earlier information. It contradicts or completely ignores previously established facts, preferences, and context.

Key evaluation criteria:
1. Facts mentioned by the user earlier that should be referenced or built upon in later turns — does the assistant recall them?
2. Contradictions with earlier statements — does the assistant contradict facts it previously stated or that the user provided?
3. Appropriate callbacks to earlier context — when relevant, does the assistant refer back to information established earlier in the conversation?
4. Consistency of the assistant's own statements — does the assistant remain consistent with its own prior answers?
5. User preferences and constraints — does the assistant remember preferences, names, numbers, or constraints the user mentioned earlier?

Respond with JSON: {"score": <float>, "reasoning": "<explanation>"}"""


class KnowledgeRetentionMetric:
    """Evaluates whether the assistant retained and correctly used information
    from earlier turns in a multi-turn conversation.

    Score interpretation: 1.0 = perfect retention, 0.0 = no retention.
    """

    def __init__(self, judge: JudgeBackend, threshold: float = 0.8) -> None:
        self.judge = judge
        self.threshold = threshold
        self.system_prompt: str = KNOWLEDGE_RETENTION_SYSTEM_PROMPT

    async def evaluate(self, conversation: ConversationalTestCase) -> CheckResult:
        transcript = conversation.format_transcript()
        prompt = (
            f"Conversation transcript:\n{transcript}\n\n"
            "Evaluate the assistant's knowledge retention across this conversation. "
            "Did the assistant remember and correctly use facts, preferences, and "
            "context from earlier turns? Identify any contradictions or forgotten "
            "information. Score it."
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
            metric_name="knowledge_retention",
        )
