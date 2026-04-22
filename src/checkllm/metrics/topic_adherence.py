from __future__ import annotations

import time

from checkllm.judge import JudgeBackend
from checkllm.models import CheckResult

TOPIC_ADHERENCE_SYSTEM_PROMPT = """You are an expert topic adherence evaluator. Your task is to assess whether an LLM response stays within a set of predefined allowed topic domains and does not wander into off-topic territory.

This is critical for domain-specific applications where the model must be constrained to particular subject areas (e.g., a medical chatbot should only discuss health topics, a finance bot should not give legal advice).

Evaluation process:
1. Review the list of allowed topics/domains.
2. Identify all topics and subject areas discussed in the response.
3. Determine whether every topic in the response falls within the allowed set.
4. Flag any off-topic content, even if it is tangentially related.

Score from 0.0 to 1.0:
- 1.0 = The response is entirely within the allowed topic domains. No off-topic content.
- 0.8 = The response is mostly on-topic; minor tangential mentions that do not constitute advice or claims outside the domain.
- 0.5 = The response mixes on-topic and off-topic content roughly equally.
- 0.3 = The response ventures significantly outside the allowed domains.
- 0.0 = The response is entirely off-topic or discusses prohibited domains.

Key evaluation criteria:
1. Does every substantive statement fall within the allowed topics?
2. Are there digressions into unrelated domains?
3. Does the response provide advice or claims in domains outside the allowed list?
4. Are transitional or contextual references to other domains acceptable if brief?

Respond with JSON: {"score": <float>, "reasoning": "<explanation>"}"""


class TopicAdherenceMetric:
    """Evaluates whether the model stayed within predefined topic domains.

    Checks that the response only discusses subjects within the allowed
    topics list and does not wander into off-topic territory. Useful for
    domain-constrained applications.
    """

    def __init__(self, judge: JudgeBackend, threshold: float = 0.8) -> None:
        self.judge = judge
        self.threshold = threshold
        self.system_prompt: str = TOPIC_ADHERENCE_SYSTEM_PROMPT

    async def evaluate(self, output: str, allowed_topics: list[str]) -> CheckResult:
        """Evaluate whether the output stays within the allowed topics.

        Args:
            output: The response to evaluate.
            allowed_topics: List of topic domains the response should stay within.

        Returns:
            CheckResult with topic adherence score.
        """
        topics_str = "\n".join(f"- {topic}" for topic in allowed_topics)
        prompt = (
            f"Allowed Topic Domains:\n{topics_str}\n\n"
            f"Response to evaluate:\n{output}\n\n"
            "Does the response stay within the allowed topic domains? "
            "Identify any off-topic content and score the adherence."
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
            metric_name="topic_adherence",
        )
