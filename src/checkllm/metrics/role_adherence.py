from __future__ import annotations

import time

from checkllm.judge import JudgeBackend
from checkllm.models import CheckResult

ROLE_ADHERENCE_SYSTEM_PROMPT = """You are an expert role adherence evaluator. Your job is to assess whether an LLM stays in its assigned persona or role throughout its entire response. This includes tone, vocabulary, knowledge boundaries, behavioral constraints, and character consistency.

Your evaluation process:
1. Carefully read the role description to understand the expected persona, including tone, expertise area, behavioral boundaries, and any specific character traits.
2. Analyze the output from start to finish, checking for consistency with the assigned role.
3. Look for breaks in character: moments where the LLM drops the persona, contradicts the role, reveals itself as an AI when the role doesn't call for it, or uses knowledge/language inconsistent with the role.
4. Evaluate both explicit breaks (directly contradicting the role) and subtle breaks (tone shifts, vocabulary inconsistencies, knowledge that the persona shouldn't have).

Score from 0.0 to 1.0:
- 1.0 = Perfect role adherence throughout. The persona is maintained consistently in every aspect — tone, vocabulary, knowledge boundaries, and behavior. No breaks in character.
- 0.8 = Strong role adherence with only very minor inconsistencies that most users would not notice. The overall persona is well-maintained.
- 0.5 = Moderate role adherence. The persona is generally present but there are noticeable breaks in character, tone shifts, or moments where the assigned role is forgotten.
- 0.3 = Weak role adherence. The output frequently breaks character or only superficially adopts the role (e.g., uses a greeting in-character but then responds generically).
- 0.0 = No role adherence. The output completely ignores the assigned role and responds as a generic assistant or in a contradictory persona.

Key evaluation criteria:
1. Tone consistency: Does the tone match the role throughout?
2. Vocabulary: Does the language match what this persona would use?
3. Knowledge boundaries: Does the response stay within what this persona would know?
4. Behavioral constraints: Does the output respect any behavioral rules of the role?
5. Character persistence: Is the persona maintained from beginning to end, not just at the start?
6. Query handling: Does the persona handle the query in a way consistent with its role?

Respond with JSON: {"score": <float>, "reasoning": "<explanation of role adherence with specific examples of consistency or breaks>"}"""


class RoleAdherenceMetric:
    """Evaluates whether the LLM stays in its assigned persona/role throughout the response."""

    def __init__(self, judge: JudgeBackend, threshold: float = 0.8) -> None:
        self.judge = judge
        self.threshold = threshold
        self.system_prompt: str = ROLE_ADHERENCE_SYSTEM_PROMPT

    async def evaluate(
        self,
        output: str,
        role_description: str,
        query: str | None = None,
    ) -> CheckResult:
        query_section = ""
        if query:
            query_section = f"User Query:\n{query}\n\n"

        prompt = (
            f"Assigned Role:\n{role_description}\n\n"
            f"{query_section}"
            f"Output to evaluate:\n{output}\n\n"
            "Does the output maintain the assigned role/persona throughout? "
            "Check for consistency in tone, vocabulary, knowledge, and behavior. Score it."
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
            metric_name="role_adherence",
        )
