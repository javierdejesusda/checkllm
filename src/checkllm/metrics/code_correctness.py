from __future__ import annotations

import time

from checkllm.judge import JudgeBackend
from checkllm.models import CheckResult

CODE_CORRECTNESS_SYSTEM_PROMPT = """You are an expert code correctness evaluator. Your task is to assess whether generated code is correct, complete, and adheres to the stated requirements.

You should evaluate the code as a senior engineer would during a code review, focusing on functional correctness rather than style preferences.

Evaluation process:
1. Read the requirements carefully to understand what the code should do.
2. Trace through the code logic to verify it implements the requirements.
3. Check for common bugs: off-by-one errors, null/undefined handling, type errors, infinite loops, missing edge cases.
4. Verify the code handles error conditions appropriately.
5. Assess whether the code is complete (no missing functions, imports, or setup).

Score from 0.0 to 1.0:
- 1.0 = The code is fully correct, handles all edge cases, and completely implements the requirements.
- 0.8 = The code is correct for the main cases; minor edge cases may not be handled but the core logic is sound.
- 0.5 = The code partially implements the requirements; some functions work but others have bugs or are missing.
- 0.3 = The code has significant logical errors that would cause failures in common cases.
- 0.0 = The code is fundamentally broken, does not compile/run, or does not address the requirements at all.

Key evaluation criteria:
1. Does the code implement all stated requirements?
2. Is the control flow logically correct?
3. Are data structures used appropriately?
4. Are boundary conditions and edge cases handled?
5. Would the code produce correct output for typical inputs?

Respond with JSON: {"score": <float>, "reasoning": "<explanation>"}"""


class CodeCorrectnessMetric:
    """Evaluates generated code for correctness against requirements.

    Assesses whether code correctly implements stated requirements,
    handles edge cases, and is free of logical bugs. Evaluates
    functional correctness rather than style.
    """

    def __init__(self, judge: JudgeBackend, threshold: float = 0.8) -> None:
        self.judge = judge
        self.threshold = threshold
        self.system_prompt: str = CODE_CORRECTNESS_SYSTEM_PROMPT

    async def evaluate(self, output: str, requirements: str) -> CheckResult:
        """Evaluate code correctness against stated requirements.

        Args:
            output: The generated code to evaluate.
            requirements: Description of what the code should do.

        Returns:
            CheckResult with correctness score.
        """
        prompt = (
            f"Requirements:\n{requirements}\n\n"
            f"Generated Code:\n{output}\n\n"
            "Does the code correctly and completely implement the requirements? "
            "Check for bugs, missing logic, and edge cases. Score it."
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
            metric_name="code_correctness",
        )
