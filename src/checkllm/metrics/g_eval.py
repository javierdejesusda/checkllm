from __future__ import annotations

import time

from checkllm.judge import JudgeBackend
from checkllm.models import CheckResult

G_EVAL_SYSTEM_PROMPT = """You are an expert LLM output evaluator using the G-Eval framework. Your job is to evaluate an LLM output against custom criteria using chain-of-thought reasoning.

Your evaluation process:
1. First, carefully read the evaluation criteria provided.
2. If evaluation steps are provided, follow them in order. If no steps are provided, generate your own detailed evaluation steps before scoring.
3. For each step, reason explicitly about how well the output meets that aspect of the criteria.
4. After completing all steps, assign a final score that reflects your step-by-step analysis.

Score from 0.0 to 1.0:
- 1.0 = The output fully satisfies all aspects of the evaluation criteria. Every step yields a positive assessment.
- 0.8 = The output satisfies most criteria with only minor shortcomings in one or two steps.
- 0.5 = The output partially meets the criteria. Some steps yield positive assessments while others reveal clear deficiencies.
- 0.3 = The output fails to meet most of the criteria. The majority of evaluation steps reveal problems.
- 0.0 = The output completely fails to meet the criteria. No evaluation step yields a positive result.

Important guidelines:
- Be rigorous and objective in your evaluation. Do not give high scores simply because the output is long or well-formatted.
- Each evaluation step should contribute meaningfully to the final score.
- Your reasoning must reference specific parts of the output and criteria.
- If the criteria are ambiguous, interpret them in the most reasonable way and note your interpretation.

Respond with JSON: {"score": <float>, "reasoning": "<step-by-step explanation leading to final score>"}"""


class GEvalMetric:
    """G-Eval: Flexible custom criteria evaluation using chain-of-thought reasoning."""

    def __init__(self, judge: JudgeBackend, threshold: float = 0.8) -> None:
        self.judge = judge
        self.threshold = threshold
        self.system_prompt: str = G_EVAL_SYSTEM_PROMPT

    async def evaluate(
        self,
        output: str,
        criteria: str,
        steps: list[str] | None = None,
    ) -> CheckResult:
        steps_section = ""
        if steps:
            formatted_steps = "\n".join(
                f"  Step {i}: {step}" for i, step in enumerate(steps, 1)
            )
            steps_section = (
                f"\nEvaluation Steps (follow these in order):\n{formatted_steps}\n"
            )
        else:
            steps_section = (
                "\nNo evaluation steps were provided. Generate your own "
                "detailed evaluation steps before scoring.\n"
            )

        prompt = (
            f"Evaluation Criteria:\n{criteria}\n"
            f"{steps_section}\n"
            f"Output to evaluate:\n{output}\n\n"
            "Evaluate the output against the criteria using chain-of-thought reasoning. "
            "Work through each evaluation step, then assign a final score."
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
            metric_name="g_eval",
        )
