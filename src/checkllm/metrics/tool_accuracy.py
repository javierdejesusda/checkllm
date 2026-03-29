from __future__ import annotations

import json
import time

from checkllm.judge import JudgeBackend
from checkllm.models import CheckResult

TOOL_ACCURACY_SYSTEM_PROMPT = """You are an expert evaluator for LLM agent tool usage. Your job is to assess whether an agent selected the correct tools with the correct parameters to accomplish a task. You will be given the agent's execution trace and the expected tool calls.

Your evaluation process:
1. Parse the agent trace to identify all tool calls the agent made, including tool names and parameters.
2. Compare each expected tool call against the agent's actual tool calls.
3. For each expected tool, check: (a) Was the correct tool selected? (b) Were the parameters correct?
4. Note any extra tool calls the agent made that were not expected (may indicate confusion or inefficiency).
5. Note any expected tool calls that were missing from the agent's trace.

Score from 0.0 to 1.0:
- 1.0 = All expected tools were called with correct parameters. No unnecessary tool calls. Perfect tool usage.
- 0.8 = All expected tools were called correctly, but there were minor parameter differences (e.g., equivalent values in different formats) or one unnecessary extra call.
- 0.5 = Some expected tools were called correctly, but others were missed or called with wrong parameters. Or the right tools were used but in a clearly suboptimal way.
- 0.3 = Most expected tools were not called, or were called with substantially wrong parameters. The agent showed poor tool selection.
- 0.0 = No expected tools were called correctly. The agent either used completely wrong tools, used no tools, or failed entirely.

Key evaluation criteria:
1. Tool selection accuracy: Were the correct tools chosen for the task?
2. Parameter correctness: Were the parameters passed to each tool accurate and complete?
3. Tool call ordering: Were tools called in a logical and effective order?
4. Completeness: Were all necessary tool calls made?
5. Efficiency: Were there unnecessary or redundant tool calls?
6. Parameter equivalence: Accept equivalent parameter values (e.g., "New York" vs "new york" for a city parameter) unless exact matching is critical.

Respond with JSON: {"score": <float>, "reasoning": "<detailed comparison of expected vs actual tool calls>"}"""


class ToolAccuracyMetric:
    """Evaluates whether an agent selected the correct tools with correct parameters."""

    def __init__(self, judge: JudgeBackend, threshold: float = 0.8) -> None:
        self.judge = judge
        self.threshold = threshold
        self.system_prompt: str = TOOL_ACCURACY_SYSTEM_PROMPT

    async def evaluate(
        self,
        output: str,
        expected_tools: list[dict[str, object]],
        query: str,
    ) -> CheckResult:
        formatted_tools = json.dumps(expected_tools, indent=2)

        prompt = (
            f"User Query:\n{query}\n\n"
            f"Expected Tool Calls:\n{formatted_tools}\n\n"
            f"Agent Trace:\n{output}\n\n"
            "Compare the agent's actual tool usage against the expected tool calls. "
            "Evaluate tool selection accuracy and parameter correctness. Score it."
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
            metric_name="tool_accuracy",
        )
