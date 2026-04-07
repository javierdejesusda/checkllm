from __future__ import annotations

import time

from checkllm.judge import JudgeBackend
from checkllm.models import CheckResult

MCP_USE_SYSTEM_PROMPT = """You are an expert MCP (Model Context Protocol) tool selection evaluator. Your job is to assess whether an agent selected appropriate MCP tools from the available set to address a user's query.

Score from 0.0 to 1.0:
- 1.0 = The agent selected exactly the right tools; all tools used were necessary and appropriate, and no relevant tools were overlooked
- 0.7 = The agent selected mostly appropriate tools with minor issues (one unnecessary tool or one missed optional tool)
- 0.5 = The agent's tool selection was partially correct; some appropriate tools were used but important ones were missed or inappropriate ones were selected
- 0.3 = The agent made poor tool selections; most tools used were inappropriate or the most critical tools were not selected
- 0.0 = The agent completely failed at tool selection; no appropriate tools were used or the agent ignored available tools entirely

Evaluation criteria:
1. Appropriateness: Were the selected tools suitable for the query?
2. Completeness: Were all necessary tools selected?
3. Efficiency: Were unnecessary or redundant tools avoided?
4. Coverage: Did the tool selection cover all aspects of the query?

Respond with JSON: {"score": <float>, "reasoning": "<explanation>"}"""


class MCPUseMetric:
    """Evaluates whether an agent selected appropriate MCP tools for a query."""

    def __init__(self, judge: JudgeBackend, threshold: float = 0.8) -> None:
        self.judge = judge
        self.threshold = threshold
        self.system_prompt: str = MCP_USE_SYSTEM_PROMPT

    async def evaluate(
        self,
        output: str,
        tools_available: list[str],
        tools_used: list[str],
        query: str,
    ) -> CheckResult:
        """Evaluate whether the agent selected appropriate MCP tools.

        Args:
            output: The agent's output or response.
            tools_available: List of MCP tool names available to the agent.
            tools_used: List of MCP tool names the agent actually invoked.
            query: The user's query or task the agent was addressing.

        Returns:
            A CheckResult with tool selection score and reasoning.
        """
        available_list = ", ".join(tools_available) if tools_available else "none"
        used_list = ", ".join(tools_used) if tools_used else "none"
        prompt = (
            f"User Query:\n{query}\n\n"
            f"Tools Available:\n{available_list}\n\n"
            f"Tools Used:\n{used_list}\n\n"
            f"Agent Output:\n{output}\n\n"
            "Did the agent select appropriate MCP tools from those available? Score it."
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
            metric_name="mcp_use",
        )
