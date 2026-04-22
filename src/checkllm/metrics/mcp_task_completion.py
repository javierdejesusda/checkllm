from __future__ import annotations

import time

from checkllm.judge import JudgeBackend
from checkllm.models import CheckResult

MCP_TASK_COMPLETION_SYSTEM_PROMPT = """You are an expert MCP (Model Context Protocol) agent evaluator. Your job is to assess whether an MCP-enabled agent successfully completed its assigned task using the available tools.

Score from 0.0 to 1.0:
- 1.0 = The task was fully completed; the output directly and comprehensively satisfies the task requirements, and the tools used were appropriate
- 0.7 = The task was mostly completed; the main objectives were met but some secondary requirements are missing or partially addressed
- 0.5 = The task was partially completed; some key objectives were achieved but significant portions are incomplete or incorrect
- 0.3 = The task was mostly incomplete; the agent made some progress but failed to achieve the primary objectives
- 0.0 = The task was not completed; the output does not address the task requirements, or the agent failed entirely

Evaluation criteria:
1. Task achievement: Does the output satisfy the stated task requirements?
2. Tool utilization: Were the available tools used appropriately to accomplish the task?
3. Output quality: Is the output accurate, complete, and well-formed?
4. Tool coverage: Were all necessary tools invoked to fully complete the task?

Respond with JSON: {"score": <float>, "reasoning": "<explanation>"}"""


class MCPTaskCompletionMetric:
    """Evaluates whether an MCP-enabled agent completed its task using available tools."""

    def __init__(self, judge: JudgeBackend, threshold: float = 0.8) -> None:
        self.judge = judge
        self.threshold = threshold
        self.system_prompt: str = MCP_TASK_COMPLETION_SYSTEM_PROMPT

    async def evaluate(
        self,
        output: str,
        task: str,
        tools_used: list[str],
    ) -> CheckResult:
        """Evaluate whether an MCP agent completed its assigned task.

        Args:
            output: The agent's final output or response.
            task: The task description the agent was asked to complete.
            tools_used: List of MCP tool names the agent invoked.

        Returns:
            A CheckResult with task completion score and reasoning.
        """
        tools_list = ", ".join(tools_used) if tools_used else "none"
        prompt = (
            f"Task:\n{task}\n\n"
            f"Tools Used:\n{tools_list}\n\n"
            f"Agent Output:\n{output}\n\n"
            "Did the agent successfully complete the task using the available MCP tools? Score it."
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
            metric_name="mcp_task_completion",
        )
