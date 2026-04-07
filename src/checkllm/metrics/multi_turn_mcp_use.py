from __future__ import annotations

import time

from checkllm.judge import JudgeBackend
from checkllm.models import CheckResult

MULTI_TURN_MCP_USE_SYSTEM_PROMPT = """You are an expert MCP (Model Context Protocol) conversation evaluator. Your job is to assess whether an agent selected appropriate MCP tools across multiple conversation turns, evaluating the overall tool selection strategy throughout the dialogue.

Score from 0.0 to 1.0:
- 1.0 = Excellent tool selection across all turns; the agent consistently chose appropriate tools, adapted its selections as the conversation evolved, and made no unnecessary or redundant calls
- 0.7 = Good tool selection overall; the agent mostly chose appropriate tools across turns with minor lapses or slight inefficiencies
- 0.5 = Moderate tool selection; the agent made appropriate selections in some turns but showed inconsistency, missed tools, or poor adaptation across the conversation
- 0.3 = Poor tool selection; the agent frequently chose inappropriate tools, missed critical tool calls, or failed to adapt selections to the conversation context
- 0.0 = Failed tool selection; the agent made no appropriate tool selections across the conversation, or ignored available tools entirely

Evaluation criteria:
1. Turn-level appropriateness: Were the right tools selected for each turn's context?
2. Conversational adaptation: Did tool selections evolve appropriately as the conversation progressed?
3. Consistency: Were tool selections coherent and non-contradictory across turns?
4. Completeness: Were all necessary tool calls made across the full conversation?
5. Efficiency: Were redundant or unnecessary tool calls avoided?

Respond with JSON: {"score": <float>, "reasoning": "<explanation>"}"""


class MultiTurnMCPUseMetric:
    """Evaluates MCP tool selection across multiple conversation turns."""

    def __init__(self, judge: JudgeBackend, threshold: float = 0.8) -> None:
        self.judge = judge
        self.threshold = threshold
        self.system_prompt: str = MULTI_TURN_MCP_USE_SYSTEM_PROMPT

    async def evaluate(
        self,
        conversation_trace: str,
        tools_available: list[str],
        tools_used: list[str],
    ) -> CheckResult:
        """Evaluate MCP tool selection quality across a multi-turn conversation.

        Args:
            conversation_trace: A formatted transcript of the multi-turn conversation,
                including tool calls made at each turn.
            tools_available: List of MCP tool names available to the agent.
            tools_used: List of MCP tool names used across all turns.

        Returns:
            A CheckResult with multi-turn tool selection score and reasoning.
        """
        available_list = ", ".join(tools_available) if tools_available else "none"
        used_list = ", ".join(tools_used) if tools_used else "none"
        prompt = (
            f"Tools Available:\n{available_list}\n\n"
            f"Tools Used Across All Turns:\n{used_list}\n\n"
            f"Conversation Trace:\n{conversation_trace}\n\n"
            "Did the agent select appropriate MCP tools across all conversation turns? Score it."
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
            metric_name="multi_turn_mcp_use",
        )
