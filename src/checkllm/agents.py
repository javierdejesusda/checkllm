"""Agent evaluation models and deterministic validation functions."""

from __future__ import annotations

from collections import Counter
from typing import Any

from pydantic import BaseModel, Field

from checkllm.models import CheckResult


class ToolCall(BaseModel):
    """A single tool/function call made by an agent."""

    name: str
    parameters: dict[str, Any] = Field(default_factory=dict)
    result: str | None = None
    timestamp_ms: int | None = None


class AgentStep(BaseModel):
    """A single step in an agent's execution."""

    thought: str | None = None  # Chain-of-thought reasoning
    action: str | None = None  # Action taken (e.g., "call_tool", "respond")
    tool_call: ToolCall | None = None
    observation: str | None = None  # Result of the action
    metadata: dict[str, Any] = Field(default_factory=dict)


class AgentTestCase(BaseModel):
    """Test case for evaluating an agent's execution."""

    query: str  # The user's original request
    steps: list[AgentStep] = Field(default_factory=list)
    final_output: str | None = None
    expected_tools: list[ToolCall] | None = None  # Expected tool calls for validation
    expected_output: str | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)

    @property
    def tool_calls(self) -> list[ToolCall]:
        """Extract all tool calls from steps."""
        return [s.tool_call for s in self.steps if s.tool_call is not None]

    @property
    def trajectory(self) -> list[str]:
        """Get the sequence of actions taken."""
        return [s.action for s in self.steps if s.action is not None]

    def format_trace(self) -> str:
        """Format the agent execution as a readable trace."""
        lines = [f"Query: {self.query}", ""]
        for i, step in enumerate(self.steps, 1):
            lines.append(f"Step {i}:")
            if step.thought:
                lines.append(f"  Thought: {step.thought}")
            if step.action:
                lines.append(f"  Action: {step.action}")
            if step.tool_call:
                lines.append(
                    f"  Tool: {step.tool_call.name}({step.tool_call.parameters})"
                )
                if step.tool_call.result:
                    lines.append(f"  Result: {step.tool_call.result}")
            if step.observation:
                lines.append(f"  Observation: {step.observation}")
            lines.append("")
        if self.final_output:
            lines.append(f"Final Output: {self.final_output}")
        return "\n".join(lines)


# ---------------------------------------------------------------------------
# Deterministic validation functions
# ---------------------------------------------------------------------------


def validate_tool_calls(test_case: AgentTestCase) -> CheckResult:
    """Compare actual tool calls against *expected_tools*.

    For each expected tool call, check whether an actual call has the same
    tool name **and** matching parameters.

    Score = correct_calls / max(len(expected), len(actual)).
    If there are no expected tools, the check passes trivially with score 1.0.
    """
    expected = test_case.expected_tools
    actual = test_case.tool_calls

    if not expected:
        return CheckResult(
            passed=True,
            score=1.0,
            reasoning="No expected tool calls specified; nothing to validate.",
            cost=0.0,
            latency_ms=0,
            metric_name="tool_call_validation",
        )

    # Greedy match: for each expected call, find a matching actual call.
    unmatched_actual = list(actual)
    correct = 0
    details: list[str] = []

    for exp in expected:
        found = False
        for idx, act in enumerate(unmatched_actual):
            if act.name == exp.name and act.parameters == exp.parameters:
                correct += 1
                unmatched_actual.pop(idx)
                found = True
                break
        if not found:
            details.append(
                f"Expected tool call not found: {exp.name}({exp.parameters})"
            )

    denominator = max(len(expected), len(actual))
    score = correct / denominator if denominator > 0 else 1.0
    passed = score == 1.0

    if unmatched_actual:
        extra_names = [c.name for c in unmatched_actual]
        details.append(f"Unexpected tool calls: {extra_names}")

    reasoning = (
        f"{correct}/{denominator} tool calls matched."
        + ((" " + " ".join(details)) if details else "")
    )

    return CheckResult(
        passed=passed,
        score=score,
        reasoning=reasoning,
        cost=0.0,
        latency_ms=0,
        metric_name="tool_call_validation",
    )


def validate_trajectory_length(
    test_case: AgentTestCase, max_steps: int
) -> CheckResult:
    """Check the agent did not take too many steps.

    Score = min(1.0, max_steps / actual_steps).
    Passes when actual_steps <= max_steps.
    """
    actual_steps = len(test_case.steps)

    if actual_steps == 0:
        return CheckResult(
            passed=True,
            score=1.0,
            reasoning="Agent took 0 steps.",
            cost=0.0,
            latency_ms=0,
            metric_name="trajectory_length",
        )

    score = min(1.0, max_steps / actual_steps)
    passed = actual_steps <= max_steps

    return CheckResult(
        passed=passed,
        score=score,
        reasoning=(
            f"Agent took {actual_steps} step(s) "
            f"(max allowed: {max_steps}). "
            f"{'Within' if passed else 'Exceeds'} limit."
        ),
        cost=0.0,
        latency_ms=0,
        metric_name="trajectory_length",
    )


def _longest_common_subsequence_length(a: list[str], b: list[str]) -> int:
    """Return the length of the longest common subsequence of *a* and *b*."""
    m, n = len(a), len(b)
    dp = [[0] * (n + 1) for _ in range(m + 1)]
    for i in range(1, m + 1):
        for j in range(1, n + 1):
            if a[i - 1] == b[j - 1]:
                dp[i][j] = dp[i - 1][j - 1] + 1
            else:
                dp[i][j] = max(dp[i - 1][j], dp[i][j - 1])
    return dp[m][n]


def validate_tool_order(
    test_case: AgentTestCase, expected_order: list[str]
) -> CheckResult:
    """Check tools were called in the expected order.

    Extra tool calls between expected ones are allowed.  Uses the longest
    common subsequence (LCS) of tool names to determine how many of the
    expected calls appear in the correct relative order.

    Score = lcs_length / len(expected_order).
    Passes when the entire expected sequence is a subsequence of actual calls.
    """
    if not expected_order:
        return CheckResult(
            passed=True,
            score=1.0,
            reasoning="No expected order specified; nothing to validate.",
            cost=0.0,
            latency_ms=0,
            metric_name="tool_order",
        )

    actual_names = [tc.name for tc in test_case.tool_calls]
    lcs_len = _longest_common_subsequence_length(actual_names, expected_order)
    score = lcs_len / len(expected_order)
    passed = lcs_len == len(expected_order)

    return CheckResult(
        passed=passed,
        score=score,
        reasoning=(
            f"LCS length {lcs_len}/{len(expected_order)} with expected order. "
            f"Actual tool sequence: {actual_names}. "
            f"Expected order: {expected_order}."
        ),
        cost=0.0,
        latency_ms=0,
        metric_name="tool_order",
    )


def validate_no_repeated_tools(
    test_case: AgentTestCase, max_repeats: int = 1
) -> CheckResult:
    """Check no tool was called more than *max_repeats* times.

    Passes when every tool's call count is at most *max_repeats*.
    Score = 1.0 when passing, otherwise the fraction of tools that are
    within the limit.
    """
    tool_names = [tc.name for tc in test_case.tool_calls]
    counts = Counter(tool_names)

    if not counts:
        return CheckResult(
            passed=True,
            score=1.0,
            reasoning="No tool calls made; nothing to validate.",
            cost=0.0,
            latency_ms=0,
            metric_name="no_repeated_tools",
        )

    violators = {name: cnt for name, cnt in counts.items() if cnt > max_repeats}
    passed = len(violators) == 0
    score = 1.0 if passed else (len(counts) - len(violators)) / len(counts)

    if passed:
        reasoning = (
            f"All tools called at most {max_repeats} time(s). "
            f"Counts: {dict(counts)}."
        )
    else:
        reasoning = (
            f"{len(violators)} tool(s) exceeded {max_repeats} call(s): "
            f"{violators}. Full counts: {dict(counts)}."
        )

    return CheckResult(
        passed=passed,
        score=score,
        reasoning=reasoning,
        cost=0.0,
        latency_ms=0,
        metric_name="no_repeated_tools",
    )
