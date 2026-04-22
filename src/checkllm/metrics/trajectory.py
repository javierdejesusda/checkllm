from __future__ import annotations

import json
import time

from checkllm.judge import JudgeBackend
from checkllm.models import CheckResult

GOAL_SUCCESS_SYSTEM_PROMPT = """You are an expert evaluator for AI agent trajectories. Your job is to assess whether the agent successfully achieved the stated goal based on the sequence of steps it took and the final output.

Score from 0.0 to 1.0:
- 1.0 = The agent fully achieved the goal. The final state completely satisfies the goal description.
- 0.8 = The agent mostly achieved the goal with minor gaps or imperfections.
- 0.5 = The agent partially achieved the goal. Some requirements are met but significant parts are missing.
- 0.3 = The agent made progress toward the goal but did not achieve the key requirements.
- 0.0 = The agent failed to achieve the goal or went in a completely wrong direction.

Key evaluation criteria:
1. Does the final output satisfy the goal description?
2. Did the trajectory lead logically toward the goal?
3. Are all requirements from the goal description addressed?
4. Is the result correct and complete?

Respond with JSON: {"score": <float>, "reasoning": "<explanation>"}"""

TOOL_SEQUENCE_SYSTEM_PROMPT = """You are an expert evaluator for AI agent tool usage sequences. Your job is to assess whether the actual sequence of tool calls is semantically equivalent to the expected sequence, even if exact names differ.

Score from 0.0 to 1.0:
- 1.0 = The sequences are semantically identical. Tools serve the same purpose in the same order.
- 0.8 = The sequences are nearly equivalent. Minor reordering of independent steps or use of equivalent tool names.
- 0.5 = The sequences share some common tools but differ significantly in order or composition.
- 0.3 = The sequences have little overlap. Most tools differ or the order is substantially wrong.
- 0.0 = The sequences are completely different. No meaningful overlap in tools or ordering.

Respond with JSON: {"score": <float>, "reasoning": "<explanation>"}"""

STEP_COUNT_SYSTEM_PROMPT = """You are an expert evaluator for AI agent efficiency. Your job is to assess whether the steps in an agent trajectory were necessary and efficient, or whether unnecessary steps were taken.

Score from 0.0 to 1.0:
- 1.0 = Every step was necessary and no redundant actions were taken.
- 0.8 = Most steps were necessary, with at most one slightly redundant action.
- 0.5 = Several steps were redundant or unnecessary, but the core actions were present.
- 0.3 = Many steps were wasted. The task could have been done much more efficiently.
- 0.0 = The trajectory is overwhelmingly inefficient, with most steps being unnecessary.

Respond with JSON: {"score": <float>, "reasoning": "<explanation>"}"""

TOOL_ARGS_MATCH_SYSTEM_PROMPT = """You are an expert evaluator for AI agent tool call arguments. Your job is to assess whether the actual tool call arguments match the expected arguments, using fuzzy matching for values that are semantically equivalent.

Score from 0.0 to 1.0:
- 1.0 = All tool calls have correct names and all arguments match exactly or are semantically equivalent.
- 0.8 = Most arguments match. Minor differences in formatting or casing that don't affect functionality.
- 0.5 = Some arguments match but others are incorrect or missing. Key parameters may be wrong.
- 0.3 = Few arguments match. Most parameters are incorrect or missing.
- 0.0 = No meaningful argument matching. All parameters are wrong.

Respond with JSON: {"score": <float>, "reasoning": "<explanation>"}"""


class TrajectoryGoalSuccessMetric:
    """Evaluates whether an agent achieved the stated goal based on its trajectory.

    Assesses the final output and trajectory of steps against the goal
    description using an LLM judge.
    """

    def __init__(self, judge: JudgeBackend, threshold: float = 0.7) -> None:
        self.judge = judge
        self.threshold = threshold
        self.system_prompt: str = GOAL_SUCCESS_SYSTEM_PROMPT

    async def evaluate(
        self,
        output: str,
        goal_description: str,
        trajectory: list[str],
    ) -> CheckResult:
        """Evaluate whether the agent achieved the stated goal.

        Args:
            output: The final output produced by the agent.
            goal_description: A description of the goal the agent was trying to achieve.
            trajectory: A list of step descriptions the agent took.

        Returns:
            A CheckResult with goal success score and reasoning.
        """
        formatted_trajectory = "\n".join(
            f"  Step {i + 1}: {step}" for i, step in enumerate(trajectory)
        )
        prompt = (
            f"Goal:\n{goal_description}\n\n"
            f"Agent Trajectory:\n{formatted_trajectory}\n\n"
            f"Final Output:\n{output}\n\n"
            "Did the agent achieve the stated goal? Evaluate the trajectory "
            "and final output against the goal description. Score it."
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
            metric_name="trajectory_goal_success",
        )


class TrajectoryToolSequenceMetric:
    """Evaluates whether tools were called in the expected order.

    Performs deterministic sequence matching first, then uses an LLM judge
    for semantic equivalence when exact matching fails.
    """

    def __init__(self, judge: JudgeBackend, threshold: float = 0.7) -> None:
        self.judge = judge
        self.threshold = threshold
        self.system_prompt: str = TOOL_SEQUENCE_SYSTEM_PROMPT

    async def evaluate(
        self,
        actual_tool_sequence: list[str],
        expected_tool_sequence: list[str],
    ) -> CheckResult:
        """Evaluate whether tools were called in the expected order.

        Args:
            actual_tool_sequence: The actual sequence of tool names called.
            expected_tool_sequence: The expected sequence of tool names.

        Returns:
            A CheckResult with sequence match score and reasoning.
        """
        start = time.perf_counter_ns()

        if actual_tool_sequence == expected_tool_sequence:
            elapsed_ms = (time.perf_counter_ns() - start) // 1_000_000
            return CheckResult(
                passed=True,
                score=1.0,
                reasoning="Tool sequences match exactly.",
                cost=0.0,
                latency_ms=int(elapsed_ms),
                metric_name="trajectory_tool_sequence",
            )

        normalized_actual = [t.lower().strip() for t in actual_tool_sequence]
        normalized_expected = [t.lower().strip() for t in expected_tool_sequence]

        if normalized_actual == normalized_expected:
            elapsed_ms = (time.perf_counter_ns() - start) // 1_000_000
            return CheckResult(
                passed=True,
                score=1.0,
                reasoning="Tool sequences match after normalization.",
                cost=0.0,
                latency_ms=int(elapsed_ms),
                metric_name="trajectory_tool_sequence",
            )

        prompt = (
            f"Expected Tool Sequence:\n{json.dumps(expected_tool_sequence)}\n\n"
            f"Actual Tool Sequence:\n{json.dumps(actual_tool_sequence)}\n\n"
            "Are these tool sequences semantically equivalent? Consider whether "
            "the tools serve the same purpose and are in a logically equivalent order. "
            "Score the similarity."
        )

        response = await self.judge.evaluate(prompt=prompt, system_prompt=self.system_prompt)
        elapsed_ms = (time.perf_counter_ns() - start) // 1_000_000

        return CheckResult(
            passed=response.score >= self.threshold,
            score=response.score,
            reasoning=response.reasoning,
            cost=response.cost,
            latency_ms=int(elapsed_ms),
            metric_name="trajectory_tool_sequence",
        )


class TrajectoryStepCountMetric:
    """Evaluates whether a task was completed within a reasonable number of steps.

    Combines a deterministic step-count check with an LLM assessment of
    whether each step was necessary.
    """

    def __init__(self, judge: JudgeBackend, threshold: float = 0.7) -> None:
        self.judge = judge
        self.threshold = threshold
        self.system_prompt: str = STEP_COUNT_SYSTEM_PROMPT

    async def evaluate(
        self,
        trajectory: list[str],
        max_steps: int,
    ) -> CheckResult:
        """Evaluate whether the agent completed the task efficiently.

        Args:
            trajectory: A list of step descriptions the agent took.
            max_steps: The maximum acceptable number of steps.

        Returns:
            A CheckResult with efficiency score and reasoning.
        """
        start = time.perf_counter_ns()
        actual_steps = len(trajectory)

        if max_steps <= 0:
            deterministic_score = 0.0
        elif actual_steps <= max_steps:
            deterministic_score = 1.0
        else:
            overshoot = actual_steps - max_steps
            deterministic_score = max(0.0, 1.0 - (overshoot / max_steps))

        formatted_trajectory = "\n".join(
            f"  Step {i + 1}: {step}" for i, step in enumerate(trajectory)
        )
        prompt = (
            f"Max Expected Steps: {max_steps}\n"
            f"Actual Steps Taken: {actual_steps}\n\n"
            f"Trajectory:\n{formatted_trajectory}\n\n"
            "Were all steps necessary? Could the task have been completed "
            "in fewer steps? Evaluate the efficiency of this trajectory."
        )

        response = await self.judge.evaluate(prompt=prompt, system_prompt=self.system_prompt)
        elapsed_ms = (time.perf_counter_ns() - start) // 1_000_000

        combined_score = (deterministic_score + response.score) / 2.0
        combined_score = max(0.0, min(1.0, combined_score))

        reasoning = (
            f"Step count: {actual_steps}/{max_steps} "
            f"(deterministic: {deterministic_score:.2f}). "
            f"Judge assessment: {response.reasoning}"
        )

        return CheckResult(
            passed=combined_score >= self.threshold,
            score=combined_score,
            reasoning=reasoning,
            cost=response.cost,
            latency_ms=int(elapsed_ms),
            metric_name="trajectory_step_count",
        )


class TrajectoryToolArgsMatchMetric:
    """Evaluates whether tool arguments were correct with fuzzy matching.

    Compares actual tool call arguments against expected arguments using
    an LLM judge for semantic equivalence.
    """

    def __init__(self, judge: JudgeBackend, threshold: float = 0.7) -> None:
        self.judge = judge
        self.threshold = threshold
        self.system_prompt: str = TOOL_ARGS_MATCH_SYSTEM_PROMPT

    async def evaluate(
        self,
        actual_tool_calls: list[dict[str, object]],
        expected_tool_calls: list[dict[str, object]],
    ) -> CheckResult:
        """Evaluate whether tool arguments match expected values.

        Args:
            actual_tool_calls: The actual tool calls with names and arguments.
            expected_tool_calls: The expected tool calls with names and arguments.

        Returns:
            A CheckResult with argument match score and reasoning.
        """
        prompt = (
            f"Expected Tool Calls:\n{json.dumps(expected_tool_calls, indent=2)}\n\n"
            f"Actual Tool Calls:\n{json.dumps(actual_tool_calls, indent=2)}\n\n"
            "Compare the actual tool calls against the expected ones. "
            "Check tool names and argument values. Use fuzzy matching for "
            "semantically equivalent values. Score the overall match."
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
            metric_name="trajectory_tool_args_match",
        )
