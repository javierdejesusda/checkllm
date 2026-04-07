"""Trajectory and trace evaluation for agent assessment.

Provides ``TrajectoryValidator`` for asserting against agent execution
trajectories (tool sequences, arguments, step counts, etc.) and
``TraceValidator`` for validating observability-style execution spans.
"""

from __future__ import annotations

import re
from typing import Any

from pydantic import BaseModel, Field

from checkllm.agents import AgentTestCase
from checkllm.models import CheckResult


class TraceSpan(BaseModel):
    """A span in an execution trace (for observability-style validation).

    Attributes:
        name: Human-readable span name (e.g. ``"retrieve_docs"``).
        span_type: Category of work: ``"llm"``, ``"tool"``, ``"retriever"``,
            ``"agent"``, or ``"custom"``.
        start_ms: Epoch-relative start timestamp in milliseconds.
        end_ms: Epoch-relative end timestamp in milliseconds.
        status: Outcome of the span -- ``"ok"`` or ``"error"``.
        attributes: Arbitrary key-value metadata attached to the span.
        children: Nested child spans for hierarchical traces.
    """

    name: str
    span_type: str
    start_ms: int
    end_ms: int
    status: str = "ok"
    attributes: dict[str, Any] = Field(default_factory=dict)
    children: list["TraceSpan"] = Field(default_factory=list)

    @property
    def duration_ms(self) -> int:
        """Compute span duration in milliseconds."""
        return self.end_ms - self.start_ms


def _flatten_spans(spans: list[TraceSpan]) -> list[TraceSpan]:
    """Recursively flatten a hierarchical span tree into a single list."""
    result: list[TraceSpan] = []
    for span in spans:
        result.append(span)
        result.extend(_flatten_spans(span.children))
    return result


def _is_subsequence(needle: list[str], haystack: list[str]) -> int:
    """Return the number of items from *needle* found as a subsequence of *haystack*.

    Items must appear in order but gaps are allowed.
    """
    matched = 0
    hay_idx = 0
    for item in needle:
        while hay_idx < len(haystack):
            if haystack[hay_idx] == item:
                matched += 1
                hay_idx += 1
                break
            hay_idx += 1
    return matched


class TrajectoryValidator:
    """Validates agent execution trajectories against expected patterns.

    Construct via the module-level :func:`validate_trajectory` helper or
    directly from an :class:`~checkllm.agents.AgentTestCase`.

    Args:
        test_case: The agent test case whose trajectory should be validated.
    """

    def __init__(self, test_case: AgentTestCase) -> None:
        self.test_case = test_case

    def tool_used(
        self,
        tool_name: str,
        min_count: int = 1,
        max_count: int | None = None,
    ) -> CheckResult:
        """Verify a specific tool was used in the trajectory.

        Args:
            tool_name: Name of the tool to look for.
            min_count: Minimum number of times the tool must appear.
            max_count: Optional upper bound on tool usage count.

        Returns:
            A ``CheckResult`` with score = clamp(actual / min_count, 0, 1).
        """
        actual = sum(1 for tc in self.test_case.tool_calls if tc.name == tool_name)
        score = min(actual / min_count, 1.0) if min_count > 0 else 1.0
        passed = actual >= min_count and (max_count is None or actual <= max_count)

        parts: list[str] = [
            f"Tool '{tool_name}' called {actual} time(s).",
            f"Expected at least {min_count}.",
        ]
        if max_count is not None:
            parts.append(f"Maximum allowed: {max_count}.")

        return CheckResult(
            passed=passed,
            score=score,
            reasoning=" ".join(parts),
            cost=0.0,
            latency_ms=0,
            metric_name="tool_used",
        )

    def tool_not_used(self, tool_name: str) -> CheckResult:
        """Verify a tool was NOT used (useful for security/safety checks).

        Args:
            tool_name: Name of the tool that must be absent.

        Returns:
            A ``CheckResult`` that passes when the tool was never called.
        """
        actual = sum(1 for tc in self.test_case.tool_calls if tc.name == tool_name)
        passed = actual == 0
        score = 1.0 if passed else 0.0

        return CheckResult(
            passed=passed,
            score=score,
            reasoning=(
                f"Tool '{tool_name}' was {'not called' if passed else f'called {actual} time(s)'}."
            ),
            cost=0.0,
            latency_ms=0,
            metric_name="tool_not_used",
        )

    def tool_args_match(
        self,
        tool_name: str,
        expected_args: dict[str, Any],
        strict: bool = False,
    ) -> CheckResult:
        """Verify tool call arguments match expected values.

        Args:
            tool_name: Name of the tool whose arguments should be checked.
            expected_args: Key-value pairs that must appear in the tool call
                parameters.
            strict: When ``True`` the tool parameters must match *exactly*
                (no extra keys allowed).  When ``False`` only the provided
                keys are checked.

        Returns:
            A ``CheckResult`` scored by the fraction of matching keys.
        """
        calls = [tc for tc in self.test_case.tool_calls if tc.name == tool_name]
        if not calls:
            return CheckResult(
                passed=False,
                score=0.0,
                reasoning=f"Tool '{tool_name}' was never called.",
                cost=0.0,
                latency_ms=0,
                metric_name="tool_args_match",
            )

        best_score = 0.0
        best_reasoning = ""
        best_passed = False

        for call in calls:
            params = call.parameters
            if strict:
                matched_keys = sum(
                    1 for k, v in expected_args.items()
                    if params.get(k) == v
                )
                total_keys = max(len(expected_args), len(params))
                extra_keys = set(params.keys()) - set(expected_args.keys())
                missing_keys = set(expected_args.keys()) - set(params.keys())
                s = matched_keys / total_keys if total_keys > 0 else 1.0
                p = (s == 1.0) and not extra_keys and not missing_keys
                reasoning = (
                    f"Strict match: {matched_keys}/{total_keys} keys matched."
                )
                if extra_keys:
                    reasoning += f" Extra keys: {sorted(extra_keys)}."
                if missing_keys:
                    reasoning += f" Missing keys: {sorted(missing_keys)}."
            else:
                matched_keys = sum(
                    1 for k, v in expected_args.items()
                    if params.get(k) == v
                )
                total_keys = len(expected_args)
                s = matched_keys / total_keys if total_keys > 0 else 1.0
                p = matched_keys == total_keys
                reasoning = f"Partial match: {matched_keys}/{total_keys} expected keys matched."

            if s > best_score or (s == best_score and p and not best_passed):
                best_score = s
                best_reasoning = reasoning
                best_passed = p

        return CheckResult(
            passed=best_passed,
            score=best_score,
            reasoning=best_reasoning,
            cost=0.0,
            latency_ms=0,
            metric_name="tool_args_match",
        )

    def tool_sequence(self, expected_sequence: list[str]) -> CheckResult:
        """Verify tools were called in the expected order (subsequence matching).

        Gaps between expected tools are allowed.

        Args:
            expected_sequence: Ordered list of tool names that must appear as
                a subsequence of actual tool calls.

        Returns:
            A ``CheckResult`` with score = matched_count / expected_count.
        """
        if not expected_sequence:
            return CheckResult(
                passed=True,
                score=1.0,
                reasoning="No expected sequence specified.",
                cost=0.0,
                latency_ms=0,
                metric_name="tool_sequence",
            )

        actual_names = [tc.name for tc in self.test_case.tool_calls]
        matched = _is_subsequence(expected_sequence, actual_names)
        score = matched / len(expected_sequence)
        passed = matched == len(expected_sequence)

        return CheckResult(
            passed=passed,
            score=score,
            reasoning=(
                f"Matched {matched}/{len(expected_sequence)} tools in sequence. "
                f"Actual: {actual_names}. Expected: {expected_sequence}."
            ),
            cost=0.0,
            latency_ms=0,
            metric_name="tool_sequence",
        )

    def tool_sequence_strict(self, expected_sequence: list[str]) -> CheckResult:
        """Verify tools were called in EXACT order with no extras between.

        Args:
            expected_sequence: The complete list of tool names that must match
                the actual tool call sequence exactly.

        Returns:
            A ``CheckResult`` that passes only on an exact match.
        """
        actual_names = [tc.name for tc in self.test_case.tool_calls]
        passed = actual_names == expected_sequence

        if not expected_sequence and not actual_names:
            score = 1.0
        elif not expected_sequence:
            score = 0.0
        else:
            matching = sum(
                1 for a, e in zip(actual_names, expected_sequence) if a == e
            )
            score = matching / max(len(actual_names), len(expected_sequence))

        return CheckResult(
            passed=passed,
            score=score,
            reasoning=(
                f"Strict sequence {'matched' if passed else 'did not match'}. "
                f"Actual: {actual_names}. Expected: {expected_sequence}."
            ),
            cost=0.0,
            latency_ms=0,
            metric_name="tool_sequence_strict",
        )

    def step_count(
        self,
        min_steps: int | None = None,
        max_steps: int | None = None,
    ) -> CheckResult:
        """Validate total step count is within bounds.

        Args:
            min_steps: Minimum number of steps required (inclusive).
            max_steps: Maximum number of steps allowed (inclusive).

        Returns:
            A ``CheckResult`` that passes when the step count is in range.
        """
        actual = len(self.test_case.steps)
        passed = True
        parts: list[str] = [f"Agent took {actual} step(s)."]

        if min_steps is not None and actual < min_steps:
            passed = False
            parts.append(f"Below minimum of {min_steps}.")
        if max_steps is not None and actual > max_steps:
            passed = False
            parts.append(f"Above maximum of {max_steps}.")

        if max_steps is not None and max_steps > 0:
            score = min(1.0, max_steps / actual) if actual > 0 else 1.0
        elif min_steps is not None and min_steps > 0:
            score = min(1.0, actual / min_steps)
        else:
            score = 1.0

        if passed:
            score = 1.0

        return CheckResult(
            passed=passed,
            score=score,
            reasoning=" ".join(parts),
            cost=0.0,
            latency_ms=0,
            metric_name="step_count",
        )

    def no_repeated_tools(self, max_consecutive: int = 1) -> CheckResult:
        """Check for excessive consecutive calls to the same tool.

        This catches agent loops where the same tool is called repeatedly
        without progress.

        Args:
            max_consecutive: Maximum allowed consecutive calls to a single
                tool before the check fails.

        Returns:
            A ``CheckResult`` that fails when any tool is called more than
            ``max_consecutive`` times in a row.
        """
        tool_names = [tc.name for tc in self.test_case.tool_calls]
        if not tool_names:
            return CheckResult(
                passed=True,
                score=1.0,
                reasoning="No tool calls made.",
                cost=0.0,
                latency_ms=0,
                metric_name="no_repeated_tools",
            )

        max_run = 1
        current_run = 1
        worst_tool = tool_names[0]

        for i in range(1, len(tool_names)):
            if tool_names[i] == tool_names[i - 1]:
                current_run += 1
                if current_run > max_run:
                    max_run = current_run
                    worst_tool = tool_names[i]
            else:
                current_run = 1

        passed = max_run <= max_consecutive
        score = min(1.0, max_consecutive / max_run) if max_run > 0 else 1.0

        return CheckResult(
            passed=passed,
            score=score,
            reasoning=(
                f"Longest consecutive run: {max_run} (tool '{worst_tool}'). "
                f"Max allowed: {max_consecutive}."
            ),
            cost=0.0,
            latency_ms=0,
            metric_name="no_repeated_tools",
        )

    def thought_contains(self, pattern: str) -> CheckResult:
        """Check that at least one thought/reasoning step contains *pattern*.

        Args:
            pattern: A regex pattern to match against agent thoughts.

        Returns:
            A ``CheckResult`` that passes when the pattern is found in at
            least one step's thought.
        """
        thoughts = [
            s.thought for s in self.test_case.steps if s.thought is not None
        ]
        if not thoughts:
            return CheckResult(
                passed=False,
                score=0.0,
                reasoning="No thoughts found in the trajectory.",
                cost=0.0,
                latency_ms=0,
                metric_name="thought_contains",
            )

        compiled = re.compile(pattern)
        matches = [t for t in thoughts if compiled.search(t)]
        passed = len(matches) > 0
        score = min(1.0, len(matches) / len(thoughts))

        return CheckResult(
            passed=passed,
            score=score,
            reasoning=(
                f"Pattern '{pattern}' matched {len(matches)}/{len(thoughts)} thought(s)."
            ),
            cost=0.0,
            latency_ms=0,
            metric_name="thought_contains",
        )

    def no_hallucinated_tools(self, allowed_tools: list[str]) -> CheckResult:
        """Verify only allowed tools were called (no made-up tool names).

        Args:
            allowed_tools: Exhaustive list of valid tool names.

        Returns:
            A ``CheckResult`` scored by the fraction of calls using
            allowed tools.
        """
        tool_names = [tc.name for tc in self.test_case.tool_calls]
        if not tool_names:
            return CheckResult(
                passed=True,
                score=1.0,
                reasoning="No tool calls made.",
                cost=0.0,
                latency_ms=0,
                metric_name="no_hallucinated_tools",
            )

        allowed_set = set(allowed_tools)
        invalid = [n for n in tool_names if n not in allowed_set]
        valid_count = len(tool_names) - len(invalid)
        score = valid_count / len(tool_names)
        passed = len(invalid) == 0

        return CheckResult(
            passed=passed,
            score=score,
            reasoning=(
                f"{valid_count}/{len(tool_names)} tool calls used allowed tools."
                + (f" Invalid: {sorted(set(invalid))}." if invalid else "")
            ),
            cost=0.0,
            latency_ms=0,
            metric_name="no_hallucinated_tools",
        )

    def tool_result_used(self, tool_name: str) -> CheckResult:
        """Verify that the result from a tool call was referenced in subsequent steps.

        Checks whether the tool's result text appears in any later step's
        thought, observation, or the final output.

        Args:
            tool_name: Name of the tool whose result should be referenced.

        Returns:
            A ``CheckResult`` that passes when the tool result is found in a
            later step or in the final output.
        """
        steps = self.test_case.steps
        tool_indices: list[int] = []
        for i, step in enumerate(steps):
            if step.tool_call and step.tool_call.name == tool_name:
                tool_indices.append(i)

        if not tool_indices:
            return CheckResult(
                passed=False,
                score=0.0,
                reasoning=f"Tool '{tool_name}' was never called.",
                cost=0.0,
                latency_ms=0,
                metric_name="tool_result_used",
            )

        used_count = 0
        for idx in tool_indices:
            result_text = steps[idx].tool_call.result  # type: ignore[union-attr]
            if not result_text:
                continue

            found = False
            for later_step in steps[idx + 1:]:
                for text in (later_step.thought, later_step.observation):
                    if text and result_text in text:
                        found = True
                        break
                if found:
                    break

            if not found and self.test_case.final_output:
                if result_text in self.test_case.final_output:
                    found = True

            if found:
                used_count += 1

        calls_with_results = sum(
            1 for i in tool_indices
            if steps[i].tool_call and steps[i].tool_call.result  # type: ignore[union-attr]
        )

        if calls_with_results == 0:
            return CheckResult(
                passed=True,
                score=1.0,
                reasoning=f"Tool '{tool_name}' returned no results to check.",
                cost=0.0,
                latency_ms=0,
                metric_name="tool_result_used",
            )

        score = used_count / calls_with_results
        passed = used_count == calls_with_results

        return CheckResult(
            passed=passed,
            score=score,
            reasoning=(
                f"{used_count}/{calls_with_results} result(s) from "
                f"'{tool_name}' referenced in subsequent steps."
            ),
            cost=0.0,
            latency_ms=0,
            metric_name="tool_result_used",
        )

    def max_cost(self, max_usd: float) -> CheckResult:
        """Validate total trajectory cost is within budget.

        Sums ``timestamp_ms`` values repurposed as cost indicators in
        metadata, or falls back to the test case metadata ``total_cost`` key.

        Args:
            max_usd: Maximum allowed cost in USD.

        Returns:
            A ``CheckResult`` that passes when total cost <= *max_usd*.
        """
        total = self.test_case.metadata.get("total_cost", 0.0)
        passed = total <= max_usd
        score = min(1.0, max_usd / total) if total > 0 else 1.0

        return CheckResult(
            passed=passed,
            score=score,
            reasoning=f"Total cost: ${total:.4f}. Budget: ${max_usd:.4f}.",
            cost=0.0,
            latency_ms=0,
            metric_name="max_cost",
        )

    def max_latency(self, max_ms: int) -> CheckResult:
        """Validate total trajectory latency.

        Uses the test case metadata ``total_latency_ms`` key.

        Args:
            max_ms: Maximum allowed latency in milliseconds.

        Returns:
            A ``CheckResult`` that passes when total latency <= *max_ms*.
        """
        total = self.test_case.metadata.get("total_latency_ms", 0)
        passed = total <= max_ms
        score = min(1.0, max_ms / total) if total > 0 else 1.0

        return CheckResult(
            passed=passed,
            score=score,
            reasoning=f"Total latency: {total}ms. Limit: {max_ms}ms.",
            cost=0.0,
            latency_ms=0,
            metric_name="max_latency",
        )

    def final_output_matches(self, pattern: str) -> CheckResult:
        """Regex match on the agent's final output.

        Args:
            pattern: A regex pattern to match against ``final_output``.

        Returns:
            A ``CheckResult`` that passes when the pattern matches.
        """
        output = self.test_case.final_output
        if output is None:
            return CheckResult(
                passed=False,
                score=0.0,
                reasoning="No final output available.",
                cost=0.0,
                latency_ms=0,
                metric_name="final_output_matches",
            )

        match = re.search(pattern, output)
        passed = match is not None

        return CheckResult(
            passed=passed,
            score=1.0 if passed else 0.0,
            reasoning=(
                f"Pattern '{pattern}' {'matched' if passed else 'did not match'} "
                f"final output."
            ),
            cost=0.0,
            latency_ms=0,
            metric_name="final_output_matches",
        )

    def trajectory_efficiency(
        self, max_steps_per_tool: float = 2.0
    ) -> CheckResult:
        """Score based on the ratio of useful steps to total steps.

        A *useful* step is one that contains a tool call.  The check passes
        when ``total_steps / max(tool_calls, 1) <= max_steps_per_tool``.

        Args:
            max_steps_per_tool: Maximum acceptable ratio of total steps to
                tool-bearing steps.

        Returns:
            A ``CheckResult`` scored by the inverse of step overhead.
        """
        total_steps = len(self.test_case.steps)
        tool_steps = len(self.test_case.tool_calls)

        if total_steps == 0:
            return CheckResult(
                passed=True,
                score=1.0,
                reasoning="No steps taken.",
                cost=0.0,
                latency_ms=0,
                metric_name="trajectory_efficiency",
            )

        denominator = max(tool_steps, 1)
        ratio = total_steps / denominator
        passed = ratio <= max_steps_per_tool
        score = min(1.0, max_steps_per_tool / ratio) if ratio > 0 else 1.0

        return CheckResult(
            passed=passed,
            score=score,
            reasoning=(
                f"Steps/tool ratio: {ratio:.1f} "
                f"({total_steps} steps, {tool_steps} tool calls). "
                f"Max allowed: {max_steps_per_tool}."
            ),
            cost=0.0,
            latency_ms=0,
            metric_name="trajectory_efficiency",
        )


class TraceValidator:
    """Validates observability-style execution spans.

    Args:
        spans: Top-level trace spans (may contain nested children).
    """

    def __init__(self, spans: list[TraceSpan]) -> None:
        self.spans = spans
        self._flat: list[TraceSpan] | None = None

    @property
    def flat_spans(self) -> list[TraceSpan]:
        """Lazily flatten the span hierarchy."""
        if self._flat is None:
            self._flat = _flatten_spans(self.spans)
        return self._flat

    def _filter_spans(
        self,
        span_type: str | None = None,
        name_pattern: str | None = None,
    ) -> list[TraceSpan]:
        """Return spans matching the given type and/or name pattern."""
        result = self.flat_spans
        if span_type is not None:
            result = [s for s in result if s.span_type == span_type]
        if name_pattern is not None:
            compiled = re.compile(name_pattern)
            result = [s for s in result if compiled.search(s.name)]
        return result

    def span_count(
        self,
        span_type: str | None = None,
        name_pattern: str | None = None,
        min_count: int = 0,
        max_count: int | None = None,
    ) -> CheckResult:
        """Assert the number of matching spans is within bounds.

        Args:
            span_type: Filter spans by type (e.g. ``"llm"``).
            name_pattern: Regex filter on span name.
            min_count: Minimum number of matching spans.
            max_count: Maximum number of matching spans (``None`` for no limit).

        Returns:
            A ``CheckResult`` based on whether the count is in range.
        """
        matched = self._filter_spans(span_type, name_pattern)
        count = len(matched)
        passed = count >= min_count and (max_count is None or count <= max_count)

        if min_count > 0:
            score = min(1.0, count / min_count)
        else:
            score = 1.0
        if not passed:
            if max_count is not None and count > max_count:
                score = min(1.0, max_count / count) if count > 0 else 0.0

        parts = [f"Found {count} matching span(s)."]
        if min_count > 0:
            parts.append(f"Minimum: {min_count}.")
        if max_count is not None:
            parts.append(f"Maximum: {max_count}.")

        return CheckResult(
            passed=passed,
            score=score,
            reasoning=" ".join(parts),
            cost=0.0,
            latency_ms=0,
            metric_name="span_count",
        )

    def span_duration(
        self,
        name_pattern: str,
        max_ms: int,
        percentile: float = 1.0,
    ) -> CheckResult:
        """Assert that matching span durations are within limits.

        Args:
            name_pattern: Regex pattern to select spans by name.
            max_ms: Maximum allowed duration in milliseconds.
            percentile: Fraction of spans (0.0-1.0) that must be within
                *max_ms*.  ``1.0`` means all spans must satisfy the limit.

        Returns:
            A ``CheckResult`` based on whether enough spans are fast enough.
        """
        matched = self._filter_spans(name_pattern=name_pattern)
        if not matched:
            return CheckResult(
                passed=True,
                score=1.0,
                reasoning=f"No spans matched pattern '{name_pattern}'.",
                cost=0.0,
                latency_ms=0,
                metric_name="span_duration",
            )

        durations = sorted([s.duration_ms for s in matched])
        within = sum(1 for d in durations if d <= max_ms)
        fraction_within = within / len(durations)
        passed = fraction_within >= percentile

        score = min(1.0, fraction_within / percentile) if percentile > 0 else 1.0

        return CheckResult(
            passed=passed,
            score=score,
            reasoning=(
                f"{within}/{len(durations)} span(s) within {max_ms}ms "
                f"({fraction_within:.0%}). Required: {percentile:.0%}."
            ),
            cost=0.0,
            latency_ms=0,
            metric_name="span_duration",
        )

    def no_error_spans(self) -> CheckResult:
        """Assert that no spans have an error status.

        Returns:
            A ``CheckResult`` that passes when all spans have ``status="ok"``.
        """
        all_spans = self.flat_spans
        if not all_spans:
            return CheckResult(
                passed=True,
                score=1.0,
                reasoning="No spans to check.",
                cost=0.0,
                latency_ms=0,
                metric_name="no_error_spans",
            )

        errors = [s for s in all_spans if s.status == "error"]
        passed = len(errors) == 0
        score = 1.0 - (len(errors) / len(all_spans))

        return CheckResult(
            passed=passed,
            score=max(0.0, score),
            reasoning=(
                f"{len(errors)} error span(s) out of {len(all_spans)} total."
                + (
                    f" Errors: {[s.name for s in errors]}."
                    if errors else ""
                )
            ),
            cost=0.0,
            latency_ms=0,
            metric_name="no_error_spans",
        )

    def total_duration(self, max_ms: int) -> CheckResult:
        """Assert the total trace duration is within a limit.

        The total duration is measured from the earliest start to the latest
        end across all spans.

        Args:
            max_ms: Maximum allowed total duration in milliseconds.

        Returns:
            A ``CheckResult`` that passes when total duration <= *max_ms*.
        """
        all_spans = self.flat_spans
        if not all_spans:
            return CheckResult(
                passed=True,
                score=1.0,
                reasoning="No spans to measure.",
                cost=0.0,
                latency_ms=0,
                metric_name="total_duration",
            )

        earliest = min(s.start_ms for s in all_spans)
        latest = max(s.end_ms for s in all_spans)
        duration = latest - earliest
        passed = duration <= max_ms
        score = min(1.0, max_ms / duration) if duration > 0 else 1.0

        return CheckResult(
            passed=passed,
            score=score,
            reasoning=f"Total duration: {duration}ms. Limit: {max_ms}ms.",
            cost=0.0,
            latency_ms=0,
            metric_name="total_duration",
        )


def validate_trajectory(test_case: AgentTestCase) -> TrajectoryValidator:
    """Create a trajectory validator for an agent test case.

    Args:
        test_case: The agent test case to validate.

    Returns:
        A ``TrajectoryValidator`` bound to the given test case.
    """
    return TrajectoryValidator(test_case)


def validate_trace(spans: list[TraceSpan]) -> TraceValidator:
    """Create a trace validator for execution spans.

    Args:
        spans: Top-level spans (may contain nested children).

    Returns:
        A ``TraceValidator`` bound to the given spans.
    """
    return TraceValidator(spans)
