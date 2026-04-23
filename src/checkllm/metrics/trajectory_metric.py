"""Deterministic trajectory metric with multiple sub-checks.

Given an ordered agent trajectory expressed as a list of
:class:`ToolCallTrace` entries (or compatible representations) and an
expected trajectory, this metric computes a composite score across four
sub-checks:

1. **Step ordering** -- Levenshtein edit distance between the actual and
   expected sequences of tool names.
2. **Loop detection** -- penalises repeated identical calls above a
   configurable threshold.
3. **Expected-tools coverage** -- fraction of expected tools that
   actually appeared.
4. **Unexpected-tools penalty** -- fraction of actual tool calls that
   were not in the expected set.

Each sub-check returns a 0.0-1.0 score; the overall score is a weighted
average.
"""

from __future__ import annotations

from collections import Counter
from collections.abc import Mapping
from typing import Any, Union

from checkllm.agents import ToolCall, ToolCallTrace
from checkllm.models import CheckResult

_TRACE_LIKE = Union[ToolCall, ToolCallTrace, Mapping[str, Any], str]


def _tool_name_and_params(call: _TRACE_LIKE) -> tuple[str, dict[str, Any]]:
    """Return ``(tool_name, parameters)`` for any accepted call representation."""
    if isinstance(call, str):
        return call, {}
    if isinstance(call, ToolCall):
        return call.name, dict(call.parameters)
    if isinstance(call, ToolCallTrace):
        return call.tool_name, dict(call.parameters)
    if isinstance(call, Mapping):
        name = call.get("tool_name") or call.get("name") or ""
        params = call.get("parameters") or call.get("args") or {}
        if not isinstance(params, Mapping):
            params = {}
        return str(name), dict(params)
    raise TypeError(f"Unsupported tool call type: {type(call).__name__}")


def _levenshtein(a: list[str], b: list[str]) -> int:
    """Compute the Levenshtein edit distance between two token sequences."""
    if a == b:
        return 0
    if not a:
        return len(b)
    if not b:
        return len(a)
    prev = list(range(len(b) + 1))
    for i, ai in enumerate(a, start=1):
        curr = [i] + [0] * len(b)
        for j, bj in enumerate(b, start=1):
            cost = 0 if ai == bj else 1
            curr[j] = min(
                curr[j - 1] + 1,
                prev[j] + 1,
                prev[j - 1] + cost,
            )
        prev = curr
    return prev[-1]


class TrajectorySubScores:
    """Container exposing individual sub-scores for a trajectory evaluation."""

    __slots__ = (
        "ordering",
        "loops",
        "coverage",
        "unexpected",
        "overall",
    )

    def __init__(
        self,
        ordering: float,
        loops: float,
        coverage: float,
        unexpected: float,
        overall: float,
    ) -> None:
        self.ordering = ordering
        self.loops = loops
        self.coverage = coverage
        self.unexpected = unexpected
        self.overall = overall

    def as_dict(self) -> dict[str, float]:
        """Return the sub-scores as a plain dictionary."""
        return {
            "ordering": self.ordering,
            "loops": self.loops,
            "coverage": self.coverage,
            "unexpected": self.unexpected,
            "overall": self.overall,
        }


class TrajectoryMetric:
    """Evaluate an ordered trajectory of :class:`ToolCallTrace` entries.

    The metric is deterministic and CI-friendly. All sub-checks return
    scores in ``[0.0, 1.0]``; the overall score is a weighted average.

    Args:
        expected_trajectory: Ordered list of expected tool names.
        loop_threshold: Number of *consecutive* identical calls above
            which the loop-detection sub-check begins to penalise. Runs
            at or below ``loop_threshold`` contribute full score.
        weights: Optional mapping overriding the default weights for the
            four sub-scores (keys: ``ordering``, ``loops``, ``coverage``,
            ``unexpected``). Weights are normalised.
        threshold: Minimum overall score to count as passing.
    """

    metric_name = "trajectory"

    _DEFAULT_WEIGHTS: dict[str, float] = {
        "ordering": 0.4,
        "loops": 0.2,
        "coverage": 0.25,
        "unexpected": 0.15,
    }

    def __init__(
        self,
        expected_trajectory: list[str],
        loop_threshold: int = 2,
        weights: Mapping[str, float] | None = None,
        threshold: float = 0.8,
    ) -> None:
        if loop_threshold < 1:
            raise ValueError("loop_threshold must be >= 1")
        self.expected_trajectory = list(expected_trajectory)
        self.loop_threshold = loop_threshold
        self.threshold = threshold

        chosen = dict(self._DEFAULT_WEIGHTS)
        if weights is not None:
            for k, v in weights.items():
                if k not in chosen:
                    raise ValueError(f"Unknown weight key: {k!r}")
                if v < 0:
                    raise ValueError(f"Weight {k!r} must be non-negative")
                chosen[k] = float(v)
        total = sum(chosen.values())
        if total <= 0:
            raise ValueError("At least one weight must be positive")
        self.weights = {k: v / total for k, v in chosen.items()}

    def _ordering_score(self, actual_names: list[str]) -> float:
        expected = self.expected_trajectory
        if not expected and not actual_names:
            return 1.0
        distance = _levenshtein(actual_names, expected)
        denom = max(len(expected), len(actual_names), 1)
        return max(0.0, 1.0 - distance / denom)

    def _loop_score(self, actual_names: list[str]) -> tuple[float, int, str]:
        if not actual_names:
            return 1.0, 1, ""
        max_run = 1
        current = 1
        worst_tool = actual_names[0]
        for i in range(1, len(actual_names)):
            if actual_names[i] == actual_names[i - 1]:
                current += 1
                if current > max_run:
                    max_run = current
                    worst_tool = actual_names[i]
            else:
                current = 1
        if max_run <= self.loop_threshold:
            return 1.0, max_run, worst_tool
        overshoot = max_run - self.loop_threshold
        score = max(0.0, 1.0 - overshoot / max(max_run, 1))
        return score, max_run, worst_tool

    def _coverage_score(self, actual_names: list[str]) -> tuple[float, list[str]]:
        expected = self.expected_trajectory
        if not expected:
            return 1.0, []
        expected_counts = Counter(expected)
        actual_counts = Counter(actual_names)
        covered = 0
        missing: list[str] = []
        for name, req in expected_counts.items():
            have = actual_counts.get(name, 0)
            covered += min(have, req)
            if have < req:
                missing.append(f"{name} (need {req}, got {have})")
        return covered / sum(expected_counts.values()), missing

    def _unexpected_score(self, actual_names: list[str]) -> tuple[float, list[str]]:
        if not actual_names:
            return 1.0, []
        expected_set = set(self.expected_trajectory)
        unexpected = [n for n in actual_names if n not in expected_set]
        if not unexpected:
            return 1.0, []
        score = max(0.0, 1.0 - len(unexpected) / len(actual_names))
        return score, sorted(set(unexpected))

    def compute_subscores(self, actual_calls: list[_TRACE_LIKE]) -> TrajectorySubScores:
        """Compute the four sub-scores and the weighted overall score.

        Args:
            actual_calls: The tool calls actually made by the agent.

        Returns:
            A :class:`TrajectorySubScores` with component and overall values.
        """
        actual_names = [_tool_name_and_params(c)[0] for c in actual_calls]
        ordering = self._ordering_score(actual_names)
        loop_score, _, _ = self._loop_score(actual_names)
        coverage, _ = self._coverage_score(actual_names)
        unexpected, _ = self._unexpected_score(actual_names)
        overall = (
            ordering * self.weights["ordering"]
            + loop_score * self.weights["loops"]
            + coverage * self.weights["coverage"]
            + unexpected * self.weights["unexpected"]
        )
        return TrajectorySubScores(
            ordering=round(ordering, 4),
            loops=round(loop_score, 4),
            coverage=round(coverage, 4),
            unexpected=round(unexpected, 4),
            overall=round(overall, 4),
        )

    def evaluate(self, actual_calls: list[_TRACE_LIKE]) -> CheckResult:
        """Score a full trajectory against the expected trajectory.

        Args:
            actual_calls: The tool calls actually made by the agent.

        Returns:
            A :class:`CheckResult` carrying the composite score and a
            breakdown of each sub-check in ``reasoning``.
        """
        actual_names = [_tool_name_and_params(c)[0] for c in actual_calls]
        ordering = self._ordering_score(actual_names)
        loop_score, max_run, worst_tool = self._loop_score(actual_names)
        coverage, missing = self._coverage_score(actual_names)
        unexpected, unexpected_names = self._unexpected_score(actual_names)

        overall = (
            ordering * self.weights["ordering"]
            + loop_score * self.weights["loops"]
            + coverage * self.weights["coverage"]
            + unexpected * self.weights["unexpected"]
        )
        passed = overall >= self.threshold

        parts = [
            f"Overall {overall:.2f} (threshold {self.threshold:.2f}).",
            f"Ordering: {ordering:.2f} (actual={actual_names}, expected={self.expected_trajectory}).",
            f"Loops: {loop_score:.2f} (max consecutive run={max_run}"
            + (f" of '{worst_tool}'" if max_run > 1 else "")
            + f", threshold={self.loop_threshold}).",
            f"Coverage: {coverage:.2f}" + (f" missing: {missing}." if missing else "."),
            f"Unexpected: {unexpected:.2f}"
            + (f" unexpected: {unexpected_names}." if unexpected_names else "."),
        ]

        return CheckResult(
            passed=passed,
            score=round(overall, 4),
            reasoning=" ".join(parts),
            cost=0.0,
            latency_ms=0,
            metric_name=self.metric_name,
            threshold=self.threshold,
        )
