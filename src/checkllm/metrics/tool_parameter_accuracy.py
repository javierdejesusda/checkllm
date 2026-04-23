"""Deterministic tool parameter / argument validation metric.

Compares the arguments supplied to each tool call against an expected
schema and expected values. Works on either raw :class:`ToolCall` objects
or the richer :class:`ToolCallTrace` records.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any, Union

from checkllm.agents import ToolCall, ToolCallTrace
from checkllm.models import CheckResult

_TRACE_LIKE = Union[ToolCall, ToolCallTrace, Mapping[str, Any]]


def _coerce(call: _TRACE_LIKE) -> tuple[str, dict[str, Any]]:
    """Return ``(tool_name, parameters)`` for any accepted call representation."""
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


def _type_matches(value: Any, expected_type: Any) -> bool:
    """Check whether *value* matches a declared schema ``expected_type``.

    Supports plain ``type`` objects, tuples of types, and JSON-Schema style
    string identifiers (``"string"``, ``"integer"``, ``"number"``, ``"boolean"``,
    ``"array"``, ``"object"``, ``"null"``).
    """
    if expected_type is None:
        return True
    if isinstance(expected_type, type):
        return isinstance(value, expected_type)
    if isinstance(expected_type, tuple):
        return any(_type_matches(value, t) for t in expected_type)
    if isinstance(expected_type, str):
        mapping: dict[str, tuple[type, ...]] = {
            "string": (str,),
            "str": (str,),
            "integer": (int,),
            "int": (int,),
            "number": (int, float),
            "float": (float, int),
            "boolean": (bool,),
            "bool": (bool,),
            "array": (list, tuple),
            "list": (list, tuple),
            "object": (dict,),
            "dict": (dict,),
            "null": (type(None),),
            "none": (type(None),),
        }
        types = mapping.get(expected_type.lower())
        if types is None:
            return True
        if bool in types and not isinstance(value, bool):
            return False
        if types == (int,) and isinstance(value, bool):
            return False
        return isinstance(value, types)
    return True


class ToolParameterAccuracyMetric:
    """Validate tool call arguments against an expected schema and values.

    Each expected tool call is described by a mapping with:

    * ``tool_name`` -- required tool name to match.
    * ``required`` -- iterable of parameter names that must be present.
    * ``optional`` -- iterable of parameter names that are permitted but
      not required.
    * ``schema`` -- mapping of parameter name to an expected type (a Python
      ``type``, a tuple of types, or a JSON-Schema style identifier).
    * ``values`` -- mapping of parameter name to an exact expected value.

    The metric scores the fraction of total individual parameter checks
    that passed across all expected tool calls.
    """

    metric_name = "tool_parameter_accuracy"

    def __init__(self, threshold: float = 0.8, strict_extras: bool = False) -> None:
        """Construct the metric.

        Args:
            threshold: Minimum overall score required to pass.
            strict_extras: When ``True``, any parameter that is neither
                ``required`` nor ``optional`` counts as a failed check.
        """
        self.threshold = threshold
        self.strict_extras = strict_extras

    def evaluate(
        self,
        actual_calls: list[_TRACE_LIKE],
        expected_calls: list[Mapping[str, Any]],
    ) -> CheckResult:
        """Score the actual calls against the expected schema/value list.

        Args:
            actual_calls: The tool calls the agent actually made.
            expected_calls: Mapping descriptions of expected tool calls.

        Returns:
            A :class:`CheckResult` summarising parameter accuracy.
        """
        if not expected_calls:
            return CheckResult(
                passed=True,
                score=1.0,
                reasoning="No expected tool calls specified; nothing to validate.",
                cost=0.0,
                latency_ms=0,
                metric_name=self.metric_name,
                threshold=self.threshold,
            )

        total_checks = 0
        passed_checks = 0
        details: list[str] = []

        actual_by_name: dict[str, list[tuple[str, dict[str, Any]]]] = {}
        for call in actual_calls:
            name, params = _coerce(call)
            actual_by_name.setdefault(name, []).append((name, params))

        for i, spec in enumerate(expected_calls):
            expected_name = str(spec.get("tool_name") or spec.get("name") or "")
            required = list(spec.get("required") or [])
            optional = list(spec.get("optional") or [])
            schema = dict(spec.get("schema") or {})
            values = dict(spec.get("values") or {})

            queue = actual_by_name.get(expected_name) or []
            if not queue:
                total_checks += 1 + len(required) + len(schema) + len(values)
                details.append(f"[{i}] missing call to '{expected_name}'")
                continue

            _, params = queue.pop(0)

            total_checks += 1
            passed_checks += 1

            for key in required:
                total_checks += 1
                if key in params:
                    passed_checks += 1
                else:
                    details.append(f"[{i}] {expected_name}: missing required param '{key}'")

            for key, expected_type in schema.items():
                total_checks += 1
                if key not in params:
                    details.append(f"[{i}] {expected_name}: param '{key}' missing for type check")
                    continue
                if _type_matches(params[key], expected_type):
                    passed_checks += 1
                else:
                    details.append(
                        f"[{i}] {expected_name}: param '{key}'={params[key]!r} "
                        f"does not satisfy type {expected_type!r}"
                    )

            for key, expected_value in values.items():
                total_checks += 1
                if params.get(key) == expected_value:
                    passed_checks += 1
                else:
                    details.append(
                        f"[{i}] {expected_name}: param '{key}'={params.get(key)!r} "
                        f"!= expected {expected_value!r}"
                    )

            if self.strict_extras:
                allowed = set(required) | set(optional) | set(schema) | set(values)
                extras = [k for k in params if k not in allowed]
                for extra in extras:
                    total_checks += 1
                    details.append(f"[{i}] {expected_name}: unexpected extra param '{extra}'")

        score = passed_checks / total_checks if total_checks > 0 else 1.0
        passed = score >= self.threshold and not details
        summary = f"{passed_checks}/{total_checks} parameter checks passed ({score:.0%})."
        if details:
            summary += " Issues: " + "; ".join(details[:8])
            if len(details) > 8:
                summary += f" (+{len(details) - 8} more)"

        return CheckResult(
            passed=passed,
            score=round(score, 4),
            reasoning=summary,
            cost=0.0,
            latency_ms=0,
            metric_name=self.metric_name,
            threshold=self.threshold,
        )
