"""CrewAI integration for checkllm.

Provides callbacks that validate CrewAI agent outputs using checkllm guards.

Usage::

    from checkllm.integrations.crewai import CheckllmCrewCallback

    callback = CheckllmCrewCallback(
        checks=["no_pii", "toxicity"],
        on_failure="log",
    )

    # Validate agent output directly
    result = callback.validate("Some agent output text")

    # Or use callbacks with CrewAI event hooks
    callback.on_agent_action(agent_output)
    callback.on_task_output(task_output)
    callback.on_crew_output(crew_output)

Requires: ``pip install crewai``
"""

from __future__ import annotations

import logging
from typing import Any

from checkllm.agents import AgentStep, AgentTestCase, ToolCall
from checkllm.api import _build_guard, _run_async
from checkllm.guardrails import Guard, ValidationResult

logger = logging.getLogger("checkllm.integrations.crewai")


def _extract_text(obj: Any) -> str:
    """Extract text from a CrewAI output object.

    Tries common attributes in order: ``.raw``, ``.output``, then
    falls back to ``str()``.

    Args:
        obj: A CrewAI output object or plain string.

    Returns:
        The extracted text content.
    """
    if isinstance(obj, str):
        return obj
    for attr in ("raw", "output"):
        value = getattr(obj, attr, None)
        if value is not None and isinstance(value, str):
            return value
    return str(obj)


class CheckllmCrewCallback:
    """CrewAI callback that validates outputs with checkllm.

    Args:
        checks: List of check specs (shorthand strings or dicts).
            Example: ``["no_pii", "max_tokens:200", "toxicity"]``
        on_failure: What to do when validation fails.
            ``"log"`` (default) logs a warning; ``"raise"`` raises
            a ``GuardrailError``.
        judge: Judge backend name or instance. Defaults to auto-detection.
        threshold: Default pass/fail threshold for judge checks.
    """

    def __init__(
        self,
        checks: list[str | dict[str, Any]],
        on_failure: str = "log",
        judge: str | None = None,
        threshold: float = 0.8,
    ) -> None:
        self.checks = checks
        self.on_failure = on_failure
        self.judge = judge
        self.threshold = threshold
        self.results: list[ValidationResult] = []
        self._guard: Guard | None = None

    def _get_guard(self) -> Guard:
        if self._guard is None:
            self._guard = _build_guard(
                checks=self.checks,
                judge=self.judge,
                threshold=self.threshold,
            )
        return self._guard

    def validate(self, output: str) -> ValidationResult:
        """Validate a single output string.

        Args:
            output: The text to validate.

        Returns:
            A ``ValidationResult`` with pass/fail details.
        """
        guard = self._get_guard()
        result = _run_async(guard.avalidate(output))
        self.results.append(result)

        if not result.valid:
            if self.on_failure == "raise":
                result.raise_on_failure()
            else:
                logger.warning(
                    "checkllm validation failed: %s",
                    result.summary(),
                )

        return result

    def on_agent_action(self, agent_output: Any) -> ValidationResult:
        """Validate an agent's action output.

        Args:
            agent_output: The output from a CrewAI agent action.

        Returns:
            A ``ValidationResult`` with pass/fail details.
        """
        try:
            text = _extract_text(agent_output)
            if text:
                return self.validate(text)
        except Exception as exc:
            logger.debug("checkllm callback error: %s", exc)
        return self.validate("")

    def on_task_output(self, task_output: Any) -> ValidationResult:
        """Validate a task completion output.

        Args:
            task_output: The output from a completed CrewAI task.

        Returns:
            A ``ValidationResult`` with pass/fail details.
        """
        try:
            text = _extract_text(task_output)
            if text:
                return self.validate(text)
        except Exception as exc:
            logger.debug("checkllm callback error: %s", exc)
        return self.validate("")

    def on_crew_output(self, crew_output: Any) -> ValidationResult:
        """Validate the final crew output.

        Args:
            crew_output: The final output from a CrewAI crew run.

        Returns:
            A ``ValidationResult`` with pass/fail details.
        """
        try:
            text = _extract_text(crew_output)
            if text:
                return self.validate(text)
        except Exception as exc:
            logger.debug("checkllm callback error: %s", exc)
        return self.validate("")


_CREWAI_INSTALL_HINT = (
    "CrewAI is not installed. Install it with `pip install crewai` to use this adapter."
)


def _coerce_args(raw: Any) -> dict[str, Any]:
    """Project a CrewAI tool-usage args entry onto a parameter dict."""
    if isinstance(raw, dict):
        return dict(raw)
    if isinstance(raw, str):
        return {"input": raw}
    return {}


def to_checkllm_tool_calls(tool_usage: Any) -> list[ToolCall]:
    """Convert a CrewAI ``tool_usage`` log into ``ToolCall`` objects.

    CrewAI exposes per-task tool calls in two related places:

    * ``TaskOutput.tools_used`` -- a list of dicts emitted by the agent
      executor with keys ``tool``/``tool_name``, ``args``/``tool_args``,
      and ``result``/``output``.
    * ``Tool.run()`` callback hooks -- which produce the same shape.

    Both shapes are accepted. Duck typing is used for object inputs.

    Args:
        tool_usage: An iterable of dicts (or duck-typed objects) describing
            tool invocations in invocation order.

    Returns:
        A list of :class:`~checkllm.agents.ToolCall`.

    Raises:
        ImportError: If CrewAI is not installed and entries cannot be
            duck-typed (i.e. lack any tool-name field).

    Example::

        from checkllm.integrations.crewai import to_checkllm_tool_calls

        result = crew.kickoff()
        calls = to_checkllm_tool_calls(result.tasks_output[0].tools_used)
    """
    calls: list[ToolCall] = []
    if not tool_usage:
        return calls
    for entry in tool_usage:
        if isinstance(entry, dict):
            tool_name = entry.get("tool") or entry.get("tool_name") or entry.get("name")
            args = entry.get("args") or entry.get("tool_args") or entry.get("parameters")
            result = entry.get("result") if entry.get("result") is not None else entry.get("output")
        else:
            tool_name = (
                getattr(entry, "tool", None)
                or getattr(entry, "tool_name", None)
                or getattr(entry, "name", None)
            )
            args = (
                getattr(entry, "args", None)
                or getattr(entry, "tool_args", None)
                or getattr(entry, "parameters", None)
            )
            result = getattr(entry, "result", None)
            if result is None:
                result = getattr(entry, "output", None)
        if not tool_name:
            try:
                import crewai  # noqa: F401
            except ImportError as exc:
                raise ImportError(_CREWAI_INSTALL_HINT) from exc
            raise TypeError(
                "Expected a CrewAI tool-usage entry with a `tool` / `tool_name` field; "
                f"got {type(entry).__name__}."
            )
        params = _coerce_args(args)
        result_str = None if result is None else str(result)
        calls.append(ToolCall(name=str(tool_name), parameters=params, result=result_str))
    return calls


def to_checkllm_test_case(
    crew_output: Any,
    *,
    query: str | None = None,
    final_output: str | None = None,
) -> AgentTestCase:
    """Convert a CrewAI ``CrewOutput`` (or ``TaskOutput``) into ``AgentTestCase``.

    For a ``CrewOutput`` with ``tasks_output``, this concatenates the
    ``tools_used`` lists of each task in order. For a single ``TaskOutput``
    with a ``tools_used`` field, that list is used directly. Duck typing
    is used so plain dicts mirroring those shapes also work.

    Args:
        crew_output: A CrewAI ``CrewOutput`` or ``TaskOutput`` (or a dict
            with the same keys).
        query: The original user instruction. Required because CrewAI
            does not store the kickoff inputs on the output object.
        final_output: Override for the final crew/task answer. If ``None``,
            falls back to ``raw``/``output`` (via :func:`_extract_text`).

    Returns:
        A populated :class:`~checkllm.agents.AgentTestCase`.

    Raises:
        ImportError: If CrewAI is not installed and the input cannot be
            duck-typed.

    Example::

        from checkllm.integrations.crewai import to_checkllm_test_case

        result = crew.kickoff()
        case = to_checkllm_test_case(result, query="Plan a 2-day trip to Lisbon.")
    """
    if isinstance(crew_output, dict):
        tasks_output = crew_output.get("tasks_output")
        tools_used = crew_output.get("tools_used")
    else:
        tasks_output = getattr(crew_output, "tasks_output", None)
        tools_used = getattr(crew_output, "tools_used", None)

    aggregated: list[Any] = []
    if tasks_output:
        for task in tasks_output:
            task_tools = (
                task.get("tools_used")
                if isinstance(task, dict)
                else getattr(task, "tools_used", None)
            )
            if task_tools:
                aggregated.extend(task_tools)
    elif tools_used:
        aggregated.extend(tools_used)

    tool_calls = to_checkllm_tool_calls(aggregated)
    steps = [AgentStep(tool_call=tc, action="call_tool") for tc in tool_calls]
    resolved_query = query if query is not None else ""
    resolved_output = final_output if final_output is not None else _extract_text(crew_output)
    if resolved_output == str(crew_output) and not isinstance(crew_output, str):
        # Heuristic: if we fell all the way to repr(), prefer None over noise.
        if not any(getattr(crew_output, a, None) for a in ("raw", "output")):
            resolved_output = None
    return AgentTestCase(query=resolved_query, steps=steps, final_output=resolved_output)
