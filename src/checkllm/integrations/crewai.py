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
