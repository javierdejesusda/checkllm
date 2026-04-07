"""OpenAI Agents SDK integration for checkllm.

Provides a run handler that validates OpenAI agent outputs.

Usage::

    from checkllm.integrations.openai_agents import CheckllmRunHandler

    handler = CheckllmRunHandler(
        checks=["no_pii", "toxicity"],
        on_failure="log",
    )

    # Validate a completed run
    handler.on_run_complete(run_result)

    # Validate tool output
    handler.on_tool_output("search", tool_output)

    # Wrap an agent function for automatic validation
    @handler.wrap_agent
    def my_agent(prompt):
        return llm.generate(prompt)

Requires: ``pip install openai-agents``
"""
from __future__ import annotations

import functools
import logging
from typing import Any, Callable, TypeVar

from checkllm.api import _build_guard, _run_async
from checkllm.guardrails import Guard, ValidationResult

logger = logging.getLogger("checkllm.integrations.openai_agents")

F = TypeVar("F", bound=Callable[..., Any])


def _extract_text(obj: Any) -> str:
    """Extract text from an OpenAI Agents SDK result object.

    Tries ``.final_output``, ``.output``, then falls back to ``str()``.

    Args:
        obj: An OpenAI agent run result or plain string.

    Returns:
        The extracted text content.
    """
    if isinstance(obj, str):
        return obj
    for attr in ("final_output", "output"):
        value = getattr(obj, attr, None)
        if value is not None:
            return str(value)
    return str(obj)


class CheckllmRunHandler:
    """OpenAI Agents SDK handler that validates outputs with checkllm.

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

    def on_run_complete(self, run_result: Any) -> ValidationResult:
        """Validate a completed agent run output.

        Args:
            run_result: The result from an OpenAI Agents SDK run.

        Returns:
            A ``ValidationResult`` with pass/fail details.
        """
        try:
            text = _extract_text(run_result)
            if text:
                return self.validate(text)
        except Exception as exc:
            logger.debug("checkllm callback error: %s", exc)
        return self.validate("")

    def on_tool_output(self, tool_name: str, output: Any) -> ValidationResult:
        """Validate an individual tool output.

        Args:
            tool_name: Name of the tool that produced the output.
            output: The tool's output value.

        Returns:
            A ``ValidationResult`` with pass/fail details.
        """
        try:
            text = str(output) if output is not None else ""
            if text:
                return self.validate(text)
        except Exception as exc:
            logger.debug("checkllm callback error (tool=%s): %s", tool_name, exc)
        return self.validate("")

    def wrap_agent(self, agent_func: F) -> F:
        """Decorator that auto-validates an agent function's return value.

        Args:
            agent_func: The agent function to wrap.

        Returns:
            A wrapped function that validates the output after each call.
        """

        @functools.wraps(agent_func)
        def wrapper(*args: Any, **kwargs: Any) -> Any:
            result = agent_func(*args, **kwargs)
            text = _extract_text(result)
            if text:
                self.validate(text)
            return result

        return wrapper  # type: ignore[return-value]
