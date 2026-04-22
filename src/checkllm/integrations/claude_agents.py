"""Claude Agent SDK integration for checkllm.

Provides handlers for validating Claude Agent SDK outputs.

Usage::

    from checkllm.integrations.claude_agents import CheckllmAgentHandler

    handler = CheckllmAgentHandler(
        checks=["no_pii", "toxicity"],
        on_failure="log",
    )

    # Validate a conversation turn
    handler.on_turn_complete(turn)

    # Validate tool execution result
    handler.on_tool_result("search", tool_result)

    # Wrap an agent function for automatic validation
    @handler.wrap_agent
    def my_agent(prompt):
        return claude.generate(prompt)

Requires: ``pip install claude-agent-sdk``
"""

from __future__ import annotations

import functools
import logging
from typing import Any, Callable, TypeVar

from checkllm.api import _build_guard, _run_async
from checkllm.guardrails import Guard, ValidationResult

logger = logging.getLogger("checkllm.integrations.claude_agents")

F = TypeVar("F", bound=Callable[..., Any])


def _extract_text(obj: Any) -> str:
    """Extract text from a Claude Agent SDK object.

    Tries ``.content``, ``.text``, ``.result``, then falls back
    to ``str()``.

    Args:
        obj: A Claude Agent SDK turn/result or plain string.

    Returns:
        The extracted text content.
    """
    if isinstance(obj, str):
        return obj
    for attr in ("content", "text", "result"):
        value = getattr(obj, attr, None)
        if value is not None and isinstance(value, str):
            return value
    return str(obj)


class CheckllmAgentHandler:
    """Claude Agent SDK handler that validates outputs with checkllm.

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

    def on_turn_complete(self, turn: Any) -> ValidationResult:
        """Validate a conversation turn.

        Args:
            turn: A Claude Agent SDK turn object.

        Returns:
            A ``ValidationResult`` with pass/fail details.
        """
        try:
            text = _extract_text(turn)
            if text:
                return self.validate(text)
        except Exception as exc:
            logger.debug("checkllm callback error: %s", exc)
        return self.validate("")

    def on_tool_result(self, tool_name: str, result: Any) -> ValidationResult:
        """Validate a tool execution result.

        Args:
            tool_name: Name of the tool that produced the result.
            result: The tool's execution result.

        Returns:
            A ``ValidationResult`` with pass/fail details.
        """
        try:
            text = _extract_text(result)
            if text:
                return self.validate(text)
        except Exception as exc:
            logger.debug(
                "checkllm callback error (tool=%s): %s",
                tool_name,
                exc,
            )
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
