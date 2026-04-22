"""PydanticAI integration for checkllm.

Provides a result validator that checks PydanticAI agent responses.

Usage::

    from checkllm.integrations.pydantic_ai import CheckllmResultValidator

    validator = CheckllmResultValidator(
        checks=["no_pii", "relevance"],
        on_failure="raise",
    )

    # Use as a PydanticAI result validator
    agent = Agent("openai:gpt-4o", result_validator=validator.as_validator())

    # Or validate manually
    result = validator.validate("Some LLM output")

Requires: ``pip install pydantic-ai``
"""

from __future__ import annotations

import logging
from typing import Any, Callable

from checkllm.api import _build_guard, _run_async
from checkllm.guardrails import Guard, ValidationResult

logger = logging.getLogger("checkllm.integrations.pydantic_ai")


def _extract_text(obj: Any) -> str:
    """Extract text from a PydanticAI result object.

    Tries ``.data`` first, then falls back to ``str()``.

    Args:
        obj: A PydanticAI RunResult or plain string.

    Returns:
        The extracted text content.
    """
    if isinstance(obj, str):
        return obj
    data = getattr(obj, "data", None)
    if data is not None:
        return str(data)
    return str(obj)


class CheckllmResultValidator:
    """PydanticAI result validator that checks responses with checkllm.

    Args:
        checks: List of check specs (shorthand strings or dicts).
            Example: ``["no_pii", "relevance", "toxicity"]``
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

    def validate_result(self, result: Any) -> ValidationResult:
        """Validate a PydanticAI RunResult.

        Args:
            result: A PydanticAI ``RunResult`` object.

        Returns:
            A ``ValidationResult`` with pass/fail details.
        """
        try:
            text = _extract_text(result)
            if text:
                return self.validate(text)
        except Exception as exc:
            logger.debug("checkllm callback error: %s", exc)
        return self.validate("")

    def as_validator(self) -> Callable[[Any], Any]:
        """Return a callable suitable for PydanticAI's ``result_validator``.

        The returned function accepts a result context and validates the
        output text, raising on failure if ``on_failure="raise"``.

        Returns:
            A callable that validates PydanticAI results.
        """

        def _validator(ctx: Any) -> Any:
            data = getattr(ctx, "data", ctx)
            text = str(data) if data is not None else ""
            self.validate(text)
            return data

        return _validator
