"""LlamaIndex integration for checkllm.

Provides a callback handler that validates query engine responses.

Usage::

    from checkllm.integrations.llamaindex import CheckllmCallbackHandler

    handler = CheckllmCallbackHandler(
        checks=["no_pii", "max_tokens:500"],
        on_failure="log",
    )

    # Validate manually after querying
    response = query_engine.query("What is Python?")
    handler.validate(str(response))

Requires: ``pip install llama-index-core``
"""

from __future__ import annotations

import logging
from typing import Any

from checkllm.api import _build_guard, _run_async
from checkllm.guardrails import Guard, ValidationResult

logger = logging.getLogger("checkllm.integrations.llamaindex")


class CheckllmCallbackHandler:
    """LlamaIndex callback handler that validates responses with checkllm.

    Parameters
    ----------
    checks:
        List of check specs (shorthand strings or dicts).
    on_failure:
        ``"log"`` (default) or ``"raise"``.
    judge:
        Judge backend name or instance.
    threshold:
        Default pass/fail threshold.
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
        """Validate a single output string."""
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

    def on_event_end(
        self, event_type: Any, payload: dict[str, Any] | None = None, **kwargs: Any
    ) -> None:
        """LlamaIndex callback: called when an event finishes."""
        if payload is None:
            return
        try:
            # LlamaIndex response patterns
            response = payload.get("response")
            if response is not None:
                text = str(response)
                if text:
                    self.validate(text)
        except Exception as exc:
            logger.debug("checkllm callback error: %s", exc)
