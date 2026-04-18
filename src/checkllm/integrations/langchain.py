"""LangChain integration for checkllm.

Provides a callback handler that validates LLM outputs using checkllm guards.

Usage::

    from checkllm.integrations.langchain import CheckllmCallbackHandler

    handler = CheckllmCallbackHandler(
        checks=["no_pii", "max_tokens:500", "toxicity"],
        on_failure="log",  # or "raise"
    )

    # Use with any LangChain chain
    chain.invoke({"input": "..."}, config={"callbacks": [handler]})

Requires: ``pip install langchain-core``
"""
from __future__ import annotations

import logging
from typing import Any

from checkllm.api import _build_guard, _run_async
from checkllm.guardrails import Guard, ValidationResult

logger = logging.getLogger("checkllm.integrations.langchain")

try:
    from langchain_core.callbacks import BaseCallbackHandler as _LangChainBase
except ImportError:
    _LangChainBase = object  # type: ignore[assignment,misc]


class CheckllmCallbackHandler(_LangChainBase):
    """LangChain callback handler that validates outputs with checkllm.

    Parameters
    ----------
    checks:
        List of check specs (shorthand strings or dicts).
        Example: ``["no_pii", "max_tokens:200", "toxicity"]``
    on_failure:
        What to do when validation fails:
        - ``"log"`` — log a warning (default)
        - ``"raise"`` — raise a ``GuardrailError``
    judge:
        Judge backend name or instance. Defaults to auto-detection.
    threshold:
        Default pass/fail threshold for judge checks.
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

    def on_llm_end(self, response: Any, **kwargs: Any) -> None:
        """LangChain callback: called when LLM finishes generating."""
        try:
            # Extract text from LangChain LLMResult
            if hasattr(response, "generations"):
                for gen_list in response.generations:
                    for gen in gen_list:
                        text = gen.text if hasattr(gen, "text") else str(gen)
                        if text:
                            self.validate(text)
            elif isinstance(response, str):
                self.validate(response)
        except Exception as exc:
            logger.debug("checkllm callback error: %s", exc)

    def on_chain_end(self, outputs: Any, **kwargs: Any) -> None:
        """LangChain callback: called when chain finishes."""
        try:
            if isinstance(outputs, dict):
                # Common patterns: {"output": "..."}, {"text": "..."}, {"result": "..."}
                for key in ("output", "text", "result", "answer"):
                    if key in outputs and isinstance(outputs[key], str):
                        self.validate(outputs[key])
                        return
            elif isinstance(outputs, str):
                self.validate(outputs)
        except Exception as exc:
            logger.debug("checkllm callback error: %s", exc)
