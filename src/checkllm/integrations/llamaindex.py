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

import json
import logging
from typing import Any

from checkllm.agents import AgentStep, AgentTestCase, ToolCall
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


_LLAMAINDEX_INSTALL_HINT = (
    "LlamaIndex is not installed. Install it with `pip install llama-index-core` "
    "(or `pip install llama-index`) to use this adapter."
)


def _parse_tool_kwargs(raw: Any) -> dict[str, Any]:
    """Project a LlamaIndex ``ToolOutput.raw_input`` onto a parameter dict.

    LlamaIndex commonly stores tool kwargs under ``raw_input["kwargs"]`` for
    ``FunctionTool``, but some integrations supply ``raw_input`` directly as
    a dict, or a JSON-encoded string. All three shapes are accepted; anything
    else returns an empty dict.
    """
    if isinstance(raw, dict):
        if "kwargs" in raw and isinstance(raw["kwargs"], dict):
            return dict(raw["kwargs"])
        return dict(raw)
    if isinstance(raw, str) and raw:
        try:
            parsed = json.loads(raw)
        except (ValueError, TypeError):
            return {}
        return parsed if isinstance(parsed, dict) else {}
    return {}


def to_checkllm_tool_calls(sources: Any) -> list[ToolCall]:
    """Convert a LlamaIndex ``AgentChatResponse.sources`` list into ``ToolCall`` objects.

    LlamaIndex agents expose tool invocations as a list of ``ToolOutput``
    objects with the attributes ``tool_name``, ``raw_input``, and
    ``content``. Duck typing is used so synthetic dicts also work.

    Args:
        sources: An iterable of ``ToolOutput``-like objects (or a single
            object). Falsy inputs return an empty list.

    Returns:
        A list of :class:`~checkllm.agents.ToolCall`.

    Raises:
        ImportError: If LlamaIndex isn't installed and the input is not
            duck-typeable.

    Example::

        from checkllm.integrations.llama_index import to_checkllm_tool_calls

        response = agent.chat("What is 2+2?")
        calls = to_checkllm_tool_calls(response.sources)
    """
    calls: list[ToolCall] = []
    if not sources:
        return calls
    for src in sources:
        if isinstance(src, dict):
            tool_name = src.get("tool_name")
            raw_input = src.get("raw_input", {})
            content = src.get("content")
        else:
            tool_name = getattr(src, "tool_name", None)
            raw_input = getattr(src, "raw_input", {})
            content = getattr(src, "content", None)
        if tool_name is None:
            try:
                import llama_index  # noqa: F401
            except ImportError as exc:
                raise ImportError(_LLAMAINDEX_INSTALL_HINT) from exc
            raise TypeError(
                "Expected a LlamaIndex ToolOutput (with `.tool_name`); "
                f"got {type(src).__name__}."
            )
        params = _parse_tool_kwargs(raw_input)
        result = None if content is None else str(content)
        calls.append(ToolCall(name=str(tool_name), parameters=params, result=result))
    return calls


def to_checkllm_test_case(
    response: Any,
    *,
    query: str | None = None,
    final_output: str | None = None,
) -> AgentTestCase:
    """Convert a LlamaIndex ``AgentChatResponse`` into a CheckLLM ``AgentTestCase``.

    Reads ``response.sources`` for tool calls and ``response.response`` (or
    ``str(response)``) for the final answer. Duck typing is used so this
    function also accepts plain dicts with the same keys.

    Args:
        response: A LlamaIndex ``AgentChatResponse``-like object exposing
            ``sources`` and ``response`` attributes/keys.
        query: The user's original message. Required because LlamaIndex
            does not store the user message on the response object.
        final_output: Override for the agent's final answer. If ``None``,
            falls back to ``response.response`` or ``str(response)``.

    Returns:
        A populated :class:`~checkllm.agents.AgentTestCase`.

    Raises:
        ImportError: If LlamaIndex isn't installed and the input is not
            duck-typeable.
        TypeError: If ``response`` exposes neither ``sources`` (attribute or
            key).

    Example::

        from checkllm.integrations.llama_index import to_checkllm_test_case

        response = agent.chat("What is 2+2?")
        case = to_checkllm_test_case(response, query="What is 2+2?")
    """
    if isinstance(response, dict):
        sources = response.get("sources", [])
        response_text = response.get("response")
    else:
        if not hasattr(response, "sources"):
            raise TypeError(
                "Expected a LlamaIndex AgentChatResponse (with `.sources`); "
                f"got {type(response).__name__}."
            )
        sources = getattr(response, "sources", [])
        response_text = getattr(response, "response", None)
    tool_calls = to_checkllm_tool_calls(sources)
    steps = [AgentStep(tool_call=tc, action="call_tool") for tc in tool_calls]
    resolved_query = query if query is not None else ""
    if final_output is not None:
        resolved_output = final_output
    elif response_text is not None:
        resolved_output = str(response_text)
    elif not isinstance(response, dict):
        resolved_output = str(response)
    else:
        resolved_output = None
    return AgentTestCase(query=resolved_query, steps=steps, final_output=resolved_output)
