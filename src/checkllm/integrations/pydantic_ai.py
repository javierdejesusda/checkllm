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

from checkllm.agents import AgentStep, AgentTestCase, ToolCall
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


_PYDANTIC_AI_INSTALL_HINT = (
    "pydantic-ai is not installed. Install it with `pip install pydantic-ai` "
    "to use this adapter."
)


def _coerce_args(raw: Any) -> dict[str, Any]:
    """Project a pydantic-ai ``ToolCallPart.args`` value onto a parameter dict.

    pydantic-ai's ``args`` is either a JSON-string or an already-parsed dict
    depending on the model wire format. Both are accepted; anything else
    yields ``{}``.
    """
    if isinstance(raw, dict):
        return dict(raw)
    if isinstance(raw, str) and raw:
        import json

        try:
            parsed = json.loads(raw)
        except (ValueError, TypeError):
            return {}
        return parsed if isinstance(parsed, dict) else {}
    return {}


def _is_tool_call_part(part: Any) -> bool:
    """Heuristic: a duck-typed pydantic-ai ToolCallPart has tool_name + args."""
    if isinstance(part, dict):
        return part.get("part_kind") == "tool-call" or ("tool_name" in part and "args" in part)
    kind = getattr(part, "part_kind", None)
    if kind == "tool-call":
        return True
    return hasattr(part, "tool_name") and hasattr(part, "args")


def _is_tool_return_part(part: Any) -> bool:
    """Heuristic: a duck-typed pydantic-ai ToolReturnPart has tool_name + content."""
    if isinstance(part, dict):
        return part.get("part_kind") == "tool-return" or (
            "tool_name" in part and "content" in part and "args" not in part
        )
    kind = getattr(part, "part_kind", None)
    if kind == "tool-return":
        return True
    return hasattr(part, "tool_name") and hasattr(part, "content") and not hasattr(part, "args")


def _iter_parts(messages: Any) -> list[Any]:
    """Flatten a pydantic-ai ``all_messages()`` list into a flat list of parts.

    Each message has a ``parts`` attribute (or ``"parts"`` key) holding a list
    of ``ToolCallPart`` / ``ToolReturnPart`` / ``TextPart``. Items that look
    like parts directly (no ``parts`` attribute) are returned as-is.
    """
    out: list[Any] = []
    for msg in messages:
        if isinstance(msg, dict):
            parts = msg.get("parts")
        else:
            parts = getattr(msg, "parts", None)
        if parts is None:
            out.append(msg)
        else:
            out.extend(parts)
    return out


def to_checkllm_tool_calls(messages: Any) -> list[ToolCall]:
    """Convert a pydantic-ai ``RunResult.all_messages()`` list into ``ToolCall`` objects.

    pydantic-ai represents tool invocations as paired ``ToolCallPart`` (the
    request) and ``ToolReturnPart`` (the response) nodes inside its
    ``ModelMessage``/``UserMessage`` parts list. This function pairs them
    by ``tool_call_id`` (when present), falling back to invocation order.

    Args:
        messages: An iterable of messages or already-flattened parts.
            Items with a ``parts`` attribute/key are flattened first.

    Returns:
        A list of :class:`~checkllm.agents.ToolCall` in invocation order.

    Raises:
        ImportError: If pydantic-ai is not installed and the input is not
            duck-typeable.

    Example::

        from checkllm.integrations.pydantic_ai import to_checkllm_tool_calls

        result = agent.run_sync("What is 2+2?")
        calls = to_checkllm_tool_calls(result.all_messages())
    """
    parts = _iter_parts(messages) if messages else []
    tool_calls_raw: list[Any] = [p for p in parts if _is_tool_call_part(p)]
    tool_returns_raw: list[Any] = [p for p in parts if _is_tool_return_part(p)]
    if not parts:
        return []
    if not tool_calls_raw and not tool_returns_raw:
        try:
            import pydantic_ai  # noqa: F401
        except ImportError as exc:
            raise ImportError(_PYDANTIC_AI_INSTALL_HINT) from exc
        # Framework is installed but inputs were unrecognizable.
        return []

    def _attr(part: Any, name: str) -> Any:
        if isinstance(part, dict):
            return part.get(name)
        return getattr(part, name, None)

    returns_by_id: dict[str, Any] = {}
    leftover_returns: list[Any] = []
    for ret in tool_returns_raw:
        rid = _attr(ret, "tool_call_id")
        if rid:
            returns_by_id[str(rid)] = ret
        else:
            leftover_returns.append(ret)

    calls: list[ToolCall] = []
    for call in tool_calls_raw:
        cid = _attr(call, "tool_call_id")
        ret_part = None
        if cid and str(cid) in returns_by_id:
            ret_part = returns_by_id.pop(str(cid))
        elif leftover_returns:
            # Fallback: pair with the first remaining return that has a
            # matching tool name; otherwise just pop FIFO.
            for i, candidate in enumerate(leftover_returns):
                if _attr(candidate, "tool_name") == _attr(call, "tool_name"):
                    ret_part = leftover_returns.pop(i)
                    break
            else:
                ret_part = leftover_returns.pop(0)

        result = None
        if ret_part is not None:
            content = _attr(ret_part, "content")
            if content is not None:
                result = str(content)

        params = _coerce_args(_attr(call, "args"))
        calls.append(
            ToolCall(
                name=str(_attr(call, "tool_name") or ""),
                parameters=params,
                result=result,
            )
        )
    return calls


def to_checkllm_test_case(
    run_result: Any,
    *,
    query: str | None = None,
    final_output: str | None = None,
) -> AgentTestCase:
    """Convert a pydantic-ai ``RunResult`` into a CheckLLM ``AgentTestCase``.

    Reads ``run_result.all_messages()`` (or ``run_result["messages"]``) for
    tool calls and ``run_result.data`` (or ``run_result["data"]``) for the
    final answer.

    Args:
        run_result: A pydantic-ai ``RunResult`` (sync or async). Must
            either expose ``all_messages()`` callable or a ``messages``
            attribute / key.
        query: The user's prompt. Required because pydantic-ai does not
            store the prompt on ``RunResult``.
        final_output: Override for the agent's final answer. If ``None``,
            falls back to ``run_result.data``.

    Returns:
        A populated :class:`~checkllm.agents.AgentTestCase`.

    Raises:
        ImportError: If pydantic-ai is not installed and the input cannot
            be duck-typed.

    Example::

        from checkllm.integrations.pydantic_ai import to_checkllm_test_case

        result = agent.run_sync("What is 2+2?")
        case = to_checkllm_test_case(result, query="What is 2+2?")
    """
    if isinstance(run_result, dict):
        messages = run_result.get("messages") or run_result.get("all_messages") or []
        data = run_result.get("data")
    else:
        all_messages_fn = getattr(run_result, "all_messages", None)
        if callable(all_messages_fn):
            messages = all_messages_fn()
        else:
            messages = getattr(run_result, "messages", None) or []
        data = getattr(run_result, "data", None)

    tool_calls = to_checkllm_tool_calls(messages)
    steps = [AgentStep(tool_call=tc, action="call_tool") for tc in tool_calls]
    resolved_query = query if query is not None else ""
    if final_output is not None:
        resolved_output = final_output
    elif data is not None:
        resolved_output = str(data)
    else:
        resolved_output = None
    return AgentTestCase(query=resolved_query, steps=steps, final_output=resolved_output)
