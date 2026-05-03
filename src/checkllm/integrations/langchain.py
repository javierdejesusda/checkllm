"""LangChain integration for checkllm.

Provides a callback handler that validates LLM outputs using checkllm guards,
plus deterministic adapters that translate LangChain agent runs into
:class:`~checkllm.agents.AgentTestCase` objects so they can be scored by
CheckLLM's :class:`~checkllm.metrics.trajectory_metric.TrajectoryMetric`.

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

from checkllm.agents import AgentStep, AgentTestCase, ToolCall
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


_LANGCHAIN_INSTALL_HINT = (
    "LangChain is not installed. Install it with `pip install langchain-core` "
    "(or `pip install langchain`) to use this adapter."
)


def _coerce_tool_input(raw: Any) -> dict[str, Any]:
    """Project a LangChain ``AgentAction.tool_input`` onto a parameter dict.

    LangChain emits tool input as either a ``str`` (single-arg tools) or a
    ``dict`` (multi-arg tools). Anything else is dropped to ``{}`` rather
    than raising, mirroring :meth:`AgentTestCase.from_trace_jsonl`.
    """
    if isinstance(raw, dict):
        return dict(raw)
    if isinstance(raw, str):
        return {"input": raw}
    return {}


def to_checkllm_tool_calls(intermediate_steps: Any) -> list[ToolCall]:
    """Convert LangChain ``intermediate_steps`` into CheckLLM ``ToolCall`` objects.

    LangChain's ``AgentExecutor.invoke(...)`` returns a dict whose
    ``intermediate_steps`` key holds a list of ``(AgentAction, observation)``
    tuples. Each ``AgentAction`` has ``tool``, ``tool_input``, and ``log``
    attributes. This function reads those attributes via ``getattr`` so it
    also accepts plain duck-typed objects (which is what tests pass).

    Args:
        intermediate_steps: An iterable of ``(action, observation)`` 2-tuples
            where ``action`` exposes ``tool`` and ``tool_input`` attributes.
            Strings, ``None``, and bare action objects are also accepted; in
            those cases the observation is set to ``None``.

    Returns:
        A list of :class:`~checkllm.agents.ToolCall` in the same order as
        the input.

    Raises:
        ImportError: If ``langchain_core`` is not installed and the input
            is not duck-typeable (i.e. items lack ``.tool``).

    Example::

        from checkllm.integrations.langchain import to_checkllm_tool_calls

        steps = [(AgentAction(tool="search", tool_input="cats"), "results...")]
        calls = to_checkllm_tool_calls(steps)
    """
    calls: list[ToolCall] = []
    if not intermediate_steps:
        return calls
    for entry in intermediate_steps:
        if isinstance(entry, tuple) and len(entry) == 2:
            action, observation = entry
        else:
            action, observation = entry, None
        tool_name = getattr(action, "tool", None)
        if tool_name is None:
            try:
                import langchain_core  # noqa: F401
            except ImportError as exc:
                raise ImportError(_LANGCHAIN_INSTALL_HINT) from exc
            raise TypeError(
                "Expected a LangChain AgentAction (with `.tool` and `.tool_input`); "
                f"got {type(action).__name__}."
            )
        params = _coerce_tool_input(getattr(action, "tool_input", {}))
        result = None if observation is None else str(observation)
        calls.append(ToolCall(name=str(tool_name), parameters=params, result=result))
    return calls


def to_checkllm_test_case(
    agent_executor_output: Any,
    *,
    query: str | None = None,
    final_output: str | None = None,
) -> AgentTestCase:
    """Convert a LangChain ``AgentExecutor`` invocation result into an ``AgentTestCase``.

    Reads ``intermediate_steps`` (the canonical tool-call log) and ``output``
    (the final answer) from the dict returned by
    ``AgentExecutor.invoke(...)`` / ``.ainvoke(...)``. Both keys are
    optional; missing values are tolerated.

    Args:
        agent_executor_output: The dict returned by
            ``AgentExecutor.invoke(...)``. May also be a plain mapping with
            ``intermediate_steps`` and ``output`` keys.
        query: The user's original request. If ``None``, falls back to the
            ``input`` key of ``agent_executor_output``, then to the empty
            string.
        final_output: Override for the final answer. If ``None``, falls back
            to ``agent_executor_output["output"]``.

    Returns:
        A populated :class:`~checkllm.agents.AgentTestCase` whose
        ``steps`` contain one ``AgentStep`` per tool invocation.

    Raises:
        ImportError: If LangChain is not installed and the input cannot be
            duck-typed.
        TypeError: If ``agent_executor_output`` is not a mapping.

    Example::

        from langchain.agents import AgentExecutor
        from checkllm.integrations.langchain import to_checkllm_test_case

        run = executor.invoke({"input": "How tall is Everest?"})
        case = to_checkllm_test_case(run, query="How tall is Everest?")
    """
    if not isinstance(agent_executor_output, dict):
        raise TypeError(
            "Expected a dict (e.g. AgentExecutor.invoke output); "
            f"got {type(agent_executor_output).__name__}."
        )
    steps_raw = agent_executor_output.get("intermediate_steps", [])
    tool_calls = to_checkllm_tool_calls(steps_raw)
    steps = [AgentStep(tool_call=tc, action="call_tool") for tc in tool_calls]
    resolved_query = query if query is not None else str(agent_executor_output.get("input", ""))
    resolved_output = (
        final_output if final_output is not None else agent_executor_output.get("output")
    )
    return AgentTestCase(query=resolved_query, steps=steps, final_output=resolved_output)
