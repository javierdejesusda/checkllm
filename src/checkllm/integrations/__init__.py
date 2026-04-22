"""Framework integrations for checkllm.

These modules provide thin wrappers that capture LLM outputs from
popular frameworks and validate them through checkllm's Guard system,
together with observability exporters for tracing and metrics backends.

Supported frameworks:

- **LangChain** -- ``LangChainHandler`` (callback handler)
- **LlamaIndex** -- ``LlamaIndexHandler`` (callback handler)
- **CrewAI** -- ``CrewAICallback`` (agent/task/crew callbacks)
- **PydanticAI** -- ``PydanticAIValidator`` (result validator)
- **OpenAI Agents SDK** -- ``OpenAIAgentsHandler`` (run handler)
- **Claude Agent SDK** -- ``ClaudeAgentHandler`` (turn/tool handler)

Observability exporters:

- **LangSmith** -- ``LangSmithTracer``
- **LangFuse** -- ``LangFuseTracer``
- **Datadog** -- ``DatadogTracer``
- **Prometheus / Grafana** -- ``PrometheusExporter``
- **Cloud sync** -- ``push_to_remote``

Tracers can also be resolved by name through :func:`get_tracer`::

    from checkllm.integrations import get_tracer

    tracer = get_tracer("langfuse")
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from checkllm.integrations.claude_agents import (
        CheckllmAgentHandler as ClaudeAgentHandler,
    )
    from checkllm.integrations.cloud_sync import SyncResult, push_to_remote
    from checkllm.integrations.crewai import CheckllmCrewCallback as CrewAICallback
    from checkllm.integrations.datadog import DatadogTracer
    from checkllm.integrations.langchain import (
        CheckllmCallbackHandler as LangChainHandler,
    )
    from checkllm.integrations.langfuse import LangFuseTracer
    from checkllm.integrations.langsmith import LangSmithTracer
    from checkllm.integrations.llamaindex import (
        CheckllmCallbackHandler as LlamaIndexHandler,
    )
    from checkllm.integrations.openai_agents import (
        CheckllmRunHandler as OpenAIAgentsHandler,
    )
    from checkllm.integrations.prometheus import PrometheusExporter
    from checkllm.integrations.pydantic_ai import (
        CheckllmResultValidator as PydanticAIValidator,
    )

__all__ = [
    "ClaudeAgentHandler",
    "CrewAICallback",
    "DatadogTracer",
    "LangChainHandler",
    "LangFuseTracer",
    "LangSmithTracer",
    "LlamaIndexHandler",
    "OpenAIAgentsHandler",
    "PrometheusExporter",
    "PydanticAIValidator",
    "SyncResult",
    "get_tracer",
    "push_to_remote",
]


_TRACER_FACTORIES: dict[str, str] = {
    "langsmith": "checkllm.integrations.langsmith:LangSmithTracer",
    "langfuse": "checkllm.integrations.langfuse:LangFuseTracer",
    "datadog": "checkllm.integrations.datadog:DatadogTracer",
    "prometheus": "checkllm.integrations.prometheus:PrometheusExporter",
}


def get_tracer(name: str, **kwargs: Any) -> Any:
    """Resolve a tracer / exporter by name.

    Args:
        name: Case-insensitive backend name. One of ``"langsmith"``,
            ``"langfuse"``, ``"datadog"``, ``"prometheus"``.
        **kwargs: Forwarded to the tracer constructor.

    Returns:
        An instantiated tracer or exporter.

    Raises:
        ValueError: If ``name`` does not match a known backend.
        ImportError: If the required optional dependency is not installed.
    """
    key = (name or "").strip().lower()
    target = _TRACER_FACTORIES.get(key)
    if target is None:
        available = ", ".join(sorted(_TRACER_FACTORIES))
        raise ValueError(f"Unknown tracer '{name}'. Available tracers: {available}")
    module_path, _, class_name = target.partition(":")
    import importlib

    module = importlib.import_module(module_path)
    cls = getattr(module, class_name)
    return cls(**kwargs)


def __getattr__(name: str) -> Any:
    """Lazy-load integration classes on first access."""
    if name == "LangChainHandler":
        from checkllm.integrations.langchain import CheckllmCallbackHandler

        return CheckllmCallbackHandler

    if name == "LlamaIndexHandler":
        from checkllm.integrations.llamaindex import (
            CheckllmCallbackHandler as _LlamaIndexHandler,
        )

        return _LlamaIndexHandler

    if name == "CrewAICallback":
        from checkllm.integrations.crewai import CheckllmCrewCallback

        return CheckllmCrewCallback

    if name == "PydanticAIValidator":
        from checkllm.integrations.pydantic_ai import CheckllmResultValidator

        return CheckllmResultValidator

    if name == "OpenAIAgentsHandler":
        from checkllm.integrations.openai_agents import CheckllmRunHandler

        return CheckllmRunHandler

    if name == "ClaudeAgentHandler":
        from checkllm.integrations.claude_agents import CheckllmAgentHandler

        return CheckllmAgentHandler

    if name == "LangSmithTracer":
        from checkllm.integrations.langsmith import LangSmithTracer

        return LangSmithTracer

    if name == "LangFuseTracer":
        from checkllm.integrations.langfuse import LangFuseTracer

        return LangFuseTracer

    if name == "DatadogTracer":
        from checkllm.integrations.datadog import DatadogTracer

        return DatadogTracer

    if name == "PrometheusExporter":
        from checkllm.integrations.prometheus import PrometheusExporter

        return PrometheusExporter

    if name == "push_to_remote":
        from checkllm.integrations.cloud_sync import push_to_remote

        return push_to_remote

    if name == "SyncResult":
        from checkllm.integrations.cloud_sync import SyncResult

        return SyncResult

    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
