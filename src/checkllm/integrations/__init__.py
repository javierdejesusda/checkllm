"""Framework integrations for checkllm.

These modules provide thin wrappers that capture LLM outputs from
popular frameworks and validate them through checkllm's Guard system.

Supported frameworks:

- **LangChain** -- ``LangChainHandler`` (callback handler)
- **LlamaIndex** -- ``LlamaIndexHandler`` (callback handler)
- **CrewAI** -- ``CrewAICallback`` (agent/task/crew callbacks)
- **PydanticAI** -- ``PydanticAIValidator`` (result validator)
- **OpenAI Agents SDK** -- ``OpenAIAgentsHandler`` (run handler)
- **Claude Agent SDK** -- ``ClaudeAgentHandler`` (turn/tool handler)
"""
from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from checkllm.integrations.claude_agents import CheckllmAgentHandler as ClaudeAgentHandler
    from checkllm.integrations.crewai import CheckllmCrewCallback as CrewAICallback
    from checkllm.integrations.langchain import CheckllmCallbackHandler as LangChainHandler
    from checkllm.integrations.llamaindex import CheckllmCallbackHandler as LlamaIndexHandler
    from checkllm.integrations.openai_agents import CheckllmRunHandler as OpenAIAgentsHandler
    from checkllm.integrations.pydantic_ai import CheckllmResultValidator as PydanticAIValidator

__all__ = [
    "ClaudeAgentHandler",
    "CrewAICallback",
    "LangChainHandler",
    "LlamaIndexHandler",
    "OpenAIAgentsHandler",
    "PydanticAIValidator",
]


def __getattr__(name: str):  # noqa: ANN001
    """Lazy-load integration classes on first access."""
    if name == "LangChainHandler":
        from checkllm.integrations.langchain import CheckllmCallbackHandler
        return CheckllmCallbackHandler

    if name == "LlamaIndexHandler":
        from checkllm.integrations.llamaindex import CheckllmCallbackHandler
        return CheckllmCallbackHandler

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

    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
