"""Tests for framework integrations.

Covers LangChain, LlamaIndex, CrewAI, PydanticAI, OpenAI Agents SDK,
and Claude Agent SDK handlers.  These tests don't require the target
frameworks installed -- they test the checkllm wrapper logic directly.
"""

import logging
from types import SimpleNamespace

import pytest
from unittest.mock import MagicMock

from checkllm.guardrails import GuardrailError
from checkllm.integrations.langchain import (
    CheckllmCallbackHandler as LangChainHandler,
)
from checkllm.integrations.llamaindex import (
    CheckllmCallbackHandler as LlamaIndexHandler,
)
from checkllm.integrations.crewai import CheckllmCrewCallback
from checkllm.integrations.pydantic_ai import CheckllmResultValidator
from checkllm.integrations.openai_agents import CheckllmRunHandler
from checkllm.integrations.claude_agents import CheckllmAgentHandler


# --- LangChain ---


def test_langchain_handler_validates_string():
    handler = LangChainHandler(checks=["contains:hello"])
    result = handler.validate("hello world")
    assert result.valid
    assert len(handler.results) == 1


def test_langchain_handler_fails_on_missing():
    handler = LangChainHandler(checks=["contains:goodbye"])
    result = handler.validate("hello world")
    assert not result.valid


def test_langchain_handler_raise_mode():
    handler = LangChainHandler(checks=["contains:goodbye"], on_failure="raise")
    from checkllm.guardrails import GuardrailError

    with pytest.raises(GuardrailError):
        handler.validate("hello world")


def test_langchain_handler_log_mode(caplog):
    handler = LangChainHandler(checks=["contains:goodbye"], on_failure="log")
    import logging

    with caplog.at_level(logging.WARNING, logger="checkllm.integrations.langchain"):
        result = handler.validate("hello world")
    assert not result.valid
    assert "failed" in caplog.text.lower()


def test_langchain_on_chain_end():
    handler = LangChainHandler(checks=["contains:hello"])
    handler.on_chain_end({"output": "hello world"})
    assert len(handler.results) == 1
    assert handler.results[0].valid


def test_langchain_on_chain_end_text_key():
    handler = LangChainHandler(checks=["contains:hello"])
    handler.on_chain_end({"text": "hello world"})
    assert len(handler.results) == 1


def test_langchain_on_chain_end_string():
    handler = LangChainHandler(checks=["contains:hello"])
    handler.on_chain_end("hello world")
    assert len(handler.results) == 1


def test_langchain_on_llm_end_string():
    handler = LangChainHandler(checks=["contains:hello"])
    handler.on_llm_end("hello world")
    assert len(handler.results) == 1


def test_langchain_on_llm_end_with_generations():
    handler = LangChainHandler(checks=["contains:hello"])
    gen = MagicMock()
    gen.text = "hello world"
    response = MagicMock()
    response.generations = [[gen]]
    handler.on_llm_end(response)
    assert len(handler.results) == 1


def test_langchain_handler_inherits_base_callback_handler():
    try:
        from langchain_core.callbacks import BaseCallbackHandler

        assert issubclass(LangChainHandler, BaseCallbackHandler), (
            "CheckllmCallbackHandler must inherit from BaseCallbackHandler "
            "so LangChain's callback system registers it correctly"
        )
    except ImportError:
        pass  # langchain_core not installed; inheritance check not applicable


# --- LlamaIndex ---


def test_llamaindex_handler_validates_string():
    handler = LlamaIndexHandler(checks=["contains:hello"])
    result = handler.validate("hello world")
    assert result.valid
    assert len(handler.results) == 1


def test_llamaindex_handler_fails():
    handler = LlamaIndexHandler(checks=["contains:goodbye"])
    result = handler.validate("hello world")
    assert not result.valid


def test_llamaindex_handler_raise_mode():
    handler = LlamaIndexHandler(checks=["contains:goodbye"], on_failure="raise")
    from checkllm.guardrails import GuardrailError

    with pytest.raises(GuardrailError):
        handler.validate("hello world")


def test_llamaindex_on_event_end():
    handler = LlamaIndexHandler(checks=["contains:hello"])
    handler.on_event_end("query", payload={"response": "hello world"})
    assert len(handler.results) == 1


def test_llamaindex_on_event_end_no_payload():
    handler = LlamaIndexHandler(checks=["contains:hello"])
    handler.on_event_end("query", payload=None)
    assert len(handler.results) == 0


def test_both_handlers_accumulate_results():
    lc = LangChainHandler(checks=["contains:a"])
    li = LlamaIndexHandler(checks=["contains:a"])
    for _ in range(3):
        lc.validate("aaa")
        li.validate("aaa")
    assert len(lc.results) == 3
    assert len(li.results) == 3


class TestCrewAICallback:
    """Tests for CheckllmCrewCallback."""

    def test_init(self):
        cb = CheckllmCrewCallback(checks=["no_pii", "toxicity"])
        assert cb.checks == ["no_pii", "toxicity"]
        assert cb.on_failure == "log"
        assert cb.threshold == 0.8
        assert cb.results == []

    def test_validate_passing(self):
        cb = CheckllmCrewCallback(checks=["contains:hello"])
        result = cb.validate("hello world")
        assert result.valid
        assert len(cb.results) == 1

    def test_validate_failing_log(self, caplog):
        cb = CheckllmCrewCallback(checks=["contains:goodbye"], on_failure="log")
        with caplog.at_level(logging.WARNING, logger="checkllm.integrations.crewai"):
            result = cb.validate("hello world")
        assert not result.valid
        assert "failed" in caplog.text.lower()

    def test_validate_failing_raise(self):
        cb = CheckllmCrewCallback(checks=["contains:goodbye"], on_failure="raise")
        with pytest.raises(GuardrailError):
            cb.validate("hello world")

    def test_on_agent_action_with_raw(self):
        cb = CheckllmCrewCallback(checks=["contains:hello"])
        agent_output = SimpleNamespace(raw="hello from agent")
        result = cb.on_agent_action(agent_output)
        assert result.valid

    def test_on_agent_action_with_output(self):
        cb = CheckllmCrewCallback(checks=["contains:hello"])
        agent_output = SimpleNamespace(output="hello from output")
        result = cb.on_agent_action(agent_output)
        assert result.valid

    def test_on_agent_action_string(self):
        cb = CheckllmCrewCallback(checks=["contains:hello"])
        result = cb.on_agent_action("hello plain string")
        assert result.valid

    def test_on_agent_action_str_fallback(self):
        cb = CheckllmCrewCallback(checks=["contains:some_field"])
        obj = SimpleNamespace(some_field=42)
        result = cb.on_agent_action(obj)
        assert result.valid

    def test_on_task_output(self):
        cb = CheckllmCrewCallback(checks=["contains:task"])
        task_output = SimpleNamespace(raw="task result text")
        result = cb.on_task_output(task_output)
        assert result.valid

    def test_on_crew_output(self):
        cb = CheckllmCrewCallback(checks=["contains:crew"])
        crew_output = SimpleNamespace(raw="crew final result")
        result = cb.on_crew_output(crew_output)
        assert result.valid

    def test_accumulates_results(self):
        cb = CheckllmCrewCallback(checks=["contains:a"])
        for _ in range(3):
            cb.validate("aaa")
        assert len(cb.results) == 3


class TestPydanticAIValidator:
    """Tests for CheckllmResultValidator."""

    def test_init(self):
        v = CheckllmResultValidator(checks=["no_pii", "relevance"])
        assert v.checks == ["no_pii", "relevance"]
        assert v.on_failure == "log"
        assert v.results == []

    def test_validate_passing(self):
        v = CheckllmResultValidator(checks=["contains:hello"])
        result = v.validate("hello world")
        assert result.valid
        assert len(v.results) == 1

    def test_validate_failing_raise(self):
        v = CheckllmResultValidator(checks=["contains:goodbye"], on_failure="raise")
        with pytest.raises(GuardrailError):
            v.validate("hello world")

    def test_validate_failing_log(self, caplog):
        v = CheckllmResultValidator(checks=["contains:goodbye"], on_failure="log")
        with caplog.at_level(logging.WARNING, logger="checkllm.integrations.pydantic_ai"):
            result = v.validate("hello world")
        assert not result.valid
        assert "failed" in caplog.text.lower()

    def test_validate_result_with_data(self):
        v = CheckllmResultValidator(checks=["contains:hello"])
        run_result = SimpleNamespace(data="hello from data")
        result = v.validate_result(run_result)
        assert result.valid

    def test_validate_result_string(self):
        v = CheckllmResultValidator(checks=["contains:hello"])
        result = v.validate_result("hello plain string")
        assert result.valid

    def test_as_validator_returns_callable(self):
        v = CheckllmResultValidator(checks=["contains:hello"])
        validator_fn = v.as_validator()
        assert callable(validator_fn)

    def test_as_validator_validates_data(self):
        v = CheckllmResultValidator(checks=["contains:hello"])
        validator_fn = v.as_validator()
        ctx = SimpleNamespace(data="hello world")
        result_data = validator_fn(ctx)
        assert result_data == "hello world"
        assert len(v.results) == 1

    def test_as_validator_raises_on_failure(self):
        v = CheckllmResultValidator(checks=["contains:goodbye"], on_failure="raise")
        validator_fn = v.as_validator()
        ctx = SimpleNamespace(data="hello world")
        with pytest.raises(GuardrailError):
            validator_fn(ctx)


class TestOpenAIAgentsHandler:
    """Tests for CheckllmRunHandler."""

    def test_init(self):
        h = CheckllmRunHandler(checks=["no_pii", "toxicity"])
        assert h.checks == ["no_pii", "toxicity"]
        assert h.on_failure == "log"
        assert h.results == []

    def test_validate_passing(self):
        h = CheckllmRunHandler(checks=["contains:hello"])
        result = h.validate("hello world")
        assert result.valid
        assert len(h.results) == 1

    def test_validate_failing_raise(self):
        h = CheckllmRunHandler(checks=["contains:goodbye"], on_failure="raise")
        with pytest.raises(GuardrailError):
            h.validate("hello world")

    def test_validate_failing_log(self, caplog):
        h = CheckllmRunHandler(checks=["contains:goodbye"], on_failure="log")
        with caplog.at_level(logging.WARNING, logger="checkllm.integrations.openai_agents"):
            result = h.validate("hello world")
        assert not result.valid
        assert "failed" in caplog.text.lower()

    def test_on_run_complete_with_final_output(self):
        h = CheckllmRunHandler(checks=["contains:hello"])
        run_result = SimpleNamespace(final_output="hello from agent")
        result = h.on_run_complete(run_result)
        assert result.valid

    def test_on_run_complete_with_output(self):
        h = CheckllmRunHandler(checks=["contains:hello"])
        run_result = SimpleNamespace(output="hello from output")
        result = h.on_run_complete(run_result)
        assert result.valid

    def test_on_run_complete_string(self):
        h = CheckllmRunHandler(checks=["contains:hello"])
        result = h.on_run_complete("hello plain string")
        assert result.valid

    def test_on_tool_output(self):
        h = CheckllmRunHandler(checks=["contains:hello"])
        result = h.on_tool_output("search", "hello results")
        assert result.valid

    def test_wrap_agent(self):
        h = CheckllmRunHandler(checks=["contains:hello"])

        @h.wrap_agent
        def my_agent(prompt):
            return "hello response"

        output = my_agent("test prompt")
        assert output == "hello response"
        assert len(h.results) == 1

    def test_wrap_agent_preserves_metadata(self):
        h = CheckllmRunHandler(checks=["contains:a"])

        @h.wrap_agent
        def my_special_agent(prompt):
            """Agent docstring."""
            return "aaa"

        assert my_special_agent.__name__ == "my_special_agent"
        assert my_special_agent.__doc__ == "Agent docstring."

    def test_wrap_agent_raises_on_failure(self):
        h = CheckllmRunHandler(checks=["contains:goodbye"], on_failure="raise")

        @h.wrap_agent
        def bad_agent(prompt):
            return "hello world"

        with pytest.raises(GuardrailError):
            bad_agent("test")


class TestClaudeAgentHandler:
    """Tests for CheckllmAgentHandler."""

    def test_init(self):
        h = CheckllmAgentHandler(checks=["no_pii", "toxicity"])
        assert h.checks == ["no_pii", "toxicity"]
        assert h.on_failure == "log"
        assert h.results == []

    def test_validate_passing(self):
        h = CheckllmAgentHandler(checks=["contains:hello"])
        result = h.validate("hello world")
        assert result.valid
        assert len(h.results) == 1

    def test_validate_failing_log(self, caplog):
        h = CheckllmAgentHandler(checks=["contains:goodbye"], on_failure="log")
        with caplog.at_level(logging.WARNING, logger="checkllm.integrations.claude_agents"):
            result = h.validate("hello world")
        assert not result.valid
        assert "failed" in caplog.text.lower()

    def test_validate_failing_raise(self):
        h = CheckllmAgentHandler(checks=["contains:goodbye"], on_failure="raise")
        with pytest.raises(GuardrailError):
            h.validate("hello world")

    def test_on_turn_complete_with_content(self):
        h = CheckllmAgentHandler(checks=["contains:hello"])
        turn = SimpleNamespace(content="hello turn content")
        result = h.on_turn_complete(turn)
        assert result.valid

    def test_on_turn_complete_with_text(self):
        h = CheckllmAgentHandler(checks=["contains:hello"])
        turn = SimpleNamespace(text="hello turn text")
        result = h.on_turn_complete(turn)
        assert result.valid

    def test_on_turn_complete_with_result(self):
        h = CheckllmAgentHandler(checks=["contains:hello"])
        turn = SimpleNamespace(result="hello turn result")
        result = h.on_turn_complete(turn)
        assert result.valid

    def test_on_turn_complete_string(self):
        h = CheckllmAgentHandler(checks=["contains:hello"])
        result = h.on_turn_complete("hello plain string")
        assert result.valid

    def test_on_turn_complete_str_fallback(self):
        h = CheckllmAgentHandler(checks=["contains:some_field"])
        turn = SimpleNamespace(some_field=42)
        result = h.on_turn_complete(turn)
        assert result.valid

    def test_on_tool_result(self):
        h = CheckllmAgentHandler(checks=["contains:hello"])
        tool_result = SimpleNamespace(content="hello tool output")
        result = h.on_tool_result("search", tool_result)
        assert result.valid

    def test_on_tool_result_str_fallback(self):
        h = CheckllmAgentHandler(checks=["contains:some_field"])
        tool_result = SimpleNamespace(some_field=42)
        result = h.on_tool_result("compute", tool_result)
        assert result.valid

    def test_wrap_agent(self):
        h = CheckllmAgentHandler(checks=["contains:hello"])

        @h.wrap_agent
        def my_agent(prompt):
            return "hello response"

        output = my_agent("test prompt")
        assert output == "hello response"
        assert len(h.results) == 1

    def test_wrap_agent_preserves_metadata(self):
        h = CheckllmAgentHandler(checks=["contains:a"])

        @h.wrap_agent
        def my_claude_agent(prompt):
            """Claude agent docstring."""
            return "aaa"

        assert my_claude_agent.__name__ == "my_claude_agent"
        assert my_claude_agent.__doc__ == "Claude agent docstring."

    def test_wrap_agent_raises_on_failure(self):
        h = CheckllmAgentHandler(checks=["contains:goodbye"], on_failure="raise")

        @h.wrap_agent
        def bad_agent(prompt):
            return "hello world"

        with pytest.raises(GuardrailError):
            bad_agent("test")


class TestLazyImports:
    """Tests for the integrations __init__.py lazy loading."""

    def test_langchain_handler_lazy(self):
        from checkllm.integrations import LangChainHandler
        from checkllm.integrations.langchain import CheckllmCallbackHandler

        assert LangChainHandler is CheckllmCallbackHandler

    def test_llamaindex_handler_lazy(self):
        from checkllm.integrations import LlamaIndexHandler
        from checkllm.integrations.llamaindex import CheckllmCallbackHandler

        assert LlamaIndexHandler is CheckllmCallbackHandler

    def test_crewai_callback_lazy(self):
        from checkllm.integrations import CrewAICallback

        assert CrewAICallback is CheckllmCrewCallback

    def test_pydantic_ai_validator_lazy(self):
        from checkllm.integrations import PydanticAIValidator

        assert PydanticAIValidator is CheckllmResultValidator

    def test_openai_agents_handler_lazy(self):
        from checkllm.integrations import OpenAIAgentsHandler

        assert OpenAIAgentsHandler is CheckllmRunHandler

    def test_claude_agent_handler_lazy(self):
        from checkllm.integrations import ClaudeAgentHandler

        assert ClaudeAgentHandler is CheckllmAgentHandler

    def test_invalid_attr_raises(self):
        import checkllm.integrations as mod

        with pytest.raises(AttributeError, match="has no attribute"):
            _ = mod.NonExistentHandler
