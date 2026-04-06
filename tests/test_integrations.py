"""Tests for framework integrations (LangChain, LlamaIndex).

These tests don't require langchain or llama-index installed —
they test the checkllm wrapper logic directly.
"""
import pytest
from unittest.mock import MagicMock

from checkllm.integrations.langchain import (
    CheckllmCallbackHandler as LangChainHandler,
)
from checkllm.integrations.llamaindex import (
    CheckllmCallbackHandler as LlamaIndexHandler,
)


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
