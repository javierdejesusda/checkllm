"""Tests for the LlamaIndex trajectory adapter.

These tests run without LlamaIndex installed: the adapter consumes
duck-typed objects that mirror ``llama_index.core.tools.ToolOutput``.
"""

from __future__ import annotations

import builtins
from types import SimpleNamespace

import pytest

from checkllm.agents import AgentTestCase, ToolCall
from checkllm.integrations.llama_index import (
    to_checkllm_test_case,
    to_checkllm_tool_calls,
)


def _tool_output(tool_name: str, raw_input, content):
    """Build a duck-typed ToolOutput."""
    return SimpleNamespace(tool_name=tool_name, raw_input=raw_input, content=content)


def _agent_chat_response(sources, response: str):
    return SimpleNamespace(sources=sources, response=response)


def test_to_checkllm_tool_calls_translates_sources():
    sources = [
        _tool_output("search", {"kwargs": {"query": "lisbon"}}, "results..."),
        _tool_output("calculator", '{"a": 1, "b": 2}', "3"),
    ]
    calls = to_checkllm_tool_calls(sources)
    assert calls[0] == ToolCall(name="search", parameters={"query": "lisbon"}, result="results...")
    # JSON-string raw_input is parsed.
    assert calls[1] == ToolCall(name="calculator", parameters={"a": 1, "b": 2}, result="3")


def test_to_checkllm_test_case_reads_sources_and_response():
    response = _agent_chat_response(
        sources=[_tool_output("search", {"kwargs": {"q": "x"}}, "ok")],
        response="The answer is 42.",
    )
    case = to_checkllm_test_case(response, query="What is the answer?")
    assert isinstance(case, AgentTestCase)
    assert case.query == "What is the answer?"
    assert case.final_output == "The answer is 42."
    assert [tc.name for tc in case.tool_calls] == ["search"]


def test_to_checkllm_tool_calls_empty():
    assert to_checkllm_tool_calls([]) == []
    assert to_checkllm_tool_calls(None) == []


def test_to_checkllm_test_case_dict_input():
    raw = {
        "sources": [{"tool_name": "lookup", "raw_input": {"id": 7}, "content": "row"}],
        "response": "done",
    }
    case = to_checkllm_test_case(raw, query="q")
    assert case.tool_calls == [
        ToolCall(name="lookup", parameters={"id": 7}, result="row"),
    ]
    assert case.final_output == "done"


def test_to_checkllm_test_case_rejects_object_without_sources():
    with pytest.raises(TypeError):
        to_checkllm_test_case(object(), query="q")


def test_import_error_path_when_llama_index_missing(monkeypatch):
    """If LlamaIndex isn't installed and inputs lack ``.tool_name``, raise ImportError."""
    real_import = builtins.__import__

    def fake_import(name, *args, **kwargs):
        if name == "llama_index":
            raise ImportError("simulated: llama_index not installed")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", fake_import)
    bad_source = SimpleNamespace()  # no tool_name attribute
    with pytest.raises(ImportError, match="LlamaIndex is not installed"):
        to_checkllm_tool_calls([bad_source])
