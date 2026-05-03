"""Tests for the LangChain trajectory adapter.

These tests run without LangChain installed: the adapter consumes
duck-typed objects that mirror ``langchain_core.agents.AgentAction``.
"""

from __future__ import annotations

import builtins
from types import SimpleNamespace

import pytest

from checkllm.agents import AgentTestCase, ToolCall
from checkllm.integrations.langchain import (
    to_checkllm_test_case,
    to_checkllm_tool_calls,
)


def _action(tool: str, tool_input):
    """Build a duck-typed AgentAction with ``tool`` and ``tool_input``."""
    return SimpleNamespace(tool=tool, tool_input=tool_input, log="")


def test_to_checkllm_tool_calls_translates_intermediate_steps():
    steps = [
        (_action("search", {"query": "everest height"}), "results: ..."),
        (_action("calculator", "8848 + 0"), "8848"),
    ]
    calls = to_checkllm_tool_calls(steps)
    assert len(calls) == 2
    assert calls[0] == ToolCall(
        name="search",
        parameters={"query": "everest height"},
        result="results: ...",
    )
    # String tool_input is wrapped under the canonical "input" key.
    assert calls[1] == ToolCall(
        name="calculator",
        parameters={"input": "8848 + 0"},
        result="8848",
    )


def test_to_checkllm_test_case_reads_intermediate_steps_and_output():
    run = {
        "input": "How tall is Everest?",
        "intermediate_steps": [
            (_action("search", {"q": "everest"}), "8848 m"),
        ],
        "output": "Mount Everest is 8,848 m tall.",
    }
    case = to_checkllm_test_case(run)
    assert isinstance(case, AgentTestCase)
    assert case.query == "How tall is Everest?"
    assert case.final_output == "Mount Everest is 8,848 m tall."
    assert [tc.name for tc in case.tool_calls] == ["search"]


def test_to_checkllm_test_case_explicit_query_and_final_output_override():
    run = {"intermediate_steps": [], "output": "fallback"}
    case = to_checkllm_test_case(run, query="my-query", final_output="my-output")
    assert case.query == "my-query"
    assert case.final_output == "my-output"


def test_to_checkllm_tool_calls_empty():
    assert to_checkllm_tool_calls([]) == []
    assert to_checkllm_tool_calls(None) == []


def test_to_checkllm_test_case_rejects_non_dict():
    with pytest.raises(TypeError):
        to_checkllm_test_case("not a dict")  # type: ignore[arg-type]


def test_import_error_path_when_langchain_missing(monkeypatch):
    """If LangChain isn't installed and the input lacks ``.tool``, raise ImportError.

    We simulate "LangChain is not installed" by making ``import langchain_core``
    fail, then pass a non-duck-typed entry that the adapter cannot understand.
    """
    real_import = builtins.__import__

    def fake_import(name, *args, **kwargs):
        if name == "langchain_core":
            raise ImportError("simulated: langchain_core not installed")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", fake_import)
    with pytest.raises(ImportError, match="LangChain is not installed"):
        to_checkllm_tool_calls([("not-an-action", "obs")])
