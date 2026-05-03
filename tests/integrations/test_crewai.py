"""Tests for the CrewAI trajectory adapter.

These tests run without CrewAI installed: the adapter consumes plain
dicts and duck-typed objects that mirror ``crewai.tasks.TaskOutput``.
"""

from __future__ import annotations

import builtins
from types import SimpleNamespace

import pytest

from checkllm.agents import AgentTestCase, ToolCall
from checkllm.integrations.crewai import (
    to_checkllm_test_case,
    to_checkllm_tool_calls,
)


def test_to_checkllm_tool_calls_translates_dict_entries():
    tool_usage = [
        {"tool": "search_web", "args": {"q": "lisbon"}, "result": "list"},
        {"tool_name": "weather", "tool_args": {"city": "Lisbon"}, "output": "sunny"},
    ]
    calls = to_checkllm_tool_calls(tool_usage)
    assert calls[0] == ToolCall(name="search_web", parameters={"q": "lisbon"}, result="list")
    assert calls[1] == ToolCall(name="weather", parameters={"city": "Lisbon"}, result="sunny")


def test_to_checkllm_test_case_reads_tasks_output_concatenated():
    crew = SimpleNamespace(
        raw="2-day plan: ...",
        tasks_output=[
            SimpleNamespace(
                tools_used=[
                    {"tool": "search_web", "args": {"q": "lisbon"}, "result": "list"},
                ]
            ),
            SimpleNamespace(
                tools_used=[
                    {"tool": "weather", "args": {"city": "Lisbon"}, "result": "sunny"},
                ]
            ),
        ],
    )
    case = to_checkllm_test_case(crew, query="Plan a 2-day trip to Lisbon.")
    assert isinstance(case, AgentTestCase)
    assert case.query == "Plan a 2-day trip to Lisbon."
    assert case.final_output == "2-day plan: ..."
    assert [tc.name for tc in case.tool_calls] == ["search_web", "weather"]


def test_to_checkllm_test_case_single_task_output_with_tools_used():
    task = SimpleNamespace(
        raw="hello",
        tools_used=[{"tool": "echo", "args": "hi", "result": "hi"}],
    )
    case = to_checkllm_test_case(task, query="say hi")
    assert case.tool_calls == [
        ToolCall(name="echo", parameters={"input": "hi"}, result="hi"),
    ]
    assert case.final_output == "hello"


def test_to_checkllm_tool_calls_empty():
    assert to_checkllm_tool_calls([]) == []
    assert to_checkllm_tool_calls(None) == []


def test_import_error_path_when_crewai_missing(monkeypatch):
    """If CrewAI isn't installed and entries lack a tool-name field, raise ImportError."""
    real_import = builtins.__import__

    def fake_import(name, *args, **kwargs):
        if name == "crewai":
            raise ImportError("simulated: crewai not installed")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", fake_import)
    with pytest.raises(ImportError, match="CrewAI is not installed"):
        to_checkllm_tool_calls([{"args": {}, "result": "x"}])  # no tool name
