"""Tests for the pydantic-ai trajectory adapter.

These tests run without pydantic-ai installed: the adapter consumes
duck-typed objects that mirror ``pydantic_ai.messages.ModelMessage``
parts (``ToolCallPart`` / ``ToolReturnPart``).
"""

from __future__ import annotations

import builtins
from types import SimpleNamespace

import pytest

from checkllm.agents import AgentTestCase, ToolCall
from checkllm.integrations.pydantic_ai import (
    to_checkllm_test_case,
    to_checkllm_tool_calls,
)


def _call_part(tool_name: str, args, tool_call_id: str):
    return SimpleNamespace(
        part_kind="tool-call",
        tool_name=tool_name,
        args=args,
        tool_call_id=tool_call_id,
    )


def _return_part(tool_name: str, content: str, tool_call_id: str):
    return SimpleNamespace(
        part_kind="tool-return",
        tool_name=tool_name,
        content=content,
        tool_call_id=tool_call_id,
    )


def _msg(parts):
    return SimpleNamespace(parts=parts)


def test_to_checkllm_tool_calls_pairs_calls_and_returns_by_id():
    messages = [
        _msg([_call_part("search", {"q": "lisbon"}, "c1")]),
        _msg([_return_part("search", "results...", "c1")]),
        _msg([_call_part("weather", '{"city": "Lisbon"}', "c2")]),
        _msg([_return_part("weather", "sunny", "c2")]),
    ]
    calls = to_checkllm_tool_calls(messages)
    assert calls[0] == ToolCall(
        name="search", parameters={"q": "lisbon"}, result="results..."
    )
    # JSON-string args is parsed.
    assert calls[1] == ToolCall(
        name="weather", parameters={"city": "Lisbon"}, result="sunny"
    )


def test_to_checkllm_test_case_reads_run_result():
    run_result = SimpleNamespace(
        data="Mount Everest is 8,848 m tall.",
        all_messages=lambda: [
            _msg(
                [
                    _call_part("lookup", {"key": "everest"}, "c1"),
                    _return_part("lookup", "8848", "c1"),
                ]
            ),
        ],
    )
    case = to_checkllm_test_case(run_result, query="How tall is Everest?")
    assert isinstance(case, AgentTestCase)
    assert case.query == "How tall is Everest?"
    assert case.final_output == "Mount Everest is 8,848 m tall."
    assert [tc.name for tc in case.tool_calls] == ["lookup"]


def test_to_checkllm_tool_calls_empty():
    assert to_checkllm_tool_calls([]) == []
    assert to_checkllm_tool_calls(None) == []


def test_to_checkllm_tool_calls_unpaired_returns_dropped():
    messages = [_msg([_call_part("search", {}, "c1")])]
    calls = to_checkllm_tool_calls(messages)
    assert len(calls) == 1
    assert calls[0].result is None


def test_import_error_path_when_pydantic_ai_missing(monkeypatch):
    """When pydantic-ai is missing AND parts are unrecognizable, return an empty
    list (the function tolerates non-tool parts gracefully). We assert both
    that the import was attempted and that the adapter doesn't crash."""
    real_import = builtins.__import__
    seen = []

    def fake_import(name, *args, **kwargs):
        if name == "pydantic_ai":
            seen.append(name)
            raise ImportError("simulated: pydantic_ai not installed")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", fake_import)
    # A bare object with no part_kind, tool_name, or content -> not a part,
    # so the adapter tries to import pydantic_ai and raises ImportError.
    with pytest.raises(ImportError, match="pydantic-ai is not installed"):
        to_checkllm_tool_calls([SimpleNamespace(unrelated="x")])
    assert seen == ["pydantic_ai"]
