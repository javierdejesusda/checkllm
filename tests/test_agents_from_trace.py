"""Tests for :meth:`AgentTestCase.from_trace_jsonl`.

These tests exercise the OTel-JSONL ingestion classmethod end-to-end using
real file I/O through ``tmp_path``; the underlying span parser is intentionally
not mocked so regressions in the whole ingest pipeline are caught here.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from checkllm.agents import AgentTestCase


def _write_jsonl(path: Path, rows: list[dict]) -> None:
    """Write ``rows`` to ``path`` as newline-delimited JSON."""
    path.write_text(
        "\n".join(json.dumps(r) for r in rows) + "\n",
        encoding="utf-8",
    )


def test_tool_span_becomes_agent_step(tmp_path: Path):
    jsonl = tmp_path / "trace.jsonl"
    lines = [
        # LLM span - no step emitted
        json.dumps({
            "name": "chat",
            "attributes": {
                "gen_ai.operation.name": "chat",
                "gen_ai.request.model": "claude-opus-4-7",
            },
            "start_time_unix_nano": 1_000_000_000,
            "end_time_unix_nano": 1_200_000_000,
            "status": {"code": "OK"},
        }),
        # Tool span - becomes one AgentStep with a ToolCall
        json.dumps({
            "name": "tool",
            "attributes": {
                "gen_ai.operation.name": "tool",
                "tool.name": "calculator",
                "tool.arguments": json.dumps({"expr": "6*7"}),
                "tool.result": "42",
            },
            "start_time_unix_nano": 1_300_000_000,
            "end_time_unix_nano": 1_400_000_000,
            "status": {"code": "OK"},
        }),
        # Second LLM span - no step emitted
        json.dumps({
            "name": "chat",
            "attributes": {"gen_ai.operation.name": "chat"},
            "start_time_unix_nano": 1_500_000_000,
            "end_time_unix_nano": 1_600_000_000,
            "status": {"code": "OK"},
        }),
    ]
    jsonl.write_text("\n".join(lines) + "\n", encoding="utf-8")

    test_case = AgentTestCase.from_trace_jsonl(
        jsonl, query="what is 6*7?", final_output="42"
    )
    assert test_case.query == "what is 6*7?"
    assert test_case.final_output == "42"
    assert len(test_case.steps) == 1
    step = test_case.steps[0]
    assert step.tool_call is not None
    assert step.tool_call.name == "calculator"
    assert step.tool_call.parameters == {"expr": "6*7"}
    assert step.tool_call.result == "42"


def test_multiple_tool_spans_become_multiple_steps(tmp_path: Path):
    """Three tool spans yield three AgentSteps in file order."""
    jsonl = tmp_path / "trace.jsonl"
    rows = [
        {
            "name": "tool",
            "attributes": {
                "gen_ai.operation.name": "tool",
                "tool.name": "search",
                "tool.arguments": json.dumps({"q": "weather"}),
                "tool.result": "sunny",
            },
            "start_time_unix_nano": 1_000_000_000,
            "end_time_unix_nano": 1_100_000_000,
            "status": {"code": "OK"},
        },
        {
            "name": "tool",
            "attributes": {
                "gen_ai.operation.name": "tool",
                "tool.name": "calculator",
                "tool.arguments": json.dumps({"expr": "2+2"}),
                "tool.result": "4",
            },
            "start_time_unix_nano": 1_200_000_000,
            "end_time_unix_nano": 1_300_000_000,
            "status": {"code": "OK"},
        },
        {
            "name": "tool",
            "attributes": {
                "gen_ai.operation.name": "tool",
                "tool.name": "translate",
                "tool.arguments": json.dumps({"text": "hola"}),
                "tool.result": "hello",
            },
            "start_time_unix_nano": 1_400_000_000,
            "end_time_unix_nano": 1_500_000_000,
            "status": {"code": "OK"},
        },
    ]
    _write_jsonl(jsonl, rows)

    test_case = AgentTestCase.from_trace_jsonl(jsonl, query="multi-tool")
    assert len(test_case.steps) == 3
    names = [s.tool_call.name for s in test_case.steps]
    assert names == ["search", "calculator", "translate"]
    results = [s.tool_call.result for s in test_case.steps]
    assert results == ["sunny", "4", "hello"]


def test_tool_arguments_as_dict_also_supported(tmp_path: Path):
    """Instrumentors may emit ``tool.arguments`` as a dict rather than a JSON string."""
    jsonl = tmp_path / "trace.jsonl"
    rows = [
        {
            "name": "tool",
            "attributes": {
                "gen_ai.operation.name": "tool",
                "tool.name": "calculator",
                # already-parsed dict, not a JSON string
                "tool.arguments": {"expr": "6*7", "precision": 2},
                "tool.result": "42",
            },
            "start_time_unix_nano": 1_000_000_000,
            "end_time_unix_nano": 1_100_000_000,
            "status": {"code": "OK"},
        },
    ]
    _write_jsonl(jsonl, rows)

    test_case = AgentTestCase.from_trace_jsonl(jsonl, query="dict args")
    assert len(test_case.steps) == 1
    assert test_case.steps[0].tool_call.parameters == {"expr": "6*7", "precision": 2}


def test_missing_tool_arguments_yields_empty_params(tmp_path: Path):
    """Absent ``tool.arguments`` must produce an empty params dict without raising."""
    jsonl = tmp_path / "trace.jsonl"
    rows = [
        {
            "name": "tool",
            "attributes": {
                "gen_ai.operation.name": "tool",
                "tool.name": "ping",
                "tool.result": "pong",
            },
            "start_time_unix_nano": 1_000_000_000,
            "end_time_unix_nano": 1_100_000_000,
            "status": {"code": "OK"},
        },
    ]
    _write_jsonl(jsonl, rows)

    test_case = AgentTestCase.from_trace_jsonl(jsonl, query="ping")
    assert len(test_case.steps) == 1
    assert test_case.steps[0].tool_call.parameters == {}
    assert test_case.steps[0].tool_call.name == "ping"


def test_empty_jsonl_yields_empty_steps(tmp_path: Path):
    """A trace with only LLM spans produces zero agent steps."""
    jsonl = tmp_path / "trace.jsonl"
    rows = [
        {
            "name": "chat",
            "attributes": {"gen_ai.operation.name": "chat"},
            "start_time_unix_nano": 1_000_000_000,
            "end_time_unix_nano": 1_100_000_000,
            "status": {"code": "OK"},
        },
        {
            "name": "chat",
            "attributes": {"gen_ai.operation.name": "chat"},
            "start_time_unix_nano": 1_200_000_000,
            "end_time_unix_nano": 1_300_000_000,
            "status": {"code": "OK"},
        },
    ]
    _write_jsonl(jsonl, rows)

    test_case = AgentTestCase.from_trace_jsonl(jsonl, query="no tools here")
    assert test_case.query == "no tools here"
    assert test_case.final_output is None
    assert len(test_case.steps) == 0


def test_tool_result_is_optional(tmp_path: Path):
    """A tool span missing ``tool.result`` yields ``tool_call.result == None``."""
    jsonl = tmp_path / "trace.jsonl"
    rows = [
        {
            "name": "tool",
            "attributes": {
                "gen_ai.operation.name": "tool",
                "tool.name": "fire_and_forget",
                "tool.arguments": json.dumps({"x": 1}),
                # no tool.result
            },
            "start_time_unix_nano": 1_000_000_000,
            "end_time_unix_nano": 1_100_000_000,
            "status": {"code": "OK"},
        },
    ]
    _write_jsonl(jsonl, rows)

    test_case = AgentTestCase.from_trace_jsonl(jsonl, query="no-result")
    assert len(test_case.steps) == 1
    assert test_case.steps[0].tool_call.name == "fire_and_forget"
    assert test_case.steps[0].tool_call.result is None
    assert test_case.steps[0].tool_call.parameters == {"x": 1}


def test_tool_arguments_as_non_dict_json_string_yields_empty_params(tmp_path: Path):
    """A JSON array in ``tool.arguments`` must degrade to ``{}`` without raising."""
    jsonl = tmp_path / "trace.jsonl"
    rows = [
        {
            "name": "tool",
            "attributes": {
                "gen_ai.operation.name": "tool",
                "tool.name": "positional_args",
                # JSON array - some instrumentors emit positional args this way
                "tool.arguments": json.dumps([1, 2, 3]),
                "tool.result": "ok",
            },
            "start_time_unix_nano": 1_000_000_000,
            "end_time_unix_nano": 1_100_000_000,
            "status": {"code": "OK"},
        },
    ]
    _write_jsonl(jsonl, rows)

    test_case = AgentTestCase.from_trace_jsonl(jsonl, query="positional")
    assert len(test_case.steps) == 1
    assert test_case.steps[0].tool_call.name == "positional_args"
    assert test_case.steps[0].tool_call.parameters == {}
    assert test_case.steps[0].tool_call.result == "ok"


def test_tool_arguments_as_bare_number_yields_empty_params(tmp_path: Path):
    """A bare-number JSON literal in ``tool.arguments`` must degrade to ``{}``."""
    jsonl = tmp_path / "trace.jsonl"
    rows = [
        {
            "name": "tool",
            "attributes": {
                "gen_ai.operation.name": "tool",
                "tool.name": "bare_number",
                "tool.arguments": "42",
                "tool.result": "ok",
            },
            "start_time_unix_nano": 1_000_000_000,
            "end_time_unix_nano": 1_100_000_000,
            "status": {"code": "OK"},
        },
    ]
    _write_jsonl(jsonl, rows)

    test_case = AgentTestCase.from_trace_jsonl(jsonl, query="bare number")
    assert len(test_case.steps) == 1
    assert test_case.steps[0].tool_call.name == "bare_number"
    assert test_case.steps[0].tool_call.parameters == {}


def test_missing_file_raises_file_not_found(tmp_path: Path):
    """A non-existent path must surface ``FileNotFoundError`` from the parser."""
    missing = tmp_path / "does_not_exist.jsonl"
    with pytest.raises(FileNotFoundError):
        AgentTestCase.from_trace_jsonl(missing, query="missing")
