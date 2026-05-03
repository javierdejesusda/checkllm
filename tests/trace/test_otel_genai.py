"""Tests for OpenTelemetry GenAI span translation."""

from __future__ import annotations

import json
from pathlib import Path

from checkllm.trace.otel_genai import otel_jsonl_to_trace_spans


def test_otel_llm_span_becomes_tracespan(tmp_path: Path):
    jsonl = tmp_path / "spans.jsonl"
    jsonl.write_text(json.dumps({
        "name": "chat",
        "kind": "CLIENT",
        "attributes": {
            "gen_ai.system": "anthropic",
            "gen_ai.request.model": "claude-opus-4-7",
            "gen_ai.usage.input_tokens": 100,
            "gen_ai.usage.output_tokens": 50,
        },
        "start_time_unix_nano": 1000_000_000,
        "end_time_unix_nano": 1500_000_000,
        "status": {"code": "OK"},
    }) + "\n")
    spans = otel_jsonl_to_trace_spans(jsonl)
    assert len(spans) == 1
    assert spans[0].span_type == "llm"
    assert spans[0].attributes["gen_ai.request.model"] == "claude-opus-4-7"
    assert spans[0].end_ms - spans[0].start_ms == 500


def test_otel_tool_span_maps_to_tool_type(tmp_path: Path):
    jsonl = tmp_path / "spans.jsonl"
    jsonl.write_text(json.dumps({
        "name": "tool",
        "kind": "INTERNAL",
        "attributes": {"gen_ai.tool.name": "search"},
        "start_time_unix_nano": 2_000_000_000,
        "end_time_unix_nano": 2_200_000_000,
        "status": {"code": "OK"},
    }) + "\n")
    spans = otel_jsonl_to_trace_spans(jsonl)
    assert len(spans) == 1
    assert spans[0].span_type == "tool"
    assert spans[0].name == "tool"
    assert spans[0].attributes["gen_ai.tool.name"] == "search"


def test_unknown_span_name_maps_to_custom(tmp_path: Path):
    jsonl = tmp_path / "spans.jsonl"
    jsonl.write_text(json.dumps({
        "name": "something_weird",
        "kind": "INTERNAL",
        "attributes": {},
        "start_time_unix_nano": 100_000_000,
        "end_time_unix_nano": 200_000_000,
        "status": {"code": "OK"},
    }) + "\n")
    spans = otel_jsonl_to_trace_spans(jsonl)
    assert len(spans) == 1
    assert spans[0].span_type == "custom"


def test_blank_lines_are_skipped(tmp_path: Path):
    jsonl = tmp_path / "spans.jsonl"
    chat_span = json.dumps({
        "name": "chat",
        "kind": "CLIENT",
        "attributes": {"gen_ai.system": "openai"},
        "start_time_unix_nano": 0,
        "end_time_unix_nano": 1_000_000,
        "status": {"code": "OK"},
    })
    # Intentionally interleave blank and whitespace-only lines.
    jsonl.write_text("\n" + chat_span + "\n   \n\n" + chat_span + "\n\n")
    spans = otel_jsonl_to_trace_spans(jsonl)
    assert len(spans) == 2
    assert all(s.span_type == "llm" for s in spans)


def test_span_without_status_is_ok(tmp_path: Path):
    jsonl = tmp_path / "spans.jsonl"
    jsonl.write_text(json.dumps({
        "name": "chat",
        "kind": "CLIENT",
        "attributes": {"gen_ai.system": "openai"},
        "start_time_unix_nano": 0,
        "end_time_unix_nano": 1_000_000,
    }) + "\n")
    spans = otel_jsonl_to_trace_spans(jsonl)
    assert len(spans) == 1
    assert spans[0].status == "ok"


def test_span_with_unset_status_is_ok(tmp_path: Path):
    jsonl = tmp_path / "spans.jsonl"
    jsonl.write_text(json.dumps({
        "name": "chat",
        "kind": "CLIENT",
        "attributes": {"gen_ai.system": "openai"},
        "start_time_unix_nano": 0,
        "end_time_unix_nano": 1_000_000,
        "status": {"code": "UNSET"},
    }) + "\n")
    spans = otel_jsonl_to_trace_spans(jsonl)
    assert len(spans) == 1
    assert spans[0].status == "ok"


def test_operation_name_attribute_overrides_span_name(tmp_path: Path):
    jsonl = tmp_path / "spans.jsonl"
    jsonl.write_text(json.dumps({
        "name": "anthropic.chat.completions.create",
        "kind": "CLIENT",
        "attributes": {"gen_ai.operation.name": "chat"},
        "start_time_unix_nano": 0,
        "end_time_unix_nano": 1_000_000,
        "status": {"code": "OK"},
    }) + "\n")
    spans = otel_jsonl_to_trace_spans(jsonl)
    assert len(spans) == 1
    assert spans[0].span_type == "llm"


def test_span_name_substring_matches_when_no_operation_name(tmp_path: Path):
    jsonl = tmp_path / "spans.jsonl"
    jsonl.write_text(json.dumps({
        "name": "ChatOpenAI",
        "kind": "CLIENT",
        "attributes": {},
        "start_time_unix_nano": 0,
        "end_time_unix_nano": 1_000_000,
        "status": {"code": "OK"},
    }) + "\n")
    spans = otel_jsonl_to_trace_spans(jsonl)
    assert len(spans) == 1
    assert spans[0].span_type == "llm"


def test_multiple_spans_preserve_order(tmp_path: Path):
    jsonl = tmp_path / "spans.jsonl"
    rows = [
        {
            "name": "chat",
            "kind": "CLIENT",
            "attributes": {"order": 0},
            "start_time_unix_nano": 0,
            "end_time_unix_nano": 1_000_000,
            "status": {"code": "OK"},
        },
        {
            "name": "tool",
            "kind": "INTERNAL",
            "attributes": {"order": 1},
            "start_time_unix_nano": 1_000_000,
            "end_time_unix_nano": 2_000_000,
            "status": {"code": "ERROR"},
        },
        {
            "name": "retrieve",
            "kind": "INTERNAL",
            "attributes": {"order": 2},
            "start_time_unix_nano": 2_000_000,
            "end_time_unix_nano": 3_000_000,
            "status": {"code": "OK"},
        },
    ]
    jsonl.write_text("\n".join(json.dumps(r) for r in rows) + "\n")
    spans = otel_jsonl_to_trace_spans(jsonl)
    assert [s.attributes["order"] for s in spans] == [0, 1, 2]
    assert [s.span_type for s in spans] == ["llm", "tool", "retriever"]
    assert spans[1].status == "error"
    assert spans[0].status == "ok"
