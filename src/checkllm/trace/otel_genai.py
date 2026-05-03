"""OpenTelemetry GenAI span translation.

Parses OTel-exported JSONL spans into CheckLLM :class:`~checkllm.trajectory.TraceSpan`
objects so that traces from any OTel-instrumented agent framework can be evaluated
by CheckLLM's trajectory and trace metrics.

The span-name-to-type mapping follows the OpenTelemetry GenAI semantic conventions
(https://opentelemetry.io/docs/specs/semconv/gen-ai/) as of 2026-04.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from checkllm.trajectory import TraceSpan

_GEN_AI_SPAN_KIND: dict[str, str] = {
    "chat": "llm",
    "completion": "llm",
    "embeddings": "llm",
    "tool": "tool",
    "retrieve": "retriever",
}


def _classify_span(name: str, attributes: dict[str, Any]) -> str:
    """Pick a CheckLLM ``span_type`` for an OTel GenAI span.

    Prefers the authoritative ``gen_ai.operation.name`` attribute and falls
    back to substring-matching the raw span name. Real instrumentation
    libraries emit names like ``"anthropic.chat"`` or ``"ChatOpenAI"``, so
    substring matching (rather than exact equality) is required to recognize
    them as ``"llm"`` spans.

    Args:
        name: Raw span name from the OTel export.
        attributes: The span's attribute dict.

    Returns:
        One of ``"llm"``, ``"tool"``, ``"retriever"``, or ``"custom"``.
    """
    operation = str(attributes.get("gen_ai.operation.name") or name).lower()
    for key, span_type in _GEN_AI_SPAN_KIND.items():
        if key in operation:
            return span_type
    return "custom"


def otel_jsonl_to_trace_spans(path: Path | str) -> list[TraceSpan]:
    """Parse an OTel-exported JSONL of spans into CheckLLM TraceSpans.

    The file is read line-by-line rather than loaded whole into memory, so
    multi-GB agent traces can be parsed without blowing the heap.

    Args:
        path: Path to a JSONL file where each line is a single OTel span
            encoded as JSON. Blank lines are skipped.

    Returns:
        A flat list of :class:`TraceSpan` objects, in file order.

    Raises:
        FileNotFoundError: If ``path`` does not exist.
        json.JSONDecodeError: If any non-blank line is not valid JSON.
        KeyError: If a span is missing required fields
            (``name``, ``start_time_unix_nano``, ``end_time_unix_nano``).
        ValueError: If ``start_time_unix_nano`` or ``end_time_unix_nano``
            cannot be parsed as an integer.
        TypeError: If ``start_time_unix_nano`` or ``end_time_unix_nano``
            is of a type that cannot be converted to ``int`` (e.g. ``None``).
    """
    path = Path(path)
    spans: list[TraceSpan] = []
    with path.open(encoding="utf-8") as fh:
        for raw in fh:
            if not raw.strip():
                continue
            row = json.loads(raw)
            attributes = dict(row.get("attributes", {}))
            name = row["name"]
            span_type = _classify_span(name, attributes)
            start_ms = int(row["start_time_unix_nano"]) // 1_000_000
            end_ms = int(row["end_time_unix_nano"]) // 1_000_000
            status_code = row.get("status", {}).get("code", "UNSET")
            status = "error" if status_code.upper() == "ERROR" else "ok"
            spans.append(
                TraceSpan(
                    name=name,
                    span_type=span_type,
                    start_ms=start_ms,
                    end_ms=end_ms,
                    status=status,
                    attributes=attributes,
                    children=[],
                )
            )
    return spans
