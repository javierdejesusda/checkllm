"""JSONL export — one JSON object per line, ideal for data pipelines."""
from __future__ import annotations

import json
from pathlib import Path

from checkllm.models import CheckResult


def export_jsonl(
    results: dict[str, list[CheckResult]],
    output_path: Path | None = None,
) -> str:
    """Export results as JSONL (one JSON object per line).

    Each line contains: test_name, metric_name, passed, score, reasoning, cost, latency_ms.

    If ``output_path`` is given, writes to file and returns the text.
    Otherwise just returns the JSONL string.
    """
    lines: list[str] = []
    for test_name, checks in results.items():
        for c in checks:
            record = {
                "test": test_name,
                "metric": c.metric_name,
                "passed": c.passed,
                "score": c.score,
                "reasoning": c.reasoning,
                "cost": c.cost,
                "latency_ms": c.latency_ms,
            }
            lines.append(json.dumps(record, ensure_ascii=False))

    text = "\n".join(lines) + "\n" if lines else ""

    if output_path:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(text)

    return text
