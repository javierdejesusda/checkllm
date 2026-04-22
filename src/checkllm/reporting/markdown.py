"""Markdown report generation — ideal for PR comments and GitHub Actions."""

from __future__ import annotations

from pathlib import Path

from checkllm.models import CheckResult


def generate_markdown_report(
    results: dict[str, list[CheckResult]],
    output_path: Path | None = None,
) -> str:
    """Generate a Markdown report from test results.

    If ``output_path`` is given, writes to file and returns the text.
    Otherwise just returns the Markdown string.
    """
    all_checks = [c for checks in results.values() for c in checks]
    passed = sum(1 for c in all_checks if c.passed)
    failed = sum(1 for c in all_checks if not c.passed)
    total_cost = sum(c.cost for c in all_checks)
    total = passed + failed
    rate = (passed / total * 100) if total > 0 else 0

    lines: list[str] = []
    lines.append("# checkllm Report")
    lines.append("")

    # Summary
    status = "PASS" if failed == 0 else "FAIL"
    lines.append(
        f"**Status:** {status} | **{passed}/{total}** checks passed ({rate:.0f}%) | **${total_cost:.4f}** total cost"
    )
    lines.append("")

    # Per-test tables
    for test_name, checks in results.items():
        test_failed = sum(1 for c in checks if not c.passed)
        badge = "PASS" if test_failed == 0 else f"{test_failed} FAILED"
        lines.append(f"## {test_name} ({badge})")
        lines.append("")
        lines.append("| Status | Metric | Score | Reasoning | Cost |")
        lines.append("|--------|--------|------:|-----------|-----:|")
        for c in checks:
            status_icon = "PASS" if c.passed else "FAIL"
            reasoning = c.reasoning[:80].replace("|", "\\|")
            lines.append(
                f"| {status_icon} | {c.metric_name} | {c.score:.2f} | {reasoning} | ${c.cost:.4f} |"
            )
        lines.append("")

    md = "\n".join(lines)

    if output_path:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(md)

    return md
