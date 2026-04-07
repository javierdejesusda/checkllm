"""GitHub integration — generate PR comments and post them via the GitHub API."""
from __future__ import annotations

import os

from checkllm.models import CheckResult
from checkllm.reporting.comparison import ComparisonReport, _paired_rows, _summary_stats


# ---------------------------------------------------------------------------
# PR comment generation
# ---------------------------------------------------------------------------

_MARKER = "<!-- checkllm-report -->"


def generate_pr_comment(
    results: dict[str, list[CheckResult]],
    comparison: ComparisonReport | None = None,
) -> str:
    """Generate a Markdown string formatted for a GitHub PR comment.

    Includes collapsible detail sections, status emoji, and an optional
    A/B comparison table when *comparison* is provided.
    """
    all_checks = [c for checks in results.values() for c in checks]
    passed = sum(1 for c in all_checks if c.passed)
    failed = sum(1 for c in all_checks if not c.passed)
    total = passed + failed
    total_cost = sum(c.cost for c in all_checks)
    rate = (passed / total * 100) if total > 0 else 0
    status_emoji = "✅" if failed == 0 else "❌"

    lines: list[str] = [_MARKER]
    lines.append(f"## {status_emoji} checkllm Report")
    lines.append("")
    lines.append(
        f"**{passed}/{total}** checks passed ({rate:.0f}%) | "
        f"**${total_cost:.4f}** total cost"
    )
    lines.append("")

    # Per-test collapsible sections
    for test_name, checks in results.items():
        test_failed = sum(1 for c in checks if not c.passed)
        badge = "✅ PASS" if test_failed == 0 else f"❌ {test_failed} FAILED"
        lines.append("<details>")
        lines.append(f"<summary><strong>{test_name}</strong> — {badge}</summary>")
        lines.append("")
        lines.append("| Status | Metric | Score | Reasoning | Cost |")
        lines.append("|--------|--------|------:|-----------|-----:|")
        for c in checks:
            icon = "✅" if c.passed else "❌"
            reasoning = c.reasoning[:80].replace("|", "\\|")
            lines.append(
                f"| {icon} | {c.metric_name} | {c.score:.2f} | {reasoning} | ${c.cost:.4f} |"
            )
        lines.append("")
        lines.append("</details>")
        lines.append("")

    # Comparison section
    if comparison is not None:
        stats_a = _summary_stats(comparison.results_a)
        stats_b = _summary_stats(comparison.results_b)

        if stats_a["avg_score"] > stats_b["avg_score"]:
            overall = comparison.label_a
        elif stats_b["avg_score"] > stats_a["avg_score"]:
            overall = comparison.label_b
        else:
            overall = "Tie"

        lines.append("<details>")
        lines.append(
            f"<summary><strong>Comparison: {comparison.label_a} vs "
            f"{comparison.label_b}</strong> — {overall}"
            + (" is better" if overall != "Tie" else "")
            + "</summary>"
        )
        lines.append("")
        lines.append(f"| Metric | {comparison.label_a} | {comparison.label_b} |")
        lines.append("|--------|------:|------:|")
        lines.append(
            f"| Pass rate | {stats_a['pass_rate']:.0%} | {stats_b['pass_rate']:.0%} |"
        )
        lines.append(
            f"| Avg score | {stats_a['avg_score']:.3f} | {stats_b['avg_score']:.3f} |"
        )
        lines.append(
            f"| Total cost | ${stats_a['total_cost']:.4f} | ${stats_b['total_cost']:.4f} |"
        )
        lines.append("")

        rows = _paired_rows(comparison)
        lines.append(
            f"| Test | Metric | {comparison.label_a} | {comparison.label_b} | Delta |"
        )
        lines.append("|------|--------|------:|------:|------:|")
        for r in rows:
            sa = f"{r['score_a']:.2f}" if r["score_a"] is not None else "-"
            sb = f"{r['score_b']:.2f}" if r["score_b"] is not None else "-"
            delta = f"{r['delta']:+.3f}" if r["delta"] is not None else "-"
            lines.append(
                f"| {r['test_name']} | {r['metric_name']} | {sa} | {sb} | {delta} |"
            )
        lines.append("")
        lines.append("</details>")
        lines.append("")

    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Post to GitHub API
# ---------------------------------------------------------------------------

def post_pr_comment(
    comment: str,
    repo: str,
    pr_number: int,
    token: str | None = None,
) -> None:
    """Post (or update) a PR comment on GitHub.

    Uses *token* or the ``GITHUB_TOKEN`` environment variable.  If a comment
    containing the checkllm marker already exists it is updated in place.

    Requires the ``httpx`` package (``pip install httpx``).
    """
    try:
        import httpx
    except ImportError as exc:
        raise ImportError(
            "The 'httpx' package is required for posting PR comments. "
            "Install it with: pip install httpx"
        ) from exc

    resolved_token = token or os.environ.get("GITHUB_TOKEN")
    if not resolved_token:
        raise ValueError(
            "A GitHub token is required. Pass it as 'token' or set GITHUB_TOKEN."
        )

    headers = {
        "Authorization": f"Bearer {resolved_token}",
        "Accept": "application/vnd.github+json",
        "X-GitHub-Api-Version": "2022-11-28",
    }
    base_url = f"https://api.github.com/repos/{repo}/issues/{pr_number}/comments"

    # Check for existing comment to update
    resp = httpx.get(base_url, headers=headers)
    resp.raise_for_status()
    existing_comments = resp.json()

    existing_id: int | None = None
    for c in existing_comments:
        if _MARKER in (c.get("body") or ""):
            existing_id = c["id"]
            break

    if existing_id is not None:
        update_url = f"https://api.github.com/repos/{repo}/issues/comments/{existing_id}"
        resp = httpx.patch(update_url, headers=headers, json={"body": comment})
        resp.raise_for_status()
    else:
        resp = httpx.post(base_url, headers=headers, json={"body": comment})
        resp.raise_for_status()
