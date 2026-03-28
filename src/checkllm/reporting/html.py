from __future__ import annotations

from pathlib import Path

from jinja2 import Environment, FileSystemLoader

from checkllm.models import CheckResult

_TEMPLATE_DIR = Path(__file__).parent.parent / "templates"


def generate_html_report(
    results: dict[str, list[CheckResult]],
    output_path: Path,
) -> None:
    """Generate a self-contained HTML report from test results."""
    all_checks = [c for checks in results.values() for c in checks]
    passed_count = sum(1 for c in all_checks if c.passed)
    failed_count = sum(1 for c in all_checks if not c.passed)
    total_cost = sum(c.cost for c in all_checks)

    env = Environment(loader=FileSystemLoader(str(_TEMPLATE_DIR)), autoescape=True)
    template = env.get_template("report.html.j2")

    html = template.render(
        results=results,
        passed_count=passed_count,
        failed_count=failed_count,
        total_cost=total_cost,
    )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(html)
