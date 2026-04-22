"""Compliance scanning engine for evaluating LLM applications.

Scans an LLM application's evaluation results against compliance framework
requirements and generates detailed compliance reports with pass/fail status,
risk levels, and remediation guidance.

Usage::

    from checkllm.frameworks import ComplianceFramework, get_framework_definition
    from checkllm.compliance_scanner import ComplianceScanner

    scanner = ComplianceScanner(ComplianceFramework.OWASP_LLM_TOP_10)
    checks_needed = scanner.get_required_checks()
    vuln_types = scanner.get_required_vulnerability_types()
    report = scanner.generate_report(results)
"""

from __future__ import annotations

import datetime
import html
from typing import Any

from pydantic import BaseModel, Field

from checkllm.frameworks import (
    ComplianceFramework,
    FrameworkRequirement,
    get_framework_definition,
)


class RequirementResult(BaseModel):
    """Evaluation result for a single framework requirement."""

    requirement: FrameworkRequirement
    status: str = "not_tested"
    findings: list[str] = Field(default_factory=list)
    score: float = 0.0


class ComplianceReport(BaseModel):
    """A compliance assessment report."""

    framework: str
    framework_version: str
    scan_date: str
    overall_score: float = 0.0
    risk_level: str = "unknown"
    requirements_met: int = 0
    requirements_total: int = 0
    requirements_failed: list[FrameworkRequirement] = Field(default_factory=list)
    requirements_passed: list[FrameworkRequirement] = Field(default_factory=list)
    requirements_not_tested: list[FrameworkRequirement] = Field(default_factory=list)
    requirement_results: list[RequirementResult] = Field(default_factory=list)
    recommendations: list[str] = Field(default_factory=list)

    def summary(self) -> str:
        """Generate a human-readable summary of the compliance report.

        Returns:
            A multi-line string with framework name, scores, risk level,
            and per-requirement status.
        """
        lines = [
            f"Compliance Report: {self.framework} (v{self.framework_version})",
            f"Scan Date: {self.scan_date}",
            f"Overall Score: {self.overall_score:.1%}",
            f"Risk Level: {self.risk_level.upper()}",
            f"Requirements: {self.requirements_met}/{self.requirements_total} passed",
            "",
        ]

        if self.requirements_failed:
            lines.append("FAILED Requirements:")
            for req in self.requirements_failed:
                lines.append(f"  [FAIL] {req.id}: {req.title} ({req.severity})")
            lines.append("")

        if self.requirements_passed:
            lines.append("PASSED Requirements:")
            for req in self.requirements_passed:
                lines.append(f"  [PASS] {req.id}: {req.title}")
            lines.append("")

        if self.requirements_not_tested:
            lines.append("NOT TESTED Requirements:")
            for req in self.requirements_not_tested:
                lines.append(f"  [----] {req.id}: {req.title}")
            lines.append("")

        if self.recommendations:
            lines.append("Recommendations:")
            for i, rec in enumerate(self.recommendations, 1):
                lines.append(f"  {i}. {rec}")

        return "\n".join(lines)

    def to_markdown(self) -> str:
        """Generate a Markdown compliance report.

        Returns:
            A Markdown-formatted string suitable for rendering as a
            compliance document.
        """
        lines = [
            f"# Compliance Report: {self.framework}",
            "",
            f"**Version:** {self.framework_version}",
            f"**Scan Date:** {self.scan_date}",
            f"**Overall Score:** {self.overall_score:.1%}",
            f"**Risk Level:** {self.risk_level.upper()}",
            f"**Requirements Met:** {self.requirements_met}/{self.requirements_total}",
            "",
            "## Summary",
            "",
            "| Status | Count |",
            "|--------|-------|",
            f"| Passed | {len(self.requirements_passed)} |",
            f"| Failed | {len(self.requirements_failed)} |",
            f"| Not Tested | {len(self.requirements_not_tested)} |",
            "",
        ]

        if self.requirements_failed:
            lines.extend(
                [
                    "## Failed Requirements",
                    "",
                ]
            )
            for req in self.requirements_failed:
                lines.extend(
                    [
                        f"### {req.id}: {req.title}",
                        "",
                        f"**Severity:** {req.severity}",
                        f"**Category:** {req.category}",
                        "",
                        req.description,
                        "",
                        f"**Remediation:** {req.remediation}",
                        "",
                    ]
                )

        if self.requirements_passed:
            lines.extend(
                [
                    "## Passed Requirements",
                    "",
                ]
            )
            for req in self.requirements_passed:
                lines.append(f"- **{req.id}:** {req.title}")
            lines.append("")

        if self.requirements_not_tested:
            lines.extend(
                [
                    "## Not Tested Requirements",
                    "",
                ]
            )
            for req in self.requirements_not_tested:
                lines.append(f"- **{req.id}:** {req.title}")
            lines.append("")

        if self.recommendations:
            lines.extend(
                [
                    "## Recommendations",
                    "",
                ]
            )
            for i, rec in enumerate(self.recommendations, 1):
                lines.append(f"{i}. {rec}")
            lines.append("")

        return "\n".join(lines)

    def to_html(self) -> str:
        """Generate an HTML compliance report.

        Returns:
            An HTML string suitable for rendering in a browser or
            embedding in a web application.
        """
        esc = html.escape
        parts = [
            "<!DOCTYPE html>",
            "<html lang='en'>",
            "<head>",
            f"<title>Compliance Report: {esc(self.framework)}</title>",
            "<style>",
            "body { font-family: system-ui, sans-serif; margin: 2em; color: #1a1a1a; }",
            "h1 { color: #1a365d; }",
            "h2 { color: #2d3748; border-bottom: 1px solid #e2e8f0; padding-bottom: 0.3em; }",
            "table { border-collapse: collapse; width: 100%; margin: 1em 0; }",
            "th, td { border: 1px solid #e2e8f0; padding: 0.5em 1em; text-align: left; }",
            "th { background-color: #f7fafc; }",
            ".pass { color: #276749; background-color: #f0fff4; }",
            ".fail { color: #9b2c2c; background-color: #fff5f5; }",
            ".not-tested { color: #744210; background-color: #fffff0; }",
            ".badge { display: inline-block; padding: 0.2em 0.6em; "
            "border-radius: 0.3em; font-size: 0.85em; font-weight: 600; }",
            ".badge-critical { background: #fed7d7; color: #9b2c2c; }",
            ".badge-high { background: #feebc8; color: #9c4221; }",
            ".badge-medium { background: #fefcbf; color: #975a16; }",
            ".badge-low { background: #c6f6d5; color: #276749; }",
            ".score { font-size: 2em; font-weight: 700; }",
            ".meta { color: #718096; }",
            "</style>",
            "</head>",
            "<body>",
            f"<h1>Compliance Report: {esc(self.framework)}</h1>",
            "<div class='meta'>",
            f"<p>Version: {esc(self.framework_version)} | Scan Date: {esc(self.scan_date)}</p>",
            "</div>",
            f"<p class='score'>{self.overall_score:.0%}</p>",
            f"<p>Risk Level: <strong>{esc(self.risk_level.upper())}</strong>"
            f" | Requirements Met: "
            f"<strong>{self.requirements_met}/{self.requirements_total}"
            f"</strong></p>",
            "",
            "<h2>Summary</h2>",
            "<table>",
            "<tr><th>Status</th><th>Count</th></tr>",
            f"<tr class='pass'><td>Passed</td><td>{len(self.requirements_passed)}</td></tr>",
            f"<tr class='fail'><td>Failed</td><td>{len(self.requirements_failed)}</td></tr>",
            f"<tr class='not-tested'><td>Not Tested</td>"
            f"<td>{len(self.requirements_not_tested)}</td></tr>",
            "</table>",
        ]

        if self.requirements_failed:
            parts.append("<h2>Failed Requirements</h2>")
            parts.append("<table>")
            parts.append(
                "<tr><th>ID</th><th>Title</th><th>Severity</th>"
                "<th>Category</th><th>Remediation</th></tr>"
            )
            for req in self.requirements_failed:
                sev_class = f"badge badge-{req.severity}"
                parts.append(
                    f"<tr><td>{esc(req.id)}</td>"
                    f"<td>{esc(req.title)}</td>"
                    f"<td><span class='{sev_class}'>{esc(req.severity)}"
                    f"</span></td>"
                    f"<td>{esc(req.category)}</td>"
                    f"<td>{esc(req.remediation)}</td></tr>"
                )
            parts.append("</table>")

        if self.requirements_passed:
            parts.append("<h2>Passed Requirements</h2>")
            parts.append("<ul>")
            for req in self.requirements_passed:
                parts.append(f"<li><strong>{esc(req.id)}:</strong> {esc(req.title)}</li>")
            parts.append("</ul>")

        if self.requirements_not_tested:
            parts.append("<h2>Not Tested Requirements</h2>")
            parts.append("<ul>")
            for req in self.requirements_not_tested:
                parts.append(f"<li><strong>{esc(req.id)}:</strong> {esc(req.title)}</li>")
            parts.append("</ul>")

        if self.recommendations:
            parts.append("<h2>Recommendations</h2>")
            parts.append("<ol>")
            for rec in self.recommendations:
                parts.append(f"<li>{esc(rec)}</li>")
            parts.append("</ol>")

        parts.extend(["</body>", "</html>"])
        return "\n".join(parts)


class ComplianceScanner:
    """Scans an LLM application against compliance frameworks.

    Args:
        framework: The compliance framework to scan against.
    """

    def __init__(self, framework: ComplianceFramework) -> None:
        self.framework = framework
        self.definition = get_framework_definition(framework)

    def get_required_checks(self) -> list[str]:
        """Return all unique metric and deterministic check names needed.

        Returns:
            A deduplicated list of metric names and deterministic check
            names required by this framework's requirements.
        """
        checks: set[str] = set()
        for req in self.definition.requirements:
            checks.update(req.metrics)
            checks.update(req.deterministic_checks)
        return sorted(checks)

    def get_required_metrics(self) -> list[str]:
        """Return all unique metric names needed for this framework.

        Returns:
            A deduplicated, sorted list of metric names.
        """
        metrics: set[str] = set()
        for req in self.definition.requirements:
            metrics.update(req.metrics)
        return sorted(metrics)

    def get_required_deterministic_checks(self) -> list[str]:
        """Return all unique deterministic check names for this framework.

        Returns:
            A deduplicated, sorted list of deterministic check names.
        """
        checks: set[str] = set()
        for req in self.definition.requirements:
            checks.update(req.deterministic_checks)
        return sorted(checks)

    def get_required_vulnerability_types(self) -> list[str]:
        """Return all unique red team vulnerability types for this framework.

        Returns:
            A deduplicated, sorted list of vulnerability type strings.
        """
        vuln_types: set[str] = set()
        for req in self.definition.requirements:
            vuln_types.update(req.vulnerability_types)
        return sorted(vuln_types)

    def get_coverage_summary(self) -> dict[str, Any]:
        """Return a coverage summary for this framework.

        Returns:
            A dictionary with counts of total requirements,
            vulnerability types, metrics, and checks.
        """
        return {
            "framework": self.definition.name,
            "version": self.definition.version,
            "total_requirements": self.definition.total_requirements,
            "vulnerability_types": len(self.get_required_vulnerability_types()),
            "metrics": len(self.get_required_metrics()),
            "deterministic_checks": len(self.get_required_deterministic_checks()),
        }

    def generate_report(
        self,
        results: dict[str, Any],
    ) -> ComplianceReport:
        """Generate a compliance report from evaluation results.

        The results dictionary maps requirement IDs, vulnerability type
        names, metric names, and check names to their outcomes. Each
        value should be a dict with at least a ``"passed"`` boolean key.
        Additional keys like ``"score"`` (float 0-1) and ``"findings"``
        (list of strings) are used when present.

        Example results format::

            {
                "prompt_injection": {"passed": True, "score": 1.0},
                "jailbreak": {"passed": False, "score": 0.3,
                              "findings": ["bypassed via roleplay"]},
                "role_adherence": {"passed": True, "score": 0.95},
                "toxicity": {"passed": True, "score": 0.98},
            }

        Args:
            results: Dictionary mapping check/vulnerability/metric names
                to outcome dictionaries.

        Returns:
            A ComplianceReport with per-requirement evaluation results.
        """
        scan_date = datetime.datetime.now(datetime.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")

        requirement_results: list[RequirementResult] = []
        passed_reqs: list[FrameworkRequirement] = []
        failed_reqs: list[FrameworkRequirement] = []
        not_tested_reqs: list[FrameworkRequirement] = []
        recommendations: list[str] = []

        for req in self.definition.requirements:
            status, score, findings = self._evaluate_requirement(req, results)

            req_result = RequirementResult(
                requirement=req,
                status=status,
                findings=findings,
                score=score,
            )
            requirement_results.append(req_result)

            if status == "pass":
                passed_reqs.append(req)
            elif status == "fail":
                failed_reqs.append(req)
                recommendations.append(f"[{req.id}] {req.title}: {req.remediation}")
            else:
                not_tested_reqs.append(req)

        total = len(self.definition.requirements)
        met = len(passed_reqs)
        overall_score = met / total if total > 0 else 0.0
        risk_level = _compute_risk_level(overall_score, failed_reqs)

        return ComplianceReport(
            framework=self.definition.name,
            framework_version=self.definition.version,
            scan_date=scan_date,
            overall_score=overall_score,
            risk_level=risk_level,
            requirements_met=met,
            requirements_total=total,
            requirements_failed=failed_reqs,
            requirements_passed=passed_reqs,
            requirements_not_tested=not_tested_reqs,
            requirement_results=requirement_results,
            recommendations=recommendations,
        )

    def _evaluate_requirement(
        self,
        req: FrameworkRequirement,
        results: dict[str, Any],
    ) -> tuple[str, float, list[str]]:
        """Evaluate a single requirement against provided results.

        Args:
            req: The requirement to evaluate.
            results: The full results dictionary.

        Returns:
            A tuple of (status, score, findings) where status is one of
            "pass", "fail", or "not_tested".
        """
        relevant_keys = req.vulnerability_types + req.metrics + req.deterministic_checks

        matched_results: list[dict[str, Any]] = []
        for key in relevant_keys:
            if key in results:
                matched_results.append(results[key])

        if not matched_results:
            if req.id in results:
                r = results[req.id]
                passed = r.get("passed", False)
                score = r.get("score", 1.0 if passed else 0.0)
                findings = r.get("findings", [])
                status = "pass" if passed else "fail"
                return status, score, findings
            return "not_tested", 0.0, []

        all_passed = True
        total_score = 0.0
        findings: list[str] = []

        for r in matched_results:
            if not r.get("passed", False):
                all_passed = False
            total_score += r.get("score", 1.0 if r.get("passed", False) else 0.0)
            findings.extend(r.get("findings", []))

        avg_score = total_score / len(matched_results)
        status = "pass" if all_passed else "fail"
        return status, avg_score, findings


def _compute_risk_level(
    overall_score: float,
    failed_reqs: list[FrameworkRequirement],
) -> str:
    """Compute the overall risk level from score and failed requirements.

    Args:
        overall_score: The fraction of requirements passed (0-1).
        failed_reqs: List of requirements that failed.

    Returns:
        One of "low", "medium", "high", or "critical".
    """
    has_critical_failure = any(r.severity == "critical" for r in failed_reqs)

    if has_critical_failure or overall_score < 0.5:
        return "critical"
    elif overall_score < 0.7:
        return "high"
    elif overall_score < 0.9:
        return "medium"
    else:
        return "low"


def scan_multiple_frameworks(
    frameworks: list[ComplianceFramework],
    results: dict[str, Any],
) -> list[ComplianceReport]:
    """Scan against multiple compliance frameworks at once.

    Args:
        frameworks: List of frameworks to evaluate against.
        results: Dictionary mapping check/vulnerability/metric names
            to outcome dictionaries.

    Returns:
        A list of ComplianceReport objects, one per framework.
    """
    reports = []
    for framework in frameworks:
        scanner = ComplianceScanner(framework)
        reports.append(scanner.generate_report(results))
    return reports
