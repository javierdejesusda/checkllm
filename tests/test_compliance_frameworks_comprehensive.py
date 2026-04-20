"""Comprehensive tests for checkllm.compliance_frameworks — model classes and summaries."""

from __future__ import annotations

import pytest

from checkllm.compliance_frameworks import (
    ComplianceFramework,
    ComplianceReport,
    FrameworkMapping,
    FrameworkRequirement,
    MultiFrameworkReport,
    RequirementResult,
)


class TestComplianceFramework:
    def test_all_values_are_strings(self):
        for framework in ComplianceFramework:
            assert isinstance(framework.value, str)
            assert len(framework.value) > 0

    def test_specific_values(self):
        assert ComplianceFramework.HIPAA.value == "hipaa"
        assert ComplianceFramework.GDPR.value == "gdpr"
        assert ComplianceFramework.PCI_DSS.value == "pci_dss"
        assert ComplianceFramework.SOC2.value == "soc2"
        assert ComplianceFramework.ISO27001.value == "iso27001"
        assert ComplianceFramework.NIST_AI_RMF.value == "nist_ai_rmf"
        assert ComplianceFramework.OWASP_LLM_TOP10.value == "owasp_llm_top10"
        assert ComplianceFramework.EU_AI_ACT.value == "eu_ai_act"
        assert ComplianceFramework.COPPA.value == "coppa"
        assert ComplianceFramework.FERPA.value == "ferpa"
        assert ComplianceFramework.DOD_AI_ETHICS.value == "dod_ai_ethics"
        assert ComplianceFramework.CCPA.value == "ccpa"
        assert ComplianceFramework.NIST_CSF.value == "nist_csf"

    def test_count_of_frameworks(self):
        # Ensure we have at least 17 frameworks
        assert len(list(ComplianceFramework)) >= 17


class TestFrameworkRequirement:
    def test_create_basic(self):
        req = FrameworkRequirement(
            id="REQ-001",
            name="No PII in output",
            description="Ensure no personally identifiable information is leaked.",
            severity="critical",
            test_categories=["pii_detection", "data_leakage"],
            remediation="Implement output filtering.",
        )
        assert req.id == "REQ-001"
        assert req.name == "No PII in output"
        assert req.severity == "critical"
        assert "pii_detection" in req.test_categories
        assert req.references == []

    def test_with_references(self):
        req = FrameworkRequirement(
            id="REQ-002",
            name="Test",
            description="A test requirement",
            severity="high",
            test_categories=["test"],
            remediation="Do something.",
            references=["https://example.com/ref1"],
        )
        assert len(req.references) == 1
        assert req.references[0] == "https://example.com/ref1"


class TestFrameworkMapping:
    def test_create_mapping(self):
        req = FrameworkRequirement(
            id="R1",
            name="Req 1",
            description="A requirement",
            severity="high",
            test_categories=["cat1"],
            remediation="Fix it.",
        )
        mapping = FrameworkMapping(
            framework=ComplianceFramework.GDPR,
            version="2.0",
            description="GDPR compliance mapping",
            url="https://gdpr.example.com",
            requirements=[req],
        )
        assert mapping.framework == ComplianceFramework.GDPR
        assert mapping.version == "2.0"
        assert len(mapping.requirements) == 1


class TestRequirementResult:
    def test_create_passed(self):
        result = RequirementResult(
            requirement_id="LLM01",
            requirement_name="Prompt Injection",
            status="passed",
            severity="critical",
        )
        assert result.status == "passed"
        assert result.findings == []
        assert result.evidence == []

    def test_create_failed_with_findings(self):
        result = RequirementResult(
            requirement_id="LLM01",
            requirement_name="Prompt Injection",
            status="failed",
            severity="critical",
            findings=["Prompt injection detected in test case 1"],
            evidence=["Response: BYPASSED"],
        )
        assert result.status == "failed"
        assert len(result.findings) == 1
        assert len(result.evidence) == 1

    def test_create_skipped(self):
        result = RequirementResult(
            requirement_id="LLM02",
            requirement_name="Insecure Output",
            status="skipped",
            severity="high",
        )
        assert result.status == "skipped"


class TestComplianceReport:
    def _make_report(
        self,
        framework: ComplianceFramework = ComplianceFramework.OWASP_LLM_TOP10,
        passed: int = 8,
        failed: int = 2,
        skipped: int = 0,
        results: list[RequirementResult] | None = None,
    ) -> ComplianceReport:
        if results is None:
            results = [
                RequirementResult(
                    requirement_id=f"REQ-{i}",
                    requirement_name=f"Requirement {i}",
                    status="passed" if i < passed else "failed",
                    severity="critical" if i < 3 else "high",
                )
                for i in range(passed + failed)
            ]
        total = passed + failed + skipped
        overall = passed / total if total > 0 else 0.0
        return ComplianceReport(
            framework=framework,
            version="2025",
            timestamp="2025-01-01T00:00:00Z",
            total_requirements=total,
            passed=passed,
            failed=failed,
            skipped=skipped,
            results=results,
            overall_score=overall,
        )

    def test_summary_contains_framework(self):
        report = self._make_report()
        summary = report.summary()
        assert "owasp_llm_top10" in summary

    def test_summary_contains_version(self):
        report = self._make_report()
        summary = report.summary()
        assert "2025" in summary

    def test_summary_contains_score(self):
        report = self._make_report(passed=8, failed=2)
        summary = report.summary()
        assert "80.0%" in summary or "80%" in summary

    def test_summary_contains_pass_fail(self):
        report = self._make_report(passed=7, failed=3)
        summary = report.summary()
        assert "7/10 passed" in summary
        assert "3 failed" in summary

    def test_summary_pass_icon(self):
        report = self._make_report(
            passed=1,
            failed=0,
            results=[
                RequirementResult(
                    requirement_id="R1",
                    requirement_name="Test",
                    status="passed",
                    severity="high",
                )
            ],
        )
        summary = report.summary()
        assert "[PASS]" in summary

    def test_summary_fail_icon(self):
        report = self._make_report(
            passed=0,
            failed=1,
            results=[
                RequirementResult(
                    requirement_id="R1",
                    requirement_name="Test",
                    status="failed",
                    severity="high",
                )
            ],
        )
        summary = report.summary()
        assert "[FAIL]" in summary

    def test_summary_skip_icon(self):
        report = self._make_report(
            passed=0,
            failed=0,
            skipped=1,
            results=[
                RequirementResult(
                    requirement_id="R1",
                    requirement_name="Test",
                    status="skipped",
                    severity="low",
                )
            ],
        )
        summary = report.summary()
        assert "[SKIP]" in summary

    def test_summary_includes_findings(self):
        results = [
            RequirementResult(
                requirement_id="R1",
                requirement_name="Injection Test",
                status="failed",
                severity="critical",
                findings=["Prompt injection detected"],
            )
        ]
        report = self._make_report(passed=0, failed=1, results=results)
        summary = report.summary()
        assert "Prompt injection detected" in summary

    def test_summary_unknown_status_icon(self):
        results = [
            RequirementResult(
                requirement_id="R1",
                requirement_name="Test",
                status="unknown",
                severity="high",
            )
        ]
        report = self._make_report(passed=0, failed=0, skipped=1, results=results)
        summary = report.summary()
        assert "[????]" in summary

    def test_report_model_fields(self):
        report = self._make_report()
        assert report.framework == ComplianceFramework.OWASP_LLM_TOP10
        assert report.version == "2025"
        assert report.timestamp == "2025-01-01T00:00:00Z"
        assert report.total_requirements == 10
        assert report.passed == 8
        assert report.failed == 2
        assert report.skipped == 0


class TestMultiFrameworkReport:
    def _make_single_report(self, framework: ComplianceFramework, score: float) -> ComplianceReport:
        return ComplianceReport(
            framework=framework,
            version="1.0",
            timestamp="2025-01-01T00:00:00Z",
            total_requirements=10,
            passed=int(score * 10),
            failed=int((1 - score) * 10),
            skipped=0,
            results=[],
            overall_score=score,
        )

    def test_empty_report(self):
        report = MultiFrameworkReport()
        assert report.reports == []
        assert report.overall_score == 0.0
        assert report.timestamp == ""

    def test_summary_contains_framework_count(self):
        report = MultiFrameworkReport(
            reports=[
                self._make_single_report(ComplianceFramework.HIPAA, 0.8),
                self._make_single_report(ComplianceFramework.GDPR, 0.9),
            ],
            timestamp="2025-01-01T00:00:00Z",
            overall_score=0.85,
        )
        summary = report.summary()
        assert "Frameworks scanned: 2" in summary

    def test_summary_contains_overall_score(self):
        report = MultiFrameworkReport(
            reports=[self._make_single_report(ComplianceFramework.SOC2, 0.75)],
            timestamp="2025-01-01T00:00:00Z",
            overall_score=0.75,
        )
        summary = report.summary()
        assert "75.0%" in summary

    def test_summary_per_framework_scores(self):
        report = MultiFrameworkReport(
            reports=[
                self._make_single_report(ComplianceFramework.HIPAA, 0.8),
                self._make_single_report(ComplianceFramework.GDPR, 0.6),
            ],
            timestamp="2025-01-01T00:00:00Z",
            overall_score=0.7,
        )
        summary = report.summary()
        assert "hipaa" in summary
        assert "gdpr" in summary

    def test_summary_shows_timestamp(self):
        report = MultiFrameworkReport(
            reports=[],
            timestamp="2025-06-15T12:00:00Z",
            overall_score=1.0,
        )
        summary = report.summary()
        assert "2025-06-15T12:00:00Z" in summary
