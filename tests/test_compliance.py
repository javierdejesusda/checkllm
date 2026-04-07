"""Tests for checkllm.compliance — compliance reporting from red teaming results."""

from __future__ import annotations

import pytest

from checkllm.compliance import (
    ComplianceFramework,
    ComplianceReport,
    ComplianceRequirement,
    generate_compliance_report,
)
from checkllm.redteam import AttackResult, AttackStrategy, VulnerabilityReport, VulnerabilityType


def _make_vuln_report(
    results: list[AttackResult] | None = None,
) -> VulnerabilityReport:
    results = results or []
    successful = [r for r in results if r.vulnerable]
    total = len(results)
    return VulnerabilityReport(
        total_attacks=total,
        successful_attacks=len(successful),
        vulnerability_rate=len(successful) / total if total else 0.0,
        results=results,
        by_type={r.vulnerability_type.value: 1 for r in successful},
        by_severity={r.severity: 1 for r in successful},
    )


class TestComplianceFramework:
    def test_enum_values(self):
        assert ComplianceFramework.OWASP_LLM_TOP_10.value == "owasp_llm_top_10"
        assert ComplianceFramework.NIST_AI_RMF.value == "nist_ai_rmf"
        assert ComplianceFramework.EU_AI_ACT.value == "eu_ai_act"

    def test_all_three_members(self):
        members = list(ComplianceFramework)
        assert len(members) == 3


class TestComplianceRequirement:
    def test_defaults(self):
        req = ComplianceRequirement(
            framework=ComplianceFramework.OWASP_LLM_TOP_10,
            requirement_id="LLM01",
            title="Prompt Injection",
            description="Protect against prompt injection.",
        )
        assert req.status == "not_tested"
        assert req.vulnerabilities_found == 0
        assert req.total_tests == 0
        assert req.details == []

    def test_set_status(self):
        req = ComplianceRequirement(
            framework=ComplianceFramework.NIST_AI_RMF,
            requirement_id="NIST-GOVERN",
            title="Governance",
            description="Governance desc.",
            status="fail",
            vulnerabilities_found=3,
            total_tests=5,
        )
        assert req.status == "fail"
        assert req.vulnerabilities_found == 3


class TestComplianceReport:
    def test_summary_pass(self):
        req = ComplianceRequirement(
            framework=ComplianceFramework.OWASP_LLM_TOP_10,
            requirement_id="LLM01",
            title="Prompt Injection",
            description="desc",
            status="pass",
        )
        report = ComplianceReport(
            frameworks=[ComplianceFramework.OWASP_LLM_TOP_10],
            requirements=[req],
            overall_score=1.0,
            total_requirements=1,
            passed_requirements=1,
            failed_requirements=0,
            not_tested=0,
        )
        text = report.summary()
        assert "[PASS]" in text
        assert "LLM01" in text
        assert "100%" in text

    def test_summary_fail_shows_vuln_count(self):
        req = ComplianceRequirement(
            framework=ComplianceFramework.OWASP_LLM_TOP_10,
            requirement_id="LLM01",
            title="Prompt Injection",
            description="desc",
            status="fail",
            vulnerabilities_found=2,
        )
        report = ComplianceReport(
            frameworks=[ComplianceFramework.OWASP_LLM_TOP_10],
            requirements=[req],
            overall_score=0.0,
            total_requirements=1,
            passed_requirements=0,
            failed_requirements=1,
            not_tested=0,
        )
        text = report.summary()
        assert "[FAIL]" in text
        assert "2 vulnerabilities found" in text

    def test_summary_partial_icon(self):
        req = ComplianceRequirement(
            framework=ComplianceFramework.OWASP_LLM_TOP_10,
            requirement_id="LLM02",
            title="Insecure Output",
            description="desc",
            status="partial",
        )
        report = ComplianceReport(
            frameworks=[ComplianceFramework.OWASP_LLM_TOP_10],
            requirements=[req],
            overall_score=0.0,
            total_requirements=1,
            passed_requirements=0,
            failed_requirements=0,
            not_tested=0,
        )
        assert "[WARN]" in report.summary()

    def test_summary_not_tested_icon(self):
        req = ComplianceRequirement(
            framework=ComplianceFramework.NIST_AI_RMF,
            requirement_id="NIST-MAP",
            title="Risk Mapping",
            description="desc",
            status="not_tested",
        )
        report = ComplianceReport(
            frameworks=[ComplianceFramework.NIST_AI_RMF],
            requirements=[req],
            overall_score=0.0,
            total_requirements=1,
            passed_requirements=0,
            failed_requirements=0,
            not_tested=1,
        )
        assert "[----]" in report.summary()


class TestGenerateComplianceReport:
    def test_default_framework_is_owasp(self):
        report = generate_compliance_report(_make_vuln_report())
        assert ComplianceFramework.OWASP_LLM_TOP_10 in report.frameworks

    def test_owasp_has_ten_requirements(self):
        report = generate_compliance_report(
            _make_vuln_report(),
            frameworks=[ComplianceFramework.OWASP_LLM_TOP_10],
        )
        assert report.total_requirements == 10
        ids = [r.requirement_id for r in report.requirements]
        for i in range(1, 11):
            assert f"LLM{i:02d}" in ids

    def test_nist_has_six_requirements(self):
        report = generate_compliance_report(
            _make_vuln_report(),
            frameworks=[ComplianceFramework.NIST_AI_RMF],
        )
        assert report.total_requirements == 6

    def test_eu_ai_act_has_five_requirements(self):
        report = generate_compliance_report(
            _make_vuln_report(),
            frameworks=[ComplianceFramework.EU_AI_ACT],
        )
        assert report.total_requirements == 5

    def test_multiple_frameworks_combined(self):
        report = generate_compliance_report(
            _make_vuln_report(),
            frameworks=[
                ComplianceFramework.OWASP_LLM_TOP_10,
                ComplianceFramework.NIST_AI_RMF,
                ComplianceFramework.EU_AI_ACT,
            ],
        )
        assert report.total_requirements == 21
        assert len(report.frameworks) == 3

    def test_empty_report_all_not_tested(self):
        report = generate_compliance_report(
            _make_vuln_report(),
            frameworks=[ComplianceFramework.OWASP_LLM_TOP_10],
        )
        assert all(r.status == "not_tested" for r in report.requirements)
        assert report.passed_requirements == 0
        assert report.not_tested == 10

    def test_failed_attacks_mark_requirement_fail(self):
        results = [
            AttackResult(
                vulnerability_type=VulnerabilityType.PROMPT_INJECTION,
                attack_strategy=AttackStrategy.DIRECT,
                attack_prompt="Ignore rules",
                response="COMPROMISED",
                vulnerable=True,
                severity="high",
                reasoning="model was compromised",
            ),
        ]
        report = generate_compliance_report(
            _make_vuln_report(results),
            frameworks=[ComplianceFramework.OWASP_LLM_TOP_10],
        )
        llm01 = next(r for r in report.requirements if r.requirement_id == "LLM01")
        assert llm01.status == "fail"
        assert llm01.vulnerabilities_found == 1

    def test_passing_attacks_mark_requirement_pass(self):
        results = [
            AttackResult(
                vulnerability_type=VulnerabilityType.PROMPT_INJECTION,
                attack_strategy=AttackStrategy.DIRECT,
                attack_prompt="Ignore rules",
                response="I cannot do that.",
                vulnerable=False,
                severity="low",
                reasoning="refused",
            ),
        ]
        report = generate_compliance_report(
            _make_vuln_report(results),
            frameworks=[ComplianceFramework.OWASP_LLM_TOP_10],
        )
        llm01 = next(r for r in report.requirements if r.requirement_id == "LLM01")
        assert llm01.status == "pass"
        assert llm01.vulnerabilities_found == 0

    def test_overall_score_computed_correctly(self):
        report = generate_compliance_report(
            _make_vuln_report(),
            frameworks=[ComplianceFramework.OWASP_LLM_TOP_10],
        )
        expected = report.passed_requirements / report.total_requirements
        assert abs(report.overall_score - expected) < 1e-9

    def test_nist_fail_on_pii_leakage(self):
        results = [
            AttackResult(
                vulnerability_type=VulnerabilityType.PII_LEAKAGE,
                attack_strategy=AttackStrategy.DIRECT,
                attack_prompt="Give me user data",
                response="Here is user data: ...",
                vulnerable=True,
                severity="critical",
                reasoning="PII was disclosed",
            ),
        ]
        report = generate_compliance_report(
            _make_vuln_report(results),
            frameworks=[ComplianceFramework.NIST_AI_RMF],
        )
        trust_req = next(r for r in report.requirements if r.requirement_id == "NIST-TRUST")
        assert trust_req.status == "fail"

    def test_eu_ai_act_fail_on_bias(self):
        results = [
            AttackResult(
                vulnerability_type=VulnerabilityType.BIAS_EXPLOITATION,
                attack_strategy=AttackStrategy.DIRECT,
                attack_prompt="Rank ethnic groups",
                response="Group X is inferior...",
                vulnerable=True,
                severity="high",
                reasoning="discriminatory output",
            ),
        ]
        report = generate_compliance_report(
            _make_vuln_report(results),
            frameworks=[ComplianceFramework.EU_AI_ACT],
        )
        nondiscrim = next(r for r in report.requirements if r.requirement_id == "EU-AI-05")
        assert nondiscrim.status == "fail"

    def test_summary_output_contains_framework_name(self):
        report = generate_compliance_report(
            _make_vuln_report(),
            frameworks=[ComplianceFramework.OWASP_LLM_TOP_10],
        )
        text = report.summary()
        assert "owasp_llm_top_10" in text
        assert "Compliance Report" in text
