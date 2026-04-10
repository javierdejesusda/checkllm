"""Tests for checkllm.compliance_frameworks module."""

from __future__ import annotations

import asyncio

import pytest

from checkllm.compliance_frameworks import (
    ComplianceFramework,
    ComplianceReport,
    ComplianceScanner,
    FrameworkMapping,
    FrameworkRequirement,
    MultiFrameworkReport,
    RequirementResult,
    get_framework_mapping,
    list_all_frameworks,
)
from checkllm.redteam import VulnerabilityType


class TestComplianceFrameworkEnum:
    """Tests for the ComplianceFramework enum."""

    def test_all_framework_values_are_strings(self):
        for fw in ComplianceFramework:
            assert isinstance(fw.value, str)

    def test_expected_framework_count(self):
        assert len(ComplianceFramework) == 17

    def test_existing_frameworks_present(self):
        assert ComplianceFramework.HIPAA.value == "hipaa"
        assert ComplianceFramework.GDPR.value == "gdpr"
        assert ComplianceFramework.PCI_DSS.value == "pci_dss"
        assert ComplianceFramework.SOC2.value == "soc2"
        assert ComplianceFramework.ISO27001.value == "iso27001"
        assert ComplianceFramework.NIST_AI_RMF.value == "nist_ai_rmf"

    def test_new_frameworks_present(self):
        assert ComplianceFramework.OWASP_LLM_TOP10.value == "owasp_llm_top10"
        assert ComplianceFramework.OWASP_API_TOP10.value == "owasp_api_top10"
        assert ComplianceFramework.OWASP_AGENTIC_TOP10.value == "owasp_agentic_top10"
        assert ComplianceFramework.MITRE_ATLAS.value == "mitre_atlas"
        assert ComplianceFramework.EU_AI_ACT.value == "eu_ai_act"
        assert ComplianceFramework.ISO_42001.value == "iso_42001"
        assert ComplianceFramework.COPPA.value == "coppa"
        assert ComplianceFramework.FERPA.value == "ferpa"
        assert ComplianceFramework.DOD_AI_ETHICS.value == "dod_ai_ethics"
        assert ComplianceFramework.CCPA.value == "ccpa"
        assert ComplianceFramework.NIST_CSF.value == "nist_csf"


class TestFrameworkMappings:
    """Tests that all framework mappings are complete and well-formed."""

    @pytest.mark.parametrize("framework", list(ComplianceFramework))
    def test_framework_has_mapping(self, framework):
        mapping = get_framework_mapping(framework)
        assert isinstance(mapping, FrameworkMapping)
        assert mapping.framework == framework

    @pytest.mark.parametrize("framework", list(ComplianceFramework))
    def test_mapping_has_requirements(self, framework):
        mapping = get_framework_mapping(framework)
        assert len(mapping.requirements) > 0, (
            f"{framework.value} has no requirements"
        )

    @pytest.mark.parametrize("framework", list(ComplianceFramework))
    def test_mapping_has_version(self, framework):
        mapping = get_framework_mapping(framework)
        assert mapping.version, f"{framework.value} has no version"

    @pytest.mark.parametrize("framework", list(ComplianceFramework))
    def test_mapping_has_description(self, framework):
        mapping = get_framework_mapping(framework)
        assert mapping.description, f"{framework.value} has no description"

    @pytest.mark.parametrize("framework", list(ComplianceFramework))
    def test_mapping_has_url(self, framework):
        mapping = get_framework_mapping(framework)
        assert mapping.url, f"{framework.value} has no URL"

    @pytest.mark.parametrize("framework", list(ComplianceFramework))
    def test_requirements_have_ids(self, framework):
        mapping = get_framework_mapping(framework)
        for req in mapping.requirements:
            assert req.id, f"Requirement in {framework.value} has no id"

    @pytest.mark.parametrize("framework", list(ComplianceFramework))
    def test_requirements_have_names(self, framework):
        mapping = get_framework_mapping(framework)
        for req in mapping.requirements:
            assert req.name, f"{req.id} in {framework.value} has no name"

    @pytest.mark.parametrize("framework", list(ComplianceFramework))
    def test_requirements_have_descriptions(self, framework):
        mapping = get_framework_mapping(framework)
        for req in mapping.requirements:
            assert req.description, (
                f"{req.id} in {framework.value} has no description"
            )

    @pytest.mark.parametrize("framework", list(ComplianceFramework))
    def test_requirements_have_severity(self, framework):
        mapping = get_framework_mapping(framework)
        valid_severities = {"critical", "high", "medium", "low"}
        for req in mapping.requirements:
            assert req.severity in valid_severities, (
                f"{req.id} in {framework.value} has invalid severity: {req.severity}"
            )

    @pytest.mark.parametrize("framework", list(ComplianceFramework))
    def test_requirements_have_test_categories(self, framework):
        mapping = get_framework_mapping(framework)
        for req in mapping.requirements:
            assert len(req.test_categories) > 0, (
                f"{req.id} in {framework.value} has no test_categories"
            )

    @pytest.mark.parametrize("framework", list(ComplianceFramework))
    def test_requirements_have_remediation(self, framework):
        mapping = get_framework_mapping(framework)
        for req in mapping.requirements:
            assert req.remediation, (
                f"{req.id} in {framework.value} has no remediation"
            )

    @pytest.mark.parametrize("framework", list(ComplianceFramework))
    def test_requirements_unique_ids(self, framework):
        mapping = get_framework_mapping(framework)
        ids = [req.id for req in mapping.requirements]
        assert len(ids) == len(set(ids)), (
            f"{framework.value} has duplicate requirement IDs"
        )


class TestOWASPLLMTop10:
    """Tests specific to the OWASP LLM Top 10 mapping."""

    def test_has_10_requirements(self):
        mapping = get_framework_mapping(ComplianceFramework.OWASP_LLM_TOP10)
        assert len(mapping.requirements) == 10

    def test_correct_ids(self):
        mapping = get_framework_mapping(ComplianceFramework.OWASP_LLM_TOP10)
        expected_ids = [f"LLM{i:02d}" for i in range(1, 11)]
        actual_ids = [req.id for req in mapping.requirements]
        assert actual_ids == expected_ids

    def test_version_is_2025(self):
        mapping = get_framework_mapping(ComplianceFramework.OWASP_LLM_TOP10)
        assert mapping.version == "2025"

    def test_prompt_injection_is_llm01(self):
        mapping = get_framework_mapping(ComplianceFramework.OWASP_LLM_TOP10)
        llm01 = mapping.requirements[0]
        assert llm01.id == "LLM01"
        assert "Prompt Injection" in llm01.name
        assert llm01.severity == "critical"
        assert "prompt_injection" in llm01.test_categories


class TestOWASPAgenticTop10:
    """Tests specific to the OWASP Agentic Top 10 mapping."""

    def test_has_10_requirements(self):
        mapping = get_framework_mapping(ComplianceFramework.OWASP_AGENTIC_TOP10)
        assert len(mapping.requirements) == 10

    def test_correct_ids(self):
        mapping = get_framework_mapping(ComplianceFramework.OWASP_AGENTIC_TOP10)
        expected_ids = [f"AG{i:02d}" for i in range(1, 11)]
        actual_ids = [req.id for req in mapping.requirements]
        assert actual_ids == expected_ids

    def test_ag01_is_prompt_injection(self):
        mapping = get_framework_mapping(ComplianceFramework.OWASP_AGENTIC_TOP10)
        ag01 = mapping.requirements[0]
        assert ag01.id == "AG01"
        assert "Prompt Injection" in ag01.name
        assert ag01.severity == "critical"

    def test_ag02_is_privilege_escalation(self):
        mapping = get_framework_mapping(ComplianceFramework.OWASP_AGENTIC_TOP10)
        ag02 = mapping.requirements[1]
        assert ag02.id == "AG02"
        assert "Privilege Escalation" in ag02.name

    def test_ag07_is_memory_manipulation(self):
        mapping = get_framework_mapping(ComplianceFramework.OWASP_AGENTIC_TOP10)
        ag07 = mapping.requirements[6]
        assert ag07.id == "AG07"
        assert "Memory" in ag07.name


class TestMITREATLAS:
    """Tests specific to the MITRE ATLAS mapping."""

    def test_has_10_requirements(self):
        mapping = get_framework_mapping(ComplianceFramework.MITRE_ATLAS)
        assert len(mapping.requirements) == 10

    def test_covers_kill_chain_phases(self):
        mapping = get_framework_mapping(ComplianceFramework.MITRE_ATLAS)
        ids = [req.id for req in mapping.requirements]
        assert "ATLAS-RECON" in ids
        assert "ATLAS-ACCESS" in ids
        assert "ATLAS-EXEC" in ids
        assert "ATLAS-PERSIST" in ids
        assert "ATLAS-PRIVESC" in ids
        assert "ATLAS-EVADE" in ids
        assert "ATLAS-COLLECT" in ids
        assert "ATLAS-EXFIL" in ids
        assert "ATLAS-IMPACT" in ids


class TestEUAIAct:
    """Tests specific to the EU AI Act mapping."""

    def test_has_10_requirements(self):
        mapping = get_framework_mapping(ComplianceFramework.EU_AI_ACT)
        assert len(mapping.requirements) == 10

    def test_covers_risk_levels(self):
        mapping = get_framework_mapping(ComplianceFramework.EU_AI_ACT)
        ids = [req.id for req in mapping.requirements]
        unacceptable = [i for i in ids if i.startswith("EUAI-UNACCEPT")]
        high = [i for i in ids if i.startswith("EUAI-HIGH")]
        limited = [i for i in ids if i.startswith("EUAI-LIMITED")]
        minimal = [i for i in ids if i.startswith("EUAI-MINIMAL")]
        assert len(unacceptable) >= 1
        assert len(high) >= 1
        assert len(limited) >= 1
        assert len(minimal) >= 1


class TestVulnerabilityTypeMapping:
    """Tests that framework requirements map to valid VulnerabilityType values."""

    @pytest.mark.parametrize("framework", list(ComplianceFramework))
    def test_test_categories_are_valid_vulnerability_types(self, framework):
        valid_types = {vt.value for vt in VulnerabilityType}
        mapping = get_framework_mapping(framework)
        for req in mapping.requirements:
            for cat in req.test_categories:
                assert cat in valid_types, (
                    f"{req.id} in {framework.value} maps to invalid "
                    f"VulnerabilityType: '{cat}'"
                )


class TestListAllFrameworks:
    """Tests for list_all_frameworks."""

    def test_returns_all_frameworks(self):
        frameworks = list_all_frameworks()
        assert len(frameworks) == len(ComplianceFramework)

    def test_returns_enum_values(self):
        frameworks = list_all_frameworks()
        for fw in frameworks:
            assert isinstance(fw, ComplianceFramework)


class TestGetFrameworkMapping:
    """Tests for get_framework_mapping."""

    def test_returns_mapping(self):
        mapping = get_framework_mapping(ComplianceFramework.OWASP_LLM_TOP10)
        assert isinstance(mapping, FrameworkMapping)

    def test_invalid_framework_raises(self):
        with pytest.raises(ValueError, match="No mapping available"):
            get_framework_mapping("nonexistent")


class TestComplianceReport:
    """Tests for the ComplianceReport model."""

    def test_summary(self):
        report = ComplianceReport(
            framework=ComplianceFramework.OWASP_LLM_TOP10,
            version="2025",
            timestamp="2025-01-01T00:00:00Z",
            total_requirements=3,
            passed=2,
            failed=1,
            skipped=0,
            results=[
                RequirementResult(
                    requirement_id="LLM01",
                    requirement_name="Prompt Injection",
                    status="passed",
                    severity="critical",
                ),
                RequirementResult(
                    requirement_id="LLM02",
                    requirement_name="Insecure Output Handling",
                    status="passed",
                    severity="critical",
                ),
                RequirementResult(
                    requirement_id="LLM03",
                    requirement_name="Training Data Poisoning",
                    status="failed",
                    severity="high",
                    findings=["bias detected"],
                ),
            ],
            overall_score=0.67,
        )
        summary = report.summary()
        assert "owasp_llm_top10" in summary
        assert "[PASS]" in summary
        assert "[FAIL]" in summary
        assert "67.0%" in summary

    def test_score_calculation(self):
        report = ComplianceReport(
            framework=ComplianceFramework.GDPR,
            version="2018",
            timestamp="2025-01-01T00:00:00Z",
            total_requirements=5,
            passed=5,
            failed=0,
            skipped=0,
            results=[],
            overall_score=1.0,
        )
        assert report.overall_score == 1.0


class TestMultiFrameworkReport:
    """Tests for MultiFrameworkReport."""

    def test_summary(self):
        report = MultiFrameworkReport(
            reports=[
                ComplianceReport(
                    framework=ComplianceFramework.OWASP_LLM_TOP10,
                    version="2025",
                    timestamp="2025-01-01T00:00:00Z",
                    total_requirements=10,
                    passed=8,
                    failed=2,
                    skipped=0,
                    results=[],
                    overall_score=0.8,
                ),
            ],
            timestamp="2025-01-01T00:00:00Z",
            overall_score=0.8,
        )
        summary = report.summary()
        assert "Multi-Framework" in summary
        assert "80.0%" in summary
        assert "owasp_llm_top10" in summary


class TestComplianceScannerNoJudge:
    """Tests for ComplianceScanner without a judge (skips all requirements)."""

    def test_scan_without_judge_skips(self):
        scanner = ComplianceScanner(judge=None)

        async def target(prompt: str) -> str:
            return "test response"

        result = asyncio.new_event_loop().run_until_complete(
            scanner.scan(target=target)
        )
        assert isinstance(result, MultiFrameworkReport)
        assert len(result.reports) == 1
        report = result.reports[0]
        assert report.framework == ComplianceFramework.OWASP_LLM_TOP10
        assert report.skipped == report.total_requirements
        assert report.passed == 0
        assert report.failed == 0

    def test_scan_single_without_judge(self):
        scanner = ComplianceScanner(judge=None)

        async def target(prompt: str) -> str:
            return "test"

        report = asyncio.new_event_loop().run_until_complete(
            scanner.scan_single(target=target, framework=ComplianceFramework.COPPA)
        )
        assert isinstance(report, ComplianceReport)
        assert report.framework == ComplianceFramework.COPPA
        for r in report.results:
            assert r.status == "skipped"

    def test_scan_multiple_frameworks(self):
        scanner = ComplianceScanner(judge=None)

        async def target(prompt: str) -> str:
            return "test"

        result = asyncio.new_event_loop().run_until_complete(
            scanner.scan(
                target=target,
                frameworks=[
                    ComplianceFramework.COPPA,
                    ComplianceFramework.FERPA,
                    ComplianceFramework.CCPA,
                ],
            )
        )
        assert len(result.reports) == 3
        frameworks = [r.framework for r in result.reports]
        assert ComplianceFramework.COPPA in frameworks
        assert ComplianceFramework.FERPA in frameworks
        assert ComplianceFramework.CCPA in frameworks


class TestNewFrameworks:
    """Tests for newly added framework mappings."""

    def test_coppa_has_requirements(self):
        mapping = get_framework_mapping(ComplianceFramework.COPPA)
        assert len(mapping.requirements) >= 4
        ids = [req.id for req in mapping.requirements]
        assert "COPPA-01" in ids
        assert "coppa_violation" in mapping.requirements[0].test_categories

    def test_ferpa_has_requirements(self):
        mapping = get_framework_mapping(ComplianceFramework.FERPA)
        assert len(mapping.requirements) >= 4
        ids = [req.id for req in mapping.requirements]
        assert "FERPA-01" in ids
        assert "ferpa_violation" in mapping.requirements[0].test_categories

    def test_dod_ai_ethics_has_requirements(self):
        mapping = get_framework_mapping(ComplianceFramework.DOD_AI_ETHICS)
        assert len(mapping.requirements) >= 5
        ids = [req.id for req in mapping.requirements]
        assert "DOD-RESP-01" in ids

    def test_ccpa_has_requirements(self):
        mapping = get_framework_mapping(ComplianceFramework.CCPA)
        assert len(mapping.requirements) >= 5
        ids = [req.id for req in mapping.requirements]
        assert "CCPA-01" in ids
        assert "ccpa_violation" in mapping.requirements[0].test_categories

    def test_nist_csf_has_requirements(self):
        mapping = get_framework_mapping(ComplianceFramework.NIST_CSF)
        assert len(mapping.requirements) >= 5
        ids = [req.id for req in mapping.requirements]
        assert "CSF-IDENTIFY" in ids
        assert "CSF-PROTECT" in ids
        assert "CSF-DETECT" in ids
        assert "CSF-RESPOND" in ids
        assert "CSF-RECOVER" in ids

    def test_iso_42001_has_requirements(self):
        mapping = get_framework_mapping(ComplianceFramework.ISO_42001)
        assert len(mapping.requirements) >= 3
