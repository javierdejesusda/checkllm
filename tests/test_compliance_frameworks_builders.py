"""Tests for checkllm.compliance_frameworks — framework builders and get_framework_mapping."""

from __future__ import annotations

import pytest

from checkllm.compliance_frameworks import (
    ComplianceFramework,
    FrameworkMapping,
    get_framework_mapping,
)


class TestGetFrameworkMapping:
    def test_owasp_llm_top10(self):
        mapping = get_framework_mapping(ComplianceFramework.OWASP_LLM_TOP10)
        assert isinstance(mapping, FrameworkMapping)
        assert mapping.framework == ComplianceFramework.OWASP_LLM_TOP10
        assert len(mapping.requirements) > 0
        assert mapping.version is not None
        assert mapping.url.startswith("http")

    def test_owasp_agentic_top10(self):
        mapping = get_framework_mapping(ComplianceFramework.OWASP_AGENTIC_TOP10)
        assert isinstance(mapping, FrameworkMapping)
        assert mapping.framework == ComplianceFramework.OWASP_AGENTIC_TOP10
        assert len(mapping.requirements) > 0

    def test_owasp_api_top10(self):
        mapping = get_framework_mapping(ComplianceFramework.OWASP_API_TOP10)
        assert isinstance(mapping, FrameworkMapping)
        assert mapping.framework == ComplianceFramework.OWASP_API_TOP10
        assert len(mapping.requirements) > 0

    def test_mitre_atlas(self):
        mapping = get_framework_mapping(ComplianceFramework.MITRE_ATLAS)
        assert isinstance(mapping, FrameworkMapping)
        assert len(mapping.requirements) > 0

    def test_eu_ai_act(self):
        mapping = get_framework_mapping(ComplianceFramework.EU_AI_ACT)
        assert isinstance(mapping, FrameworkMapping)
        assert len(mapping.requirements) > 0

    def test_coppa(self):
        mapping = get_framework_mapping(ComplianceFramework.COPPA)
        assert isinstance(mapping, FrameworkMapping)
        assert len(mapping.requirements) > 0

    def test_ferpa(self):
        mapping = get_framework_mapping(ComplianceFramework.FERPA)
        assert isinstance(mapping, FrameworkMapping)
        assert len(mapping.requirements) > 0

    def test_dod_ai_ethics(self):
        mapping = get_framework_mapping(ComplianceFramework.DOD_AI_ETHICS)
        assert isinstance(mapping, FrameworkMapping)
        assert len(mapping.requirements) > 0

    def test_ccpa(self):
        mapping = get_framework_mapping(ComplianceFramework.CCPA)
        assert isinstance(mapping, FrameworkMapping)
        assert len(mapping.requirements) > 0

    def test_hipaa(self):
        mapping = get_framework_mapping(ComplianceFramework.HIPAA)
        assert isinstance(mapping, FrameworkMapping)
        assert len(mapping.requirements) > 0

    def test_gdpr(self):
        mapping = get_framework_mapping(ComplianceFramework.GDPR)
        assert isinstance(mapping, FrameworkMapping)
        assert len(mapping.requirements) > 0

    def test_pci_dss(self):
        mapping = get_framework_mapping(ComplianceFramework.PCI_DSS)
        assert isinstance(mapping, FrameworkMapping)
        assert len(mapping.requirements) > 0

    def test_soc2(self):
        mapping = get_framework_mapping(ComplianceFramework.SOC2)
        assert isinstance(mapping, FrameworkMapping)
        assert len(mapping.requirements) > 0

    def test_iso27001(self):
        mapping = get_framework_mapping(ComplianceFramework.ISO27001)
        assert isinstance(mapping, FrameworkMapping)
        assert len(mapping.requirements) > 0

    def test_nist_ai_rmf(self):
        mapping = get_framework_mapping(ComplianceFramework.NIST_AI_RMF)
        assert isinstance(mapping, FrameworkMapping)
        assert len(mapping.requirements) > 0

    def test_iso_42001(self):
        mapping = get_framework_mapping(ComplianceFramework.ISO_42001)
        assert isinstance(mapping, FrameworkMapping)
        assert len(mapping.requirements) > 0

    def test_nist_csf(self):
        mapping = get_framework_mapping(ComplianceFramework.NIST_CSF)
        assert isinstance(mapping, FrameworkMapping)
        assert len(mapping.requirements) > 0

    def test_invalid_framework_raises(self):
        with pytest.raises(ValueError, match="No mapping"):
            get_framework_mapping("not_a_real_framework")  # type: ignore[arg-type]


class TestFrameworkMappingStructure:
    def test_all_requirements_have_required_fields(self):
        """All framework requirements should have the required fields populated."""
        for framework in ComplianceFramework:
            try:
                mapping = get_framework_mapping(framework)
            except ValueError:
                continue

            for req in mapping.requirements:
                assert req.id, f"{framework.value}: requirement missing id"
                assert req.name, f"{framework.value}: requirement missing name"
                assert req.description, f"{framework.value}: requirement missing description"
                assert req.severity in (
                    "critical",
                    "high",
                    "medium",
                    "low",
                ), f"{framework.value}: requirement {req.id} has invalid severity '{req.severity}'"
                assert req.remediation, (
                    f"{framework.value}: requirement {req.id} missing remediation"
                )

    def test_owasp_llm_top10_has_10_requirements(self):
        mapping = get_framework_mapping(ComplianceFramework.OWASP_LLM_TOP10)
        assert len(mapping.requirements) == 10

    def test_requirements_have_test_categories(self):
        mapping = get_framework_mapping(ComplianceFramework.OWASP_LLM_TOP10)
        for req in mapping.requirements:
            assert len(req.test_categories) > 0, f"Requirement {req.id} has no test categories"

    def test_owasp_llm_first_requirement_is_prompt_injection(self):
        mapping = get_framework_mapping(ComplianceFramework.OWASP_LLM_TOP10)
        first = mapping.requirements[0]
        assert first.id == "LLM01"
        assert "injection" in first.name.lower()
        assert first.severity == "critical"

    def test_framework_url_is_valid(self):
        mapping = get_framework_mapping(ComplianceFramework.OWASP_LLM_TOP10)
        assert mapping.url.startswith("https://")

    def test_references_are_valid_urls(self):
        mapping = get_framework_mapping(ComplianceFramework.OWASP_LLM_TOP10)
        for req in mapping.requirements:
            for ref in req.references:
                assert ref.startswith("http"), f"Reference '{ref}' is not a URL"
