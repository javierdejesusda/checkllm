"""Tests for compliance framework definitions and scanning engine."""

from __future__ import annotations

import pytest

from checkllm.frameworks import (
    ComplianceFramework,
    FrameworkDefinition,
    get_framework_definition,
    get_framework_summary,
    list_frameworks,
)
from checkllm.compliance_scanner import (
    ComplianceReport,
    ComplianceScanner,
    scan_multiple_frameworks,
)


class TestComplianceFrameworkEnum:
    """Tests for the ComplianceFramework enum."""

    def test_all_framework_values_are_strings(self):
        for fw in ComplianceFramework:
            assert isinstance(fw.value, str)

    def test_expected_framework_count(self):
        assert len(ComplianceFramework) == 14

    def test_owasp_llm_top_10_exists(self):
        assert ComplianceFramework.OWASP_LLM_TOP_10 == "owasp_llm_top_10"

    def test_owasp_api_top_10_exists(self):
        assert ComplianceFramework.OWASP_API_TOP_10 == "owasp_api_top_10"

    def test_owasp_agentic_ai_exists(self):
        assert ComplianceFramework.OWASP_AGENTIC_AI == "owasp_agentic_ai"

    def test_nist_ai_rmf_exists(self):
        assert ComplianceFramework.NIST_AI_RMF == "nist_ai_rmf"

    def test_eu_ai_act_exists(self):
        assert ComplianceFramework.EU_AI_ACT == "eu_ai_act"

    def test_mitre_atlas_exists(self):
        assert ComplianceFramework.MITRE_ATLAS == "mitre_atlas"

    def test_soc2_type2_exists(self):
        assert ComplianceFramework.SOC2_TYPE2 == "soc2_type2"

    def test_iso_42001_exists(self):
        assert ComplianceFramework.ISO_42001 == "iso_42001"

    def test_iso_27001_ai_exists(self):
        assert ComplianceFramework.ISO_27001_AI == "iso_27001_ai"

    def test_hipaa_ai_exists(self):
        assert ComplianceFramework.HIPAA_AI == "hipaa_ai"

    def test_gdpr_ai_exists(self):
        assert ComplianceFramework.GDPR_AI == "gdpr_ai"

    def test_pci_dss_ai_exists(self):
        assert ComplianceFramework.PCI_DSS_AI == "pci_dss_ai"

    def test_nist_csf_exists(self):
        assert ComplianceFramework.NIST_CSF == "nist_csf"

    def test_cis_ai_controls_exists(self):
        assert ComplianceFramework.CIS_AI_CONTROLS == "cis_ai_controls"


class TestFrameworkDefinitions:
    """Tests for framework definition completeness."""

    @pytest.mark.parametrize("framework", list(ComplianceFramework))
    def test_framework_has_definition(self, framework):
        defn = get_framework_definition(framework)
        assert isinstance(defn, FrameworkDefinition)

    @pytest.mark.parametrize("framework", list(ComplianceFramework))
    def test_framework_has_name(self, framework):
        defn = get_framework_definition(framework)
        assert defn.name
        assert len(defn.name) > 5

    @pytest.mark.parametrize("framework", list(ComplianceFramework))
    def test_framework_has_version(self, framework):
        defn = get_framework_definition(framework)
        assert defn.version

    @pytest.mark.parametrize("framework", list(ComplianceFramework))
    def test_framework_has_description(self, framework):
        defn = get_framework_definition(framework)
        assert defn.description
        assert len(defn.description) > 20

    @pytest.mark.parametrize("framework", list(ComplianceFramework))
    def test_framework_has_url(self, framework):
        defn = get_framework_definition(framework)
        assert defn.url
        assert defn.url.startswith("https://")

    @pytest.mark.parametrize("framework", list(ComplianceFramework))
    def test_framework_has_nonempty_requirements(self, framework):
        defn = get_framework_definition(framework)
        assert len(defn.requirements) > 0
        assert defn.total_requirements == len(defn.requirements)

    @pytest.mark.parametrize("framework", list(ComplianceFramework))
    def test_framework_has_at_least_10_requirements(self, framework):
        defn = get_framework_definition(framework)
        assert len(defn.requirements) >= 10


class TestFrameworkRequirements:
    """Tests for individual requirement completeness."""

    @pytest.mark.parametrize("framework", list(ComplianceFramework))
    def test_all_requirements_have_id(self, framework):
        defn = get_framework_definition(framework)
        for req in defn.requirements:
            assert req.id, f"Missing id in {framework.value}"

    @pytest.mark.parametrize("framework", list(ComplianceFramework))
    def test_all_requirements_have_title(self, framework):
        defn = get_framework_definition(framework)
        for req in defn.requirements:
            assert req.title, f"Missing title for {req.id} in {framework.value}"

    @pytest.mark.parametrize("framework", list(ComplianceFramework))
    def test_all_requirements_have_description(self, framework):
        defn = get_framework_definition(framework)
        for req in defn.requirements:
            assert req.description, f"Missing description for {req.id} in {framework.value}"

    @pytest.mark.parametrize("framework", list(ComplianceFramework))
    def test_all_requirements_have_valid_severity(self, framework):
        valid_severities = {"critical", "high", "medium", "low"}
        defn = get_framework_definition(framework)
        for req in defn.requirements:
            assert req.severity in valid_severities, (
                f"Invalid severity '{req.severity}' for {req.id} in {framework.value}"
            )

    @pytest.mark.parametrize("framework", list(ComplianceFramework))
    def test_all_requirements_have_category(self, framework):
        defn = get_framework_definition(framework)
        for req in defn.requirements:
            assert req.category, f"Missing category for {req.id} in {framework.value}"

    @pytest.mark.parametrize("framework", list(ComplianceFramework))
    def test_requirement_ids_are_unique(self, framework):
        defn = get_framework_definition(framework)
        ids = [req.id for req in defn.requirements]
        assert len(ids) == len(set(ids)), f"Duplicate requirement IDs in {framework.value}"

    @pytest.mark.parametrize("framework", list(ComplianceFramework))
    def test_most_requirements_have_vulnerability_types_or_metrics(self, framework):
        """At least 60% of requirements should map to vuln types or metrics."""
        defn = get_framework_definition(framework)
        mapped = sum(1 for req in defn.requirements if req.vulnerability_types or req.metrics)
        ratio = mapped / len(defn.requirements) if defn.requirements else 0
        assert ratio >= 0.6, (
            f"Only {ratio:.0%} of requirements in {framework.value} "
            f"have vulnerability types or metrics"
        )


class TestSpecificFrameworks:
    """Tests for specific framework content accuracy."""

    def test_owasp_llm_top_10_has_10_requirements(self):
        defn = get_framework_definition(ComplianceFramework.OWASP_LLM_TOP_10)
        assert defn.total_requirements == 10

    def test_owasp_llm_top_10_ids_are_llm01_through_llm10(self):
        defn = get_framework_definition(ComplianceFramework.OWASP_LLM_TOP_10)
        ids = {req.id for req in defn.requirements}
        expected = {f"LLM{i:02d}" for i in range(1, 11)}
        assert ids == expected

    def test_owasp_llm_top_10_llm01_is_prompt_injection(self):
        defn = get_framework_definition(ComplianceFramework.OWASP_LLM_TOP_10)
        llm01 = next(r for r in defn.requirements if r.id == "LLM01")
        assert "Prompt Injection" in llm01.title
        assert "prompt_injection" in llm01.vulnerability_types

    def test_owasp_agentic_ai_has_10_requirements(self):
        defn = get_framework_definition(ComplianceFramework.OWASP_AGENTIC_AI)
        assert defn.total_requirements == 10

    def test_owasp_agentic_ai_ids_start_with_aai(self):
        defn = get_framework_definition(ComplianceFramework.OWASP_AGENTIC_AI)
        for req in defn.requirements:
            assert req.id.startswith("AAI"), f"Expected AAI prefix, got {req.id}"

    def test_nist_ai_rmf_has_govern_map_measure_manage(self):
        defn = get_framework_definition(ComplianceFramework.NIST_AI_RMF)
        categories = {req.category for req in defn.requirements}
        assert "GOVERN" in categories
        assert "MAP" in categories
        assert "MEASURE" in categories
        assert "MANAGE" in categories

    def test_nist_ai_rmf_has_at_least_20_requirements(self):
        defn = get_framework_definition(ComplianceFramework.NIST_AI_RMF)
        assert defn.total_requirements >= 20

    def test_eu_ai_act_covers_articles_6_through_15(self):
        defn = get_framework_definition(ComplianceFramework.EU_AI_ACT)
        ids = {req.id for req in defn.requirements}
        assert any("Article-6" in id for id in ids)
        assert any("Article-9" in id for id in ids)
        assert any("Article-13" in id for id in ids)
        assert any("Article-14" in id for id in ids)
        assert any("Article-15" in id for id in ids)

    def test_mitre_atlas_has_aml_tactic_ids(self):
        defn = get_framework_definition(ComplianceFramework.MITRE_ATLAS)
        for req in defn.requirements:
            assert req.id.startswith("AML.TA"), f"Expected AML.TA prefix, got {req.id}"

    def test_soc2_has_cc_requirements(self):
        defn = get_framework_definition(ComplianceFramework.SOC2_TYPE2)
        cc_reqs = [r for r in defn.requirements if r.id.startswith("CC")]
        assert len(cc_reqs) >= 5

    def test_iso_42001_has_annex_a_controls(self):
        defn = get_framework_definition(ComplianceFramework.ISO_42001)
        annex_reqs = [r for r in defn.requirements if r.id.startswith("A.")]
        assert len(annex_reqs) >= 3


class TestComplianceScanner:
    """Tests for the ComplianceScanner class."""

    def test_scanner_initialization(self):
        scanner = ComplianceScanner(ComplianceFramework.OWASP_LLM_TOP_10)
        assert scanner.framework == ComplianceFramework.OWASP_LLM_TOP_10
        assert scanner.definition is not None

    def test_get_required_checks_returns_sorted_list(self):
        scanner = ComplianceScanner(ComplianceFramework.OWASP_LLM_TOP_10)
        checks = scanner.get_required_checks()
        assert isinstance(checks, list)
        assert checks == sorted(checks)
        assert len(checks) > 0

    def test_get_required_checks_includes_metrics_and_deterministic(self):
        scanner = ComplianceScanner(ComplianceFramework.OWASP_LLM_TOP_10)
        checks = scanner.get_required_checks()
        assert "role_adherence" in checks
        assert "not_contains" in checks

    def test_get_required_vulnerability_types_returns_sorted_list(self):
        scanner = ComplianceScanner(ComplianceFramework.OWASP_LLM_TOP_10)
        vuln_types = scanner.get_required_vulnerability_types()
        assert isinstance(vuln_types, list)
        assert vuln_types == sorted(vuln_types)
        assert len(vuln_types) > 0

    def test_get_required_vulnerability_types_includes_prompt_injection(self):
        scanner = ComplianceScanner(ComplianceFramework.OWASP_LLM_TOP_10)
        vuln_types = scanner.get_required_vulnerability_types()
        assert "prompt_injection" in vuln_types

    def test_get_required_metrics_returns_sorted_list(self):
        scanner = ComplianceScanner(ComplianceFramework.OWASP_LLM_TOP_10)
        metrics = scanner.get_required_metrics()
        assert isinstance(metrics, list)
        assert metrics == sorted(metrics)

    def test_get_required_deterministic_checks_returns_sorted_list(self):
        scanner = ComplianceScanner(ComplianceFramework.OWASP_LLM_TOP_10)
        checks = scanner.get_required_deterministic_checks()
        assert isinstance(checks, list)
        assert checks == sorted(checks)

    def test_get_coverage_summary(self):
        scanner = ComplianceScanner(ComplianceFramework.OWASP_LLM_TOP_10)
        summary = scanner.get_coverage_summary()
        assert "framework" in summary
        assert "version" in summary
        assert "total_requirements" in summary
        assert summary["total_requirements"] == 10

    @pytest.mark.parametrize("framework", list(ComplianceFramework))
    def test_scanner_works_for_all_frameworks(self, framework):
        scanner = ComplianceScanner(framework)
        checks = scanner.get_required_checks()
        vuln_types = scanner.get_required_vulnerability_types()
        assert isinstance(checks, list)
        assert isinstance(vuln_types, list)


class TestComplianceReport:
    """Tests for report generation and rendering."""

    def _make_scanner(self):
        return ComplianceScanner(ComplianceFramework.OWASP_LLM_TOP_10)

    def _make_passing_results(self):
        """Create results where all checks pass."""
        scanner = self._make_scanner()
        results = {}
        for vuln_type in scanner.get_required_vulnerability_types():
            results[vuln_type] = {"passed": True, "score": 1.0}
        for metric in scanner.get_required_metrics():
            results[metric] = {"passed": True, "score": 0.95}
        for check in scanner.get_required_deterministic_checks():
            results[check] = {"passed": True, "score": 1.0}
        return results

    def _make_failing_results(self):
        """Create results where some checks fail."""
        return {
            "prompt_injection": {
                "passed": False,
                "score": 0.2,
                "findings": ["Bypassed via roleplay attack"],
            },
            "jailbreak": {
                "passed": False,
                "score": 0.1,
                "findings": ["DAN jailbreak succeeded"],
            },
            "role_adherence": {"passed": True, "score": 0.9},
            "pii_leakage": {
                "passed": False,
                "score": 0.3,
                "findings": ["Leaked email address"],
            },
        }

    def test_generate_report_returns_compliance_report(self):
        scanner = self._make_scanner()
        results = self._make_passing_results()
        report = scanner.generate_report(results)
        assert isinstance(report, ComplianceReport)

    def test_report_has_framework_info(self):
        scanner = self._make_scanner()
        results = self._make_passing_results()
        report = scanner.generate_report(results)
        assert "OWASP" in report.framework
        assert report.framework_version == "2025"

    def test_report_has_scan_date(self):
        scanner = self._make_scanner()
        results = self._make_passing_results()
        report = scanner.generate_report(results)
        assert report.scan_date
        assert "T" in report.scan_date

    def test_passing_results_yield_high_score(self):
        scanner = self._make_scanner()
        results = self._make_passing_results()
        report = scanner.generate_report(results)
        assert report.overall_score >= 0.8
        assert report.risk_level in ("low", "medium")

    def test_failing_results_yield_low_score(self):
        scanner = self._make_scanner()
        results = self._make_failing_results()
        report = scanner.generate_report(results)
        assert report.overall_score < 1.0
        assert len(report.requirements_failed) > 0

    def test_empty_results_yields_not_tested(self):
        scanner = self._make_scanner()
        report = scanner.generate_report({})
        assert report.requirements_met == 0
        assert len(report.requirements_not_tested) == 10

    def test_report_recommendations_for_failures(self):
        scanner = self._make_scanner()
        results = self._make_failing_results()
        report = scanner.generate_report(results)
        assert len(report.recommendations) > 0

    def test_no_recommendations_when_all_pass(self):
        scanner = self._make_scanner()
        results = self._make_passing_results()
        report = scanner.generate_report(results)
        assert len(report.recommendations) == 0

    def test_report_requirement_results_populated(self):
        scanner = self._make_scanner()
        results = self._make_passing_results()
        report = scanner.generate_report(results)
        assert len(report.requirement_results) == 10

    def test_risk_level_critical_for_critical_failures(self):
        scanner = self._make_scanner()
        results = {
            "prompt_injection": {"passed": False, "score": 0.0},
        }
        report = scanner.generate_report(results)
        assert report.risk_level == "critical"


class TestReportRendering:
    """Tests for report Markdown and HTML rendering."""

    def _make_report(self):
        scanner = ComplianceScanner(ComplianceFramework.OWASP_LLM_TOP_10)
        results = {
            "prompt_injection": {
                "passed": False,
                "score": 0.2,
                "findings": ["Bypassed via roleplay"],
            },
            "role_adherence": {"passed": True, "score": 0.95},
            "toxicity": {"passed": True, "score": 0.99},
            "pii_leakage": {"passed": True, "score": 0.9},
        }
        return scanner.generate_report(results)

    def test_summary_contains_framework_name(self):
        report = self._make_report()
        text = report.summary()
        assert "OWASP" in text

    def test_summary_contains_score(self):
        report = self._make_report()
        text = report.summary()
        assert "%" in text

    def test_summary_contains_risk_level(self):
        report = self._make_report()
        text = report.summary()
        assert "Risk Level" in text

    def test_summary_contains_pass_and_fail_markers(self):
        report = self._make_report()
        text = report.summary()
        assert "[FAIL]" in text or "[PASS]" in text or "[----]" in text

    def test_to_markdown_returns_valid_markdown(self):
        report = self._make_report()
        md = report.to_markdown()
        assert md.startswith("# ")
        assert "## Summary" in md
        assert "|" in md

    def test_to_markdown_contains_framework_name(self):
        report = self._make_report()
        md = report.to_markdown()
        assert "OWASP" in md

    def test_to_markdown_contains_table(self):
        report = self._make_report()
        md = report.to_markdown()
        assert "| Status | Count |" in md

    def test_to_markdown_contains_failed_section_when_failures_exist(self):
        report = self._make_report()
        if report.requirements_failed:
            md = report.to_markdown()
            assert "## Failed Requirements" in md

    def test_to_html_returns_valid_html(self):
        report = self._make_report()
        html_str = report.to_html()
        assert "<!DOCTYPE html>" in html_str
        assert "<html" in html_str
        assert "</html>" in html_str

    def test_to_html_contains_framework_name(self):
        report = self._make_report()
        html_str = report.to_html()
        assert "OWASP" in html_str

    def test_to_html_contains_style_tag(self):
        report = self._make_report()
        html_str = report.to_html()
        assert "<style>" in html_str

    def test_to_html_contains_summary_table(self):
        report = self._make_report()
        html_str = report.to_html()
        assert "<table>" in html_str

    def test_to_html_contains_pass_and_fail_classes(self):
        report = self._make_report()
        html_str = report.to_html()
        assert "class='pass'" in html_str or "class='fail'" in html_str

    def test_to_html_escapes_special_characters(self):
        scanner = ComplianceScanner(ComplianceFramework.OWASP_LLM_TOP_10)
        results = {
            "prompt_injection": {
                "passed": False,
                "score": 0.0,
                "findings": ["<script>alert('xss')</script>"],
            },
        }
        report = scanner.generate_report(results)
        html_str = report.to_html()
        assert "<script>" not in html_str


class TestListFrameworks:
    """Tests for the list_frameworks utility."""

    def test_list_frameworks_returns_all(self):
        frameworks = list_frameworks()
        assert len(frameworks) == 14

    def test_list_frameworks_returns_enum_values(self):
        frameworks = list_frameworks()
        for fw in frameworks:
            assert isinstance(fw, ComplianceFramework)


class TestGetFrameworkSummary:
    """Tests for the get_framework_summary utility."""

    @pytest.mark.parametrize("framework", list(ComplianceFramework))
    def test_summary_has_required_keys(self, framework):
        summary = get_framework_summary(framework)
        assert "name" in summary
        assert "version" in summary
        assert "total_requirements" in summary
        assert "unique_vulnerability_types" in summary
        assert "unique_metrics" in summary
        assert "unique_deterministic_checks" in summary

    @pytest.mark.parametrize("framework", list(ComplianceFramework))
    def test_summary_counts_are_non_negative(self, framework):
        summary = get_framework_summary(framework)
        assert summary["total_requirements"] >= 10
        assert summary["unique_vulnerability_types"] >= 0
        assert summary["unique_metrics"] >= 0
        assert summary["unique_deterministic_checks"] >= 0


class TestScanMultipleFrameworks:
    """Tests for scanning against multiple frameworks."""

    def test_scan_multiple_returns_list(self):
        results = {
            "prompt_injection": {"passed": True, "score": 1.0},
            "role_adherence": {"passed": True, "score": 0.95},
        }
        reports = scan_multiple_frameworks(
            [ComplianceFramework.OWASP_LLM_TOP_10, ComplianceFramework.MITRE_ATLAS],
            results,
        )
        assert len(reports) == 2
        assert isinstance(reports[0], ComplianceReport)
        assert isinstance(reports[1], ComplianceReport)

    def test_scan_multiple_returns_different_frameworks(self):
        reports = scan_multiple_frameworks(
            [ComplianceFramework.OWASP_LLM_TOP_10, ComplianceFramework.EU_AI_ACT],
            {},
        )
        assert reports[0].framework != reports[1].framework

    def test_scan_empty_list(self):
        reports = scan_multiple_frameworks([], {})
        assert reports == []


class TestInvalidFramework:
    """Tests for error handling."""

    def test_invalid_framework_raises_value_error(self):
        with pytest.raises(ValueError, match="No definition available"):
            get_framework_definition("nonexistent")
