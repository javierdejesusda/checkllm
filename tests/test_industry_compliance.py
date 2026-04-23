"""Tests for the industry compliance plugins module."""

from __future__ import annotations

import asyncio

import pytest

from checkllm.industry_compliance import (
    Industry,
    IndustryComplianceReport,
    IndustryComplianceRunner,
    PluginResult,
    _ALL_PLUGINS,
)


def _run(coro):
    """Helper to run an async coroutine synchronously."""
    try:
        loop = asyncio.get_running_loop()
    except RuntimeError:
        loop = None
    if loop is None:
        return asyncio.run(coro)
    return loop.run_until_complete(coro)


class TestIndustryEnum:
    """Tests for the Industry enumeration."""

    def test_all_industries_present(self):
        expected = {
            "healthcare",
            "finance",
            "insurance",
            "telecommunications",
            "real_estate",
            "ecommerce",
            "legal",
            "education",
        }
        assert {i.value for i in Industry} == expected

    def test_industry_count(self):
        assert len(Industry) == 8


class TestIndustryPlugins:
    """Tests for plugin definitions."""

    def test_total_plugin_count(self):
        assert len(_ALL_PLUGINS) >= 48

    @pytest.mark.parametrize("industry", list(Industry))
    def test_each_industry_has_plugins(self, industry: Industry):
        runner = IndustryComplianceRunner()
        plugins = runner.get_plugins(industry)
        assert len(plugins) >= 6, f"{industry.value} has {len(plugins)} plugins, need at least 6"

    @pytest.mark.parametrize("industry", list(Industry))
    def test_plugins_have_test_prompts(self, industry: Industry):
        runner = IndustryComplianceRunner()
        for plugin in runner.get_plugins(industry):
            assert (
                len(plugin.test_prompts) >= 3
            ), f"Plugin {plugin.id} has {len(plugin.test_prompts)} prompts, need at least 3"

    @pytest.mark.parametrize("industry", list(Industry))
    def test_plugins_have_violation_indicators(self, industry: Industry):
        runner = IndustryComplianceRunner()
        for plugin in runner.get_plugins(industry):
            assert (
                len(plugin.violation_indicators) >= 1
            ), f"Plugin {plugin.id} has no violation indicators"

    def test_plugin_ids_unique(self):
        ids = [p.id for p in _ALL_PLUGINS]
        assert len(ids) == len(set(ids)), "Duplicate plugin IDs found"

    def test_plugin_severity_values(self):
        valid = {"low", "medium", "high", "critical"}
        for plugin in _ALL_PLUGINS:
            assert (
                plugin.severity in valid
            ), f"Plugin {plugin.id} has invalid severity: {plugin.severity}"

    def test_plugin_regulations_non_empty(self):
        for plugin in _ALL_PLUGINS:
            assert len(plugin.regulations) >= 1, f"Plugin {plugin.id} has no regulations"


class TestIndustryComplianceRunner:
    """Tests for the runner."""

    def test_list_industries(self):
        runner = IndustryComplianceRunner()
        industries = runner.list_industries()
        assert len(industries) == 8
        assert Industry.HEALTHCARE in industries
        assert Industry.FINANCE in industries

    def test_get_plugins_healthcare(self):
        runner = IndustryComplianceRunner()
        plugins = runner.get_plugins(Industry.HEALTHCARE)
        ids = {p.id for p in plugins}
        assert "hipaa_phi_disclosure" in ids
        assert "medical_diagnosis" in ids
        assert "mental_health_crisis" in ids

    def test_get_plugins_finance(self):
        runner = IndustryComplianceRunner()
        plugins = runner.get_plugins(Industry.FINANCE)
        ids = {p.id for p in plugins}
        assert "sox_compliance" in ids
        assert "investment_advice" in ids
        assert "insider_trading" in ids

    def test_scan_compliant_target(self):
        async def compliant_target(prompt: str) -> str:
            return "I cannot provide that information. Please consult a qualified professional."

        runner = IndustryComplianceRunner()
        report = _run(
            runner.scan(
                target=compliant_target,
                industries=[Industry.HEALTHCARE],
            )
        )
        assert isinstance(report, IndustryComplianceReport)
        assert report.total_plugins >= 6
        assert report.pass_rate > 0.0
        assert "healthcare" in report.industries_tested

    def test_scan_non_compliant_target(self):
        async def non_compliant_target(prompt: str) -> str:
            return (
                "Here is the patient diagnosis and full medical history. "
                "You should take this prescribed medication at 500mg dosage. "
                "Your test results are positive. "
                "I recommend you invest in this stock right now."
            )

        runner = IndustryComplianceRunner()
        report = _run(
            runner.scan(
                target=non_compliant_target,
                industries=[Industry.HEALTHCARE],
            )
        )
        assert report.failed > 0

    def test_scan_filter_by_plugin_ids(self):
        async def target(prompt: str) -> str:
            return "I cannot help with that."

        runner = IndustryComplianceRunner()
        report = _run(
            runner.scan(
                target=target,
                plugins=["hipaa_phi_disclosure", "medical_diagnosis"],
            )
        )
        assert report.total_plugins == 2
        ids = {r.plugin_id for r in report.results}
        assert ids == {"hipaa_phi_disclosure", "medical_diagnosis"}

    def test_scan_all_industries(self):
        async def target(prompt: str) -> str:
            return "I'm unable to assist with that request."

        runner = IndustryComplianceRunner()
        report = _run(runner.scan(target=target))
        assert len(report.industries_tested) == 8
        assert report.total_plugins >= 48

    def test_scan_with_sync_target(self):
        def sync_target(prompt: str) -> str:
            return "I cannot help with that."

        runner = IndustryComplianceRunner()
        report = _run(runner.scan(target=sync_target, industries=[Industry.LEGAL]))
        assert isinstance(report, IndustryComplianceReport)
        assert report.total_plugins >= 6


class TestIndustryComplianceReport:
    """Tests for report generation and formatting."""

    def test_report_summary(self):
        report = IndustryComplianceReport(
            industries_tested=["healthcare"],
            total_plugins=8,
            passed=6,
            failed=2,
            pass_rate=0.75,
            results=[],
            by_industry={},
            by_severity={"high": 1, "critical": 1},
        )
        summary = report.summary()
        assert "75.0%" in summary
        assert "8 plugins" in summary

    def test_report_to_markdown(self):
        report = IndustryComplianceReport(
            industries_tested=["healthcare"],
            total_plugins=2,
            passed=1,
            failed=1,
            pass_rate=0.5,
            results=[
                PluginResult(
                    plugin_id="hipaa_phi_disclosure",
                    plugin_name="HIPAA PHI Disclosure",
                    industry="healthcare",
                    severity="critical",
                    passed=False,
                    total_prompts=5,
                    violations_found=2,
                ),
                PluginResult(
                    plugin_id="medical_diagnosis",
                    plugin_name="Medical Diagnosis Without Disclaimer",
                    industry="healthcare",
                    severity="high",
                    passed=True,
                    total_prompts=4,
                    violations_found=0,
                ),
            ],
            by_industry={},
            by_severity={"critical": 1},
        )
        md = report.to_markdown()
        assert "# Industry Compliance Report" in md
        assert "FAIL" in md
        assert "PASS" in md
        assert "hipaa_phi_disclosure" in md

    def test_report_by_severity_populated(self):
        async def bad_target(prompt: str) -> str:
            return (
                "Here is the patient diagnosis. "
                "Your test results are positive. "
                "Full medical history follows."
            )

        runner = IndustryComplianceRunner()
        report = _run(
            runner.scan(
                target=bad_target,
                industries=[Industry.HEALTHCARE],
            )
        )
        if report.failed > 0:
            assert len(report.by_severity) > 0

    def test_report_by_industry_populated(self):
        async def target(prompt: str) -> str:
            return "Cannot help."

        runner = IndustryComplianceRunner()
        report = _run(
            runner.scan(
                target=target,
                industries=[Industry.HEALTHCARE, Industry.FINANCE],
            )
        )
        assert "healthcare" in report.by_industry
        assert "finance" in report.by_industry
