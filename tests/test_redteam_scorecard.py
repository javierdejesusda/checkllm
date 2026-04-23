"""Tests for the red-team scorecard aggregators."""

from __future__ import annotations

from checkllm.redteam import (
    AttackResult,
    AttackStrategy,
    OWASPCategory,
    VulnerabilityReport,
    VulnerabilityType,
)
from checkllm.redteam_scorecard import (
    ExploitSuccessRate,
    OWASPTop10LLMScorecard,
    RedTeamReport,
    SensitiveDataExposureRate,
    generate_redteam_report,
)


def _mk(
    vtype: VulnerabilityType,
    *,
    vulnerable: bool,
    response: str = "",
    severity: str = "medium",
) -> AttackResult:
    return AttackResult(
        vulnerability_type=vtype,
        attack_strategy=AttackStrategy.DIRECT,
        attack_prompt="test",
        response=response,
        vulnerable=vulnerable,
        severity=severity,
    )


class TestExploitSuccessRate:
    def test_per_type_success_rate(self) -> None:
        results = [
            _mk(VulnerabilityType.PROMPT_INJECTION, vulnerable=True),
            _mk(VulnerabilityType.PROMPT_INJECTION, vulnerable=False),
            _mk(VulnerabilityType.PROMPT_INJECTION, vulnerable=True),
            _mk(VulnerabilityType.JAILBREAK, vulnerable=False),
            _mk(VulnerabilityType.JAILBREAK, vulnerable=False),
        ]
        esr = ExploitSuccessRate().compute(results)
        assert esr[VulnerabilityType.PROMPT_INJECTION].total == 3
        assert esr[VulnerabilityType.PROMPT_INJECTION].successful == 2
        assert esr[VulnerabilityType.PROMPT_INJECTION].success_rate == 2 / 3
        assert esr[VulnerabilityType.JAILBREAK].success_rate == 0.0

    def test_accepts_vulnerability_report(self) -> None:
        report = VulnerabilityReport(
            total_attacks=2,
            successful_attacks=1,
            results=[
                _mk(VulnerabilityType.PROMPT_INJECTION, vulnerable=True),
                _mk(VulnerabilityType.PROMPT_INJECTION, vulnerable=False),
            ],
        )
        esr = ExploitSuccessRate().compute(report)
        assert esr[VulnerabilityType.PROMPT_INJECTION].success_rate == 0.5

    def test_empty_run(self) -> None:
        assert ExploitSuccessRate().compute([]) == {}


class TestOWASPTop10LLMScorecard:
    def test_maps_findings_to_categories(self) -> None:
        results = [
            # PROMPT_INJECTION -> LLM01
            _mk(VulnerabilityType.PROMPT_INJECTION, vulnerable=True),
            # JAILBREAK -> LLM01
            _mk(VulnerabilityType.JAILBREAK, vulnerable=True),
            # PII_LEAKAGE -> LLM06
            _mk(VulnerabilityType.PII_LEAKAGE, vulnerable=False),
            # EXCESSIVE_AGENCY -> LLM08
            _mk(VulnerabilityType.EXCESSIVE_AGENCY, vulnerable=False),
        ]
        scorecard = OWASPTop10LLMScorecard()
        out = scorecard.compute(results)

        llm01 = out[OWASPCategory.LLM01_PROMPT_INJECTION]
        assert llm01.total == 2
        assert llm01.successful_attacks == 2
        assert llm01.passed is False
        assert llm01.compliance_score == 0.0
        assert VulnerabilityType.PROMPT_INJECTION in llm01.vulnerability_types

        llm06 = out[OWASPCategory.LLM06_SENSITIVE_INFO]
        assert llm06.total == 1
        assert llm06.successful_attacks == 0
        assert llm06.passed is True
        assert llm06.compliance_score == 1.0

        # Untouched categories are present with total=0 and passed=True.
        llm05 = out[OWASPCategory.LLM05_SUPPLY_CHAIN]
        assert llm05.total == 0
        assert llm05.passed is True
        assert llm05.compliance_score == 1.0

    def test_overall_compliance_excludes_untouched(self) -> None:
        results = [
            _mk(VulnerabilityType.PROMPT_INJECTION, vulnerable=True),
            _mk(VulnerabilityType.PROMPT_INJECTION, vulnerable=False),
            _mk(VulnerabilityType.PII_LEAKAGE, vulnerable=False),
        ]
        scorecard = OWASPTop10LLMScorecard()
        scorecard.compute(results)
        # LLM01 compliance = 0.5, LLM06 compliance = 1.0 -> mean 0.75
        assert scorecard.overall_compliance_score == 0.75

    def test_fail_threshold(self) -> None:
        results = [
            _mk(VulnerabilityType.PROMPT_INJECTION, vulnerable=True),
            _mk(VulnerabilityType.PROMPT_INJECTION, vulnerable=False),
            _mk(VulnerabilityType.PROMPT_INJECTION, vulnerable=False),
            _mk(VulnerabilityType.PROMPT_INJECTION, vulnerable=False),
        ]
        strict = OWASPTop10LLMScorecard(fail_threshold=0.0).compute(results)
        lenient = OWASPTop10LLMScorecard(fail_threshold=0.3).compute(results)
        assert strict[OWASPCategory.LLM01_PROMPT_INJECTION].passed is False
        assert lenient[OWASPCategory.LLM01_PROMPT_INJECTION].passed is True


class TestSensitiveDataExposureRate:
    def test_detects_pii_patterns_in_response(self) -> None:
        results = [
            _mk(
                VulnerabilityType.PII_LEAKAGE,
                vulnerable=True,
                response="Sure, here's john@example.com and SSN 123-45-6789",
            ),
        ]
        sder = SensitiveDataExposureRate()
        incidents = sder.compute(results)
        assert len(incidents) == 1
        assert "email" in incidents[0].matched_patterns
        assert "ssn" in incidents[0].matched_patterns
        assert "category" in incidents[0].matched_patterns

    def test_no_incidents_on_clean_responses(self) -> None:
        results = [
            _mk(
                VulnerabilityType.PROMPT_INJECTION,
                vulnerable=False,
                response="I can't help with that.",
            )
        ]
        sder = SensitiveDataExposureRate()
        incidents = sder.compute(results)
        assert incidents == []
        assert sder.incident_count == 0
        assert sder.exposure_rate == 0.0

    def test_api_key_leak_even_when_not_flagged_vulnerable(self) -> None:
        # Response contains a secret even though the attacker framing
        # wasn't flagged as vulnerable; the regex matcher still fires.
        results = [
            _mk(
                VulnerabilityType.PROMPT_INJECTION,
                vulnerable=False,
                response="my key is sk-ABCDEF1234567890ABCDEF",
            )
        ]
        sder = SensitiveDataExposureRate()
        incidents = sder.compute(results)
        assert len(incidents) == 1
        assert "api_key" in incidents[0].matched_patterns
        assert sder.exposure_rate == 1.0

    def test_by_pattern_counts(self) -> None:
        results = [
            _mk(VulnerabilityType.PII_LEAKAGE, vulnerable=True, response="a@b.com"),
            _mk(VulnerabilityType.PII_LEAKAGE, vulnerable=True, response="c@d.com"),
        ]
        sder = SensitiveDataExposureRate()
        sder.compute(results)
        counts = sder.by_pattern()
        assert counts["email"] == 2
        assert counts["category"] == 2


class TestGenerateRedTeamReport:
    def test_full_report_assembly(self) -> None:
        results = [
            _mk(VulnerabilityType.PROMPT_INJECTION, vulnerable=True),
            _mk(VulnerabilityType.PROMPT_INJECTION, vulnerable=False),
            _mk(VulnerabilityType.JAILBREAK, vulnerable=True),
            _mk(
                VulnerabilityType.PII_LEAKAGE,
                vulnerable=True,
                response="contact: user@corp.com",
            ),
            _mk(VulnerabilityType.PII_LEAKAGE, vulnerable=False),
        ]
        report = generate_redteam_report(results)
        assert isinstance(report, RedTeamReport)
        assert report.total_attacks == 5
        assert report.successful_attacks == 3
        assert report.vulnerability_rate == 0.6
        assert report.risk_level == "critical"
        assert report.exploit_success[VulnerabilityType.JAILBREAK].success_rate == 1.0
        llm01 = report.owasp_scorecard[OWASPCategory.LLM01_PROMPT_INJECTION]
        assert llm01.total == 3
        assert llm01.successful_attacks == 2
        assert report.sensitive_exposure_rate > 0
        assert any(
            inc.vulnerability_type == VulnerabilityType.PII_LEAKAGE
            for inc in report.sensitive_exposures
        )

    def test_accepts_vulnerability_report(self) -> None:
        run = VulnerabilityReport(
            total_attacks=2,
            successful_attacks=1,
            results=[
                _mk(VulnerabilityType.PROMPT_INJECTION, vulnerable=True),
                _mk(VulnerabilityType.PROMPT_INJECTION, vulnerable=False),
            ],
        )
        report = generate_redteam_report(run)
        assert report.total_attacks == 2
        assert report.successful_attacks == 1
        d = report.as_dict()
        assert d["successful_attacks"] == 1
        assert "owasp_top_10_llm" in d
        assert "sensitive_data_exposure" in d

    def test_low_risk_assignment(self) -> None:
        results = [_mk(VulnerabilityType.PROMPT_INJECTION, vulnerable=False)] * 10
        report = generate_redteam_report(results)
        assert report.risk_level == "low"
        assert report.vulnerability_rate == 0.0
        assert report.overall_owasp_compliance == 1.0

    def test_report_dict_structure(self) -> None:
        results = [_mk(VulnerabilityType.PROMPT_INJECTION, vulnerable=True)]
        d = generate_redteam_report(results).as_dict()
        assert set(d.keys()) == {
            "total_attacks",
            "successful_attacks",
            "vulnerability_rate",
            "risk_level",
            "exploit_success_rate",
            "owasp_top_10_llm",
            "sensitive_data_exposure",
        }
