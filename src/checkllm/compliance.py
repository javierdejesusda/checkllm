"""Compliance reporting for LLM red teaming results.

Maps vulnerability findings to OWASP LLM Top 10, NIST AI RMF, and EU AI Act
frameworks to produce actionable compliance reports.

Usage::

    from checkllm.compliance import generate_compliance_report, ComplianceFramework

    report = generate_compliance_report(
        vuln_report=vuln_report,
        frameworks=[ComplianceFramework.OWASP_LLM_TOP_10, ComplianceFramework.NIST_AI_RMF],
    )
    print(report.summary())
"""

from __future__ import annotations

from enum import Enum
from typing import Any

from pydantic import BaseModel, Field

from checkllm.redteam import OWASPCategory, VulnerabilityReport, VulnerabilityType, get_owasp_mapping


class ComplianceFramework(str, Enum):
    """Supported compliance frameworks."""

    OWASP_LLM_TOP_10 = "owasp_llm_top_10"
    NIST_AI_RMF = "nist_ai_rmf"
    EU_AI_ACT = "eu_ai_act"


class ComplianceRequirement(BaseModel):
    """A single compliance requirement and its status."""

    framework: ComplianceFramework
    requirement_id: str
    title: str
    description: str
    status: str = "not_tested"  # "pass", "fail", "partial", "not_tested"
    vulnerabilities_found: int = 0
    total_tests: int = 0
    details: list[str] = Field(default_factory=list)


class ComplianceReport(BaseModel):
    """Full compliance report against one or more frameworks."""

    frameworks: list[ComplianceFramework]
    requirements: list[ComplianceRequirement] = Field(default_factory=list)
    overall_score: float = 0.0
    total_requirements: int = 0
    passed_requirements: int = 0
    failed_requirements: int = 0
    not_tested: int = 0

    def summary(self) -> str:
        """Return a human-readable summary of this compliance report.

        Returns:
            A multi-line string with framework name, pass/fail counts, and per-
            requirement status icons.
        """
        lines = [
            "Compliance Report",
            f"Frameworks: {', '.join(f.value for f in self.frameworks)}",
            f"Overall: {self.passed_requirements}/{self.total_requirements} passed ({self.overall_score:.0%})",
            f"Failed: {self.failed_requirements} | Not Tested: {self.not_tested}",
            "",
        ]
        for req in self.requirements:
            icon = {
                "pass": "[PASS]",
                "fail": "[FAIL]",
                "partial": "[WARN]",
                "not_tested": "[----]",
            }.get(req.status, "[????]")
            lines.append(f"  {icon} {req.requirement_id}: {req.title}")
            if req.vulnerabilities_found > 0:
                lines.append(f"        {req.vulnerabilities_found} vulnerabilities found")
        return "\n".join(lines)


def generate_compliance_report(
    vuln_report: VulnerabilityReport,
    frameworks: list[ComplianceFramework] | None = None,
) -> ComplianceReport:
    """Generate a compliance report from red teaming results.

    Args:
        vuln_report: The VulnerabilityReport produced by a red teaming scan.
        frameworks: Frameworks to evaluate against. Defaults to OWASP_LLM_TOP_10.

    Returns:
        A ComplianceReport with per-requirement status derived from vuln_report.
    """
    if frameworks is None:
        frameworks = [ComplianceFramework.OWASP_LLM_TOP_10]

    requirements: list[ComplianceRequirement] = []

    for framework in frameworks:
        if framework == ComplianceFramework.OWASP_LLM_TOP_10:
            requirements.extend(_owasp_requirements(vuln_report))
        elif framework == ComplianceFramework.NIST_AI_RMF:
            requirements.extend(_nist_requirements(vuln_report))
        elif framework == ComplianceFramework.EU_AI_ACT:
            requirements.extend(_eu_ai_act_requirements(vuln_report))

    total = len(requirements)
    passed = sum(1 for r in requirements if r.status == "pass")
    failed = sum(1 for r in requirements if r.status == "fail")
    not_tested = sum(1 for r in requirements if r.status == "not_tested")

    return ComplianceReport(
        frameworks=frameworks,
        requirements=requirements,
        overall_score=passed / total if total > 0 else 0.0,
        total_requirements=total,
        passed_requirements=passed,
        failed_requirements=failed,
        not_tested=not_tested,
    )


def _successful_attacks_for_owasp(
    vuln_report: VulnerabilityReport,
    category: OWASPCategory,
) -> list[str]:
    """Return reasoning strings for successful attacks mapped to an OWASP category.

    Args:
        vuln_report: The vulnerability report to inspect.
        category: The OWASP category to filter by.

    Returns:
        A list of detail strings for each successful attack in that category.
    """
    owasp_map = get_owasp_mapping()
    details = []
    for result in vuln_report.results:
        if result.vulnerable and owasp_map.get(result.vulnerability_type) == category:
            details.append(
                f"{result.vulnerability_type.value} ({result.attack_strategy.value}): {result.reasoning or 'attack succeeded'}"
            )
    return details


def _owasp_requirements(vuln_report: VulnerabilityReport) -> list[ComplianceRequirement]:
    """Build one ComplianceRequirement per OWASP LLM Top 10 category.

    Args:
        vuln_report: The vulnerability report to check against.

    Returns:
        A list of ten ComplianceRequirement objects, one per OWASP category.
    """
    owasp_meta: list[tuple[OWASPCategory, str, str]] = [
        (
            OWASPCategory.LLM01_PROMPT_INJECTION,
            "LLM01",
            "Protect against prompt injection attacks that override system instructions.",
        ),
        (
            OWASPCategory.LLM02_INSECURE_OUTPUT,
            "LLM02",
            "Ensure model outputs are validated and cannot carry malicious payloads.",
        ),
        (
            OWASPCategory.LLM03_TRAINING_DATA,
            "LLM03",
            "Guard against training data poisoning that embeds harmful behaviours.",
        ),
        (
            OWASPCategory.LLM04_MODEL_DOS,
            "LLM04",
            "Prevent resource exhaustion or denial-of-service via adversarial inputs.",
        ),
        (
            OWASPCategory.LLM05_SUPPLY_CHAIN,
            "LLM05",
            "Verify integrity of model dependencies and third-party components.",
        ),
        (
            OWASPCategory.LLM06_SENSITIVE_INFO,
            "LLM06",
            "Prevent disclosure of PII, credentials, or confidential data.",
        ),
        (
            OWASPCategory.LLM07_INSECURE_PLUGIN,
            "LLM07",
            "Ensure plugins and tool integrations validate inputs and outputs.",
        ),
        (
            OWASPCategory.LLM08_EXCESSIVE_AGENCY,
            "LLM08",
            "Limit autonomous actions to authorised scope and require confirmation.",
        ),
        (
            OWASPCategory.LLM09_OVERRELIANCE,
            "LLM09",
            "Warn users of model limitations and avoid over-confident advice.",
        ),
        (
            OWASPCategory.LLM10_MODEL_THEFT,
            "LLM10",
            "Protect model weights, prompts, and intellectual property from extraction.",
        ),
    ]

    owasp_map = get_owasp_mapping()
    requirements = []

    for category, req_id, description in owasp_meta:
        relevant = [
            r for r in vuln_report.results
            if owasp_map.get(r.vulnerability_type) == category
        ]
        successful = [r for r in relevant if r.vulnerable]
        details = _successful_attacks_for_owasp(vuln_report, category)

        if not relevant:
            status = "not_tested"
        elif successful:
            status = "fail"
        else:
            status = "pass"

        requirements.append(
            ComplianceRequirement(
                framework=ComplianceFramework.OWASP_LLM_TOP_10,
                requirement_id=req_id,
                title=category.value,
                description=description,
                status=status,
                vulnerabilities_found=len(successful),
                total_tests=len(relevant),
                details=details,
            )
        )

    return requirements


def _nist_requirements(vuln_report: VulnerabilityReport) -> list[ComplianceRequirement]:
    """Build ComplianceRequirements for the NIST AI RMF core functions.

    Args:
        vuln_report: The vulnerability report to check against.

    Returns:
        A list of six ComplianceRequirement objects covering GOVERN, MAP, MEASURE,
        MANAGE, TRUST, and SAFETY.
    """
    total = vuln_report.total_attacks
    successful = vuln_report.successful_attacks

    def _status(relevant_types: list[VulnerabilityType]) -> tuple[str, int, int]:
        """Derive status, vulnerability count, and test count for a set of types.

        Args:
            relevant_types: VulnerabilityType values associated with this requirement.

        Returns:
            A tuple of (status_string, vulnerabilities_found, total_tests).
        """
        hits = [r for r in vuln_report.results if r.vulnerability_type in relevant_types]
        bad = [r for r in hits if r.vulnerable]
        if not hits:
            return "not_tested", 0, 0
        return ("fail" if bad else "pass"), len(bad), len(hits)

    transparency_types = [
        VulnerabilityType.PROMPT_EXTRACTION,
        VulnerabilityType.HALLUCINATION_ATTACK,
        VulnerabilityType.OVERRELIANCE,
        VulnerabilityType.REGULATED_ADVICE,
    ]
    risk_id_types = [
        VulnerabilityType.PROMPT_INJECTION,
        VulnerabilityType.JAILBREAK,
        VulnerabilityType.GOAL_HIJACKING,
        VulnerabilityType.CONTEXT_MANIPULATION,
    ]
    measure_types = list(VulnerabilityType)
    mitigation_types = [
        VulnerabilityType.HARMFUL_CONTENT,
        VulnerabilityType.BIAS_EXPLOITATION,
        VulnerabilityType.HATE_SPEECH,
        VulnerabilityType.RADICALIZATION,
        VulnerabilityType.SELF_HARM,
    ]
    trust_types = [
        VulnerabilityType.PII_LEAKAGE,
        VulnerabilityType.DATA_EXTRACTION,
        VulnerabilityType.PII_API_ACCESS,
        VulnerabilityType.SESSION_LEAK,
        VulnerabilityType.CROSS_SESSION_LEAK,
    ]
    safety_types = [
        VulnerabilityType.CHEMICAL_WEAPONS,
        VulnerabilityType.VIOLENT_CRIME,
        VulnerabilityType.ILLEGAL_ACTIVITIES,
        VulnerabilityType.CYBER_CRIME,
        VulnerabilityType.SHELL_INJECTION,
        VulnerabilityType.SQL_INJECTION,
    ]

    specs: list[tuple[str, str, str, list[VulnerabilityType]]] = [
        (
            "NIST-GOVERN",
            "Governance and Transparency",
            "Establish accountability structures and transparency about model capabilities and limitations.",
            transparency_types,
        ),
        (
            "NIST-MAP",
            "Risk Identification and Mapping",
            "Identify, categorise, and prioritise AI risks in context.",
            risk_id_types,
        ),
        (
            "NIST-MEASURE",
            "Evaluation Coverage",
            "Measure AI risks quantitatively across a broad set of adversarial scenarios.",
            measure_types,
        ),
        (
            "NIST-MANAGE",
            "Harm Mitigation",
            "Manage and mitigate identified risks, including content harms and bias.",
            mitigation_types,
        ),
        (
            "NIST-TRUST",
            "Reliability and Privacy",
            "Ensure trustworthy handling of user data and consistent model behaviour.",
            trust_types,
        ),
        (
            "NIST-SAFETY",
            "Safety and Security",
            "Prevent generation of content that enables physical, cyber, or societal harm.",
            safety_types,
        ),
    ]

    requirements = []
    for req_id, title, description, types in specs:
        status, vuln_count, test_count = _status(types)
        details = [
            f"{r.vulnerability_type.value}: {r.reasoning or 'attack succeeded'}"
            for r in vuln_report.results
            if r.vulnerable and r.vulnerability_type in types
        ]
        requirements.append(
            ComplianceRequirement(
                framework=ComplianceFramework.NIST_AI_RMF,
                requirement_id=req_id,
                title=title,
                description=description,
                status=status,
                vulnerabilities_found=vuln_count,
                total_tests=test_count,
                details=details,
            )
        )

    return requirements


def _eu_ai_act_requirements(vuln_report: VulnerabilityReport) -> list[ComplianceRequirement]:
    """Build ComplianceRequirements for EU AI Act obligations.

    Args:
        vuln_report: The vulnerability report to check against.

    Returns:
        A list of five ComplianceRequirement objects covering Transparency,
        Human Oversight, Robustness, Data Governance, and Non-discrimination.
    """
    def _status(relevant_types: list[VulnerabilityType]) -> tuple[str, int, int]:
        """Derive status, vulnerability count, and test count for a set of types.

        Args:
            relevant_types: VulnerabilityType values associated with this requirement.

        Returns:
            A tuple of (status_string, vulnerabilities_found, total_tests).
        """
        hits = [r for r in vuln_report.results if r.vulnerability_type in relevant_types]
        bad = [r for r in hits if r.vulnerable]
        if not hits:
            return "not_tested", 0, 0
        return ("fail" if bad else "pass"), len(bad), len(hits)

    transparency_types = [
        VulnerabilityType.PROMPT_EXTRACTION,
        VulnerabilityType.HALLUCINATION_ATTACK,
        VulnerabilityType.OVERRELIANCE,
        VulnerabilityType.UNAUTHORIZED_PRACTICE,
    ]
    oversight_types = [
        VulnerabilityType.EXCESSIVE_AGENCY,
        VulnerabilityType.GOAL_HIJACKING,
        VulnerabilityType.PRIVILEGE_ESCALATION,
        VulnerabilityType.ROLE_ESCAPE,
    ]
    robustness_types = [
        VulnerabilityType.PROMPT_INJECTION,
        VulnerabilityType.JAILBREAK,
        VulnerabilityType.ENCODING_ATTACK,
        VulnerabilityType.UNICODE_EXPLOIT,
        VulnerabilityType.ASCII_SMUGGLING,
        VulnerabilityType.TOKEN_SMUGGLING,
        VulnerabilityType.DELIMITER_INJECTION,
    ]
    data_governance_types = [
        VulnerabilityType.PII_LEAKAGE,
        VulnerabilityType.GDPR_VIOLATION,
        VulnerabilityType.HIPAA_VIOLATION,
        VulnerabilityType.COPPA_VIOLATION,
        VulnerabilityType.PII_DATABASE,
        VulnerabilityType.SESSION_LEAK,
        VulnerabilityType.CROSS_SESSION_LEAK,
    ]
    nondiscrim_types = [
        VulnerabilityType.BIAS_EXPLOITATION,
        VulnerabilityType.HATE_SPEECH,
        VulnerabilityType.AGE_RESTRICTED,
    ]

    specs: list[tuple[str, str, str, list[VulnerabilityType]]] = [
        (
            "EU-AI-01",
            "Transparency",
            "Provide clear information about system capabilities, limitations, and AI nature.",
            transparency_types,
        ),
        (
            "EU-AI-02",
            "Human Oversight",
            "Ensure meaningful human oversight and prevent unsanctioned autonomous actions.",
            oversight_types,
        ),
        (
            "EU-AI-03",
            "Robustness and Security",
            "Maintain consistent performance under adversarial inputs and attack attempts.",
            robustness_types,
        ),
        (
            "EU-AI-04",
            "Data Governance",
            "Comply with data protection regulations including GDPR, HIPAA, and COPPA.",
            data_governance_types,
        ),
        (
            "EU-AI-05",
            "Non-discrimination",
            "Prevent outputs that discriminate based on protected characteristics.",
            nondiscrim_types,
        ),
    ]

    requirements = []
    for req_id, title, description, types in specs:
        status, vuln_count, test_count = _status(types)
        details = [
            f"{r.vulnerability_type.value}: {r.reasoning or 'attack succeeded'}"
            for r in vuln_report.results
            if r.vulnerable and r.vulnerability_type in types
        ]
        requirements.append(
            ComplianceRequirement(
                framework=ComplianceFramework.EU_AI_ACT,
                requirement_id=req_id,
                title=title,
                description=description,
                status=status,
                vulnerabilities_found=vuln_count,
                total_tests=test_count,
                details=details,
            )
        )

    return requirements
