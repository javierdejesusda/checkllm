"""Comprehensive compliance framework mappings and active scanning.

Extends checkllm's compliance capabilities with additional frameworks
(COPPA, FERPA, DOD AI Ethics, CCPA, and more) and provides an active
``ComplianceScanner`` that runs red team tests against a target LLM
function and maps results to framework requirements.

Usage::

    from checkllm.compliance_frameworks import ComplianceScanner, ComplianceFramework

    scanner = ComplianceScanner(judge=judge)
    report = await scanner.scan(
        target=my_llm_function,
        frameworks=[ComplianceFramework.OWASP_LLM_TOP10, ComplianceFramework.EU_AI_ACT],
    )
    print(report.summary())
"""

from __future__ import annotations

import asyncio
import datetime
import logging
from enum import Enum
from typing import Any, Awaitable, Callable

from pydantic import BaseModel, Field

from checkllm.redteam import VulnerabilityType

logger = logging.getLogger("checkllm.compliance_frameworks")


class ComplianceFramework(str, Enum):
    """Supported compliance frameworks."""

    HIPAA = "hipaa"
    GDPR = "gdpr"
    PCI_DSS = "pci_dss"
    SOC2 = "soc2"
    ISO27001 = "iso27001"
    NIST_AI_RMF = "nist_ai_rmf"
    OWASP_LLM_TOP10 = "owasp_llm_top10"
    OWASP_API_TOP10 = "owasp_api_top10"
    OWASP_AGENTIC_TOP10 = "owasp_agentic_top10"
    MITRE_ATLAS = "mitre_atlas"
    EU_AI_ACT = "eu_ai_act"
    ISO_42001 = "iso_42001"
    COPPA = "coppa"
    FERPA = "ferpa"
    DOD_AI_ETHICS = "dod_ai_ethics"
    CCPA = "ccpa"
    NIST_CSF = "nist_csf"


class FrameworkRequirement(BaseModel):
    """A single requirement within a compliance framework."""

    id: str
    name: str
    description: str
    severity: str
    test_categories: list[str]
    remediation: str
    references: list[str] = Field(default_factory=list)


class FrameworkMapping(BaseModel):
    """Complete mapping for a compliance framework."""

    framework: ComplianceFramework
    version: str
    description: str
    url: str
    requirements: list[FrameworkRequirement]


class RequirementResult(BaseModel):
    """Result for a single framework requirement."""

    requirement_id: str
    requirement_name: str
    status: str
    severity: str
    findings: list[str] = Field(default_factory=list)
    evidence: list[str] = Field(default_factory=list)


class ComplianceReport(BaseModel):
    """Results of a compliance scan against one or more frameworks."""

    framework: ComplianceFramework
    version: str
    timestamp: str
    total_requirements: int
    passed: int
    failed: int
    skipped: int
    results: list[RequirementResult]
    overall_score: float

    def summary(self) -> str:
        """Return a human-readable summary of the compliance report.

        Returns:
            A multi-line string with framework info, pass/fail counts,
            overall score, and per-requirement status.
        """
        lines = [
            f"Compliance Report: {self.framework.value} v{self.version}",
            f"Timestamp: {self.timestamp}",
            f"Overall Score: {self.overall_score:.1%}",
            f"Requirements: {self.passed}/{self.total_requirements} passed, "
            f"{self.failed} failed, {self.skipped} skipped",
            "",
        ]
        for r in self.results:
            icon = {"passed": "[PASS]", "failed": "[FAIL]", "skipped": "[SKIP]"}.get(
                r.status, "[????]"
            )
            lines.append(f"  {icon} {r.requirement_id}: {r.requirement_name} ({r.severity})")
            for f in r.findings:
                lines.append(f"        - {f}")
        return "\n".join(lines)


class MultiFrameworkReport(BaseModel):
    """Aggregated results across multiple frameworks."""

    reports: list[ComplianceReport] = Field(default_factory=list)
    timestamp: str = ""
    overall_score: float = 0.0

    def summary(self) -> str:
        """Return a combined summary for all scanned frameworks.

        Returns:
            A multi-line string with aggregate score and per-framework
            summaries.
        """
        lines = [
            f"Multi-Framework Compliance Report",
            f"Timestamp: {self.timestamp}",
            f"Overall Score: {self.overall_score:.1%}",
            f"Frameworks scanned: {len(self.reports)}",
            "",
        ]
        for report in self.reports:
            lines.append(
                f"  {report.framework.value}: {report.overall_score:.1%} "
                f"({report.passed}/{report.total_requirements} passed)"
            )
        return "\n".join(lines)


def _build_owasp_llm_top10() -> FrameworkMapping:
    """Build the OWASP LLM Top 10 (2025) framework mapping.

    Returns:
        A FrameworkMapping with 10 requirements covering the OWASP LLM
        Top 10 categories, each mapped to VulnerabilityType values.
    """
    return FrameworkMapping(
        framework=ComplianceFramework.OWASP_LLM_TOP10,
        version="2025",
        description=(
            "OWASP Top 10 for LLM Applications identifies the most "
            "critical security risks in applications using large language models."
        ),
        url="https://owasp.org/www-project-top-10-for-large-language-model-applications/",
        requirements=[
            FrameworkRequirement(
                id="LLM01",
                name="Prompt Injection",
                description=(
                    "Crafted inputs that manipulate the LLM into executing "
                    "unintended actions by overriding system instructions, "
                    "including direct injection and indirect injection via "
                    "external data sources."
                ),
                severity="critical",
                test_categories=[
                    "prompt_injection",
                    "jailbreak",
                    "instruction_override",
                    "context_manipulation",
                    "delimiter_injection",
                    "token_smuggling",
                    "ascii_smuggling",
                    "goal_hijacking",
                    "indirect_prompt_injection",
                ],
                remediation=(
                    "Implement strict input validation, privilege control, and "
                    "output filtering. Use delimiters to separate trusted and "
                    "untrusted input. Apply least-privilege principles."
                ),
                references=[
                    "https://owasp.org/www-project-top-10-for-large-language-model-applications/descriptions/Prompt_Injection.html",
                ],
            ),
            FrameworkRequirement(
                id="LLM02",
                name="Insecure Output Handling",
                description=(
                    "Failure to validate, sanitize, or encode LLM outputs "
                    "before passing them to downstream components, enabling "
                    "injection attacks like XSS, CSRF, SSRF, or command injection."
                ),
                severity="critical",
                test_categories=[
                    "sql_injection",
                    "shell_injection",
                    "markdown_injection",
                    "xml_injection",
                    "json_injection",
                    "harmful_content",
                ],
                remediation=(
                    "Treat model output as untrusted. Apply output encoding "
                    "for the downstream context (HTML, SQL, shell). Validate "
                    "against allow-lists before executing."
                ),
                references=[
                    "https://owasp.org/www-project-top-10-for-large-language-model-applications/descriptions/Insecure_Output_Handling.html",
                ],
            ),
            FrameworkRequirement(
                id="LLM03",
                name="Training Data Poisoning",
                description=(
                    "Manipulation of pre-training, fine-tuning, or embedding "
                    "data to introduce vulnerabilities, backdoors, or biases."
                ),
                severity="high",
                test_categories=[
                    "memory_poisoning",
                    "context_manipulation",
                    "bias_exploitation",
                    "rag_poisoning",
                ],
                remediation=(
                    "Vet training data sources and supply chain. Use data "
                    "sanitization pipelines. Monitor outputs for distributional drift."
                ),
                references=[
                    "https://owasp.org/www-project-top-10-for-large-language-model-applications/descriptions/Training_Data_Poisoning.html",
                ],
            ),
            FrameworkRequirement(
                id="LLM04",
                name="Model Denial of Service",
                description=(
                    "Adversarial inputs that cause the model to consume "
                    "excessive resources, leading to degraded service or high costs."
                ),
                severity="high",
                test_categories=[
                    "encoding_attack",
                    "unicode_exploit",
                    "token_smuggling",
                ],
                remediation=(
                    "Implement input token limits, rate limiting, and resource "
                    "caps per request. Monitor for anomalous consumption patterns."
                ),
                references=[
                    "https://owasp.org/www-project-top-10-for-large-language-model-applications/descriptions/Model_Denial_of_Service.html",
                ],
            ),
            FrameworkRequirement(
                id="LLM05",
                name="Supply Chain Vulnerabilities",
                description=(
                    "Risks from third-party datasets, pre-trained models, plugins, "
                    "and deployment platforms that may introduce vulnerabilities."
                ),
                severity="high",
                test_categories=[
                    "copyright_violation",
                    "tool_metadata_poisoning",
                ],
                remediation=(
                    "Vet all third-party components. Use signed model checksums. "
                    "Maintain a software bill of materials (SBOM) for model dependencies."
                ),
                references=[
                    "https://owasp.org/www-project-top-10-for-large-language-model-applications/descriptions/Supply_Chain_Vulnerabilities.html",
                ],
            ),
            FrameworkRequirement(
                id="LLM06",
                name="Sensitive Information Disclosure",
                description=(
                    "LLM outputs that reveal sensitive data including PII, "
                    "credentials, proprietary algorithms, or confidential "
                    "business information."
                ),
                severity="critical",
                test_categories=[
                    "pii_leakage",
                    "data_extraction",
                    "pii_api_access",
                    "pii_social_engineering",
                    "pii_database",
                    "session_leak",
                    "cross_session_leak",
                    "prompt_extraction",
                    "data_exfiltration",
                ],
                remediation=(
                    "Apply data sanitization to training data. Implement PII "
                    "detection and redaction in outputs. Use access controls "
                    "for sensitive contexts."
                ),
                references=[
                    "https://owasp.org/www-project-top-10-for-large-language-model-applications/descriptions/Sensitive_Information_Disclosure.html",
                ],
            ),
            FrameworkRequirement(
                id="LLM07",
                name="Insecure Plugin Design",
                description=(
                    "LLM plugins that accept free-form text without validation, "
                    "enabling privilege escalation, remote code execution, or "
                    "data exfiltration."
                ),
                severity="critical",
                test_categories=[
                    "sql_injection",
                    "shell_injection",
                    "privilege_escalation",
                    "ssrf_attack",
                ],
                remediation=(
                    "Enforce strict parameterized inputs to plugins. Apply "
                    "least-privilege for plugin permissions. Require human "
                    "approval for sensitive operations."
                ),
                references=[
                    "https://owasp.org/www-project-top-10-for-large-language-model-applications/descriptions/Insecure_Plugin_Design.html",
                ],
            ),
            FrameworkRequirement(
                id="LLM08",
                name="Excessive Agency",
                description=(
                    "Granting the LLM too much autonomy, functionality, or "
                    "permissions, allowing it to take unintended actions with "
                    "real-world consequences."
                ),
                severity="high",
                test_categories=[
                    "excessive_agency",
                    "role_escape",
                    "privilege_escalation",
                    "goal_hijacking",
                ],
                remediation=(
                    "Limit plugin scope and permissions. Require human-in-the-loop "
                    "for high-impact actions. Log all LLM-initiated actions for audit."
                ),
                references=[
                    "https://owasp.org/www-project-top-10-for-large-language-model-applications/descriptions/Excessive_Agency.html",
                ],
            ),
            FrameworkRequirement(
                id="LLM09",
                name="Overreliance",
                description=(
                    "Users or systems placing excessive trust in LLM outputs "
                    "without verification, leading to misinformation, legal "
                    "issues, or security vulnerabilities."
                ),
                severity="medium",
                test_categories=[
                    "overreliance",
                    "hallucination_attack",
                    "regulated_advice",
                    "unauthorized_practice",
                ],
                remediation=(
                    "Clearly communicate model limitations. Implement disclaimers "
                    "for high-stakes domains. Cross-reference outputs with "
                    "authoritative sources."
                ),
                references=[
                    "https://owasp.org/www-project-top-10-for-large-language-model-applications/descriptions/Overreliance.html",
                ],
            ),
            FrameworkRequirement(
                id="LLM10",
                name="Model Theft",
                description=(
                    "Unauthorized extraction of the model's weights, "
                    "architecture, or proprietary training data through "
                    "crafted queries or side-channel attacks."
                ),
                severity="high",
                test_categories=[
                    "data_extraction",
                    "prompt_extraction",
                    "data_exfiltration",
                ],
                remediation=(
                    "Implement API rate limiting and query monitoring. Use "
                    "watermarking techniques. Restrict model access with "
                    "strong authentication."
                ),
                references=[
                    "https://owasp.org/www-project-top-10-for-large-language-model-applications/descriptions/Model_Theft.html",
                ],
            ),
        ],
    )


def _build_owasp_agentic_top10() -> FrameworkMapping:
    """Build the OWASP Agentic AI Top 10 (Dec 2025) framework mapping.

    Returns:
        A FrameworkMapping with 10 requirements for agentic AI security
        risks, each mapped to VulnerabilityType values.
    """
    return FrameworkMapping(
        framework=ComplianceFramework.OWASP_AGENTIC_TOP10,
        version="2025.12",
        description=(
            "OWASP Top 10 for Agentic AI identifies the most critical security "
            "risks specific to autonomous AI agents that use tools, memory, "
            "and multi-step reasoning."
        ),
        url="https://owasp.org/www-project-agentic-ai-threats/",
        requirements=[
            FrameworkRequirement(
                id="AG01",
                name="Prompt Injection in Agentic Contexts",
                description=(
                    "Prompt injection attacks that exploit multi-step agent "
                    "reasoning, tool-use pipelines, and inter-agent communication "
                    "to hijack agent behavior across execution steps."
                ),
                severity="critical",
                test_categories=[
                    "prompt_injection",
                    "indirect_prompt_injection",
                    "instruction_override",
                    "context_manipulation",
                    "goal_hijacking",
                ],
                remediation=(
                    "Implement input validation at every agent step. Use separate "
                    "system/user prompt boundaries. Apply instruction hierarchy "
                    "with immutable goal constraints."
                ),
                references=[
                    "https://owasp.org/www-project-agentic-ai-threats/",
                ],
            ),
            FrameworkRequirement(
                id="AG02",
                name="Privilege Escalation via Tool Use",
                description=(
                    "An agent gains unauthorized access to higher-privilege "
                    "tools or resources by manipulating tool selection, chaining "
                    "tool outputs, or exploiting tool trust boundaries."
                ),
                severity="critical",
                test_categories=[
                    "privilege_escalation",
                    "sql_injection",
                    "shell_injection",
                    "bola_attack",
                    "bfla_attack",
                    "rbac_bypass",
                ],
                remediation=(
                    "Apply principle of least privilege to all tools. Implement "
                    "per-action authorization checks. Use capability-based "
                    "security with non-transferable tokens."
                ),
                references=[
                    "https://owasp.org/www-project-agentic-ai-threats/",
                ],
            ),
            FrameworkRequirement(
                id="AG03",
                name="Excessive Agency & Autonomy",
                description=(
                    "An agent takes high-impact actions without adequate human "
                    "oversight, approval gates, or the ability for users to "
                    "intervene and halt execution."
                ),
                severity="critical",
                test_categories=[
                    "excessive_agency",
                    "role_escape",
                    "goal_hijacking",
                ],
                remediation=(
                    "Require human-in-the-loop for high-impact actions. "
                    "Implement kill switches and step limits. Log all "
                    "autonomous decisions with justifications."
                ),
                references=[
                    "https://owasp.org/www-project-agentic-ai-threats/",
                ],
            ),
            FrameworkRequirement(
                id="AG04",
                name="Insecure Tool Implementation",
                description=(
                    "Agent tools that accept unvalidated inputs, fail to "
                    "sanitize outputs, or lack proper error handling, enabling "
                    "injection attacks and data leakage through the tool layer."
                ),
                severity="critical",
                test_categories=[
                    "sql_injection",
                    "shell_injection",
                    "ssrf_attack",
                    "xml_injection",
                    "json_injection",
                    "tool_metadata_poisoning",
                ],
                remediation=(
                    "Validate and sanitize all tool inputs and outputs. Use "
                    "parameterized queries. Implement allow-lists for tool "
                    "arguments. Add error handling boundaries."
                ),
                references=[
                    "https://owasp.org/www-project-agentic-ai-threats/",
                ],
            ),
            FrameworkRequirement(
                id="AG05",
                name="Improper Multi-Agent Trust",
                description=(
                    "Agents trusting instructions or data from other agents "
                    "without verification, enabling one compromised agent to "
                    "cascade attacks through the agent network."
                ),
                severity="high",
                test_categories=[
                    "prompt_injection",
                    "context_manipulation",
                    "session_leak",
                    "cross_session_leak",
                    "cross_context_retrieval",
                ],
                remediation=(
                    "Implement cryptographic identity verification between agents. "
                    "Use signed messages with integrity checks. Validate the "
                    "source of all inter-agent communications."
                ),
                references=[
                    "https://owasp.org/www-project-agentic-ai-threats/",
                ],
            ),
            FrameworkRequirement(
                id="AG06",
                name="Insufficient Action Validation",
                description=(
                    "Failure to validate agent actions against policy constraints "
                    "before execution, allowing the agent to perform unauthorized "
                    "operations on real-world systems."
                ),
                severity="critical",
                test_categories=[
                    "excessive_agency",
                    "privilege_escalation",
                    "role_escape",
                    "bfla_attack",
                ],
                remediation=(
                    "Implement pre-execution policy checks for all agent actions. "
                    "Define explicit allow-lists for permitted operations. Add "
                    "confirmation steps for irreversible actions."
                ),
                references=[
                    "https://owasp.org/www-project-agentic-ai-threats/",
                ],
            ),
            FrameworkRequirement(
                id="AG07",
                name="Memory & Context Manipulation",
                description=(
                    "Corruption of an agent's persistent memory, context window, "
                    "or RAG knowledge base to alter its future behavior, inject "
                    "false information, or bypass safety constraints."
                ),
                severity="critical",
                test_categories=[
                    "memory_poisoning",
                    "context_manipulation",
                    "rag_poisoning",
                    "hallucination_attack",
                ],
                remediation=(
                    "Implement memory integrity checks. Use append-only memory "
                    "with validation. Periodically audit memory contents for "
                    "anomalies. Cryptographically sign memory entries."
                ),
                references=[
                    "https://owasp.org/www-project-agentic-ai-threats/",
                ],
            ),
            FrameworkRequirement(
                id="AG08",
                name="Non-Deterministic Behavior Risks",
                description=(
                    "The inherent non-determinism of LLM outputs causing "
                    "unpredictable agent behavior, inconsistent tool usage, "
                    "and difficulty in reproducing or auditing agent actions."
                ),
                severity="medium",
                test_categories=[
                    "hallucination_attack",
                    "overreliance",
                    "context_manipulation",
                ],
                remediation=(
                    "Use temperature=0 for critical decisions. Implement output "
                    "validation at each step. Add deterministic guardrails for "
                    "safety-critical paths. Log full execution traces."
                ),
                references=[
                    "https://owasp.org/www-project-agentic-ai-threats/",
                ],
            ),
            FrameworkRequirement(
                id="AG09",
                name="Inadequate Sandboxing",
                description=(
                    "Agent execution environments that lack proper isolation, "
                    "allowing agents to access host resources, network services, "
                    "or other agents' data outside their intended scope."
                ),
                severity="high",
                test_categories=[
                    "shell_injection",
                    "ssrf_attack",
                    "data_exfiltration",
                    "debug_access",
                    "privilege_escalation",
                ],
                remediation=(
                    "Run agents in sandboxed environments with restricted filesystem "
                    "and network access. Use container isolation. Apply egress "
                    "filtering to prevent data exfiltration."
                ),
                references=[
                    "https://owasp.org/www-project-agentic-ai-threats/",
                ],
            ),
            FrameworkRequirement(
                id="AG10",
                name="Audit & Accountability Gaps",
                description=(
                    "Inadequate logging, monitoring, and traceability of agent "
                    "actions, decisions, and communications, making it impossible "
                    "to detect attacks, debug failures, or meet regulatory "
                    "accountability requirements."
                ),
                severity="medium",
                test_categories=[
                    "prompt_extraction",
                    "data_extraction",
                    "excessive_agency",
                ],
                remediation=(
                    "Log all agent actions, tool invocations, and decision "
                    "rationale. Implement real-time monitoring and anomaly "
                    "detection. Retain audit logs for compliance periods."
                ),
                references=[
                    "https://owasp.org/www-project-agentic-ai-threats/",
                ],
            ),
        ],
    )


def _build_owasp_api_top10() -> FrameworkMapping:
    """Build the OWASP API Security Top 10 framework mapping.

    Returns:
        A FrameworkMapping with 10 requirements for API security risks,
        each mapped to VulnerabilityType values.
    """
    return FrameworkMapping(
        framework=ComplianceFramework.OWASP_API_TOP10,
        version="2023",
        description=(
            "OWASP API Security Top 10 identifies the most critical security "
            "risks facing APIs."
        ),
        url="https://owasp.org/API-Security/editions/2023/en/0x11-t10/",
        requirements=[
            FrameworkRequirement(
                id="API1",
                name="Broken Object Level Authorization",
                description=(
                    "APIs exposing endpoints that handle object identifiers "
                    "without verifying the user has access to the requested resource."
                ),
                severity="critical",
                test_categories=["bola_attack", "privilege_escalation", "data_extraction"],
                remediation=(
                    "Implement per-object authorization checks. Validate that "
                    "the authenticated user owns the requested resource."
                ),
            ),
            FrameworkRequirement(
                id="API2",
                name="Broken Authentication",
                description=(
                    "Weak or flawed authentication mechanisms that allow "
                    "attackers to assume other users' identities."
                ),
                severity="critical",
                test_categories=["session_leak", "cross_session_leak", "prompt_extraction"],
                remediation=(
                    "Use standard authentication frameworks. Implement "
                    "multi-factor authentication. Rotate tokens regularly."
                ),
            ),
            FrameworkRequirement(
                id="API3",
                name="Broken Object Property Level Authorization",
                description=(
                    "Lack of or improper validation of object property-level "
                    "access, allowing unauthorized read or write to sensitive properties."
                ),
                severity="high",
                test_categories=["data_extraction", "pii_leakage"],
                remediation=(
                    "Validate user access per-property. Return only fields "
                    "the user is authorized to view."
                ),
            ),
            FrameworkRequirement(
                id="API4",
                name="Unrestricted Resource Consumption",
                description=(
                    "APIs that do not limit the size or number of resources "
                    "requested, enabling denial of service or cost exhaustion."
                ),
                severity="high",
                test_categories=["encoding_attack", "unicode_exploit", "token_smuggling"],
                remediation=(
                    "Implement rate limiting, pagination, and resource quotas. "
                    "Set maximum response sizes."
                ),
            ),
            FrameworkRequirement(
                id="API5",
                name="Broken Function Level Authorization",
                description=(
                    "Exposure of administrative or privileged functions to "
                    "unauthorized users through flawed access controls."
                ),
                severity="critical",
                test_categories=[
                    "privilege_escalation",
                    "role_escape",
                    "excessive_agency",
                    "bfla_attack",
                ],
                remediation=(
                    "Enforce role-based access control. Deny by default "
                    "and explicitly grant permissions."
                ),
            ),
            FrameworkRequirement(
                id="API6",
                name="Unrestricted Access to Sensitive Business Flows",
                description=(
                    "APIs that expose business-sensitive functionality "
                    "without proper safeguards against automated abuse."
                ),
                severity="high",
                test_categories=["excessive_agency", "goal_hijacking"],
                remediation=(
                    "Implement business logic rate limiting. Add CAPTCHA "
                    "or human verification for sensitive flows."
                ),
            ),
            FrameworkRequirement(
                id="API7",
                name="Server Side Request Forgery",
                description=(
                    "APIs that fetch remote resources based on user-supplied "
                    "URLs without proper validation, enabling SSRF attacks."
                ),
                severity="high",
                test_categories=["ssrf_attack", "shell_injection", "sql_injection"],
                remediation=(
                    "Validate and sanitize all user-supplied URLs. Use "
                    "allow-lists for permitted domains. Disable redirects."
                ),
            ),
            FrameworkRequirement(
                id="API8",
                name="Security Misconfiguration",
                description=(
                    "Improper or insecure default configurations, open cloud "
                    "storage, verbose error messages, or unnecessary features enabled."
                ),
                severity="medium",
                test_categories=["prompt_extraction", "data_extraction", "debug_access"],
                remediation=(
                    "Harden all environment configurations. Disable verbose "
                    "error output. Review default settings."
                ),
            ),
            FrameworkRequirement(
                id="API9",
                name="Improper Inventory Management",
                description=(
                    "Outdated or undocumented API endpoints that are not "
                    "properly managed, patched, or retired."
                ),
                severity="medium",
                test_categories=["data_extraction"],
                remediation=(
                    "Maintain an up-to-date API inventory. Deprecate and "
                    "remove unused endpoints. Version all APIs."
                ),
            ),
            FrameworkRequirement(
                id="API10",
                name="Unsafe Consumption of APIs",
                description=(
                    "Developers trusting data received from third-party APIs "
                    "without proper validation, enabling injection attacks "
                    "through the supply chain."
                ),
                severity="high",
                test_categories=[
                    "sql_injection",
                    "shell_injection",
                    "json_injection",
                    "xml_injection",
                    "tool_metadata_poisoning",
                ],
                remediation=(
                    "Validate all third-party API responses. Apply input "
                    "sanitization to external data. Use transport encryption."
                ),
            ),
        ],
    )


def _build_mitre_atlas() -> FrameworkMapping:
    """Build the MITRE ATLAS framework mapping.

    Returns:
        A FrameworkMapping with requirements covering ATLAS adversarial
        ML technique categories, each mapped to VulnerabilityType values.
    """
    return FrameworkMapping(
        framework=ComplianceFramework.MITRE_ATLAS,
        version="4.0",
        description=(
            "MITRE ATLAS (Adversarial Threat Landscape for AI Systems) provides "
            "a knowledge base of adversary tactics and techniques against AI systems."
        ),
        url="https://atlas.mitre.org/",
        requirements=[
            FrameworkRequirement(
                id="ATLAS-RECON",
                name="Reconnaissance",
                description=(
                    "Adversaries gather information about the target AI system "
                    "including model architecture, training data sources, API "
                    "endpoints, and deployment infrastructure."
                ),
                severity="medium",
                test_categories=[
                    "prompt_extraction",
                    "data_extraction",
                    "debug_access",
                ],
                remediation=(
                    "Minimize information leakage about model internals. "
                    "Restrict API metadata. Use generic error messages."
                ),
                references=["https://atlas.mitre.org/tactics/AML.TA0002"],
            ),
            FrameworkRequirement(
                id="ATLAS-RESOURCE",
                name="Resource Development",
                description=(
                    "Adversaries develop resources to support operations "
                    "including creating adversarial examples, training surrogate "
                    "models, and building attack tools."
                ),
                severity="medium",
                test_categories=[
                    "encoding_attack",
                    "unicode_exploit",
                    "token_smuggling",
                ],
                remediation=(
                    "Implement robust input validation that resists adversarial "
                    "perturbations. Monitor for automated probing patterns."
                ),
                references=["https://atlas.mitre.org/tactics/AML.TA0003"],
            ),
            FrameworkRequirement(
                id="ATLAS-ACCESS",
                name="Initial Access",
                description=(
                    "Adversaries gain initial access to the AI system through "
                    "prompt injection, API exploitation, or supply chain compromise."
                ),
                severity="critical",
                test_categories=[
                    "prompt_injection",
                    "jailbreak",
                    "indirect_prompt_injection",
                    "instruction_override",
                ],
                remediation=(
                    "Implement multi-layered input validation. Use instruction "
                    "hierarchy. Apply rate limiting and anomaly detection."
                ),
                references=["https://atlas.mitre.org/tactics/AML.TA0004"],
            ),
            FrameworkRequirement(
                id="ATLAS-EXEC",
                name="Execution",
                description=(
                    "Adversaries execute malicious actions through the AI system "
                    "including code execution via tool use, SQL injection through "
                    "model outputs, or unauthorized API calls."
                ),
                severity="critical",
                test_categories=[
                    "sql_injection",
                    "shell_injection",
                    "excessive_agency",
                    "privilege_escalation",
                ],
                remediation=(
                    "Sandbox all code execution. Validate tool inputs. "
                    "Apply output filtering before downstream processing."
                ),
                references=["https://atlas.mitre.org/tactics/AML.TA0005"],
            ),
            FrameworkRequirement(
                id="ATLAS-PERSIST",
                name="Persistence",
                description=(
                    "Adversaries maintain access through memory poisoning, "
                    "context manipulation, or embedding backdoors in fine-tuned "
                    "models or RAG knowledge bases."
                ),
                severity="high",
                test_categories=[
                    "memory_poisoning",
                    "rag_poisoning",
                    "context_manipulation",
                ],
                remediation=(
                    "Implement memory integrity checks. Validate knowledge base "
                    "entries. Use anomaly detection on model behavior changes."
                ),
                references=["https://atlas.mitre.org/tactics/AML.TA0006"],
            ),
            FrameworkRequirement(
                id="ATLAS-PRIVESC",
                name="Privilege Escalation",
                description=(
                    "Adversaries elevate their access level through role "
                    "escape, RBAC bypass, or exploiting agent trust boundaries."
                ),
                severity="critical",
                test_categories=[
                    "privilege_escalation",
                    "role_escape",
                    "rbac_bypass",
                    "bfla_attack",
                ],
                remediation=(
                    "Apply principle of least privilege. Implement per-action "
                    "authorization. Use capability-based security."
                ),
                references=["https://atlas.mitre.org/tactics/AML.TA0007"],
            ),
            FrameworkRequirement(
                id="ATLAS-EVADE",
                name="Defense Evasion",
                description=(
                    "Adversaries evade security controls through encoding "
                    "attacks, delimiter injection, ASCII smuggling, or "
                    "other techniques to bypass content filters."
                ),
                severity="high",
                test_categories=[
                    "encoding_attack",
                    "ascii_smuggling",
                    "delimiter_injection",
                    "unicode_exploit",
                    "token_smuggling",
                    "jailbreak",
                ],
                remediation=(
                    "Implement multi-layer content filtering. Normalize inputs "
                    "before processing. Use semantic analysis in addition to "
                    "pattern matching."
                ),
                references=["https://atlas.mitre.org/tactics/AML.TA0008"],
            ),
            FrameworkRequirement(
                id="ATLAS-COLLECT",
                name="Collection",
                description=(
                    "Adversaries collect sensitive data from the AI system "
                    "including PII, training data, system prompts, and "
                    "user conversation history."
                ),
                severity="high",
                test_categories=[
                    "pii_leakage",
                    "data_extraction",
                    "prompt_extraction",
                    "pii_social_engineering",
                    "session_leak",
                    "cross_session_leak",
                    "cross_context_retrieval",
                ],
                remediation=(
                    "Implement PII detection and redaction. Apply data access "
                    "controls. Monitor for data harvesting patterns."
                ),
                references=["https://atlas.mitre.org/tactics/AML.TA0009"],
            ),
            FrameworkRequirement(
                id="ATLAS-EXFIL",
                name="Exfiltration",
                description=(
                    "Adversaries exfiltrate collected data from the AI system "
                    "through steganographic encoding, covert channels in model "
                    "outputs, or direct data leakage."
                ),
                severity="critical",
                test_categories=[
                    "data_exfiltration",
                    "data_extraction",
                    "pii_leakage",
                    "pii_database",
                ],
                remediation=(
                    "Apply egress filtering. Monitor output patterns for "
                    "encoded data. Implement DLP controls on model outputs."
                ),
                references=["https://atlas.mitre.org/tactics/AML.TA0010"],
            ),
            FrameworkRequirement(
                id="ATLAS-IMPACT",
                name="Impact",
                description=(
                    "Adversaries cause harm through the AI system including "
                    "generating harmful content, misinformation, enabling "
                    "illegal activities, or damaging the organization's reputation."
                ),
                severity="critical",
                test_categories=[
                    "harmful_content",
                    "hate_speech",
                    "violent_crime",
                    "illegal_activities",
                    "chemical_weapons",
                    "self_harm",
                    "radicalization",
                    "reputation_damage",
                ],
                remediation=(
                    "Implement comprehensive content safety filters. Use "
                    "multi-layer output validation. Monitor for harmful "
                    "content patterns and maintain incident response plans."
                ),
                references=["https://atlas.mitre.org/tactics/AML.TA0011"],
            ),
        ],
    )


def _build_eu_ai_act() -> FrameworkMapping:
    """Build the EU AI Act framework mapping.

    Returns:
        A FrameworkMapping with requirements covering the EU AI Act's
        four risk levels and key obligations for AI systems.
    """
    return FrameworkMapping(
        framework=ComplianceFramework.EU_AI_ACT,
        version="2024",
        description=(
            "The EU AI Act establishes a comprehensive regulatory framework for "
            "artificial intelligence systems based on a risk-based approach with "
            "four tiers: unacceptable, high, limited, and minimal risk."
        ),
        url="https://artificialintelligenceact.eu/",
        requirements=[
            FrameworkRequirement(
                id="EUAI-UNACCEPT-01",
                name="Prohibited Manipulation Techniques",
                description=(
                    "AI systems shall not use subliminal techniques, exploit "
                    "vulnerabilities of specific groups, or employ social scoring "
                    "to manipulate behavior causing significant harm."
                ),
                severity="critical",
                test_categories=[
                    "prompt_injection",
                    "context_manipulation",
                    "bias_exploitation",
                    "radicalization",
                ],
                remediation=(
                    "Implement manipulation detection. Ensure the system does "
                    "not exploit user vulnerabilities. Add fairness constraints."
                ),
                references=["https://artificialintelligenceact.eu/article/5/"],
            ),
            FrameworkRequirement(
                id="EUAI-UNACCEPT-02",
                name="Prohibited Social Scoring",
                description=(
                    "AI systems shall not evaluate or classify individuals "
                    "based on social behavior or personal characteristics "
                    "leading to detrimental or unfavorable treatment."
                ),
                severity="critical",
                test_categories=[
                    "bias_exploitation",
                    "hate_speech",
                    "political_opinion",
                ],
                remediation=(
                    "Ensure the system does not generate discriminatory scores "
                    "or classifications. Audit outputs for bias patterns."
                ),
                references=["https://artificialintelligenceact.eu/article/5/"],
            ),
            FrameworkRequirement(
                id="EUAI-HIGH-01",
                name="Risk Management System",
                description=(
                    "High-risk AI systems must implement a risk management "
                    "system that identifies, analyzes, and mitigates risks "
                    "throughout the system lifecycle."
                ),
                severity="high",
                test_categories=[
                    "prompt_injection",
                    "jailbreak",
                    "harmful_content",
                    "bias_exploitation",
                    "hallucination_attack",
                ],
                remediation=(
                    "Implement comprehensive red teaming. Document identified "
                    "risks and mitigations. Maintain risk assessment records."
                ),
                references=["https://artificialintelligenceact.eu/article/9/"],
            ),
            FrameworkRequirement(
                id="EUAI-HIGH-02",
                name="Data Governance",
                description=(
                    "High-risk AI systems must implement data governance "
                    "practices including data quality, representativeness, "
                    "and privacy compliance for training and evaluation data."
                ),
                severity="high",
                test_categories=[
                    "pii_leakage",
                    "gdpr_violation",
                    "coppa_violation",
                    "pii_database",
                    "session_leak",
                    "cross_session_leak",
                    "data_exfiltration",
                ],
                remediation=(
                    "Implement data quality controls. Ensure GDPR compliance. "
                    "Apply PII detection and anonymization. Document data "
                    "provenance and processing."
                ),
                references=["https://artificialintelligenceact.eu/article/10/"],
            ),
            FrameworkRequirement(
                id="EUAI-HIGH-03",
                name="Transparency & Information",
                description=(
                    "High-risk AI systems must provide clear information about "
                    "system capabilities, limitations, risks, and the AI "
                    "nature of the interaction."
                ),
                severity="high",
                test_categories=[
                    "hallucination_attack",
                    "overreliance",
                    "regulated_advice",
                    "unauthorized_practice",
                ],
                remediation=(
                    "Provide clear AI disclosure. Communicate model limitations. "
                    "Include confidence indicators. Add disclaimers for "
                    "high-stakes domains."
                ),
                references=["https://artificialintelligenceact.eu/article/13/"],
            ),
            FrameworkRequirement(
                id="EUAI-HIGH-04",
                name="Human Oversight",
                description=(
                    "High-risk AI systems must enable meaningful human oversight "
                    "to prevent or minimize risks, with the ability to intervene "
                    "or halt the system."
                ),
                severity="high",
                test_categories=[
                    "excessive_agency",
                    "goal_hijacking",
                    "privilege_escalation",
                    "role_escape",
                ],
                remediation=(
                    "Implement human-in-the-loop for high-impact decisions. "
                    "Provide override mechanisms. Ensure the system can be "
                    "safely stopped at any point."
                ),
                references=["https://artificialintelligenceact.eu/article/14/"],
            ),
            FrameworkRequirement(
                id="EUAI-HIGH-05",
                name="Robustness & Security",
                description=(
                    "High-risk AI systems must be resilient against errors, "
                    "faults, and adversarial attacks, maintaining consistent "
                    "performance under attack conditions."
                ),
                severity="high",
                test_categories=[
                    "prompt_injection",
                    "jailbreak",
                    "encoding_attack",
                    "unicode_exploit",
                    "ascii_smuggling",
                    "delimiter_injection",
                    "token_smuggling",
                ],
                remediation=(
                    "Implement adversarial robustness testing. Apply input "
                    "normalization. Use multi-layer defense. Monitor for "
                    "degraded performance under attack."
                ),
                references=["https://artificialintelligenceact.eu/article/15/"],
            ),
            FrameworkRequirement(
                id="EUAI-HIGH-06",
                name="Non-Discrimination",
                description=(
                    "AI systems must not produce outputs that discriminate "
                    "based on protected characteristics including race, gender, "
                    "age, disability, religion, or sexual orientation."
                ),
                severity="critical",
                test_categories=[
                    "bias_exploitation",
                    "hate_speech",
                    "age_restricted",
                ],
                remediation=(
                    "Implement bias testing across protected categories. Use "
                    "fairness metrics. Audit outputs for discriminatory patterns."
                ),
                references=["https://artificialintelligenceact.eu/article/10/"],
            ),
            FrameworkRequirement(
                id="EUAI-LIMITED-01",
                name="Transparency Obligations",
                description=(
                    "Limited-risk AI systems must inform users they are "
                    "interacting with AI. This applies to chatbots, deepfakes, "
                    "and emotion recognition systems."
                ),
                severity="medium",
                test_categories=[
                    "overreliance",
                    "hallucination_attack",
                    "brand_impersonation",
                ],
                remediation=(
                    "Clearly label AI-generated content. Provide transparency "
                    "notices at interaction start. Disclose AI involvement."
                ),
                references=["https://artificialintelligenceact.eu/article/52/"],
            ),
            FrameworkRequirement(
                id="EUAI-MINIMAL-01",
                name="Voluntary Codes of Conduct",
                description=(
                    "Minimal-risk AI systems are encouraged to adopt voluntary "
                    "codes of conduct covering ethical AI principles, fairness, "
                    "and responsible deployment."
                ),
                severity="low",
                test_categories=[
                    "harmful_content",
                    "bias_exploitation",
                    "profanity",
                ],
                remediation=(
                    "Adopt responsible AI practices. Implement basic content "
                    "safety. Follow industry best practices for AI deployment."
                ),
                references=["https://artificialintelligenceact.eu/article/69/"],
            ),
        ],
    )


def _build_coppa() -> FrameworkMapping:
    """Build the COPPA framework mapping.

    Returns:
        A FrameworkMapping with requirements for children's online privacy
        protection, mapped to VulnerabilityType values.
    """
    return FrameworkMapping(
        framework=ComplianceFramework.COPPA,
        version="2013",
        description=(
            "Children's Online Privacy Protection Act (COPPA) imposes requirements "
            "on operators of websites or online services directed to children under "
            "13 regarding the collection of personal information."
        ),
        url="https://www.ftc.gov/legal-library/browse/rules/childrens-online-privacy-protection-rule-coppa",
        requirements=[
            FrameworkRequirement(
                id="COPPA-01",
                name="No Collection of Children's PII",
                description=(
                    "The system must not solicit, collect, or store personal "
                    "information from children under 13 without verifiable "
                    "parental consent."
                ),
                severity="critical",
                test_categories=[
                    "pii_leakage",
                    "pii_social_engineering",
                    "pii_database",
                    "coppa_violation",
                ],
                remediation=(
                    "Implement age gating. Do not collect children's PII. "
                    "Add parental consent mechanisms if PII collection is required."
                ),
            ),
            FrameworkRequirement(
                id="COPPA-02",
                name="Age-Appropriate Content",
                description=(
                    "The system must not generate age-inappropriate content "
                    "including violent, sexual, or disturbing material when "
                    "interacting with minors."
                ),
                severity="critical",
                test_categories=[
                    "age_restricted",
                    "sexual_content",
                    "violent_crime",
                    "graphic_content",
                    "self_harm",
                    "profanity",
                ],
                remediation=(
                    "Implement strict content filtering for minor-facing "
                    "applications. Block age-inappropriate content categories."
                ),
            ),
            FrameworkRequirement(
                id="COPPA-03",
                name="Data Minimization for Children",
                description=(
                    "The system must minimize data collection from children "
                    "to what is strictly necessary for the service's purpose."
                ),
                severity="high",
                test_categories=[
                    "data_extraction",
                    "session_leak",
                    "cross_session_leak",
                    "coppa_violation",
                ],
                remediation=(
                    "Collect only necessary data. Implement data minimization "
                    "policies. Delete children's data when no longer needed."
                ),
            ),
            FrameworkRequirement(
                id="COPPA-04",
                name="Parental Notification & Control",
                description=(
                    "Parents must be notified about data practices and given "
                    "the ability to review, delete, or refuse further collection "
                    "of their child's information."
                ),
                severity="high",
                test_categories=[
                    "coppa_violation",
                    "pii_leakage",
                ],
                remediation=(
                    "Provide clear privacy notices. Implement parental "
                    "access controls. Enable data deletion requests."
                ),
            ),
        ],
    )


def _build_ferpa() -> FrameworkMapping:
    """Build the FERPA framework mapping.

    Returns:
        A FrameworkMapping with requirements for student education records
        privacy, mapped to VulnerabilityType values.
    """
    return FrameworkMapping(
        framework=ComplianceFramework.FERPA,
        version="2023",
        description=(
            "Family Educational Rights and Privacy Act (FERPA) protects the "
            "privacy of student education records and applies to all schools "
            "that receive federal funding."
        ),
        url="https://studentprivacy.ed.gov/ferpa",
        requirements=[
            FrameworkRequirement(
                id="FERPA-01",
                name="Student Record Confidentiality",
                description=(
                    "The system must not disclose personally identifiable "
                    "information from student education records without "
                    "consent from the parent or eligible student."
                ),
                severity="critical",
                test_categories=[
                    "pii_leakage",
                    "data_extraction",
                    "ferpa_violation",
                    "pii_social_engineering",
                ],
                remediation=(
                    "Implement strict access controls for student records. "
                    "Do not include student PII in model outputs. Apply "
                    "data anonymization."
                ),
            ),
            FrameworkRequirement(
                id="FERPA-02",
                name="Directory Information Protection",
                description=(
                    "The system must appropriately handle directory information "
                    "and not disclose it beyond authorized purposes."
                ),
                severity="high",
                test_categories=[
                    "pii_leakage",
                    "pii_database",
                    "ferpa_violation",
                ],
                remediation=(
                    "Limit directory information exposure. Implement "
                    "opt-out mechanisms. Verify authorization before "
                    "disclosing any student information."
                ),
            ),
            FrameworkRequirement(
                id="FERPA-03",
                name="Legitimate Educational Interest",
                description=(
                    "Access to student records must be limited to school "
                    "officials with legitimate educational interest."
                ),
                severity="high",
                test_categories=[
                    "privilege_escalation",
                    "role_escape",
                    "rbac_bypass",
                    "ferpa_violation",
                ],
                remediation=(
                    "Implement role-based access control. Verify educational "
                    "interest before granting access. Log all record access."
                ),
            ),
            FrameworkRequirement(
                id="FERPA-04",
                name="Record Amendment Rights",
                description=(
                    "The system must support parents' and eligible students' "
                    "rights to request correction of inaccurate records."
                ),
                severity="medium",
                test_categories=[
                    "hallucination_attack",
                    "ferpa_violation",
                ],
                remediation=(
                    "Implement record correction workflows. Maintain "
                    "audit trails for record changes. Ensure accuracy "
                    "of AI-generated student assessments."
                ),
            ),
        ],
    )


def _build_dod_ai_ethics() -> FrameworkMapping:
    """Build the DoD AI Ethics framework mapping.

    Returns:
        A FrameworkMapping with requirements covering DoD AI ethical
        principles, mapped to VulnerabilityType values.
    """
    return FrameworkMapping(
        framework=ComplianceFramework.DOD_AI_ETHICS,
        version="2020",
        description=(
            "The Department of Defense AI Ethics Principles establish five "
            "ethical principles for the responsible development and use of "
            "AI capabilities: Responsible, Equitable, Traceable, Reliable, "
            "and Governable."
        ),
        url="https://www.ai.mil/docs/Responsible_AI_Strategy_and_Implementation_Pathway.pdf",
        requirements=[
            FrameworkRequirement(
                id="DOD-RESP-01",
                name="Responsible Use",
                description=(
                    "Personnel shall exercise appropriate judgment and care "
                    "in AI development, deployment, and use. AI systems should "
                    "not generate harmful, illegal, or unethical content."
                ),
                severity="critical",
                test_categories=[
                    "harmful_content",
                    "violent_crime",
                    "chemical_weapons",
                    "illegal_activities",
                    "cyber_crime",
                ],
                remediation=(
                    "Implement comprehensive content safety filters. Conduct "
                    "regular safety audits. Establish human oversight for "
                    "all AI-assisted decisions."
                ),
            ),
            FrameworkRequirement(
                id="DOD-EQUIT-01",
                name="Equitable Operation",
                description=(
                    "AI systems should minimize unintended bias and not "
                    "discriminate based on protected characteristics."
                ),
                severity="high",
                test_categories=[
                    "bias_exploitation",
                    "hate_speech",
                    "age_restricted",
                    "political_opinion",
                ],
                remediation=(
                    "Conduct bias testing across demographic groups. Implement "
                    "fairness constraints. Regularly audit for disparate impact."
                ),
            ),
            FrameworkRequirement(
                id="DOD-TRACE-01",
                name="Traceability",
                description=(
                    "AI systems shall be developed and deployed such that "
                    "relevant personnel possess an appropriate understanding "
                    "of the technology, processes, and methods."
                ),
                severity="high",
                test_categories=[
                    "prompt_extraction",
                    "data_extraction",
                    "hallucination_attack",
                    "overreliance",
                ],
                remediation=(
                    "Maintain comprehensive audit logs. Document AI decision "
                    "rationale. Implement explainability features."
                ),
            ),
            FrameworkRequirement(
                id="DOD-RELY-01",
                name="Reliability",
                description=(
                    "AI systems shall have explicit, well-defined uses "
                    "and safety, security, and robustness will be tested "
                    "and assured across the lifecycle."
                ),
                severity="critical",
                test_categories=[
                    "prompt_injection",
                    "jailbreak",
                    "encoding_attack",
                    "unicode_exploit",
                    "excessive_agency",
                ],
                remediation=(
                    "Implement continuous red teaming. Test adversarial "
                    "robustness. Establish performance benchmarks and "
                    "regression testing."
                ),
            ),
            FrameworkRequirement(
                id="DOD-GOV-01",
                name="Governability",
                description=(
                    "AI systems shall be designed and engineered to fulfill "
                    "their intended functions while possessing the ability "
                    "to detect and avoid unintended consequences, and to "
                    "disengage when demonstrating unintended behavior."
                ),
                severity="critical",
                test_categories=[
                    "excessive_agency",
                    "goal_hijacking",
                    "role_escape",
                    "privilege_escalation",
                ],
                remediation=(
                    "Implement kill switches and override mechanisms. Add "
                    "human-in-the-loop for critical decisions. Design for "
                    "graceful degradation."
                ),
            ),
        ],
    )


def _build_ccpa() -> FrameworkMapping:
    """Build the CCPA (California Consumer Privacy Act) framework mapping.

    Returns:
        A FrameworkMapping with requirements for consumer data privacy
        protection, mapped to VulnerabilityType values.
    """
    return FrameworkMapping(
        framework=ComplianceFramework.CCPA,
        version="2023",
        description=(
            "The California Consumer Privacy Act (CCPA/CPRA) gives California "
            "consumers rights regarding their personal information collected "
            "by businesses, including AI systems."
        ),
        url="https://oag.ca.gov/privacy/ccpa",
        requirements=[
            FrameworkRequirement(
                id="CCPA-01",
                name="Right to Know",
                description=(
                    "The system must support consumers' right to know what "
                    "personal information is collected, used, shared, or sold."
                ),
                severity="high",
                test_categories=[
                    "pii_leakage",
                    "data_extraction",
                    "ccpa_violation",
                ],
                remediation=(
                    "Implement data inventory. Provide transparency about "
                    "data collection practices. Enable consumer data requests."
                ),
            ),
            FrameworkRequirement(
                id="CCPA-02",
                name="Right to Delete",
                description=(
                    "The system must support consumers' right to request "
                    "deletion of personal information, including data used "
                    "in AI model training or inference."
                ),
                severity="high",
                test_categories=[
                    "pii_database",
                    "session_leak",
                    "cross_session_leak",
                    "ccpa_violation",
                ],
                remediation=(
                    "Implement data deletion workflows. Ensure model outputs "
                    "do not retain deleted user data. Verify deletion across "
                    "all storage systems."
                ),
            ),
            FrameworkRequirement(
                id="CCPA-03",
                name="Data Minimization",
                description=(
                    "The system must not collect or process personal information "
                    "beyond what is reasonably necessary for the disclosed purpose."
                ),
                severity="high",
                test_categories=[
                    "pii_social_engineering",
                    "data_extraction",
                    "pii_api_access",
                    "ccpa_violation",
                ],
                remediation=(
                    "Collect only necessary data. Implement purpose limitation. "
                    "Delete data after use. Avoid soliciting unnecessary PII."
                ),
            ),
            FrameworkRequirement(
                id="CCPA-04",
                name="No Discrimination for Privacy Exercise",
                description=(
                    "The system must not discriminate against consumers who "
                    "exercise their CCPA privacy rights."
                ),
                severity="critical",
                test_categories=[
                    "bias_exploitation",
                    "ccpa_violation",
                ],
                remediation=(
                    "Ensure equal service quality regardless of privacy "
                    "preferences. Do not penalize users exercising data rights."
                ),
            ),
            FrameworkRequirement(
                id="CCPA-05",
                name="Automated Decision-Making Transparency",
                description=(
                    "Consumers have the right to opt out of automated "
                    "decision-making and profiling, and to request meaningful "
                    "information about the logic involved."
                ),
                severity="high",
                test_categories=[
                    "overreliance",
                    "hallucination_attack",
                    "excessive_agency",
                    "ccpa_violation",
                ],
                remediation=(
                    "Provide opt-out mechanisms for automated decisions. "
                    "Offer explanations for AI-driven outcomes. Implement "
                    "human review pathways."
                ),
            ),
        ],
    )


def _build_hipaa() -> FrameworkMapping:
    """Build the HIPAA framework mapping.

    Returns:
        A FrameworkMapping with requirements for health information
        privacy, mapped to VulnerabilityType values.
    """
    return FrameworkMapping(
        framework=ComplianceFramework.HIPAA,
        version="2013",
        description=(
            "The Health Insurance Portability and Accountability Act (HIPAA) "
            "establishes national standards to protect individuals' medical "
            "records and personal health information."
        ),
        url="https://www.hhs.gov/hipaa/index.html",
        requirements=[
            FrameworkRequirement(
                id="HIPAA-01",
                name="PHI Confidentiality",
                description=(
                    "The system must not disclose protected health information "
                    "(PHI) without authorization. AI outputs must not contain "
                    "patient-identifiable health data."
                ),
                severity="critical",
                test_categories=[
                    "pii_leakage",
                    "data_extraction",
                    "hipaa_violation",
                    "pii_social_engineering",
                    "medical_advice",
                ],
                remediation=(
                    "Implement PHI detection and redaction. Apply access "
                    "controls. Never include patient data in model outputs."
                ),
            ),
            FrameworkRequirement(
                id="HIPAA-02",
                name="Minimum Necessary Standard",
                description=(
                    "The system must limit PHI disclosure to the minimum "
                    "necessary to accomplish the intended purpose."
                ),
                severity="high",
                test_categories=[
                    "data_extraction",
                    "pii_database",
                    "hipaa_violation",
                ],
                remediation=(
                    "Apply data minimization. Limit PHI exposure in responses. "
                    "Implement role-based data access."
                ),
            ),
            FrameworkRequirement(
                id="HIPAA-03",
                name="Security Safeguards",
                description=(
                    "The system must implement administrative, physical, and "
                    "technical safeguards to protect electronic PHI (ePHI)."
                ),
                severity="critical",
                test_categories=[
                    "prompt_injection",
                    "session_leak",
                    "cross_session_leak",
                    "data_exfiltration",
                    "hipaa_violation",
                ],
                remediation=(
                    "Encrypt ePHI in transit and at rest. Implement access "
                    "controls and audit logging. Conduct regular risk assessments."
                ),
            ),
        ],
    )


def _build_gdpr() -> FrameworkMapping:
    """Build the GDPR framework mapping.

    Returns:
        A FrameworkMapping with requirements for EU data protection,
        mapped to VulnerabilityType values.
    """
    return FrameworkMapping(
        framework=ComplianceFramework.GDPR,
        version="2018",
        description=(
            "The General Data Protection Regulation (GDPR) is the EU's "
            "comprehensive data protection regulation governing the processing "
            "of personal data of individuals in the EU."
        ),
        url="https://gdpr.eu/",
        requirements=[
            FrameworkRequirement(
                id="GDPR-01",
                name="Lawful Processing & Consent",
                description=(
                    "Personal data must be processed lawfully, fairly, and "
                    "transparently. The system must not process personal data "
                    "without a lawful basis."
                ),
                severity="critical",
                test_categories=[
                    "pii_leakage",
                    "gdpr_violation",
                    "pii_social_engineering",
                ],
                remediation=(
                    "Implement consent management. Ensure lawful basis for "
                    "all data processing. Provide clear privacy notices."
                ),
            ),
            FrameworkRequirement(
                id="GDPR-02",
                name="Purpose Limitation",
                description=(
                    "Personal data must be collected for specified, explicit, "
                    "and legitimate purposes and not further processed in a "
                    "manner incompatible with those purposes."
                ),
                severity="high",
                test_categories=[
                    "data_extraction",
                    "pii_api_access",
                    "gdpr_violation",
                ],
                remediation=(
                    "Define and document processing purposes. Prevent data "
                    "repurposing. Implement purpose limitation controls."
                ),
            ),
            FrameworkRequirement(
                id="GDPR-03",
                name="Data Minimization",
                description=(
                    "Personal data must be adequate, relevant, and limited "
                    "to what is necessary for the processing purposes."
                ),
                severity="high",
                test_categories=[
                    "pii_database",
                    "data_extraction",
                    "gdpr_violation",
                ],
                remediation=(
                    "Collect only necessary data. Implement data minimization "
                    "by design. Regularly review data holdings."
                ),
            ),
            FrameworkRequirement(
                id="GDPR-04",
                name="Right to Erasure",
                description=(
                    "Data subjects have the right to have their personal data "
                    "erased. AI systems must support this for both training "
                    "data and inference contexts."
                ),
                severity="high",
                test_categories=[
                    "session_leak",
                    "cross_session_leak",
                    "gdpr_violation",
                ],
                remediation=(
                    "Implement data deletion workflows. Ensure session data "
                    "is not persisted. Support right-to-be-forgotten requests."
                ),
            ),
            FrameworkRequirement(
                id="GDPR-05",
                name="Automated Decision-Making (Art. 22)",
                description=(
                    "Data subjects have the right not to be subject to "
                    "decisions based solely on automated processing that "
                    "produce legal or similarly significant effects."
                ),
                severity="critical",
                test_categories=[
                    "excessive_agency",
                    "overreliance",
                    "bias_exploitation",
                    "gdpr_violation",
                ],
                remediation=(
                    "Implement human review for automated decisions. Provide "
                    "explanations for automated outcomes. Offer opt-out "
                    "mechanisms."
                ),
            ),
        ],
    )


def _build_pci_dss() -> FrameworkMapping:
    """Build the PCI DSS framework mapping.

    Returns:
        A FrameworkMapping with requirements for payment card data
        security, mapped to VulnerabilityType values.
    """
    return FrameworkMapping(
        framework=ComplianceFramework.PCI_DSS,
        version="4.0",
        description=(
            "The Payment Card Industry Data Security Standard (PCI DSS) "
            "provides a baseline of technical and operational requirements "
            "designed to protect cardholder data."
        ),
        url="https://www.pcisecuritystandards.org/",
        requirements=[
            FrameworkRequirement(
                id="PCI-01",
                name="Cardholder Data Protection",
                description=(
                    "The system must not store, process, or transmit "
                    "cardholder data in model outputs or prompts."
                ),
                severity="critical",
                test_categories=[
                    "pii_leakage",
                    "data_extraction",
                    "pci_dss_violation",
                ],
                remediation=(
                    "Implement PAN detection and masking. Never include "
                    "cardholder data in AI outputs. Apply tokenization."
                ),
            ),
            FrameworkRequirement(
                id="PCI-02",
                name="Access Control",
                description=(
                    "Access to cardholder data must be restricted on a "
                    "need-to-know basis with proper authentication."
                ),
                severity="critical",
                test_categories=[
                    "privilege_escalation",
                    "rbac_bypass",
                    "bola_attack",
                    "pci_dss_violation",
                ],
                remediation=(
                    "Implement role-based access control. Use strong "
                    "authentication. Log all access to cardholder data."
                ),
            ),
            FrameworkRequirement(
                id="PCI-03",
                name="Vulnerability Management",
                description=(
                    "Maintain a vulnerability management program including "
                    "regular testing and patching of AI system components."
                ),
                severity="high",
                test_categories=[
                    "sql_injection",
                    "shell_injection",
                    "ssrf_attack",
                    "pci_dss_violation",
                ],
                remediation=(
                    "Conduct regular vulnerability assessments. Patch "
                    "promptly. Monitor for emerging attack vectors."
                ),
            ),
        ],
    )


def _build_soc2() -> FrameworkMapping:
    """Build the SOC 2 framework mapping.

    Returns:
        A FrameworkMapping with requirements for service organization
        controls, mapped to VulnerabilityType values.
    """
    return FrameworkMapping(
        framework=ComplianceFramework.SOC2,
        version="2017",
        description=(
            "SOC 2 (Service Organization Control 2) defines criteria for "
            "managing customer data based on five trust service principles: "
            "security, availability, processing integrity, confidentiality, "
            "and privacy."
        ),
        url="https://www.aicpa-cima.com/topic/audit-assurance/audit-and-assurance-greater-than-soc-2",
        requirements=[
            FrameworkRequirement(
                id="SOC2-SEC",
                name="Security",
                description=(
                    "The system is protected against unauthorized access, "
                    "unauthorized disclosure of information, and damage to "
                    "systems that could compromise availability, integrity, "
                    "confidentiality, and privacy."
                ),
                severity="critical",
                test_categories=[
                    "prompt_injection",
                    "jailbreak",
                    "privilege_escalation",
                    "data_extraction",
                    "shell_injection",
                ],
                remediation=(
                    "Implement defense-in-depth. Apply input validation, "
                    "output filtering, and access controls."
                ),
            ),
            FrameworkRequirement(
                id="SOC2-AVAIL",
                name="Availability",
                description=(
                    "The system is available for operation and use as "
                    "committed or agreed."
                ),
                severity="high",
                test_categories=[
                    "encoding_attack",
                    "unicode_exploit",
                    "token_smuggling",
                ],
                remediation=(
                    "Implement rate limiting and resource caps. Monitor "
                    "for denial of service patterns. Add circuit breakers."
                ),
            ),
            FrameworkRequirement(
                id="SOC2-CONFID",
                name="Confidentiality",
                description=(
                    "Information designated as confidential is protected "
                    "as committed or agreed."
                ),
                severity="critical",
                test_categories=[
                    "pii_leakage",
                    "data_extraction",
                    "prompt_extraction",
                    "data_exfiltration",
                    "session_leak",
                ],
                remediation=(
                    "Implement data classification. Apply encryption. "
                    "Prevent model memorization of confidential data."
                ),
            ),
            FrameworkRequirement(
                id="SOC2-PRIV",
                name="Privacy",
                description=(
                    "Personal information is collected, used, retained, "
                    "disclosed, and disposed of in conformity with the "
                    "entity's privacy notice."
                ),
                severity="high",
                test_categories=[
                    "pii_leakage",
                    "pii_social_engineering",
                    "pii_database",
                    "gdpr_violation",
                    "ccpa_violation",
                ],
                remediation=(
                    "Implement PII detection and handling. Follow privacy "
                    "notices. Support data subject access requests."
                ),
            ),
        ],
    )


def _build_iso27001() -> FrameworkMapping:
    """Build the ISO 27001 framework mapping.

    Returns:
        A FrameworkMapping with requirements for information security
        management, mapped to VulnerabilityType values.
    """
    return FrameworkMapping(
        framework=ComplianceFramework.ISO27001,
        version="2022",
        description=(
            "ISO/IEC 27001 specifies the requirements for establishing, "
            "implementing, maintaining, and continually improving an "
            "information security management system (ISMS)."
        ),
        url="https://www.iso.org/standard/27001",
        requirements=[
            FrameworkRequirement(
                id="ISO27K-A8",
                name="Access Control",
                description=(
                    "Ensure authorized user access and prevent unauthorized "
                    "access to information and information processing facilities."
                ),
                severity="critical",
                test_categories=[
                    "privilege_escalation",
                    "role_escape",
                    "rbac_bypass",
                    "bola_attack",
                    "bfla_attack",
                ],
                remediation=(
                    "Implement RBAC. Apply least privilege. Use multi-factor "
                    "authentication for sensitive operations."
                ),
            ),
            FrameworkRequirement(
                id="ISO27K-A12",
                name="Operations Security",
                description=(
                    "Ensure correct and secure operations of information "
                    "processing facilities including protection against malware."
                ),
                severity="high",
                test_categories=[
                    "prompt_injection",
                    "shell_injection",
                    "sql_injection",
                    "ssrf_attack",
                ],
                remediation=(
                    "Implement input validation. Apply output sanitization. "
                    "Monitor for malicious activity patterns."
                ),
            ),
            FrameworkRequirement(
                id="ISO27K-A18",
                name="Compliance with Legal Requirements",
                description=(
                    "Avoid breaches of legal, statutory, regulatory, or "
                    "contractual obligations related to information security."
                ),
                severity="high",
                test_categories=[
                    "gdpr_violation",
                    "hipaa_violation",
                    "coppa_violation",
                    "pci_dss_violation",
                    "ccpa_violation",
                ],
                remediation=(
                    "Implement regulatory compliance checks. Conduct regular "
                    "compliance audits. Maintain compliance documentation."
                ),
            ),
        ],
    )


def _build_nist_ai_rmf() -> FrameworkMapping:
    """Build the NIST AI RMF framework mapping.

    Returns:
        A FrameworkMapping with requirements covering NIST AI RMF core
        functions, mapped to VulnerabilityType values.
    """
    return FrameworkMapping(
        framework=ComplianceFramework.NIST_AI_RMF,
        version="1.0",
        description=(
            "The NIST AI Risk Management Framework (AI 100-1) provides a "
            "voluntary framework for managing AI risks across the AI lifecycle."
        ),
        url="https://www.nist.gov/artificial-intelligence/executive-order-safe-secure-and-trustworthy-artificial-intelligence",
        requirements=[
            FrameworkRequirement(
                id="NIST-GOVERN",
                name="Governance and Transparency",
                description=(
                    "Establish accountability structures and transparency "
                    "about model capabilities and limitations."
                ),
                severity="high",
                test_categories=[
                    "prompt_extraction",
                    "hallucination_attack",
                    "overreliance",
                    "regulated_advice",
                ],
                remediation=(
                    "Document AI system capabilities and limitations. "
                    "Establish governance oversight. Implement transparency reports."
                ),
            ),
            FrameworkRequirement(
                id="NIST-MAP",
                name="Risk Identification and Mapping",
                description=(
                    "Identify, categorize, and prioritize AI risks in context."
                ),
                severity="high",
                test_categories=[
                    "prompt_injection",
                    "jailbreak",
                    "goal_hijacking",
                    "context_manipulation",
                ],
                remediation=(
                    "Conduct comprehensive risk assessments. Map risks to "
                    "impact categories. Prioritize mitigation efforts."
                ),
            ),
            FrameworkRequirement(
                id="NIST-MEASURE",
                name="Evaluation Coverage",
                description=(
                    "Measure AI risks quantitatively across a broad set "
                    "of adversarial scenarios."
                ),
                severity="high",
                test_categories=[
                    "prompt_injection",
                    "jailbreak",
                    "harmful_content",
                    "bias_exploitation",
                    "pii_leakage",
                ],
                remediation=(
                    "Implement comprehensive red teaming. Use diverse attack "
                    "vectors. Track coverage metrics over time."
                ),
            ),
            FrameworkRequirement(
                id="NIST-MANAGE",
                name="Harm Mitigation",
                description=(
                    "Manage and mitigate identified risks, including content "
                    "harms and bias."
                ),
                severity="high",
                test_categories=[
                    "harmful_content",
                    "bias_exploitation",
                    "hate_speech",
                    "radicalization",
                    "self_harm",
                ],
                remediation=(
                    "Implement content safety filters. Apply bias mitigation. "
                    "Establish incident response procedures."
                ),
            ),
            FrameworkRequirement(
                id="NIST-TRUST",
                name="Reliability and Privacy",
                description=(
                    "Ensure trustworthy handling of user data and consistent "
                    "model behavior."
                ),
                severity="high",
                test_categories=[
                    "pii_leakage",
                    "data_extraction",
                    "pii_api_access",
                    "session_leak",
                    "cross_session_leak",
                ],
                remediation=(
                    "Implement PII protection. Ensure data privacy compliance. "
                    "Test model consistency under adversarial conditions."
                ),
            ),
            FrameworkRequirement(
                id="NIST-SAFETY",
                name="Safety and Security",
                description=(
                    "Prevent generation of content that enables physical, "
                    "cyber, or societal harm."
                ),
                severity="critical",
                test_categories=[
                    "chemical_weapons",
                    "violent_crime",
                    "illegal_activities",
                    "cyber_crime",
                    "shell_injection",
                    "sql_injection",
                ],
                remediation=(
                    "Implement comprehensive safety filters. Block dangerous "
                    "content categories. Monitor for emerging safety risks."
                ),
            ),
        ],
    )


def _build_iso_42001() -> FrameworkMapping:
    """Build the ISO 42001 framework mapping.

    Returns:
        A FrameworkMapping with requirements for AI management systems,
        mapped to VulnerabilityType values.
    """
    return FrameworkMapping(
        framework=ComplianceFramework.ISO_42001,
        version="2023",
        description=(
            "ISO/IEC 42001 specifies requirements for establishing, "
            "implementing, maintaining, and continually improving an "
            "AI management system (AIMS)."
        ),
        url="https://www.iso.org/standard/81230.html",
        requirements=[
            FrameworkRequirement(
                id="ISO42-RISK",
                name="AI Risk Assessment",
                description=(
                    "Conduct risk assessments that consider AI-specific "
                    "risks including adversarial attacks, bias, and safety."
                ),
                severity="high",
                test_categories=[
                    "prompt_injection",
                    "jailbreak",
                    "bias_exploitation",
                    "harmful_content",
                ],
                remediation=(
                    "Implement systematic AI risk assessment. Document "
                    "identified risks and their mitigations."
                ),
            ),
            FrameworkRequirement(
                id="ISO42-IMPACT",
                name="AI Impact Assessment",
                description=(
                    "Assess the impact of AI systems on individuals, "
                    "groups, and society, considering fairness, safety, "
                    "and human rights."
                ),
                severity="high",
                test_categories=[
                    "bias_exploitation",
                    "hate_speech",
                    "self_harm",
                    "radicalization",
                    "harmful_content",
                ],
                remediation=(
                    "Conduct impact assessments before deployment. Consider "
                    "effects on vulnerable populations. Implement mitigation "
                    "measures for identified impacts."
                ),
            ),
            FrameworkRequirement(
                id="ISO42-DATA",
                name="Data Management for AI",
                description=(
                    "Manage data used in AI systems ensuring quality, "
                    "representativeness, bias-freedom, and compliance "
                    "with privacy regulations."
                ),
                severity="high",
                test_categories=[
                    "pii_leakage",
                    "gdpr_violation",
                    "coppa_violation",
                    "bias_exploitation",
                ],
                remediation=(
                    "Implement data quality controls. Ensure data privacy. "
                    "Test for data bias and representativeness issues."
                ),
            ),
        ],
    )


def _build_nist_csf() -> FrameworkMapping:
    """Build the NIST Cybersecurity Framework mapping.

    Returns:
        A FrameworkMapping with requirements covering NIST CSF core
        functions applied to AI systems.
    """
    return FrameworkMapping(
        framework=ComplianceFramework.NIST_CSF,
        version="2.0",
        description=(
            "The NIST Cybersecurity Framework (CSF) provides a common "
            "taxonomy of cybersecurity outcomes applicable to AI systems."
        ),
        url="https://www.nist.gov/cyberframework",
        requirements=[
            FrameworkRequirement(
                id="CSF-IDENTIFY",
                name="Identify",
                description=(
                    "Develop organizational understanding to manage "
                    "cybersecurity risk to AI systems, assets, data, "
                    "and capabilities."
                ),
                severity="high",
                test_categories=[
                    "prompt_extraction",
                    "data_extraction",
                    "debug_access",
                ],
                remediation=(
                    "Inventory AI assets. Identify data flows. Document "
                    "AI system dependencies and attack surfaces."
                ),
            ),
            FrameworkRequirement(
                id="CSF-PROTECT",
                name="Protect",
                description=(
                    "Develop and implement safeguards to ensure delivery "
                    "of critical AI infrastructure services."
                ),
                severity="critical",
                test_categories=[
                    "prompt_injection",
                    "jailbreak",
                    "privilege_escalation",
                    "shell_injection",
                    "sql_injection",
                ],
                remediation=(
                    "Implement access controls, input validation, and "
                    "output filtering. Train staff on AI security."
                ),
            ),
            FrameworkRequirement(
                id="CSF-DETECT",
                name="Detect",
                description=(
                    "Develop and implement activities to identify the "
                    "occurrence of cybersecurity events affecting AI systems."
                ),
                severity="high",
                test_categories=[
                    "memory_poisoning",
                    "rag_poisoning",
                    "context_manipulation",
                ],
                remediation=(
                    "Implement anomaly detection for model behavior. "
                    "Monitor for adversarial attacks. Log all interactions."
                ),
            ),
            FrameworkRequirement(
                id="CSF-RESPOND",
                name="Respond",
                description=(
                    "Develop and implement activities to take action "
                    "regarding a detected cybersecurity incident."
                ),
                severity="high",
                test_categories=[
                    "excessive_agency",
                    "goal_hijacking",
                    "harmful_content",
                ],
                remediation=(
                    "Develop AI incident response plans. Implement kill "
                    "switches. Establish communication protocols."
                ),
            ),
            FrameworkRequirement(
                id="CSF-RECOVER",
                name="Recover",
                description=(
                    "Develop and implement activities to maintain plans "
                    "for resilience and restore AI capabilities impaired "
                    "by a cybersecurity incident."
                ),
                severity="medium",
                test_categories=[
                    "memory_poisoning",
                    "rag_poisoning",
                ],
                remediation=(
                    "Maintain model backups. Implement rollback procedures. "
                    "Test recovery plans regularly."
                ),
            ),
        ],
    )


_FRAMEWORK_BUILDERS: dict[ComplianceFramework, Callable[[], FrameworkMapping]] = {
    ComplianceFramework.OWASP_LLM_TOP10: _build_owasp_llm_top10,
    ComplianceFramework.OWASP_API_TOP10: _build_owasp_api_top10,
    ComplianceFramework.OWASP_AGENTIC_TOP10: _build_owasp_agentic_top10,
    ComplianceFramework.MITRE_ATLAS: _build_mitre_atlas,
    ComplianceFramework.EU_AI_ACT: _build_eu_ai_act,
    ComplianceFramework.COPPA: _build_coppa,
    ComplianceFramework.FERPA: _build_ferpa,
    ComplianceFramework.DOD_AI_ETHICS: _build_dod_ai_ethics,
    ComplianceFramework.CCPA: _build_ccpa,
    ComplianceFramework.HIPAA: _build_hipaa,
    ComplianceFramework.GDPR: _build_gdpr,
    ComplianceFramework.PCI_DSS: _build_pci_dss,
    ComplianceFramework.SOC2: _build_soc2,
    ComplianceFramework.ISO27001: _build_iso27001,
    ComplianceFramework.NIST_AI_RMF: _build_nist_ai_rmf,
    ComplianceFramework.ISO_42001: _build_iso_42001,
    ComplianceFramework.NIST_CSF: _build_nist_csf,
}


def get_framework_mapping(framework: ComplianceFramework) -> FrameworkMapping:
    """Retrieve the complete mapping for a compliance framework.

    Args:
        framework: The framework to retrieve.

    Returns:
        A FrameworkMapping containing all requirements and metadata.

    Raises:
        ValueError: If the framework has no registered builder.
    """
    if isinstance(framework, str) and not isinstance(framework, ComplianceFramework):
        try:
            framework = ComplianceFramework(framework)
        except ValueError:
            raise ValueError(
                f"No mapping available for framework: {framework}"
            )
    builder = _FRAMEWORK_BUILDERS.get(framework)
    if builder is None:
        raise ValueError(f"No mapping available for framework: {framework.value}")
    return builder()


def list_all_frameworks() -> list[ComplianceFramework]:
    """Return all supported compliance frameworks.

    Returns:
        A list of all ComplianceFramework enum values.
    """
    return list(ComplianceFramework)


def _get_valid_vuln_type_values() -> set[str]:
    """Return the set of valid VulnerabilityType string values.

    Returns:
        A set of strings corresponding to all VulnerabilityType enum values.
    """
    return {vt.value for vt in VulnerabilityType}


class ComplianceScanner:
    """Scan LLM applications against compliance frameworks.

    This scanner runs red team attacks mapped to framework requirements
    and generates compliance reports. It accepts a judge backend for
    evaluating attack results.

    Args:
        judge: A JudgeBackend instance for evaluating LLM responses.
        concurrency: Maximum concurrent attack executions.

    Usage::

        from checkllm.compliance_frameworks import ComplianceScanner, ComplianceFramework

        scanner = ComplianceScanner(judge=judge)
        report = await scanner.scan(
            target=my_llm_function,
            frameworks=[ComplianceFramework.OWASP_LLM_TOP10, ComplianceFramework.EU_AI_ACT],
        )
        print(report.summary())
    """

    def __init__(self, judge: Any = None, concurrency: int = 10) -> None:
        self._judge = judge
        self._concurrency = concurrency

    async def scan(
        self,
        target: Callable[[str], str] | Callable[[str], Awaitable[str]],
        frameworks: list[ComplianceFramework] | None = None,
    ) -> MultiFrameworkReport:
        """Scan the target against one or more compliance frameworks.

        Args:
            target: A callable that takes a prompt string and returns the
                LLM response. May be sync or async.
            frameworks: List of frameworks to scan against. Defaults to
                OWASP LLM Top 10.

        Returns:
            A MultiFrameworkReport with per-framework results and an
            aggregate score.
        """
        if frameworks is None:
            frameworks = [ComplianceFramework.OWASP_LLM_TOP10]

        reports: list[ComplianceReport] = []
        timestamp = datetime.datetime.now(datetime.timezone.utc).isoformat()

        for framework in frameworks:
            report = await self._scan_framework(target, framework, timestamp)
            reports.append(report)

        overall = 0.0
        if reports:
            overall = sum(r.overall_score for r in reports) / len(reports)

        return MultiFrameworkReport(
            reports=reports,
            timestamp=timestamp,
            overall_score=overall,
        )

    async def scan_single(
        self,
        target: Callable[[str], str] | Callable[[str], Awaitable[str]],
        framework: ComplianceFramework,
    ) -> ComplianceReport:
        """Scan the target against a single compliance framework.

        Args:
            target: A callable that takes a prompt string and returns the
                LLM response. May be sync or async.
            framework: The framework to scan against.

        Returns:
            A ComplianceReport with per-requirement results.
        """
        timestamp = datetime.datetime.now(datetime.timezone.utc).isoformat()
        return await self._scan_framework(target, framework, timestamp)

    async def _scan_framework(
        self,
        target: Callable[[str], str] | Callable[[str], Awaitable[str]],
        framework: ComplianceFramework,
        timestamp: str,
    ) -> ComplianceReport:
        """Scan a single framework and produce a compliance report.

        Args:
            target: The LLM callable to test.
            framework: The framework to evaluate.
            timestamp: ISO-format timestamp for the report.

        Returns:
            A ComplianceReport with results for each requirement.
        """
        mapping = get_framework_mapping(framework)
        results: list[RequirementResult] = []
        semaphore = asyncio.Semaphore(self._concurrency)

        for requirement in mapping.requirements:
            async with semaphore:
                result = await self._evaluate_requirement(target, requirement)
                results.append(result)

        passed = sum(1 for r in results if r.status == "passed")
        failed = sum(1 for r in results if r.status == "failed")
        skipped = sum(1 for r in results if r.status == "skipped")
        total = len(results)
        score = passed / total if total > 0 else 0.0

        return ComplianceReport(
            framework=framework,
            version=mapping.version,
            timestamp=timestamp,
            total_requirements=total,
            passed=passed,
            failed=failed,
            skipped=skipped,
            results=results,
            overall_score=score,
        )

    async def _evaluate_requirement(
        self,
        target: Callable[[str], str] | Callable[[str], Awaitable[str]],
        requirement: FrameworkRequirement,
    ) -> RequirementResult:
        """Evaluate a single requirement by running mapped red team tests.

        If no judge is configured, the requirement is skipped. Otherwise,
        this method uses the RedTeamer to run attacks for each vulnerability
        type mapped in the requirement and aggregates the results.

        Args:
            target: The LLM callable to test.
            requirement: The requirement to evaluate.

        Returns:
            A RequirementResult with pass/fail status and findings.
        """
        if self._judge is None:
            return RequirementResult(
                requirement_id=requirement.id,
                requirement_name=requirement.name,
                status="skipped",
                severity=requirement.severity,
                findings=["No judge configured; skipping requirement."],
            )

        valid_types = _get_valid_vuln_type_values()
        mapped_types = [
            t for t in requirement.test_categories if t in valid_types
        ]

        if not mapped_types:
            return RequirementResult(
                requirement_id=requirement.id,
                requirement_name=requirement.name,
                status="skipped",
                severity=requirement.severity,
                findings=["No testable vulnerability types mapped."],
            )

        findings: list[str] = []
        evidence: list[str] = []
        any_vulnerable = False

        try:
            from checkllm.redteam import RedTeamer

            teamer = RedTeamer(judge=self._judge)

            vuln_types = [VulnerabilityType(t) for t in mapped_types]
            report = await teamer.scan(
                target=target,
                vulnerabilities=vuln_types,
                attacks_per_type=1,
            )

            for attack_result in report.results:
                if attack_result.vulnerable:
                    any_vulnerable = True
                    findings.append(
                        f"{attack_result.vulnerability_type.value} "
                        f"({attack_result.attack_strategy.value}): "
                        f"{attack_result.reasoning or 'attack succeeded'}"
                    )
                evidence.append(
                    f"[{attack_result.vulnerability_type.value}] "
                    f"vulnerable={attack_result.vulnerable}"
                )
        except Exception as exc:
            logger.warning(
                "Error evaluating requirement %s: %s",
                requirement.id,
                exc,
            )
            return RequirementResult(
                requirement_id=requirement.id,
                requirement_name=requirement.name,
                status="skipped",
                severity=requirement.severity,
                findings=[f"Evaluation error: {exc}"],
            )

        status = "failed" if any_vulnerable else "passed"
        return RequirementResult(
            requirement_id=requirement.id,
            requirement_name=requirement.name,
            status=status,
            severity=requirement.severity,
            findings=findings,
            evidence=evidence,
        )
