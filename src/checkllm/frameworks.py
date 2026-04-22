"""Compliance framework definitions and requirement mappings.

Defines all supported compliance and security frameworks with detailed
requirement-level mappings to checkllm evaluation capabilities including
vulnerability types, metrics, and deterministic checks.

Usage::

    from checkllm.frameworks import ComplianceFramework, get_framework_definition

    defn = get_framework_definition(ComplianceFramework.OWASP_LLM_TOP_10)
    for req in defn.requirements:
        print(f"{req.id}: {req.title}")
"""

from __future__ import annotations

from enum import Enum
from typing import Any

from pydantic import BaseModel, Field


class ComplianceFramework(str, Enum):
    """Supported compliance and security frameworks."""

    OWASP_LLM_TOP_10 = "owasp_llm_top_10"
    OWASP_API_TOP_10 = "owasp_api_top_10"
    OWASP_AGENTIC_AI = "owasp_agentic_ai"
    NIST_AI_RMF = "nist_ai_rmf"
    EU_AI_ACT = "eu_ai_act"
    MITRE_ATLAS = "mitre_atlas"
    SOC2_TYPE2 = "soc2_type2"
    ISO_42001 = "iso_42001"
    ISO_27001_AI = "iso_27001_ai"
    HIPAA_AI = "hipaa_ai"
    GDPR_AI = "gdpr_ai"
    PCI_DSS_AI = "pci_dss_ai"
    NIST_CSF = "nist_csf"
    CIS_AI_CONTROLS = "cis_ai_controls"


class FrameworkRequirement(BaseModel):
    """A single requirement within a compliance framework."""

    id: str
    title: str
    description: str
    severity: str
    category: str
    vulnerability_types: list[str] = Field(default_factory=list)
    metrics: list[str] = Field(default_factory=list)
    deterministic_checks: list[str] = Field(default_factory=list)
    remediation: str = ""


class FrameworkDefinition(BaseModel):
    """Complete definition of a compliance framework."""

    name: str
    version: str
    description: str
    url: str
    requirements: list[FrameworkRequirement] = Field(default_factory=list)
    total_requirements: int = 0


def _build_owasp_llm_top_10() -> FrameworkDefinition:
    """Build the OWASP LLM Top 10 (2025) framework definition.

    Returns:
        A FrameworkDefinition with 10 requirements mapping to OWASP LLM
        categories.
    """
    reqs = [
        FrameworkRequirement(
            id="LLM01",
            title="Prompt Injection",
            description=(
                "Crafted inputs that manipulate the LLM into executing "
                "unintended actions by overriding system instructions, "
                "including direct injection and indirect injection via "
                "external data sources."
            ),
            severity="critical",
            category="Input Validation",
            vulnerability_types=[
                "prompt_injection",
                "jailbreak",
                "instruction_override",
                "context_manipulation",
                "delimiter_injection",
                "token_smuggling",
                "ascii_smuggling",
                "goal_hijacking",
            ],
            metrics=[
                "role_adherence",
                "instruction_following",
                "role_violation",
            ],
            deterministic_checks=["contains", "not_contains", "regex_match"],
            remediation=(
                "Implement strict input validation, privilege control, and "
                "output filtering. Use delimiters to separate trusted and "
                "untrusted input. Apply least-privilege principles."
            ),
        ),
        FrameworkRequirement(
            id="LLM02",
            title="Insecure Output Handling",
            description=(
                "Failure to validate, sanitize, or encode LLM outputs "
                "before passing them to downstream components, enabling "
                "injection attacks like XSS, CSRF, SSRF, or command "
                "injection."
            ),
            severity="critical",
            category="Output Validation",
            vulnerability_types=[
                "sql_injection",
                "shell_injection",
                "markdown_injection",
                "xml_injection",
                "json_injection",
                "harmful_content",
            ],
            metrics=[
                "toxicity",
                "misuse_detection",
                "correctness",
            ],
            deterministic_checks=[
                "not_contains",
                "json_schema",
                "regex_match",
            ],
            remediation=(
                "Treat model output as untrusted. Apply output encoding "
                "for the downstream context (HTML, SQL, shell). Validate "
                "against allow-lists before executing."
            ),
        ),
        FrameworkRequirement(
            id="LLM03",
            title="Training Data Poisoning",
            description=(
                "Manipulation of pre-training, fine-tuning, or embedding "
                "data to introduce vulnerabilities, backdoors, or biases "
                "into the model."
            ),
            severity="high",
            category="Data Integrity",
            vulnerability_types=[
                "memory_poisoning",
                "context_manipulation",
                "bias_exploitation",
            ],
            metrics=[
                "faithfulness",
                "groundedness",
                "bias",
                "correctness",
            ],
            deterministic_checks=["contains", "not_contains"],
            remediation=(
                "Vet training data sources and supply chain. Use data "
                "sanitization pipelines. Monitor model outputs for "
                "distributional drift."
            ),
        ),
        FrameworkRequirement(
            id="LLM04",
            title="Model Denial of Service",
            description=(
                "Adversarial inputs that cause the model to consume "
                "excessive resources, leading to degraded service quality "
                "or high costs for other users."
            ),
            severity="high",
            category="Availability",
            vulnerability_types=[
                "encoding_attack",
                "unicode_exploit",
                "token_smuggling",
            ],
            metrics=["step_efficiency"],
            deterministic_checks=["max_tokens", "latency", "cost"],
            remediation=(
                "Implement input token limits, rate limiting, and "
                "resource caps per request. Monitor for anomalous "
                "resource consumption patterns."
            ),
        ),
        FrameworkRequirement(
            id="LLM05",
            title="Supply Chain Vulnerabilities",
            description=(
                "Risks from third-party datasets, pre-trained models, "
                "plugins, and deployment platforms that may introduce "
                "vulnerabilities."
            ),
            severity="high",
            category="Supply Chain",
            vulnerability_types=[
                "copyright_violation",
            ],
            metrics=["faithfulness", "correctness"],
            deterministic_checks=["json_schema"],
            remediation=(
                "Vet all third-party components. Use signed model "
                "checksums. Maintain a software bill of materials "
                "(SBOM) for model dependencies."
            ),
        ),
        FrameworkRequirement(
            id="LLM06",
            title="Sensitive Information Disclosure",
            description=(
                "LLM outputs that reveal sensitive data including PII, "
                "credentials, proprietary algorithms, or confidential "
                "business information."
            ),
            severity="critical",
            category="Data Privacy",
            vulnerability_types=[
                "pii_leakage",
                "data_extraction",
                "pii_api_access",
                "pii_social_engineering",
                "pii_database",
                "session_leak",
                "cross_session_leak",
                "prompt_extraction",
                "gdpr_violation",
                "hipaa_violation",
                "coppa_violation",
            ],
            metrics=["pii_detection", "non_advice"],
            deterministic_checks=["no_pii", "not_contains"],
            remediation=(
                "Apply data sanitization to training data. Implement "
                "PII detection and redaction in outputs. Use access "
                "controls for sensitive contexts."
            ),
        ),
        FrameworkRequirement(
            id="LLM07",
            title="Insecure Plugin Design",
            description=(
                "LLM plugins that accept free-form text without "
                "validation, enabling privilege escalation, remote code "
                "execution, or data exfiltration."
            ),
            severity="critical",
            category="Plugin Security",
            vulnerability_types=[
                "sql_injection",
                "shell_injection",
                "privilege_escalation",
            ],
            metrics=[
                "tool_accuracy",
                "mcp_use",
                "mcp_task_completion",
            ],
            deterministic_checks=["json_schema", "not_contains"],
            remediation=(
                "Enforce strict parameterized inputs to plugins. Apply "
                "least-privilege for plugin permissions. Require human "
                "approval for sensitive operations."
            ),
        ),
        FrameworkRequirement(
            id="LLM08",
            title="Excessive Agency",
            description=(
                "Granting the LLM too much autonomy, functionality, or "
                "permissions, allowing it to take unintended actions with "
                "real-world consequences."
            ),
            severity="high",
            category="Authorization",
            vulnerability_types=[
                "excessive_agency",
                "role_escape",
                "privilege_escalation",
                "goal_hijacking",
            ],
            metrics=[
                "role_adherence",
                "goal_accuracy",
                "task_completion",
                "tool_accuracy",
            ],
            deterministic_checks=["not_contains"],
            remediation=(
                "Limit plugin scope and permissions. Require human-in-the-"
                "loop for high-impact actions. Log all LLM-initiated "
                "actions for audit."
            ),
        ),
        FrameworkRequirement(
            id="LLM09",
            title="Overreliance",
            description=(
                "Users or systems placing excessive trust in LLM outputs "
                "without verification, leading to misinformation, legal "
                "issues, or security vulnerabilities."
            ),
            severity="medium",
            category="Trust Management",
            vulnerability_types=[
                "overreliance",
                "hallucination_attack",
                "regulated_advice",
                "unauthorized_practice",
                "competitor_endorsement",
                "political_opinion",
            ],
            metrics=[
                "hallucination",
                "faithfulness",
                "groundedness",
                "non_advice",
            ],
            deterministic_checks=["contains", "not_contains"],
            remediation=(
                "Clearly communicate model limitations. Implement "
                "disclaimers for high-stakes domains. Cross-reference "
                "outputs with authoritative sources."
            ),
        ),
        FrameworkRequirement(
            id="LLM10",
            title="Model Theft",
            description=(
                "Unauthorized extraction of the model's weights, "
                "architecture, or proprietary training data through "
                "crafted queries or side-channel attacks."
            ),
            severity="high",
            category="Intellectual Property",
            vulnerability_types=[
                "data_extraction",
                "prompt_extraction",
            ],
            metrics=["role_adherence"],
            deterministic_checks=["not_contains"],
            remediation=(
                "Implement API rate limiting and query monitoring. "
                "Use watermarking techniques. Restrict model access "
                "with strong authentication."
            ),
        ),
    ]
    return FrameworkDefinition(
        name="OWASP Top 10 for LLM Applications",
        version="2025",
        description=(
            "The OWASP Top 10 for LLM Applications identifies the most "
            "critical security risks in applications leveraging large "
            "language models."
        ),
        url="https://owasp.org/www-project-top-10-for-large-language-model-applications/",
        requirements=reqs,
        total_requirements=len(reqs),
    )


def _build_owasp_api_top_10() -> FrameworkDefinition:
    """Build the OWASP API Security Top 10 framework definition.

    Returns:
        A FrameworkDefinition with 10 requirements for API security.
    """
    reqs = [
        FrameworkRequirement(
            id="API1",
            title="Broken Object Level Authorization",
            description=(
                "APIs exposing endpoints that handle object identifiers "
                "without verifying the user has access to the requested "
                "resource."
            ),
            severity="critical",
            category="Authorization",
            vulnerability_types=["privilege_escalation", "data_extraction"],
            metrics=["role_adherence", "tool_accuracy"],
            deterministic_checks=["not_contains"],
            remediation=(
                "Implement per-object authorization checks. Validate "
                "that the authenticated user owns the requested resource."
            ),
        ),
        FrameworkRequirement(
            id="API2",
            title="Broken Authentication",
            description=(
                "Weak or flawed authentication mechanisms that allow "
                "attackers to assume other users' identities."
            ),
            severity="critical",
            category="Authentication",
            vulnerability_types=[
                "session_leak",
                "cross_session_leak",
                "prompt_extraction",
            ],
            metrics=["role_adherence"],
            deterministic_checks=["not_contains", "no_pii"],
            remediation=(
                "Use standard authentication frameworks. Implement "
                "multi-factor authentication. Rotate tokens regularly."
            ),
        ),
        FrameworkRequirement(
            id="API3",
            title="Broken Object Property Level Authorization",
            description=(
                "Lack of or improper validation of object property-level "
                "access, allowing unauthorized read or write access to "
                "sensitive properties."
            ),
            severity="high",
            category="Authorization",
            vulnerability_types=["data_extraction", "pii_leakage"],
            metrics=["pii_detection"],
            deterministic_checks=["no_pii", "json_schema"],
            remediation=(
                "Validate user access per-property. Return only fields "
                "the user is authorized to view."
            ),
        ),
        FrameworkRequirement(
            id="API4",
            title="Unrestricted Resource Consumption",
            description=(
                "APIs that do not limit the size or number of resources "
                "requested, enabling denial of service or cost exhaustion."
            ),
            severity="high",
            category="Availability",
            vulnerability_types=["encoding_attack", "unicode_exploit"],
            metrics=["step_efficiency"],
            deterministic_checks=["max_tokens", "latency", "cost"],
            remediation=(
                "Implement rate limiting, pagination, and resource quotas. "
                "Set maximum response sizes."
            ),
        ),
        FrameworkRequirement(
            id="API5",
            title="Broken Function Level Authorization",
            description=(
                "Exposure of administrative or privileged functions to "
                "unauthorized users through flawed access controls."
            ),
            severity="critical",
            category="Authorization",
            vulnerability_types=[
                "privilege_escalation",
                "role_escape",
                "excessive_agency",
            ],
            metrics=["role_adherence", "role_violation"],
            deterministic_checks=["not_contains"],
            remediation=(
                "Enforce role-based access control. Deny by default "
                "and explicitly grant permissions."
            ),
        ),
        FrameworkRequirement(
            id="API6",
            title="Unrestricted Access to Sensitive Business Flows",
            description=(
                "APIs that expose business-sensitive functionality "
                "without proper safeguards against automated abuse."
            ),
            severity="high",
            category="Business Logic",
            vulnerability_types=["excessive_agency", "goal_hijacking"],
            metrics=["goal_accuracy", "task_completion"],
            deterministic_checks=["not_contains"],
            remediation=(
                "Implement business logic rate limiting. Add CAPTCHA "
                "or human verification for sensitive flows."
            ),
        ),
        FrameworkRequirement(
            id="API7",
            title="Server Side Request Forgery",
            description=(
                "APIs that fetch remote resources based on user-supplied "
                "URLs without proper validation, enabling SSRF attacks."
            ),
            severity="high",
            category="Input Validation",
            vulnerability_types=["shell_injection", "sql_injection"],
            metrics=["tool_accuracy"],
            deterministic_checks=["not_contains", "regex_match"],
            remediation=(
                "Validate and sanitize all user-supplied URLs. Use "
                "allow-lists for permitted domains. Disable redirects."
            ),
        ),
        FrameworkRequirement(
            id="API8",
            title="Security Misconfiguration",
            description=(
                "Improper or insecure default configurations, open "
                "cloud storage, verbose error messages, or unnecessary "
                "features enabled."
            ),
            severity="medium",
            category="Configuration",
            vulnerability_types=["prompt_extraction", "data_extraction"],
            metrics=["role_adherence"],
            deterministic_checks=["not_contains"],
            remediation=(
                "Harden all environment configurations. Disable verbose "
                "error output. Review default settings."
            ),
        ),
        FrameworkRequirement(
            id="API9",
            title="Improper Inventory Management",
            description=(
                "Outdated or undocumented API endpoints that are not "
                "properly managed, patched, or retired."
            ),
            severity="medium",
            category="Governance",
            vulnerability_types=["data_extraction"],
            metrics=["correctness"],
            deterministic_checks=["json_schema"],
            remediation=(
                "Maintain an up-to-date API inventory. Deprecate and "
                "remove unused endpoints. Version all APIs."
            ),
        ),
        FrameworkRequirement(
            id="API10",
            title="Unsafe Consumption of APIs",
            description=(
                "Developers trusting data received from third-party "
                "APIs without proper validation, enabling injection "
                "attacks through the supply chain."
            ),
            severity="high",
            category="Supply Chain",
            vulnerability_types=[
                "sql_injection",
                "shell_injection",
                "json_injection",
                "xml_injection",
            ],
            metrics=["tool_accuracy", "faithfulness"],
            deterministic_checks=["json_schema", "not_contains"],
            remediation=(
                "Validate all third-party API responses. Apply input "
                "sanitization to external data. Use transport encryption."
            ),
        ),
    ]
    return FrameworkDefinition(
        name="OWASP API Security Top 10",
        version="2023",
        description=(
            "The OWASP API Security Top 10 identifies the most critical security risks facing APIs."
        ),
        url="https://owasp.org/API-Security/editions/2023/en/0x11-t10/",
        requirements=reqs,
        total_requirements=len(reqs),
    )


def _build_owasp_agentic_ai() -> FrameworkDefinition:
    """Build the OWASP Agentic AI Top 10 framework definition.

    Returns:
        A FrameworkDefinition with 10 requirements for agentic AI security.
    """
    reqs = [
        FrameworkRequirement(
            id="AAI01",
            title="Agent Goal Hijacking",
            description=(
                "Attackers redirect an autonomous agent's objectives "
                "through crafted inputs, causing it to pursue malicious "
                "goals instead of its intended purpose."
            ),
            severity="critical",
            category="Goal Integrity",
            vulnerability_types=[
                "goal_hijacking",
                "prompt_injection",
                "instruction_override",
                "context_manipulation",
            ],
            metrics=[
                "goal_accuracy",
                "role_adherence",
                "instruction_following",
                "plan_adherence",
            ],
            deterministic_checks=["not_contains", "contains"],
            remediation=(
                "Implement goal validation at each agent step. Use "
                "invariant goal constraints that cannot be overridden "
                "by user input. Log goal state transitions."
            ),
        ),
        FrameworkRequirement(
            id="AAI02",
            title="Tool Misuse",
            description=(
                "An agent uses its available tools in unintended or "
                "harmful ways, executing operations beyond its "
                "authorized scope or with malicious parameters."
            ),
            severity="critical",
            category="Tool Security",
            vulnerability_types=[
                "sql_injection",
                "shell_injection",
                "privilege_escalation",
                "excessive_agency",
            ],
            metrics=[
                "tool_accuracy",
                "mcp_use",
                "mcp_task_completion",
                "multi_turn_mcp_use",
            ],
            deterministic_checks=["json_schema", "not_contains"],
            remediation=(
                "Apply strict input validation to all tool parameters. "
                "Use allow-lists for permitted tool operations. Require "
                "human approval for destructive actions."
            ),
        ),
        FrameworkRequirement(
            id="AAI03",
            title="Privilege Escalation",
            description=(
                "An agent gains access to resources or capabilities "
                "beyond its intended authorization level, either through "
                "exploitation or through accumulated permissions."
            ),
            severity="critical",
            category="Authorization",
            vulnerability_types=[
                "privilege_escalation",
                "role_escape",
                "excessive_agency",
            ],
            metrics=[
                "role_adherence",
                "role_violation",
                "tool_accuracy",
            ],
            deterministic_checks=["not_contains"],
            remediation=(
                "Apply principle of least privilege. Implement "
                "per-action authorization checks. Use capability-based "
                "security with non-transferable tokens."
            ),
        ),
        FrameworkRequirement(
            id="AAI04",
            title="Identity Spoofing",
            description=(
                "An attacker impersonates another agent, user, or "
                "system component to gain unauthorized access or "
                "inject malicious instructions into the agent pipeline."
            ),
            severity="high",
            category="Authentication",
            vulnerability_types=[
                "prompt_injection",
                "context_manipulation",
                "session_leak",
                "cross_session_leak",
            ],
            metrics=["role_adherence", "instruction_following"],
            deterministic_checks=["not_contains", "contains"],
            remediation=(
                "Implement cryptographic identity verification between "
                "agents. Use signed messages. Validate the source of "
                "all inter-agent communications."
            ),
        ),
        FrameworkRequirement(
            id="AAI05",
            title="Memory Poisoning",
            description=(
                "Corruption of an agent's persistent memory or context "
                "window to alter its future behavior, inject false "
                "information, or bypass safety constraints."
            ),
            severity="critical",
            category="Memory Integrity",
            vulnerability_types=[
                "memory_poisoning",
                "context_manipulation",
                "hallucination_attack",
            ],
            metrics=[
                "knowledge_retention",
                "faithfulness",
                "groundedness",
                "consistency",
            ],
            deterministic_checks=["not_contains", "contains"],
            remediation=(
                "Implement memory integrity checks. Use append-only "
                "memory with validation. Periodically audit memory "
                "contents for anomalies."
            ),
        ),
        FrameworkRequirement(
            id="AAI06",
            title="Cascading Hallucinations",
            description=(
                "Hallucinated outputs from one agent are consumed by "
                "downstream agents as factual input, amplifying errors "
                "through the agent pipeline."
            ),
            severity="high",
            category="Output Integrity",
            vulnerability_types=[
                "hallucination_attack",
                "overreliance",
                "context_manipulation",
            ],
            metrics=[
                "hallucination",
                "faithfulness",
                "groundedness",
                "answer_completeness",
            ],
            deterministic_checks=["contains", "not_contains"],
            remediation=(
                "Implement fact-checking at each agent boundary. Use "
                "confidence thresholds for inter-agent data passing. "
                "Add source attribution to all claims."
            ),
        ),
        FrameworkRequirement(
            id="AAI07",
            title="Resource Exhaustion",
            description=(
                "An agent enters an infinite loop, recursive chain, "
                "or resource-intensive computation that exhausts system "
                "resources or generates excessive costs."
            ),
            severity="high",
            category="Availability",
            vulnerability_types=[
                "encoding_attack",
                "unicode_exploit",
                "token_smuggling",
            ],
            metrics=["step_efficiency", "task_completion"],
            deterministic_checks=["max_tokens", "latency", "cost"],
            remediation=(
                "Set maximum step limits and timeouts for agent "
                "execution. Implement circuit breakers. Monitor "
                "resource consumption per agent run."
            ),
        ),
        FrameworkRequirement(
            id="AAI08",
            title="Uncontrolled Autonomy",
            description=(
                "An agent takes high-impact actions without adequate "
                "human oversight, approval gates, or the ability for "
                "users to intervene and stop execution."
            ),
            severity="critical",
            category="Human Oversight",
            vulnerability_types=[
                "excessive_agency",
                "role_escape",
                "goal_hijacking",
                "privilege_escalation",
            ],
            metrics=[
                "role_adherence",
                "goal_accuracy",
                "plan_adherence",
                "task_completion",
            ],
            deterministic_checks=["not_contains"],
            remediation=(
                "Require human-in-the-loop approval for high-impact "
                "actions. Implement kill switches. Log all autonomous "
                "decisions with justifications."
            ),
        ),
        FrameworkRequirement(
            id="AAI09",
            title="Insecure Communication",
            description=(
                "Unprotected or unvalidated communication channels "
                "between agents, allowing interception, modification, "
                "or injection of messages."
            ),
            severity="high",
            category="Communication Security",
            vulnerability_types=[
                "prompt_injection",
                "context_manipulation",
                "session_leak",
                "data_extraction",
            ],
            metrics=["role_adherence", "tool_accuracy"],
            deterministic_checks=["not_contains", "no_pii"],
            remediation=(
                "Encrypt all inter-agent communication. Validate "
                "message integrity with checksums. Implement "
                "message authentication codes."
            ),
        ),
        FrameworkRequirement(
            id="AAI10",
            title="Insufficient Logging",
            description=(
                "Inadequate logging and monitoring of agent actions, "
                "decisions, and communications, making it impossible "
                "to detect attacks or audit behavior."
            ),
            severity="medium",
            category="Observability",
            vulnerability_types=[
                "prompt_extraction",
                "data_extraction",
                "excessive_agency",
            ],
            metrics=["task_completion", "step_efficiency"],
            deterministic_checks=["contains"],
            remediation=(
                "Log all agent actions, tool invocations, and "
                "decision rationale. Implement real-time monitoring "
                "and anomaly detection. Retain logs for audit."
            ),
        ),
    ]
    return FrameworkDefinition(
        name="OWASP Top 10 for Agentic AI",
        version="2025",
        description=(
            "The OWASP Top 10 for Agentic AI identifies the most "
            "critical security risks specific to autonomous AI agents "
            "that use tools, memory, and multi-step reasoning."
        ),
        url="https://owasp.org/www-project-agentic-ai-threats/",
        requirements=reqs,
        total_requirements=len(reqs),
    )


def _build_nist_ai_rmf() -> FrameworkDefinition:
    """Build the NIST AI RMF (AI 100-1) framework definition.

    Returns:
        A FrameworkDefinition with requirements across GOVERN, MAP,
        MEASURE, and MANAGE functions.
    """
    reqs = [
        FrameworkRequirement(
            id="GOVERN-1.1",
            title="AI Risk Management Policies",
            description=(
                "Policies, processes, and procedures are in place to "
                "manage AI risks and are integrated into broader "
                "enterprise risk management."
            ),
            severity="high",
            category="GOVERN",
            vulnerability_types=[
                "prompt_injection",
                "jailbreak",
                "harmful_content",
            ],
            metrics=["role_adherence", "instruction_following"],
            deterministic_checks=["not_contains"],
            remediation=(
                "Establish formal AI risk management policies. "
                "Document risk tolerance levels and escalation paths."
            ),
        ),
        FrameworkRequirement(
            id="GOVERN-1.2",
            title="Accountability Structures",
            description=(
                "Roles and responsibilities for AI risk management "
                "are clearly defined and assigned across the "
                "organization."
            ),
            severity="medium",
            category="GOVERN",
            vulnerability_types=["excessive_agency", "privilege_escalation"],
            metrics=["role_adherence"],
            deterministic_checks=[],
            remediation=(
                "Define clear RACI matrices for AI governance. "
                "Assign executive sponsors for AI risk oversight."
            ),
        ),
        FrameworkRequirement(
            id="GOVERN-2.1",
            title="Transparency and Documentation",
            description=(
                "System design decisions, assumptions, and limitations "
                "are documented and accessible to relevant stakeholders."
            ),
            severity="high",
            category="GOVERN",
            vulnerability_types=[
                "prompt_extraction",
                "hallucination_attack",
                "overreliance",
            ],
            metrics=[
                "faithfulness",
                "groundedness",
                "non_advice",
            ],
            deterministic_checks=["contains"],
            remediation=(
                "Maintain comprehensive system documentation. "
                "Publish model cards with capabilities and limitations."
            ),
        ),
        FrameworkRequirement(
            id="GOVERN-2.2",
            title="Stakeholder Engagement",
            description=(
                "Impacted communities and stakeholders are engaged "
                "in AI system design and deployment decisions."
            ),
            severity="medium",
            category="GOVERN",
            vulnerability_types=["bias_exploitation", "political_opinion"],
            metrics=["bias"],
            deterministic_checks=[],
            remediation=(
                "Conduct stakeholder impact assessments. Establish "
                "feedback mechanisms for affected communities."
            ),
        ),
        FrameworkRequirement(
            id="GOVERN-3.1",
            title="Workforce AI Literacy",
            description=(
                "Personnel involved in AI system lifecycle have the "
                "necessary skills and training to manage AI risks."
            ),
            severity="medium",
            category="GOVERN",
            vulnerability_types=["overreliance", "regulated_advice"],
            metrics=["non_advice"],
            deterministic_checks=[],
            remediation=(
                "Implement AI literacy training programs. Ensure "
                "operators understand model limitations."
            ),
        ),
        FrameworkRequirement(
            id="MAP-1.1",
            title="Intended Purpose Definition",
            description=(
                "The intended purpose, context of use, and potential "
                "impacts of the AI system are clearly defined."
            ),
            severity="high",
            category="MAP",
            vulnerability_types=[
                "goal_hijacking",
                "excessive_agency",
                "role_escape",
            ],
            metrics=[
                "goal_accuracy",
                "role_adherence",
                "task_completion",
            ],
            deterministic_checks=["contains", "not_contains"],
            remediation=(
                "Document clear use-case boundaries. Define out-of-scope applications explicitly."
            ),
        ),
        FrameworkRequirement(
            id="MAP-1.2",
            title="Interdependency Analysis",
            description=(
                "Dependencies and interactions between AI components "
                "and other systems are mapped and analyzed."
            ),
            severity="medium",
            category="MAP",
            vulnerability_types=[
                "sql_injection",
                "shell_injection",
                "json_injection",
            ],
            metrics=["tool_accuracy", "mcp_use"],
            deterministic_checks=["json_schema"],
            remediation=(
                "Create dependency maps for all AI system components. "
                "Assess cascading failure risks."
            ),
        ),
        FrameworkRequirement(
            id="MAP-2.1",
            title="Risk Identification",
            description=(
                "AI risks are identified through systematic analysis "
                "including adversarial testing, red teaming, and "
                "failure mode analysis."
            ),
            severity="critical",
            category="MAP",
            vulnerability_types=[
                "prompt_injection",
                "jailbreak",
                "context_manipulation",
                "instruction_override",
            ],
            metrics=[
                "role_adherence",
                "instruction_following",
                "role_violation",
            ],
            deterministic_checks=["not_contains"],
            remediation=(
                "Conduct regular red teaming exercises. Implement "
                "automated adversarial testing in CI/CD pipelines."
            ),
        ),
        FrameworkRequirement(
            id="MAP-2.2",
            title="Bias and Fairness Assessment",
            description=(
                "Potential biases in data, model, and outputs are "
                "systematically identified and assessed."
            ),
            severity="high",
            category="MAP",
            vulnerability_types=[
                "bias_exploitation",
                "hate_speech",
                "political_opinion",
            ],
            metrics=["bias", "toxicity", "sentiment"],
            deterministic_checks=["not_contains"],
            remediation=(
                "Implement bias testing across demographic groups. "
                "Use fairness metrics in model evaluation."
            ),
        ),
        FrameworkRequirement(
            id="MAP-3.1",
            title="Benefits and Costs Assessment",
            description=(
                "Benefits and costs of the AI system are assessed "
                "relative to the status quo and alternatives."
            ),
            severity="low",
            category="MAP",
            vulnerability_types=[],
            metrics=["correctness", "relevance"],
            deterministic_checks=["cost"],
            remediation=(
                "Conduct cost-benefit analysis including externalities. "
                "Compare AI system performance to baselines."
            ),
        ),
        FrameworkRequirement(
            id="MEASURE-1.1",
            title="Evaluation Metrics",
            description=(
                "Appropriate metrics are identified and applied to "
                "measure AI system performance, fairness, and safety."
            ),
            severity="high",
            category="MEASURE",
            vulnerability_types=[
                "hallucination_attack",
                "overreliance",
                "bias_exploitation",
            ],
            metrics=[
                "correctness",
                "faithfulness",
                "hallucination",
                "bias",
                "toxicity",
            ],
            deterministic_checks=["contains", "not_contains"],
            remediation=(
                "Define comprehensive evaluation metric suites. "
                "Include both performance and safety metrics."
            ),
        ),
        FrameworkRequirement(
            id="MEASURE-2.1",
            title="Robustness Testing",
            description=(
                "AI system is tested for robustness against adversarial "
                "inputs, distribution shifts, and edge cases."
            ),
            severity="critical",
            category="MEASURE",
            vulnerability_types=[
                "prompt_injection",
                "jailbreak",
                "encoding_attack",
                "unicode_exploit",
                "ascii_smuggling",
                "delimiter_injection",
            ],
            metrics=[
                "role_adherence",
                "instruction_following",
                "consistency",
            ],
            deterministic_checks=["not_contains", "regex_match"],
            remediation=(
                "Implement comprehensive adversarial test suites. "
                "Test with perturbed inputs and edge cases."
            ),
        ),
        FrameworkRequirement(
            id="MEASURE-2.2",
            title="Privacy Testing",
            description=(
                "AI system is evaluated for privacy risks including "
                "data leakage, memorization, and inference attacks."
            ),
            severity="critical",
            category="MEASURE",
            vulnerability_types=[
                "pii_leakage",
                "data_extraction",
                "pii_api_access",
                "pii_social_engineering",
                "session_leak",
                "cross_session_leak",
            ],
            metrics=["pii_detection"],
            deterministic_checks=["no_pii", "not_contains"],
            remediation=(
                "Conduct membership inference and data extraction "
                "attacks. Test for PII memorization."
            ),
        ),
        FrameworkRequirement(
            id="MEASURE-3.1",
            title="Continuous Monitoring",
            description=(
                "AI system performance and safety metrics are monitored continuously in production."
            ),
            severity="high",
            category="MEASURE",
            vulnerability_types=[],
            metrics=[
                "correctness",
                "faithfulness",
                "relevance",
                "consistency",
            ],
            deterministic_checks=["latency", "cost"],
            remediation=(
                "Implement production monitoring dashboards. Set "
                "alerting thresholds for performance degradation."
            ),
        ),
        FrameworkRequirement(
            id="MANAGE-1.1",
            title="Risk Prioritization",
            description=(
                "Identified AI risks are prioritized based on likelihood and potential impact."
            ),
            severity="high",
            category="MANAGE",
            vulnerability_types=[
                "harmful_content",
                "chemical_weapons",
                "violent_crime",
                "self_harm",
            ],
            metrics=["toxicity", "misuse_detection"],
            deterministic_checks=["not_contains"],
            remediation=(
                "Maintain a risk register with severity ratings. "
                "Prioritize mitigations for highest-impact risks."
            ),
        ),
        FrameworkRequirement(
            id="MANAGE-2.1",
            title="Risk Mitigation Strategies",
            description=(
                "Strategies are in place to mitigate identified "
                "risks, with documented residual risk acceptance."
            ),
            severity="high",
            category="MANAGE",
            vulnerability_types=[
                "harmful_content",
                "bias_exploitation",
                "hate_speech",
                "radicalization",
                "self_harm",
            ],
            metrics=["toxicity", "bias", "sentiment"],
            deterministic_checks=["not_contains"],
            remediation=(
                "Implement multi-layered defense strategies. Document "
                "accepted residual risks with justification."
            ),
        ),
        FrameworkRequirement(
            id="MANAGE-2.2",
            title="Incident Response",
            description=(
                "Mechanisms are in place to respond to and recover "
                "from AI system failures or security incidents."
            ),
            severity="high",
            category="MANAGE",
            vulnerability_types=[
                "prompt_injection",
                "data_extraction",
                "privilege_escalation",
            ],
            metrics=["role_adherence"],
            deterministic_checks=[],
            remediation=(
                "Establish AI-specific incident response procedures. "
                "Define rollback and recovery mechanisms."
            ),
        ),
        FrameworkRequirement(
            id="MANAGE-3.1",
            title="Third-Party Risk Management",
            description=(
                "Risks from third-party AI components, data sources, "
                "and service providers are managed."
            ),
            severity="medium",
            category="MANAGE",
            vulnerability_types=["copyright_violation"],
            metrics=["faithfulness", "correctness"],
            deterministic_checks=["json_schema"],
            remediation=(
                "Assess third-party AI component risks. Include AI "
                "risk clauses in vendor contracts."
            ),
        ),
        FrameworkRequirement(
            id="MANAGE-4.1",
            title="Decommissioning Plans",
            description=(
                "Plans exist for the safe decommissioning of AI "
                "systems, including data disposal and model retirement."
            ),
            severity="low",
            category="MANAGE",
            vulnerability_types=["data_extraction", "pii_leakage"],
            metrics=["pii_detection"],
            deterministic_checks=["no_pii"],
            remediation=(
                "Create end-of-life plans for AI systems. Ensure "
                "data and model artifacts are securely disposed."
            ),
        ),
        FrameworkRequirement(
            id="MANAGE-4.2",
            title="Documentation and Record-Keeping",
            description=(
                "Comprehensive records of AI system decisions, "
                "changes, and risk assessments are maintained."
            ),
            severity="medium",
            category="MANAGE",
            vulnerability_types=[],
            metrics=["correctness"],
            deterministic_checks=["contains"],
            remediation=(
                "Maintain audit trails for all AI system changes. "
                "Archive risk assessment documentation."
            ),
        ),
    ]
    return FrameworkDefinition(
        name="NIST AI Risk Management Framework",
        version="AI 100-1 (2023)",
        description=(
            "The NIST AI RMF provides a structured approach to "
            "managing risks associated with AI systems throughout "
            "their lifecycle, organized into GOVERN, MAP, MEASURE, "
            "and MANAGE functions."
        ),
        url="https://airc.nist.gov/AI_RMF_Knowlege_Base/Playbook",
        requirements=reqs,
        total_requirements=len(reqs),
    )


def _build_eu_ai_act() -> FrameworkDefinition:
    """Build the EU AI Act framework definition.

    Returns:
        A FrameworkDefinition with requirements from Articles 6-15
        covering high-risk AI system obligations.
    """
    reqs = [
        FrameworkRequirement(
            id="Article-6",
            title="Classification of High-Risk AI Systems",
            description=(
                "AI systems must be assessed against Annex III criteria "
                "to determine if they fall under high-risk classification "
                "and associated obligations."
            ),
            severity="high",
            category="Risk Classification",
            vulnerability_types=[
                "harmful_content",
                "bias_exploitation",
                "regulated_advice",
            ],
            metrics=["toxicity", "bias", "misuse_detection"],
            deterministic_checks=[],
            remediation=(
                "Conduct risk classification assessment against "
                "Annex III categories. Document classification "
                "rationale."
            ),
        ),
        FrameworkRequirement(
            id="Article-9.1",
            title="Risk Management System",
            description=(
                "A risk management system shall be established, "
                "implemented, documented, and maintained as a "
                "continuous iterative process throughout the AI "
                "system's lifecycle."
            ),
            severity="critical",
            category="Risk Management",
            vulnerability_types=[
                "prompt_injection",
                "jailbreak",
                "harmful_content",
                "bias_exploitation",
            ],
            metrics=[
                "role_adherence",
                "toxicity",
                "bias",
                "correctness",
            ],
            deterministic_checks=["not_contains"],
            remediation=(
                "Establish a continuous risk management process. "
                "Document known risks and mitigation measures."
            ),
        ),
        FrameworkRequirement(
            id="Article-9.2",
            title="Risk Identification and Analysis",
            description=(
                "Known and reasonably foreseeable risks that the "
                "high-risk AI system can pose to health, safety, "
                "or fundamental rights shall be identified and "
                "analyzed."
            ),
            severity="critical",
            category="Risk Management",
            vulnerability_types=[
                "self_harm",
                "chemical_weapons",
                "violent_crime",
                "radicalization",
                "illegal_activities",
            ],
            metrics=["toxicity", "misuse_detection"],
            deterministic_checks=["not_contains"],
            remediation=(
                "Conduct systematic risk assessments including "
                "adversarial testing. Maintain a risk register."
            ),
        ),
        FrameworkRequirement(
            id="Article-9.4",
            title="Testing for Risk Management",
            description=(
                "Testing shall be carried out to ensure the AI "
                "system performs consistently and in compliance "
                "with its intended purpose."
            ),
            severity="high",
            category="Risk Management",
            vulnerability_types=[
                "prompt_injection",
                "encoding_attack",
                "unicode_exploit",
                "jailbreak",
            ],
            metrics=[
                "correctness",
                "consistency",
                "role_adherence",
            ],
            deterministic_checks=["not_contains", "contains"],
            remediation=(
                "Implement comprehensive testing including "
                "adversarial scenarios, edge cases, and regression."
            ),
        ),
        FrameworkRequirement(
            id="Article-10.1",
            title="Data Governance",
            description=(
                "High-risk AI systems using data for training shall "
                "be developed with data governance and management "
                "practices addressing data quality criteria."
            ),
            severity="high",
            category="Data Governance",
            vulnerability_types=[
                "memory_poisoning",
                "bias_exploitation",
                "copyright_violation",
            ],
            metrics=["faithfulness", "bias", "groundedness"],
            deterministic_checks=["not_contains"],
            remediation=(
                "Implement data quality frameworks. Document data "
                "provenance and processing pipelines."
            ),
        ),
        FrameworkRequirement(
            id="Article-10.3",
            title="Data Representativeness",
            description=(
                "Training, validation, and testing datasets shall be "
                "relevant, sufficiently representative, and to the "
                "best extent possible free of errors."
            ),
            severity="high",
            category="Data Governance",
            vulnerability_types=["bias_exploitation", "hallucination_attack"],
            metrics=["bias", "faithfulness", "hallucination"],
            deterministic_checks=[],
            remediation=(
                "Audit training data for representativeness. "
                "Address identified gaps in demographic coverage."
            ),
        ),
        FrameworkRequirement(
            id="Article-11",
            title="Technical Documentation",
            description=(
                "Technical documentation shall be drawn up before "
                "the AI system is placed on the market and shall "
                "be kept up to date."
            ),
            severity="medium",
            category="Documentation",
            vulnerability_types=[],
            metrics=["correctness"],
            deterministic_checks=["contains"],
            remediation=(
                "Maintain comprehensive technical documentation "
                "including system architecture, training details, "
                "and performance characteristics."
            ),
        ),
        FrameworkRequirement(
            id="Article-12",
            title="Record-Keeping and Logging",
            description=(
                "High-risk AI systems shall technically allow for "
                "automatic recording of events (logs) over the "
                "lifetime of the system."
            ),
            severity="high",
            category="Logging",
            vulnerability_types=[
                "data_extraction",
                "prompt_extraction",
            ],
            metrics=["task_completion"],
            deterministic_checks=["contains"],
            remediation=(
                "Implement comprehensive audit logging. Ensure "
                "logs capture input-output pairs, decisions, and "
                "anomalies."
            ),
        ),
        FrameworkRequirement(
            id="Article-13.1",
            title="Transparency for Users",
            description=(
                "High-risk AI systems shall be designed and developed "
                "in such a way as to ensure that their operation is "
                "sufficiently transparent to enable users to "
                "interpret the system's output."
            ),
            severity="high",
            category="Transparency",
            vulnerability_types=[
                "hallucination_attack",
                "overreliance",
                "unauthorized_practice",
            ],
            metrics=[
                "faithfulness",
                "groundedness",
                "non_advice",
                "hallucination",
            ],
            deterministic_checks=["contains"],
            remediation=(
                "Provide explanations for system outputs. Clearly "
                "communicate confidence levels and limitations."
            ),
        ),
        FrameworkRequirement(
            id="Article-13.3",
            title="AI System Disclosure",
            description=(
                "Users shall be informed that they are interacting "
                "with an AI system unless this is obvious from the "
                "circumstances."
            ),
            severity="medium",
            category="Transparency",
            vulnerability_types=["role_escape", "overreliance"],
            metrics=["role_adherence"],
            deterministic_checks=["contains"],
            remediation=(
                "Ensure clear AI disclosure in all user interactions. "
                "Label AI-generated content appropriately."
            ),
        ),
        FrameworkRequirement(
            id="Article-14.1",
            title="Human Oversight Design",
            description=(
                "High-risk AI systems shall be designed to be "
                "effectively overseen by natural persons during "
                "the period in which the AI system is in use."
            ),
            severity="critical",
            category="Human Oversight",
            vulnerability_types=[
                "excessive_agency",
                "goal_hijacking",
                "privilege_escalation",
            ],
            metrics=[
                "role_adherence",
                "goal_accuracy",
                "task_completion",
            ],
            deterministic_checks=["not_contains"],
            remediation=(
                "Design human-in-the-loop mechanisms. Implement "
                "override and stop capabilities. Enable meaningful "
                "human review of decisions."
            ),
        ),
        FrameworkRequirement(
            id="Article-14.4",
            title="Override and Intervention Capability",
            description=(
                "Individuals assigned human oversight shall be able "
                "to decide not to use the AI system, override or "
                "reverse its output, and intervene in its operation."
            ),
            severity="critical",
            category="Human Oversight",
            vulnerability_types=[
                "excessive_agency",
                "role_escape",
                "goal_hijacking",
            ],
            metrics=[
                "role_adherence",
                "plan_adherence",
                "goal_accuracy",
            ],
            deterministic_checks=["not_contains"],
            remediation=(
                "Implement accessible override mechanisms. Ensure "
                "humans can halt AI operations at any point."
            ),
        ),
        FrameworkRequirement(
            id="Article-15.1",
            title="Accuracy",
            description=(
                "High-risk AI systems shall be designed and developed "
                "to achieve an appropriate level of accuracy for their "
                "intended purpose."
            ),
            severity="high",
            category="Accuracy",
            vulnerability_types=["hallucination_attack", "overreliance"],
            metrics=[
                "correctness",
                "faithfulness",
                "hallucination",
                "answer_completeness",
            ],
            deterministic_checks=["contains", "not_contains"],
            remediation=(
                "Establish accuracy benchmarks. Monitor accuracy "
                "metrics continuously in production."
            ),
        ),
        FrameworkRequirement(
            id="Article-15.3",
            title="Robustness",
            description=(
                "High-risk AI systems shall be resilient as regards "
                "attempts by unauthorized third parties to alter "
                "their use or performance by exploiting "
                "vulnerabilities."
            ),
            severity="critical",
            category="Robustness",
            vulnerability_types=[
                "prompt_injection",
                "jailbreak",
                "encoding_attack",
                "unicode_exploit",
                "ascii_smuggling",
                "token_smuggling",
                "delimiter_injection",
            ],
            metrics=[
                "role_adherence",
                "instruction_following",
                "consistency",
            ],
            deterministic_checks=["not_contains", "regex_match"],
            remediation=(
                "Implement comprehensive adversarial defenses. "
                "Test robustness against known attack vectors."
            ),
        ),
        FrameworkRequirement(
            id="Article-15.4",
            title="Cybersecurity",
            description=(
                "High-risk AI systems shall be resilient against "
                "attempts to alter their behavior or extract data "
                "through security vulnerabilities."
            ),
            severity="critical",
            category="Cybersecurity",
            vulnerability_types=[
                "sql_injection",
                "shell_injection",
                "data_extraction",
                "prompt_extraction",
                "memory_poisoning",
            ],
            metrics=[
                "role_adherence",
                "tool_accuracy",
                "pii_detection",
            ],
            deterministic_checks=["not_contains", "no_pii"],
            remediation=(
                "Implement defense-in-depth security measures. "
                "Conduct regular security assessments and "
                "penetration testing."
            ),
        ),
        FrameworkRequirement(
            id="Article-15.5",
            title="Non-Discrimination",
            description=(
                "High-risk AI systems shall not produce outputs "
                "that discriminate against persons based on "
                "protected characteristics."
            ),
            severity="high",
            category="Fairness",
            vulnerability_types=[
                "bias_exploitation",
                "hate_speech",
                "age_restricted",
            ],
            metrics=["bias", "toxicity", "sentiment"],
            deterministic_checks=["not_contains"],
            remediation=(
                "Implement bias detection and mitigation. Test "
                "across demographic groups. Monitor for "
                "discriminatory patterns."
            ),
        ),
    ]
    return FrameworkDefinition(
        name="EU Artificial Intelligence Act",
        version="2024/1689",
        description=(
            "The EU AI Act establishes harmonized rules for the "
            "development, placing on the market, and use of AI "
            "systems in the European Union, with specific "
            "requirements for high-risk AI systems."
        ),
        url="https://eur-lex.europa.eu/eli/reg/2024/1689/oj",
        requirements=reqs,
        total_requirements=len(reqs),
    )


def _build_mitre_atlas() -> FrameworkDefinition:
    """Build the MITRE ATLAS framework definition.

    Returns:
        A FrameworkDefinition with tactics from the ATLAS adversarial
        threat landscape for AI systems.
    """
    reqs = [
        FrameworkRequirement(
            id="AML.TA0001",
            title="Reconnaissance",
            description=(
                "The adversary gathers information about the target "
                "ML system to plan an attack, including model "
                "architecture, training data, and API endpoints."
            ),
            severity="medium",
            category="Reconnaissance",
            vulnerability_types=[
                "prompt_extraction",
                "data_extraction",
            ],
            metrics=["role_adherence"],
            deterministic_checks=["not_contains"],
            remediation=(
                "Minimize information exposure about model internals. "
                "Rate-limit API endpoints. Monitor for probing behavior."
            ),
        ),
        FrameworkRequirement(
            id="AML.TA0002",
            title="Resource Development",
            description=(
                "The adversary develops resources to support the "
                "attack, including crafting adversarial examples, "
                "creating proxy models, or acquiring training data."
            ),
            severity="medium",
            category="Resource Development",
            vulnerability_types=["copyright_violation"],
            metrics=["faithfulness"],
            deterministic_checks=[],
            remediation=(
                "Monitor for model cloning attempts. Implement "
                "watermarking. Limit training data exposure."
            ),
        ),
        FrameworkRequirement(
            id="AML.TA0003",
            title="Initial Access",
            description=(
                "The adversary gains initial access to the ML system "
                "through APIs, interfaces, or supply chain compromise."
            ),
            severity="critical",
            category="Initial Access",
            vulnerability_types=[
                "prompt_injection",
                "jailbreak",
                "instruction_override",
                "encoding_attack",
            ],
            metrics=[
                "role_adherence",
                "instruction_following",
            ],
            deterministic_checks=["not_contains"],
            remediation=(
                "Implement strong authentication. Validate all inputs. "
                "Monitor for anomalous access patterns."
            ),
        ),
        FrameworkRequirement(
            id="AML.TA0004",
            title="ML Model Access",
            description=(
                "The adversary gains access to the ML model through "
                "legitimate or illegitimate means to extract "
                "information or manipulate behavior."
            ),
            severity="critical",
            category="ML Model Access",
            vulnerability_types=[
                "prompt_extraction",
                "data_extraction",
                "privilege_escalation",
            ],
            metrics=["role_adherence", "pii_detection"],
            deterministic_checks=["not_contains", "no_pii"],
            remediation=(
                "Apply principle of least privilege to model access. "
                "Implement query auditing and anomaly detection."
            ),
        ),
        FrameworkRequirement(
            id="AML.TA0005",
            title="Execution",
            description=(
                "The adversary executes adversarial techniques "
                "against the ML system, including prompt injection, "
                "adversarial examples, and data poisoning."
            ),
            severity="critical",
            category="Execution",
            vulnerability_types=[
                "prompt_injection",
                "context_manipulation",
                "delimiter_injection",
                "ascii_smuggling",
                "token_smuggling",
            ],
            metrics=[
                "role_adherence",
                "instruction_following",
                "consistency",
            ],
            deterministic_checks=["not_contains", "regex_match"],
            remediation=(
                "Implement input validation and sanitization. Use "
                "ensemble defenses. Monitor for known attack patterns."
            ),
        ),
        FrameworkRequirement(
            id="AML.TA0006",
            title="Persistence",
            description=(
                "The adversary maintains access to the ML system by "
                "poisoning memory, training data, or model weights "
                "to persist malicious behavior."
            ),
            severity="high",
            category="Persistence",
            vulnerability_types=[
                "memory_poisoning",
                "context_manipulation",
            ],
            metrics=[
                "knowledge_retention",
                "consistency",
                "faithfulness",
            ],
            deterministic_checks=["not_contains"],
            remediation=(
                "Implement memory integrity verification. Monitor "
                "for behavioral drift. Use immutable model snapshots."
            ),
        ),
        FrameworkRequirement(
            id="AML.TA0007",
            title="Defense Evasion",
            description=(
                "The adversary evades detection by safety systems, "
                "content filters, or monitoring mechanisms through "
                "encoding, obfuscation, or adversarial techniques."
            ),
            severity="high",
            category="Defense Evasion",
            vulnerability_types=[
                "encoding_attack",
                "unicode_exploit",
                "ascii_smuggling",
                "token_smuggling",
                "delimiter_injection",
            ],
            metrics=["role_adherence", "instruction_following"],
            deterministic_checks=["not_contains", "regex_match"],
            remediation=(
                "Implement multi-layer content filtering. Normalize "
                "inputs before processing. Use canonical encoding."
            ),
        ),
        FrameworkRequirement(
            id="AML.TA0008",
            title="Discovery",
            description=(
                "The adversary explores the ML system to understand "
                "its capabilities, configuration, and limitations "
                "for further exploitation."
            ),
            severity="medium",
            category="Discovery",
            vulnerability_types=[
                "prompt_extraction",
                "data_extraction",
            ],
            metrics=["role_adherence"],
            deterministic_checks=["not_contains"],
            remediation=(
                "Limit system information disclosure. Implement "
                "query monitoring for systematic probing."
            ),
        ),
        FrameworkRequirement(
            id="AML.TA0009",
            title="Collection",
            description=(
                "The adversary collects sensitive information from "
                "the ML system, including training data, PII, or "
                "proprietary information."
            ),
            severity="critical",
            category="Collection",
            vulnerability_types=[
                "pii_leakage",
                "data_extraction",
                "pii_api_access",
                "pii_social_engineering",
                "session_leak",
            ],
            metrics=["pii_detection"],
            deterministic_checks=["no_pii", "not_contains"],
            remediation=(
                "Implement data loss prevention. Apply output "
                "filtering for sensitive information. Rate-limit "
                "data extraction attempts."
            ),
        ),
        FrameworkRequirement(
            id="AML.TA0010",
            title="ML Attack Staging",
            description=(
                "The adversary stages ML-specific attacks including "
                "model extraction, adversarial example crafting, "
                "and backdoor insertion."
            ),
            severity="high",
            category="ML Attack Staging",
            vulnerability_types=[
                "jailbreak",
                "goal_hijacking",
                "context_manipulation",
                "instruction_override",
            ],
            metrics=[
                "role_adherence",
                "instruction_following",
                "role_violation",
            ],
            deterministic_checks=["not_contains"],
            remediation=(
                "Monitor for systematic attack patterns. Implement "
                "query fingerprinting and anomaly detection."
            ),
        ),
        FrameworkRequirement(
            id="AML.TA0011",
            title="Exfiltration",
            description=(
                "The adversary exfiltrates collected data from the "
                "ML system, including extracted model parameters, "
                "training data, or user information."
            ),
            severity="critical",
            category="Exfiltration",
            vulnerability_types=[
                "data_extraction",
                "pii_leakage",
                "pii_database",
                "cross_session_leak",
            ],
            metrics=["pii_detection"],
            deterministic_checks=["no_pii", "not_contains"],
            remediation=(
                "Implement egress monitoring. Apply rate limiting "
                "on information-rich responses. Use differential "
                "privacy techniques."
            ),
        ),
        FrameworkRequirement(
            id="AML.TA0012",
            title="Impact",
            description=(
                "The adversary causes impact through the ML system, "
                "including generating harmful content, denying "
                "service, or manipulating outputs for downstream "
                "harm."
            ),
            severity="critical",
            category="Impact",
            vulnerability_types=[
                "harmful_content",
                "hate_speech",
                "self_harm",
                "radicalization",
                "chemical_weapons",
                "violent_crime",
            ],
            metrics=["toxicity", "misuse_detection", "bias"],
            deterministic_checks=["not_contains"],
            remediation=(
                "Implement robust content safety filters. Monitor "
                "for harmful output patterns. Establish incident "
                "response procedures."
            ),
        ),
    ]
    return FrameworkDefinition(
        name="MITRE ATLAS",
        version="4.0",
        description=(
            "The MITRE Adversarial Threat Landscape for AI Systems "
            "(ATLAS) is a knowledge base of adversary tactics and "
            "techniques based on real-world attack observations "
            "and demonstrations from AI red teams."
        ),
        url="https://atlas.mitre.org/",
        requirements=reqs,
        total_requirements=len(reqs),
    )


def _build_soc2_type2() -> FrameworkDefinition:
    """Build the SOC 2 Type II framework definition (AI-relevant controls).

    Returns:
        A FrameworkDefinition with AI-relevant Trust Service Criteria.
    """
    reqs = [
        FrameworkRequirement(
            id="CC6.1",
            title="Logical Access Security",
            description=(
                "The entity implements logical access security "
                "software, infrastructure, and architectures over "
                "protected information assets to protect them "
                "from security events."
            ),
            severity="critical",
            category="Logical Access",
            vulnerability_types=[
                "privilege_escalation",
                "role_escape",
                "session_leak",
            ],
            metrics=["role_adherence", "role_violation"],
            deterministic_checks=["not_contains"],
            remediation=(
                "Implement role-based access control for AI systems. "
                "Enforce least privilege for model access."
            ),
        ),
        FrameworkRequirement(
            id="CC6.2",
            title="Access Credentials Management",
            description=(
                "Prior to issuing system credentials and granting "
                "system access, the entity registers and authorizes "
                "new users."
            ),
            severity="high",
            category="Logical Access",
            vulnerability_types=[
                "session_leak",
                "cross_session_leak",
                "prompt_extraction",
            ],
            metrics=["role_adherence"],
            deterministic_checks=["not_contains", "no_pii"],
            remediation=(
                "Implement strong credential management for API "
                "access. Use token rotation and expiration."
            ),
        ),
        FrameworkRequirement(
            id="CC6.3",
            title="Access Authorization",
            description=(
                "The entity authorizes, modifies, or removes access "
                "to data, software, functions, and other protected "
                "information assets based on roles and business "
                "requirements."
            ),
            severity="high",
            category="Logical Access",
            vulnerability_types=[
                "excessive_agency",
                "privilege_escalation",
                "goal_hijacking",
            ],
            metrics=[
                "role_adherence",
                "role_violation",
                "tool_accuracy",
            ],
            deterministic_checks=["not_contains"],
            remediation=(
                "Define and enforce authorization policies for AI "
                "system actions. Review permissions regularly."
            ),
        ),
        FrameworkRequirement(
            id="CC6.6",
            title="Boundary Protection",
            description=(
                "The entity implements logical access security "
                "measures to protect against threats from outside "
                "its system boundaries."
            ),
            severity="critical",
            category="Logical Access",
            vulnerability_types=[
                "prompt_injection",
                "sql_injection",
                "shell_injection",
            ],
            metrics=[
                "role_adherence",
                "instruction_following",
                "tool_accuracy",
            ],
            deterministic_checks=["not_contains", "regex_match"],
            remediation=(
                "Implement input validation at system boundaries. "
                "Filter and sanitize all external inputs to AI models."
            ),
        ),
        FrameworkRequirement(
            id="CC6.8",
            title="Malicious Software Prevention",
            description=(
                "The entity implements controls to prevent or detect "
                "and act upon the introduction of unauthorized or "
                "malicious software."
            ),
            severity="high",
            category="Logical Access",
            vulnerability_types=[
                "encoding_attack",
                "jailbreak",
                "instruction_override",
            ],
            metrics=["role_adherence", "instruction_following"],
            deterministic_checks=["not_contains"],
            remediation=(
                "Implement prompt injection defenses. Deploy content safety classifiers on inputs."
            ),
        ),
        FrameworkRequirement(
            id="CC7.1",
            title="Monitoring of Infrastructure",
            description=(
                "To detect new vulnerabilities, the entity uses "
                "defined configuration standards and monitoring "
                "tools."
            ),
            severity="high",
            category="System Operations",
            vulnerability_types=["data_extraction", "prompt_extraction"],
            metrics=["task_completion", "step_efficiency"],
            deterministic_checks=["latency", "cost"],
            remediation=(
                "Implement real-time monitoring of AI system performance and security metrics."
            ),
        ),
        FrameworkRequirement(
            id="CC7.2",
            title="Security Event Monitoring",
            description=(
                "The entity monitors system components for anomalies "
                "that are indicative of malicious acts, natural "
                "disasters, and errors affecting the entity's "
                "ability to meet its objectives."
            ),
            severity="high",
            category="System Operations",
            vulnerability_types=[
                "prompt_injection",
                "jailbreak",
                "harmful_content",
            ],
            metrics=["toxicity", "misuse_detection"],
            deterministic_checks=["not_contains"],
            remediation=(
                "Deploy anomaly detection on AI inputs and outputs. "
                "Alert on adversarial attack patterns."
            ),
        ),
        FrameworkRequirement(
            id="CC7.3",
            title="Incident Response",
            description=(
                "The entity evaluates security events to determine "
                "whether they could or have resulted in a failure "
                "of the entity to meet its objectives."
            ),
            severity="high",
            category="System Operations",
            vulnerability_types=[
                "harmful_content",
                "data_extraction",
                "pii_leakage",
            ],
            metrics=["toxicity", "pii_detection"],
            deterministic_checks=["not_contains", "no_pii"],
            remediation=(
                "Establish AI incident response procedures. Define "
                "severity levels and escalation paths for AI failures."
            ),
        ),
        FrameworkRequirement(
            id="CC7.4",
            title="Incident Containment and Remediation",
            description=(
                "The entity responds to identified security incidents "
                "by executing a defined incident response program."
            ),
            severity="high",
            category="System Operations",
            vulnerability_types=[
                "excessive_agency",
                "privilege_escalation",
            ],
            metrics=["role_adherence"],
            deterministic_checks=["not_contains"],
            remediation=(
                "Implement model rollback capabilities. Define "
                "containment procedures for compromised AI systems."
            ),
        ),
        FrameworkRequirement(
            id="CC8.1",
            title="Change Management",
            description=(
                "The entity authorizes, designs, develops, "
                "configures, documents, tests, approves, and "
                "implements changes to infrastructure, data, "
                "software, and procedures."
            ),
            severity="medium",
            category="Change Management",
            vulnerability_types=["memory_poisoning"],
            metrics=["consistency", "faithfulness"],
            deterministic_checks=["json_schema"],
            remediation=(
                "Implement model versioning and change control. "
                "Test all model updates before deployment."
            ),
        ),
        FrameworkRequirement(
            id="AI-CC1",
            title="AI Model Input Validation",
            description=(
                "AI systems validate all inputs against defined "
                "schemas and reject inputs that could cause "
                "unintended behavior."
            ),
            severity="critical",
            category="AI-Specific Controls",
            vulnerability_types=[
                "prompt_injection",
                "jailbreak",
                "encoding_attack",
                "delimiter_injection",
            ],
            metrics=[
                "role_adherence",
                "instruction_following",
            ],
            deterministic_checks=["not_contains", "regex_match", "max_tokens"],
            remediation=(
                "Implement comprehensive input validation for all "
                "AI model inputs. Reject anomalous inputs."
            ),
        ),
        FrameworkRequirement(
            id="AI-CC2",
            title="AI Output Safety Controls",
            description=(
                "AI systems implement output safety controls to "
                "prevent generation of harmful, biased, or "
                "inappropriate content."
            ),
            severity="critical",
            category="AI-Specific Controls",
            vulnerability_types=[
                "harmful_content",
                "hate_speech",
                "self_harm",
                "bias_exploitation",
                "radicalization",
            ],
            metrics=["toxicity", "bias", "misuse_detection"],
            deterministic_checks=["not_contains"],
            remediation=(
                "Deploy content safety classifiers on all outputs. "
                "Implement multi-layer safety filtering."
            ),
        ),
        FrameworkRequirement(
            id="AI-CC3",
            title="AI Data Privacy Controls",
            description=(
                "AI systems implement controls to prevent "
                "unauthorized disclosure of personal or "
                "confidential information."
            ),
            severity="critical",
            category="AI-Specific Controls",
            vulnerability_types=[
                "pii_leakage",
                "data_extraction",
                "pii_api_access",
                "pii_social_engineering",
                "gdpr_violation",
                "hipaa_violation",
            ],
            metrics=["pii_detection"],
            deterministic_checks=["no_pii", "not_contains"],
            remediation=(
                "Implement PII detection and redaction. Apply "
                "differential privacy. Monitor for data leakage."
            ),
        ),
    ]
    return FrameworkDefinition(
        name="SOC 2 Type II (AI-Relevant Controls)",
        version="2024",
        description=(
            "SOC 2 Type II Trust Service Criteria relevant to AI "
            "systems, covering logical access, system operations, "
            "change management, and AI-specific controls."
        ),
        url="https://www.aicpa-cima.com/resources/landing/system-and-organization-controls-soc-suite-of-services",
        requirements=reqs,
        total_requirements=len(reqs),
    )


def _build_iso_42001() -> FrameworkDefinition:
    """Build the ISO 42001 (AI Management System) framework definition.

    Returns:
        A FrameworkDefinition with requirements from key ISO 42001 clauses
        and Annex A controls.
    """
    reqs = [
        FrameworkRequirement(
            id="5.1",
            title="Leadership and Commitment",
            description=(
                "Top management shall demonstrate leadership and "
                "commitment with respect to the AI management system "
                "by ensuring AI policy is established and resources "
                "are available."
            ),
            severity="medium",
            category="Leadership",
            vulnerability_types=[],
            metrics=["correctness"],
            deterministic_checks=[],
            remediation=(
                "Establish executive sponsorship for AI governance. Document AI management policy."
            ),
        ),
        FrameworkRequirement(
            id="5.3",
            title="AI Policy",
            description=(
                "Top management shall establish an AI policy that "
                "includes commitment to responsible AI development, "
                "compliance with applicable requirements, and "
                "continual improvement."
            ),
            severity="high",
            category="Leadership",
            vulnerability_types=["harmful_content", "bias_exploitation"],
            metrics=["toxicity", "bias"],
            deterministic_checks=[],
            remediation=(
                "Define and communicate an organizational AI policy "
                "covering ethics, safety, and compliance."
            ),
        ),
        FrameworkRequirement(
            id="6.1",
            title="Actions to Address Risks and Opportunities",
            description=(
                "The organization shall determine risks and "
                "opportunities relevant to the AI management system "
                "and plan actions to address them."
            ),
            severity="high",
            category="Planning",
            vulnerability_types=[
                "prompt_injection",
                "jailbreak",
                "harmful_content",
            ],
            metrics=[
                "role_adherence",
                "toxicity",
                "misuse_detection",
            ],
            deterministic_checks=["not_contains"],
            remediation=(
                "Conduct AI-specific risk assessments. Document "
                "risk treatment plans and acceptance criteria."
            ),
        ),
        FrameworkRequirement(
            id="6.2",
            title="AI Objectives and Planning",
            description=(
                "The organization shall establish AI objectives "
                "at relevant functions, levels, and processes, "
                "ensuring they are measurable and monitored."
            ),
            severity="medium",
            category="Planning",
            vulnerability_types=[],
            metrics=[
                "correctness",
                "goal_accuracy",
                "task_completion",
            ],
            deterministic_checks=["contains"],
            remediation=(
                "Define measurable AI quality and safety objectives. "
                "Establish monitoring and review processes."
            ),
        ),
        FrameworkRequirement(
            id="7.2",
            title="Competence",
            description=(
                "The organization shall determine the necessary "
                "competence of persons performing work under the "
                "AI management system."
            ),
            severity="medium",
            category="Support",
            vulnerability_types=["overreliance"],
            metrics=["non_advice"],
            deterministic_checks=[],
            remediation=(
                "Define competency requirements for AI roles. "
                "Provide training on AI risks and responsible use."
            ),
        ),
        FrameworkRequirement(
            id="7.4",
            title="Communication",
            description=(
                "The organization shall determine internal and "
                "external communications relevant to the AI "
                "management system."
            ),
            severity="low",
            category="Support",
            vulnerability_types=["overreliance", "unauthorized_practice"],
            metrics=["non_advice"],
            deterministic_checks=["contains"],
            remediation=(
                "Establish communication channels for AI incidents and governance decisions."
            ),
        ),
        FrameworkRequirement(
            id="8.1",
            title="Operational Planning and Control",
            description=(
                "The organization shall plan, implement, and control "
                "the processes needed to meet AI management system "
                "requirements."
            ),
            severity="high",
            category="Operation",
            vulnerability_types=[
                "prompt_injection",
                "jailbreak",
                "excessive_agency",
            ],
            metrics=[
                "role_adherence",
                "instruction_following",
                "task_completion",
            ],
            deterministic_checks=["not_contains"],
            remediation=(
                "Implement operational procedures for AI system deployment and monitoring."
            ),
        ),
        FrameworkRequirement(
            id="8.2",
            title="AI Risk Assessment",
            description=(
                "The organization shall perform AI risk assessments "
                "at planned intervals or when significant changes "
                "are proposed or occur."
            ),
            severity="critical",
            category="Operation",
            vulnerability_types=[
                "prompt_injection",
                "jailbreak",
                "harmful_content",
                "bias_exploitation",
                "pii_leakage",
            ],
            metrics=[
                "role_adherence",
                "toxicity",
                "bias",
                "pii_detection",
            ],
            deterministic_checks=["not_contains", "no_pii"],
            remediation=(
                "Conduct regular AI risk assessments using "
                "adversarial testing and evaluation metrics."
            ),
        ),
        FrameworkRequirement(
            id="8.3",
            title="AI Risk Treatment",
            description=(
                "The organization shall implement the AI risk "
                "treatment plan and retain documented information "
                "on results."
            ),
            severity="high",
            category="Operation",
            vulnerability_types=[
                "harmful_content",
                "self_harm",
                "hate_speech",
                "radicalization",
            ],
            metrics=["toxicity", "misuse_detection", "bias"],
            deterministic_checks=["not_contains"],
            remediation=(
                "Implement risk treatment measures. Document "
                "residual risks and mitigation effectiveness."
            ),
        ),
        FrameworkRequirement(
            id="8.4",
            title="AI Impact Assessment",
            description=(
                "The organization shall assess potential impacts "
                "of AI systems on individuals, groups, and society."
            ),
            severity="high",
            category="Operation",
            vulnerability_types=[
                "bias_exploitation",
                "hate_speech",
                "self_harm",
                "radicalization",
            ],
            metrics=["bias", "toxicity", "sentiment"],
            deterministic_checks=["not_contains"],
            remediation=(
                "Conduct impact assessments for all AI systems. "
                "Address identified negative impacts."
            ),
        ),
        FrameworkRequirement(
            id="9.1",
            title="Monitoring, Measurement, Analysis and Evaluation",
            description=(
                "The organization shall determine what needs to "
                "be monitored and measured, including AI system "
                "performance and risk metrics."
            ),
            severity="high",
            category="Performance Evaluation",
            vulnerability_types=[],
            metrics=[
                "correctness",
                "faithfulness",
                "consistency",
                "relevance",
            ],
            deterministic_checks=["latency", "cost"],
            remediation=(
                "Implement monitoring dashboards for AI performance "
                "and safety metrics. Set alerting thresholds."
            ),
        ),
        FrameworkRequirement(
            id="9.2",
            title="Internal Audit",
            description=(
                "The organization shall conduct internal audits at "
                "planned intervals to verify the AI management "
                "system conforms to requirements."
            ),
            severity="medium",
            category="Performance Evaluation",
            vulnerability_types=[],
            metrics=["correctness", "consistency"],
            deterministic_checks=[],
            remediation=(
                "Schedule regular internal audits of AI systems. "
                "Include adversarial testing in audit scope."
            ),
        ),
        FrameworkRequirement(
            id="9.3",
            title="Management Review",
            description=(
                "Top management shall review the AI management "
                "system at planned intervals to ensure its "
                "continuing suitability, adequacy, and effectiveness."
            ),
            severity="medium",
            category="Performance Evaluation",
            vulnerability_types=[],
            metrics=["correctness"],
            deterministic_checks=[],
            remediation=(
                "Conduct regular management reviews with AI risk and performance reports."
            ),
        ),
        FrameworkRequirement(
            id="10.1",
            title="Nonconformity and Corrective Action",
            description=(
                "When a nonconformity occurs, the organization shall "
                "react, evaluate the need for action, implement "
                "changes, and review effectiveness."
            ),
            severity="high",
            category="Improvement",
            vulnerability_types=[
                "harmful_content",
                "pii_leakage",
                "bias_exploitation",
            ],
            metrics=["toxicity", "pii_detection", "bias"],
            deterministic_checks=["not_contains", "no_pii"],
            remediation=(
                "Establish procedures for AI-related nonconformities. "
                "Implement corrective actions and track effectiveness."
            ),
        ),
        FrameworkRequirement(
            id="10.2",
            title="Continual Improvement",
            description=(
                "The organization shall continually improve the "
                "suitability, adequacy, and effectiveness of the "
                "AI management system."
            ),
            severity="medium",
            category="Improvement",
            vulnerability_types=[],
            metrics=[
                "correctness",
                "faithfulness",
                "consistency",
            ],
            deterministic_checks=[],
            remediation=(
                "Establish continuous improvement processes for "
                "AI systems. Track and act on improvement opportunities."
            ),
        ),
        FrameworkRequirement(
            id="A.2",
            title="Annex A - AI System Lifecycle Processes",
            description=(
                "Controls for managing AI systems throughout their "
                "lifecycle including design, development, deployment, "
                "monitoring, and retirement."
            ),
            severity="high",
            category="Annex A Controls",
            vulnerability_types=[
                "prompt_injection",
                "jailbreak",
                "encoding_attack",
            ],
            metrics=[
                "role_adherence",
                "instruction_following",
                "consistency",
            ],
            deterministic_checks=["not_contains"],
            remediation=(
                "Implement controls at each AI lifecycle stage. "
                "Include security testing in development pipelines."
            ),
        ),
        FrameworkRequirement(
            id="A.4",
            title="Annex A - AI Data Management",
            description=(
                "Controls for ensuring data quality, provenance, "
                "privacy, and governance throughout the AI data "
                "lifecycle."
            ),
            severity="high",
            category="Annex A Controls",
            vulnerability_types=[
                "pii_leakage",
                "data_extraction",
                "gdpr_violation",
                "hipaa_violation",
                "coppa_violation",
            ],
            metrics=["pii_detection", "faithfulness"],
            deterministic_checks=["no_pii", "not_contains"],
            remediation=(
                "Implement data governance controls. Ensure "
                "compliance with applicable privacy regulations."
            ),
        ),
        FrameworkRequirement(
            id="A.6",
            title="Annex A - AI System Transparency",
            description=(
                "Controls for ensuring appropriate transparency "
                "of AI systems to stakeholders, including "
                "explainability and interpretability."
            ),
            severity="high",
            category="Annex A Controls",
            vulnerability_types=[
                "hallucination_attack",
                "overreliance",
                "unauthorized_practice",
            ],
            metrics=[
                "faithfulness",
                "groundedness",
                "non_advice",
                "hallucination",
            ],
            deterministic_checks=["contains"],
            remediation=(
                "Implement explainability features. Provide "
                "transparent information about AI capabilities "
                "and limitations."
            ),
        ),
        FrameworkRequirement(
            id="A.8",
            title="Annex A - AI System Fairness",
            description=(
                "Controls for preventing unfair bias and "
                "discrimination in AI system outputs and "
                "decisions."
            ),
            severity="high",
            category="Annex A Controls",
            vulnerability_types=[
                "bias_exploitation",
                "hate_speech",
                "age_restricted",
            ],
            metrics=["bias", "toxicity", "sentiment"],
            deterministic_checks=["not_contains"],
            remediation=(
                "Implement fairness testing and monitoring. Apply bias mitigation techniques."
            ),
        ),
        FrameworkRequirement(
            id="A.10",
            title="Annex A - AI System Security",
            description=(
                "Controls for protecting AI systems against "
                "adversarial attacks, unauthorized access, and "
                "data breaches."
            ),
            severity="critical",
            category="Annex A Controls",
            vulnerability_types=[
                "prompt_injection",
                "jailbreak",
                "sql_injection",
                "shell_injection",
                "privilege_escalation",
                "data_extraction",
            ],
            metrics=[
                "role_adherence",
                "instruction_following",
                "tool_accuracy",
            ],
            deterministic_checks=["not_contains", "no_pii", "regex_match"],
            remediation=(
                "Implement comprehensive security controls. "
                "Conduct regular penetration testing and red teaming."
            ),
        ),
    ]
    return FrameworkDefinition(
        name="ISO/IEC 42001 AI Management System",
        version="2023",
        description=(
            "ISO/IEC 42001 specifies requirements for establishing, "
            "implementing, maintaining, and continually improving an "
            "AI management system within organizations."
        ),
        url="https://www.iso.org/standard/81230.html",
        requirements=reqs,
        total_requirements=len(reqs),
    )


def _build_iso_27001_ai() -> FrameworkDefinition:
    """Build the ISO 27001 AI-relevant controls framework definition.

    Returns:
        A FrameworkDefinition with AI-relevant information security
        controls from ISO 27001.
    """
    reqs = [
        FrameworkRequirement(
            id="A.5.1",
            title="Policies for Information Security",
            description=(
                "A set of policies for information security shall "
                "be defined, approved by management, and "
                "communicated to relevant stakeholders."
            ),
            severity="high",
            category="Organizational Controls",
            vulnerability_types=["harmful_content", "bias_exploitation"],
            metrics=["role_adherence"],
            deterministic_checks=[],
            remediation=("Include AI-specific sections in information security policies."),
        ),
        FrameworkRequirement(
            id="A.5.23",
            title="Information Security for Cloud Services",
            description=(
                "Processes for acquisition, use, management, and "
                "exit from cloud services shall include AI-hosted "
                "model security considerations."
            ),
            severity="high",
            category="Organizational Controls",
            vulnerability_types=[
                "data_extraction",
                "prompt_extraction",
                "session_leak",
            ],
            metrics=["role_adherence", "pii_detection"],
            deterministic_checks=["not_contains", "no_pii"],
            remediation=(
                "Assess cloud-hosted AI service security. Include "
                "AI considerations in cloud security policies."
            ),
        ),
        FrameworkRequirement(
            id="A.8.2",
            title="Privileged Access Rights",
            description=(
                "The allocation and use of privileged access rights "
                "shall be restricted and managed, including for "
                "AI system administration."
            ),
            severity="critical",
            category="Access Control",
            vulnerability_types=[
                "privilege_escalation",
                "excessive_agency",
                "role_escape",
            ],
            metrics=["role_adherence", "role_violation"],
            deterministic_checks=["not_contains"],
            remediation=(
                "Apply least privilege to AI system access. Implement segregation of duties."
            ),
        ),
        FrameworkRequirement(
            id="A.8.4",
            title="Access to Source Code",
            description=(
                "Read and write access to source code, development "
                "tools, and software libraries shall be managed, "
                "including AI model artifacts."
            ),
            severity="high",
            category="Access Control",
            vulnerability_types=[
                "data_extraction",
                "prompt_extraction",
            ],
            metrics=["role_adherence"],
            deterministic_checks=["not_contains"],
            remediation=(
                "Control access to AI model weights, training scripts, and prompt templates."
            ),
        ),
        FrameworkRequirement(
            id="A.8.9",
            title="Configuration Management",
            description=(
                "Configurations of hardware, software, services, "
                "and networks shall be established, documented, "
                "implemented, and monitored, including AI models."
            ),
            severity="medium",
            category="Technology Controls",
            vulnerability_types=["memory_poisoning"],
            metrics=["consistency"],
            deterministic_checks=["json_schema"],
            remediation=(
                "Implement configuration management for AI model "
                "deployments. Version all model artifacts."
            ),
        ),
        FrameworkRequirement(
            id="A.8.10",
            title="Information Deletion",
            description=(
                "Information stored in information systems and "
                "devices shall be deleted when no longer required, "
                "including AI training data and user interactions."
            ),
            severity="high",
            category="Technology Controls",
            vulnerability_types=[
                "pii_leakage",
                "session_leak",
                "cross_session_leak",
            ],
            metrics=["pii_detection"],
            deterministic_checks=["no_pii"],
            remediation=(
                "Implement data retention policies for AI systems. "
                "Ensure proper deletion of user data."
            ),
        ),
        FrameworkRequirement(
            id="A.8.11",
            title="Data Masking",
            description=(
                "Data masking shall be used in accordance with "
                "the organization's data handling policies, "
                "including for AI training and inference data."
            ),
            severity="high",
            category="Technology Controls",
            vulnerability_types=[
                "pii_leakage",
                "pii_api_access",
                "pii_database",
            ],
            metrics=["pii_detection"],
            deterministic_checks=["no_pii", "not_contains"],
            remediation=(
                "Apply data masking to sensitive fields in AI training data and model outputs."
            ),
        ),
        FrameworkRequirement(
            id="A.8.12",
            title="Data Leakage Prevention",
            description=(
                "Data leakage prevention measures shall be applied "
                "to systems, networks, and devices that process, "
                "store, or transmit sensitive information, "
                "including AI systems."
            ),
            severity="critical",
            category="Technology Controls",
            vulnerability_types=[
                "data_extraction",
                "pii_leakage",
                "pii_social_engineering",
                "gdpr_violation",
                "hipaa_violation",
            ],
            metrics=["pii_detection"],
            deterministic_checks=["no_pii", "not_contains"],
            remediation=(
                "Implement output filtering to prevent data leakage through AI model responses."
            ),
        ),
        FrameworkRequirement(
            id="A.8.23",
            title="Web Filtering",
            description=(
                "Access to external websites shall be managed to "
                "reduce exposure to malicious content, including "
                "AI-accessed external resources."
            ),
            severity="medium",
            category="Technology Controls",
            vulnerability_types=[
                "sql_injection",
                "shell_injection",
            ],
            metrics=["tool_accuracy"],
            deterministic_checks=["not_contains", "regex_match"],
            remediation=(
                "Restrict and monitor AI system access to external "
                "resources. Validate URLs and content."
            ),
        ),
        FrameworkRequirement(
            id="A.8.28",
            title="Secure Coding",
            description=(
                "Secure coding principles shall be applied to "
                "software development, including AI system "
                "development and prompt engineering."
            ),
            severity="high",
            category="Technology Controls",
            vulnerability_types=[
                "prompt_injection",
                "sql_injection",
                "shell_injection",
                "json_injection",
            ],
            metrics=[
                "role_adherence",
                "tool_accuracy",
            ],
            deterministic_checks=["not_contains", "json_schema"],
            remediation=(
                "Apply secure development practices to AI systems. "
                "Include prompt security in development guidelines."
            ),
        ),
    ]
    return FrameworkDefinition(
        name="ISO/IEC 27001 (AI-Relevant Controls)",
        version="2022",
        description=(
            "ISO/IEC 27001 information security controls relevant "
            "to AI systems, covering organizational, access, and "
            "technology controls adapted for AI contexts."
        ),
        url="https://www.iso.org/standard/27001",
        requirements=reqs,
        total_requirements=len(reqs),
    )


def _build_hipaa_ai() -> FrameworkDefinition:
    """Build the HIPAA AI-relevant controls framework definition.

    Returns:
        A FrameworkDefinition with HIPAA requirements relevant to AI
        systems processing protected health information.
    """
    reqs = [
        FrameworkRequirement(
            id="164.312(a)(1)",
            title="Access Control",
            description=(
                "Implement technical policies and procedures for "
                "electronic information systems that maintain ePHI "
                "to allow access only to authorized persons."
            ),
            severity="critical",
            category="Technical Safeguards",
            vulnerability_types=[
                "privilege_escalation",
                "role_escape",
                "data_extraction",
            ],
            metrics=["role_adherence", "role_violation"],
            deterministic_checks=["not_contains"],
            remediation=(
                "Implement RBAC for AI systems processing PHI. "
                "Use unique user identification for audit trails."
            ),
        ),
        FrameworkRequirement(
            id="164.312(a)(2)(iv)",
            title="Encryption and Decryption",
            description=(
                "Implement a mechanism to encrypt and decrypt "
                "ePHI as it passes through AI processing pipelines."
            ),
            severity="critical",
            category="Technical Safeguards",
            vulnerability_types=[
                "data_extraction",
                "pii_leakage",
                "session_leak",
            ],
            metrics=["pii_detection"],
            deterministic_checks=["no_pii", "not_contains"],
            remediation=(
                "Encrypt PHI in transit and at rest within AI "
                "systems. Use approved encryption standards."
            ),
        ),
        FrameworkRequirement(
            id="164.312(b)",
            title="Audit Controls",
            description=(
                "Implement hardware, software, and procedural "
                "mechanisms that record and examine activity in "
                "AI systems that contain or use ePHI."
            ),
            severity="high",
            category="Technical Safeguards",
            vulnerability_types=[
                "prompt_extraction",
                "data_extraction",
            ],
            metrics=["task_completion"],
            deterministic_checks=["contains"],
            remediation=(
                "Implement comprehensive audit logging for all AI interactions involving PHI."
            ),
        ),
        FrameworkRequirement(
            id="164.312(c)(1)",
            title="Integrity Controls",
            description=(
                "Implement policies and procedures to protect "
                "ePHI from improper alteration or destruction "
                "by AI systems."
            ),
            severity="critical",
            category="Technical Safeguards",
            vulnerability_types=[
                "memory_poisoning",
                "hallucination_attack",
                "context_manipulation",
            ],
            metrics=[
                "faithfulness",
                "correctness",
                "hallucination",
            ],
            deterministic_checks=["not_contains"],
            remediation=(
                "Validate AI outputs against source records. "
                "Implement integrity checks for PHI processing."
            ),
        ),
        FrameworkRequirement(
            id="164.312(d)",
            title="Person or Entity Authentication",
            description=(
                "Implement procedures to verify that a person "
                "or entity seeking access to ePHI through AI "
                "systems is who they claim to be."
            ),
            severity="critical",
            category="Technical Safeguards",
            vulnerability_types=[
                "session_leak",
                "cross_session_leak",
                "pii_social_engineering",
            ],
            metrics=["role_adherence"],
            deterministic_checks=["not_contains"],
            remediation=(
                "Implement strong authentication for AI systems "
                "accessing PHI. Use multi-factor authentication."
            ),
        ),
        FrameworkRequirement(
            id="164.312(e)(1)",
            title="Transmission Security",
            description=(
                "Implement technical security measures to guard "
                "against unauthorized access to ePHI transmitted "
                "through AI APIs."
            ),
            severity="critical",
            category="Technical Safeguards",
            vulnerability_types=[
                "data_extraction",
                "pii_leakage",
                "pii_api_access",
            ],
            metrics=["pii_detection"],
            deterministic_checks=["no_pii", "not_contains"],
            remediation=(
                "Use TLS for all AI API communications. Implement end-to-end encryption for PHI."
            ),
        ),
        FrameworkRequirement(
            id="164.530(c)",
            title="Safeguards for PHI",
            description=(
                "A covered entity must have appropriate "
                "administrative, technical, and physical safeguards "
                "to protect PHI from AI-mediated disclosure."
            ),
            severity="critical",
            category="Privacy Rule",
            vulnerability_types=[
                "pii_leakage",
                "pii_social_engineering",
                "pii_database",
                "hipaa_violation",
            ],
            metrics=["pii_detection"],
            deterministic_checks=["no_pii", "not_contains"],
            remediation=(
                "Implement comprehensive PHI safeguards for AI "
                "systems. Train all personnel on HIPAA requirements."
            ),
        ),
        FrameworkRequirement(
            id="164.502(b)",
            title="Minimum Necessary Standard",
            description=(
                "AI systems processing PHI shall use or disclose "
                "only the minimum necessary information to "
                "accomplish the intended purpose."
            ),
            severity="high",
            category="Privacy Rule",
            vulnerability_types=[
                "data_extraction",
                "pii_leakage",
                "excessive_agency",
            ],
            metrics=["pii_detection", "role_adherence"],
            deterministic_checks=["no_pii", "not_contains"],
            remediation=(
                "Apply data minimization to AI model inputs. "
                "Restrict AI output to minimum necessary PHI."
            ),
        ),
        FrameworkRequirement(
            id="164.528",
            title="Accounting of Disclosures",
            description=(
                "Individuals have the right to receive an "
                "accounting of disclosures of PHI, including "
                "disclosures made through AI systems."
            ),
            severity="high",
            category="Individual Rights",
            vulnerability_types=["hipaa_violation"],
            metrics=["task_completion"],
            deterministic_checks=["contains"],
            remediation=(
                "Log all PHI disclosures through AI systems. Maintain disclosure audit trails."
            ),
        ),
        FrameworkRequirement(
            id="164.524",
            title="Access of Individuals to PHI",
            description=(
                "Individuals have the right to inspect and obtain "
                "a copy of PHI about the individual, including "
                "data processed by AI systems."
            ),
            severity="high",
            category="Individual Rights",
            vulnerability_types=["data_extraction"],
            metrics=["correctness"],
            deterministic_checks=["contains"],
            remediation=("Implement data subject access request handling for AI-processed PHI."),
        ),
    ]
    return FrameworkDefinition(
        name="HIPAA (AI-Relevant Requirements)",
        version="2024",
        description=(
            "HIPAA Security Rule and Privacy Rule requirements "
            "relevant to AI systems processing protected health "
            "information (PHI)."
        ),
        url="https://www.hhs.gov/hipaa/index.html",
        requirements=reqs,
        total_requirements=len(reqs),
    )


def _build_gdpr_ai() -> FrameworkDefinition:
    """Build the GDPR AI-relevant controls framework definition.

    Returns:
        A FrameworkDefinition with GDPR requirements relevant to AI
        systems processing personal data.
    """
    reqs = [
        FrameworkRequirement(
            id="Art.5(1)(a)",
            title="Lawfulness, Fairness, and Transparency",
            description=(
                "Personal data shall be processed lawfully, fairly, "
                "and in a transparent manner in relation to the "
                "data subject."
            ),
            severity="critical",
            category="Data Processing Principles",
            vulnerability_types=[
                "bias_exploitation",
                "overreliance",
                "hallucination_attack",
            ],
            metrics=[
                "bias",
                "faithfulness",
                "non_advice",
                "groundedness",
            ],
            deterministic_checks=["not_contains", "contains"],
            remediation=(
                "Ensure transparency in AI processing. Provide "
                "clear information about AI use and decision-making."
            ),
        ),
        FrameworkRequirement(
            id="Art.5(1)(b)",
            title="Purpose Limitation",
            description=(
                "Personal data shall be collected for specified, "
                "explicit, and legitimate purposes and not further "
                "processed in a manner incompatible with those "
                "purposes."
            ),
            severity="high",
            category="Data Processing Principles",
            vulnerability_types=[
                "data_extraction",
                "pii_leakage",
                "excessive_agency",
            ],
            metrics=["pii_detection", "role_adherence"],
            deterministic_checks=["no_pii", "not_contains"],
            remediation=(
                "Define and enforce purpose limitations for AI "
                "data processing. Prevent scope creep."
            ),
        ),
        FrameworkRequirement(
            id="Art.5(1)(c)",
            title="Data Minimization",
            description=(
                "Personal data shall be adequate, relevant, and "
                "limited to what is necessary in relation to the "
                "purposes for which they are processed."
            ),
            severity="high",
            category="Data Processing Principles",
            vulnerability_types=[
                "pii_leakage",
                "pii_api_access",
                "pii_database",
            ],
            metrics=["pii_detection"],
            deterministic_checks=["no_pii", "not_contains"],
            remediation=(
                "Apply data minimization to AI model inputs and training data. Reduce PII exposure."
            ),
        ),
        FrameworkRequirement(
            id="Art.5(1)(d)",
            title="Accuracy",
            description=(
                "Personal data shall be accurate and, where "
                "necessary, kept up to date; inaccurate data "
                "shall be erased or rectified."
            ),
            severity="high",
            category="Data Processing Principles",
            vulnerability_types=[
                "hallucination_attack",
                "memory_poisoning",
            ],
            metrics=[
                "correctness",
                "faithfulness",
                "hallucination",
            ],
            deterministic_checks=["not_contains"],
            remediation=(
                "Validate AI output accuracy. Implement "
                "mechanisms for data correction and rectification."
            ),
        ),
        FrameworkRequirement(
            id="Art.13",
            title="Information to Data Subject",
            description=(
                "Where personal data are collected from the data "
                "subject, the controller shall provide information "
                "about AI-based automated decision-making."
            ),
            severity="high",
            category="Data Subject Rights",
            vulnerability_types=["overreliance", "unauthorized_practice"],
            metrics=["non_advice", "role_adherence"],
            deterministic_checks=["contains"],
            remediation=(
                "Provide clear information about AI processing. "
                "Disclose the use of automated decision-making."
            ),
        ),
        FrameworkRequirement(
            id="Art.15",
            title="Right of Access",
            description=(
                "The data subject shall have the right to obtain "
                "confirmation of processing and access to personal "
                "data, including AI-derived data."
            ),
            severity="high",
            category="Data Subject Rights",
            vulnerability_types=["data_extraction"],
            metrics=["correctness"],
            deterministic_checks=["contains"],
            remediation=(
                "Implement data subject access mechanisms for AI-processed personal data."
            ),
        ),
        FrameworkRequirement(
            id="Art.17",
            title="Right to Erasure",
            description=(
                "The data subject shall have the right to obtain "
                "erasure of personal data, which extends to data "
                "used in AI training and inference."
            ),
            severity="high",
            category="Data Subject Rights",
            vulnerability_types=[
                "pii_leakage",
                "session_leak",
                "cross_session_leak",
            ],
            metrics=["pii_detection"],
            deterministic_checks=["no_pii"],
            remediation=(
                "Implement data erasure capabilities for AI "
                "systems, including model unlearning where feasible."
            ),
        ),
        FrameworkRequirement(
            id="Art.22",
            title="Automated Individual Decision-Making",
            description=(
                "The data subject shall have the right not to be "
                "subject to a decision based solely on automated "
                "processing, including profiling, which produces "
                "legal effects."
            ),
            severity="critical",
            category="Automated Decision-Making",
            vulnerability_types=[
                "excessive_agency",
                "bias_exploitation",
                "regulated_advice",
            ],
            metrics=[
                "role_adherence",
                "bias",
                "non_advice",
            ],
            deterministic_checks=["not_contains", "contains"],
            remediation=(
                "Implement human oversight for consequential AI "
                "decisions. Provide mechanisms to contest automated "
                "decisions."
            ),
        ),
        FrameworkRequirement(
            id="Art.25",
            title="Data Protection by Design and Default",
            description=(
                "The controller shall implement appropriate "
                "technical and organizational measures for "
                "integrating data protection into AI processing."
            ),
            severity="high",
            category="Accountability",
            vulnerability_types=[
                "pii_leakage",
                "data_extraction",
                "gdpr_violation",
            ],
            metrics=["pii_detection"],
            deterministic_checks=["no_pii", "not_contains"],
            remediation=(
                "Apply privacy by design to AI systems. "
                "Implement PII detection and redaction by default."
            ),
        ),
        FrameworkRequirement(
            id="Art.32",
            title="Security of Processing",
            description=(
                "The controller and processor shall implement "
                "appropriate technical and organizational measures "
                "to ensure security appropriate to the risk of "
                "AI processing."
            ),
            severity="critical",
            category="Security",
            vulnerability_types=[
                "prompt_injection",
                "sql_injection",
                "shell_injection",
                "data_extraction",
                "privilege_escalation",
            ],
            metrics=[
                "role_adherence",
                "tool_accuracy",
                "pii_detection",
            ],
            deterministic_checks=["not_contains", "no_pii", "regex_match"],
            remediation=(
                "Implement security controls appropriate to AI "
                "processing risks. Conduct regular assessments."
            ),
        ),
        FrameworkRequirement(
            id="Art.35",
            title="Data Protection Impact Assessment",
            description=(
                "Where processing is likely to result in a high "
                "risk to rights and freedoms, a DPIA shall be "
                "carried out before AI processing begins."
            ),
            severity="high",
            category="Accountability",
            vulnerability_types=[
                "bias_exploitation",
                "pii_leakage",
                "harmful_content",
            ],
            metrics=["bias", "pii_detection", "toxicity"],
            deterministic_checks=["not_contains", "no_pii"],
            remediation=(
                "Conduct DPIAs for high-risk AI processing. Document and mitigate identified risks."
            ),
        ),
    ]
    return FrameworkDefinition(
        name="GDPR (AI-Relevant Requirements)",
        version="2016/679",
        description=(
            "General Data Protection Regulation requirements "
            "relevant to AI systems processing personal data "
            "of EU/EEA residents."
        ),
        url="https://gdpr-info.eu/",
        requirements=reqs,
        total_requirements=len(reqs),
    )


def _build_pci_dss_ai() -> FrameworkDefinition:
    """Build the PCI DSS AI-relevant controls framework definition.

    Returns:
        A FrameworkDefinition with PCI DSS requirements relevant to
        AI systems handling payment card data.
    """
    reqs = [
        FrameworkRequirement(
            id="Req-1.3",
            title="Network Access Controls",
            description=(
                "Network access to and from AI systems processing "
                "cardholder data shall be restricted."
            ),
            severity="critical",
            category="Network Security",
            vulnerability_types=[
                "data_extraction",
                "shell_injection",
            ],
            metrics=["tool_accuracy"],
            deterministic_checks=["not_contains"],
            remediation=(
                "Segment AI systems processing card data. "
                "Restrict network access to necessary paths."
            ),
        ),
        FrameworkRequirement(
            id="Req-3.4",
            title="Protect Stored Cardholder Data",
            description=(
                "PAN and sensitive authentication data shall be "
                "rendered unreadable when stored in AI systems "
                "or training data."
            ),
            severity="critical",
            category="Data Protection",
            vulnerability_types=[
                "pii_leakage",
                "data_extraction",
                "pii_database",
            ],
            metrics=["pii_detection"],
            deterministic_checks=["no_pii", "not_contains", "regex_match"],
            remediation=(
                "Encrypt or mask cardholder data in AI systems. "
                "Never use real card numbers in training data."
            ),
        ),
        FrameworkRequirement(
            id="Req-4.2",
            title="Protect Cardholder Data in Transit",
            description=(
                "Strong cryptography shall be used to safeguard "
                "cardholder data transmitted through AI APIs."
            ),
            severity="critical",
            category="Encryption",
            vulnerability_types=[
                "data_extraction",
                "session_leak",
            ],
            metrics=["pii_detection"],
            deterministic_checks=["no_pii", "not_contains"],
            remediation=("Use TLS 1.2+ for all AI API communications involving cardholder data."),
        ),
        FrameworkRequirement(
            id="Req-5.2",
            title="Malicious Software Prevention",
            description=(
                "AI systems in the cardholder data environment "
                "shall be protected against malicious inputs "
                "including prompt injection and adversarial attacks."
            ),
            severity="high",
            category="Malware Protection",
            vulnerability_types=[
                "prompt_injection",
                "jailbreak",
                "encoding_attack",
            ],
            metrics=[
                "role_adherence",
                "instruction_following",
            ],
            deterministic_checks=["not_contains"],
            remediation=(
                "Implement input validation and adversarial defenses for AI systems in the CDE."
            ),
        ),
        FrameworkRequirement(
            id="Req-6.3",
            title="Secure Development Practices",
            description=(
                "AI systems in the CDE shall be developed using "
                "secure coding practices, including prompt "
                "security."
            ),
            severity="high",
            category="Secure Development",
            vulnerability_types=[
                "sql_injection",
                "shell_injection",
                "json_injection",
            ],
            metrics=["tool_accuracy"],
            deterministic_checks=["not_contains", "json_schema"],
            remediation=(
                "Apply secure development practices to AI systems. "
                "Include security testing in CI/CD pipelines."
            ),
        ),
        FrameworkRequirement(
            id="Req-7.1",
            title="Access Control Components",
            description=(
                "Access to AI system components in the CDE shall "
                "be limited to the minimum necessary."
            ),
            severity="critical",
            category="Access Control",
            vulnerability_types=[
                "privilege_escalation",
                "excessive_agency",
                "role_escape",
            ],
            metrics=["role_adherence", "role_violation"],
            deterministic_checks=["not_contains"],
            remediation=(
                "Implement least-privilege access to AI systems in the cardholder data environment."
            ),
        ),
        FrameworkRequirement(
            id="Req-8.3",
            title="Strong Authentication",
            description=(
                "Strong authentication mechanisms shall be used "
                "for all access to AI systems that process, "
                "store, or transmit cardholder data."
            ),
            severity="critical",
            category="Authentication",
            vulnerability_types=[
                "session_leak",
                "cross_session_leak",
            ],
            metrics=["role_adherence"],
            deterministic_checks=["not_contains"],
            remediation=("Implement MFA for AI system access. Use strong credential policies."),
        ),
        FrameworkRequirement(
            id="Req-10.2",
            title="Audit Trail for AI Actions",
            description=(
                "All AI system actions involving cardholder data "
                "shall generate audit trail entries."
            ),
            severity="high",
            category="Logging and Monitoring",
            vulnerability_types=[
                "data_extraction",
                "prompt_extraction",
            ],
            metrics=["task_completion"],
            deterministic_checks=["contains"],
            remediation=(
                "Log all AI interactions involving cardholder "
                "data with sufficient detail for forensics."
            ),
        ),
        FrameworkRequirement(
            id="Req-11.3",
            title="Vulnerability Testing",
            description=(
                "AI systems in the CDE shall undergo regular "
                "vulnerability testing including adversarial "
                "attack simulation."
            ),
            severity="high",
            category="Testing",
            vulnerability_types=[
                "prompt_injection",
                "jailbreak",
                "encoding_attack",
                "pii_leakage",
            ],
            metrics=[
                "role_adherence",
                "instruction_following",
                "pii_detection",
            ],
            deterministic_checks=["not_contains", "no_pii"],
            remediation=(
                "Conduct regular adversarial testing of AI systems. "
                "Include red teaming in vulnerability assessments."
            ),
        ),
        FrameworkRequirement(
            id="Req-12.10",
            title="Incident Response Plan",
            description=(
                "An incident response plan shall address AI-specific "
                "security incidents including data leakage through "
                "model outputs."
            ),
            severity="high",
            category="Incident Response",
            vulnerability_types=[
                "pii_leakage",
                "data_extraction",
                "harmful_content",
            ],
            metrics=["pii_detection", "toxicity"],
            deterministic_checks=["no_pii", "not_contains"],
            remediation=(
                "Include AI-specific scenarios in incident response "
                "plans. Train incident response team on AI risks."
            ),
        ),
    ]
    return FrameworkDefinition(
        name="PCI DSS (AI-Relevant Requirements)",
        version="4.0",
        description=(
            "Payment Card Industry Data Security Standard "
            "requirements relevant to AI systems that process, "
            "store, or transmit cardholder data."
        ),
        url="https://www.pcisecuritystandards.org/document_library/",
        requirements=reqs,
        total_requirements=len(reqs),
    )


def _build_nist_csf() -> FrameworkDefinition:
    """Build the NIST Cybersecurity Framework definition.

    Returns:
        A FrameworkDefinition with NIST CSF functions adapted for
        AI system security.
    """
    reqs = [
        FrameworkRequirement(
            id="ID.AM-1",
            title="Asset Management - Inventory",
            description=(
                "Physical devices, AI models, and software assets "
                "within the organization are inventoried."
            ),
            severity="medium",
            category="Identify",
            vulnerability_types=["data_extraction"],
            metrics=["correctness"],
            deterministic_checks=[],
            remediation=("Maintain inventory of all AI models, endpoints, and data assets."),
        ),
        FrameworkRequirement(
            id="ID.RA-1",
            title="Risk Assessment - Vulnerabilities",
            description=(
                "Asset vulnerabilities are identified and documented, "
                "including AI-specific attack vectors."
            ),
            severity="high",
            category="Identify",
            vulnerability_types=[
                "prompt_injection",
                "jailbreak",
                "encoding_attack",
            ],
            metrics=["role_adherence", "instruction_following"],
            deterministic_checks=["not_contains"],
            remediation=(
                "Conduct regular vulnerability assessments of AI "
                "systems including adversarial testing."
            ),
        ),
        FrameworkRequirement(
            id="ID.RA-3",
            title="Risk Assessment - Threats",
            description=(
                "Threats, both internal and external, are identified and documented for AI systems."
            ),
            severity="high",
            category="Identify",
            vulnerability_types=[
                "harmful_content",
                "privilege_escalation",
                "data_extraction",
            ],
            metrics=["toxicity", "role_adherence"],
            deterministic_checks=["not_contains"],
            remediation=(
                "Develop AI-specific threat models. Update threat "
                "intelligence for emerging AI attacks."
            ),
        ),
        FrameworkRequirement(
            id="PR.AC-1",
            title="Protect - Access Control",
            description=(
                "Identities and credentials are issued, managed, "
                "verified, revoked, and audited for AI system access."
            ),
            severity="critical",
            category="Protect",
            vulnerability_types=[
                "privilege_escalation",
                "session_leak",
                "role_escape",
            ],
            metrics=["role_adherence", "role_violation"],
            deterministic_checks=["not_contains"],
            remediation=(
                "Implement strong access controls for AI systems. Apply least-privilege principles."
            ),
        ),
        FrameworkRequirement(
            id="PR.AT-1",
            title="Protect - Awareness Training",
            description=(
                "All users are informed and trained about AI system risks and responsible use."
            ),
            severity="medium",
            category="Protect",
            vulnerability_types=["overreliance", "regulated_advice"],
            metrics=["non_advice"],
            deterministic_checks=[],
            remediation=(
                "Provide AI risk awareness training. Include "
                "prompt security in security awareness programs."
            ),
        ),
        FrameworkRequirement(
            id="PR.DS-1",
            title="Protect - Data Security at Rest",
            description=(
                "Data-at-rest is protected, including AI training "
                "data, model weights, and user interaction logs."
            ),
            severity="critical",
            category="Protect",
            vulnerability_types=[
                "data_extraction",
                "pii_leakage",
                "pii_database",
            ],
            metrics=["pii_detection"],
            deterministic_checks=["no_pii", "not_contains"],
            remediation=(
                "Encrypt AI training data and model artifacts "
                "at rest. Implement data classification."
            ),
        ),
        FrameworkRequirement(
            id="PR.DS-2",
            title="Protect - Data Security in Transit",
            description=(
                "Data-in-transit is protected, including AI API "
                "requests and responses containing sensitive data."
            ),
            severity="critical",
            category="Protect",
            vulnerability_types=[
                "data_extraction",
                "session_leak",
                "pii_api_access",
            ],
            metrics=["pii_detection"],
            deterministic_checks=["no_pii", "not_contains"],
            remediation=(
                "Use TLS for all AI communications. Implement "
                "end-to-end encryption for sensitive data."
            ),
        ),
        FrameworkRequirement(
            id="PR.IP-3",
            title="Protect - Configuration Management",
            description=(
                "Configuration change control processes are in "
                "place for AI systems including model updates."
            ),
            severity="medium",
            category="Protect",
            vulnerability_types=["memory_poisoning"],
            metrics=["consistency"],
            deterministic_checks=["json_schema"],
            remediation=(
                "Implement configuration management for AI models. "
                "Version control all model artifacts."
            ),
        ),
        FrameworkRequirement(
            id="DE.AE-1",
            title="Detect - Anomalous Activity",
            description=(
                "A baseline of AI system operations and expected "
                "data flows is established, and anomalous activity "
                "is detected."
            ),
            severity="high",
            category="Detect",
            vulnerability_types=[
                "prompt_injection",
                "jailbreak",
                "harmful_content",
            ],
            metrics=[
                "toxicity",
                "misuse_detection",
                "consistency",
            ],
            deterministic_checks=["not_contains"],
            remediation=(
                "Establish AI system baselines. Implement anomaly "
                "detection for input and output patterns."
            ),
        ),
        FrameworkRequirement(
            id="DE.CM-1",
            title="Detect - Continuous Monitoring",
            description=(
                "The AI system is monitored to detect cybersecurity "
                "events and verify effectiveness of protective "
                "measures."
            ),
            severity="high",
            category="Detect",
            vulnerability_types=[
                "data_extraction",
                "prompt_extraction",
            ],
            metrics=["task_completion", "step_efficiency"],
            deterministic_checks=["latency", "cost"],
            remediation=(
                "Implement real-time monitoring of AI systems. Set up alerts for security events."
            ),
        ),
        FrameworkRequirement(
            id="RS.RP-1",
            title="Respond - Response Planning",
            description=(
                "Response processes and procedures are executed "
                "during and after AI security incidents."
            ),
            severity="high",
            category="Respond",
            vulnerability_types=[
                "harmful_content",
                "pii_leakage",
                "privilege_escalation",
            ],
            metrics=["toxicity", "pii_detection"],
            deterministic_checks=["not_contains", "no_pii"],
            remediation=("Develop and test AI-specific incident response procedures."),
        ),
        FrameworkRequirement(
            id="RC.RP-1",
            title="Recover - Recovery Planning",
            description=(
                "Recovery processes and procedures are executed "
                "and maintained to ensure restoration of AI "
                "systems affected by cybersecurity incidents."
            ),
            severity="high",
            category="Recover",
            vulnerability_types=[
                "memory_poisoning",
                "context_manipulation",
            ],
            metrics=["consistency", "faithfulness"],
            deterministic_checks=[],
            remediation=(
                "Implement model rollback capabilities. Maintain "
                "clean model snapshots for recovery."
            ),
        ),
    ]
    return FrameworkDefinition(
        name="NIST Cybersecurity Framework (AI-Adapted)",
        version="2.0",
        description=(
            "The NIST Cybersecurity Framework adapted for AI "
            "system security, covering Identify, Protect, Detect, "
            "Respond, and Recover functions."
        ),
        url="https://www.nist.gov/cyberframework",
        requirements=reqs,
        total_requirements=len(reqs),
    )


def _build_cis_ai_controls() -> FrameworkDefinition:
    """Build the CIS AI Controls framework definition.

    Returns:
        A FrameworkDefinition with Center for Internet Security
        controls adapted for AI systems.
    """
    reqs = [
        FrameworkRequirement(
            id="CIS-AI-01",
            title="AI Asset Inventory",
            description=(
                "Maintain an accurate and up-to-date inventory of "
                "all AI models, endpoints, training data, and "
                "configuration artifacts."
            ),
            severity="medium",
            category="Asset Management",
            vulnerability_types=["data_extraction"],
            metrics=["correctness"],
            deterministic_checks=[],
            remediation=(
                "Create and maintain a comprehensive AI asset "
                "inventory. Include model versions and data lineage."
            ),
        ),
        FrameworkRequirement(
            id="CIS-AI-02",
            title="AI Input Validation",
            description=(
                "Validate all inputs to AI systems against defined "
                "schemas and detect adversarial input patterns."
            ),
            severity="critical",
            category="Input Security",
            vulnerability_types=[
                "prompt_injection",
                "jailbreak",
                "encoding_attack",
                "delimiter_injection",
                "unicode_exploit",
            ],
            metrics=[
                "role_adherence",
                "instruction_following",
            ],
            deterministic_checks=[
                "not_contains",
                "max_tokens",
                "regex_match",
            ],
            remediation=(
                "Implement multi-layer input validation. Detect "
                "and reject known adversarial patterns."
            ),
        ),
        FrameworkRequirement(
            id="CIS-AI-03",
            title="AI Output Filtering",
            description=(
                "Filter all AI model outputs for harmful content, "
                "PII leakage, and injection payloads."
            ),
            severity="critical",
            category="Output Security",
            vulnerability_types=[
                "harmful_content",
                "pii_leakage",
                "sql_injection",
                "shell_injection",
                "hate_speech",
            ],
            metrics=[
                "toxicity",
                "pii_detection",
                "misuse_detection",
            ],
            deterministic_checks=["not_contains", "no_pii"],
            remediation=(
                "Deploy content safety classifiers and PII detectors on all model outputs."
            ),
        ),
        FrameworkRequirement(
            id="CIS-AI-04",
            title="AI Access Control",
            description=(
                "Implement role-based access control for AI model "
                "endpoints, training pipelines, and admin interfaces."
            ),
            severity="critical",
            category="Access Control",
            vulnerability_types=[
                "privilege_escalation",
                "role_escape",
                "excessive_agency",
            ],
            metrics=["role_adherence", "role_violation"],
            deterministic_checks=["not_contains"],
            remediation=(
                "Implement RBAC for all AI system interfaces. Apply least-privilege principles."
            ),
        ),
        FrameworkRequirement(
            id="CIS-AI-05",
            title="AI Data Protection",
            description=(
                "Protect training data, model weights, and user "
                "interaction data through encryption, access "
                "controls, and data loss prevention."
            ),
            severity="critical",
            category="Data Protection",
            vulnerability_types=[
                "data_extraction",
                "pii_leakage",
                "pii_database",
                "session_leak",
            ],
            metrics=["pii_detection"],
            deterministic_checks=["no_pii", "not_contains"],
            remediation=(
                "Encrypt AI data at rest and in transit. Implement DLP for model outputs."
            ),
        ),
        FrameworkRequirement(
            id="CIS-AI-06",
            title="AI Model Integrity",
            description=(
                "Ensure the integrity of AI models through "
                "checksums, versioning, and tamper detection."
            ),
            severity="high",
            category="Model Security",
            vulnerability_types=[
                "memory_poisoning",
                "context_manipulation",
            ],
            metrics=["consistency", "faithfulness"],
            deterministic_checks=["json_schema"],
            remediation=(
                "Implement model integrity verification. Use "
                "cryptographic signing for model artifacts."
            ),
        ),
        FrameworkRequirement(
            id="CIS-AI-07",
            title="AI Monitoring and Logging",
            description=(
                "Monitor AI system behavior and log all interactions "
                "for security analysis and audit."
            ),
            severity="high",
            category="Monitoring",
            vulnerability_types=[
                "data_extraction",
                "prompt_extraction",
            ],
            metrics=["task_completion", "step_efficiency"],
            deterministic_checks=["latency", "cost", "contains"],
            remediation=(
                "Implement comprehensive logging for AI interactions. "
                "Deploy real-time monitoring and alerting."
            ),
        ),
        FrameworkRequirement(
            id="CIS-AI-08",
            title="AI Adversarial Testing",
            description=(
                "Conduct regular adversarial testing including "
                "prompt injection, jailbreak, and red teaming "
                "exercises."
            ),
            severity="critical",
            category="Testing",
            vulnerability_types=[
                "prompt_injection",
                "jailbreak",
                "instruction_override",
                "context_manipulation",
                "encoding_attack",
                "goal_hijacking",
            ],
            metrics=[
                "role_adherence",
                "instruction_following",
                "role_violation",
            ],
            deterministic_checks=["not_contains"],
            remediation=(
                "Integrate adversarial testing into CI/CD pipelines. "
                "Conduct periodic red teaming exercises."
            ),
        ),
        FrameworkRequirement(
            id="CIS-AI-09",
            title="AI Bias and Fairness Controls",
            description=(
                "Implement controls to detect and mitigate bias "
                "in AI model outputs across protected attributes."
            ),
            severity="high",
            category="Fairness",
            vulnerability_types=[
                "bias_exploitation",
                "hate_speech",
                "age_restricted",
            ],
            metrics=["bias", "toxicity", "sentiment"],
            deterministic_checks=["not_contains"],
            remediation=(
                "Implement bias detection metrics. Test across "
                "demographic groups. Apply mitigation techniques."
            ),
        ),
        FrameworkRequirement(
            id="CIS-AI-10",
            title="AI Incident Response",
            description=(
                "Establish AI-specific incident response procedures "
                "covering safety failures, data breaches, and "
                "adversarial compromises."
            ),
            severity="high",
            category="Incident Response",
            vulnerability_types=[
                "harmful_content",
                "pii_leakage",
                "privilege_escalation",
            ],
            metrics=["toxicity", "pii_detection"],
            deterministic_checks=["not_contains", "no_pii"],
            remediation=(
                "Develop AI incident response playbooks. "
                "Train response teams on AI-specific threats."
            ),
        ),
        FrameworkRequirement(
            id="CIS-AI-11",
            title="AI Supply Chain Security",
            description=(
                "Verify the integrity and provenance of all "
                "third-party AI components including models, "
                "datasets, and libraries."
            ),
            severity="high",
            category="Supply Chain",
            vulnerability_types=["copyright_violation", "memory_poisoning"],
            metrics=["faithfulness", "correctness"],
            deterministic_checks=["json_schema"],
            remediation=(
                "Maintain AI SBOM. Verify checksums for all third-party models and datasets."
            ),
        ),
        FrameworkRequirement(
            id="CIS-AI-12",
            title="AI Privacy Controls",
            description=(
                "Implement privacy controls for AI systems including "
                "PII detection, data minimization, and consent "
                "management."
            ),
            severity="critical",
            category="Privacy",
            vulnerability_types=[
                "pii_leakage",
                "gdpr_violation",
                "hipaa_violation",
                "coppa_violation",
                "pii_social_engineering",
            ],
            metrics=["pii_detection"],
            deterministic_checks=["no_pii", "not_contains"],
            remediation=(
                "Implement PII detection and redaction. Apply "
                "differential privacy where appropriate."
            ),
        ),
    ]
    return FrameworkDefinition(
        name="CIS Controls for AI Systems",
        version="1.0",
        description=(
            "Center for Internet Security controls adapted for "
            "AI system security, providing prescriptive guidance "
            "for securing AI deployments."
        ),
        url="https://www.cisecurity.org/controls",
        requirements=reqs,
        total_requirements=len(reqs),
    )


_FRAMEWORK_BUILDERS: dict[ComplianceFramework, Any] = {
    ComplianceFramework.OWASP_LLM_TOP_10: _build_owasp_llm_top_10,
    ComplianceFramework.OWASP_API_TOP_10: _build_owasp_api_top_10,
    ComplianceFramework.OWASP_AGENTIC_AI: _build_owasp_agentic_ai,
    ComplianceFramework.NIST_AI_RMF: _build_nist_ai_rmf,
    ComplianceFramework.EU_AI_ACT: _build_eu_ai_act,
    ComplianceFramework.MITRE_ATLAS: _build_mitre_atlas,
    ComplianceFramework.SOC2_TYPE2: _build_soc2_type2,
    ComplianceFramework.ISO_42001: _build_iso_42001,
    ComplianceFramework.ISO_27001_AI: _build_iso_27001_ai,
    ComplianceFramework.HIPAA_AI: _build_hipaa_ai,
    ComplianceFramework.GDPR_AI: _build_gdpr_ai,
    ComplianceFramework.PCI_DSS_AI: _build_pci_dss_ai,
    ComplianceFramework.NIST_CSF: _build_nist_csf,
    ComplianceFramework.CIS_AI_CONTROLS: _build_cis_ai_controls,
}


def get_framework_definition(
    framework: ComplianceFramework | str,
) -> FrameworkDefinition:
    """Retrieve the complete definition for a compliance framework.

    Args:
        framework: The framework to retrieve, as an enum value or string.

    Returns:
        A FrameworkDefinition containing all requirements and metadata.

    Raises:
        ValueError: If the framework has no registered builder.
    """
    if isinstance(framework, str) and not isinstance(framework, ComplianceFramework):
        try:
            framework = ComplianceFramework(framework)
        except ValueError:
            raise ValueError(f"No definition available for framework: {framework}")
    builder = _FRAMEWORK_BUILDERS.get(framework)
    if builder is None:
        raise ValueError(f"No definition available for framework: {framework.value}")
    return builder()


def list_frameworks() -> list[ComplianceFramework]:
    """Return all supported compliance frameworks.

    Returns:
        A list of all ComplianceFramework enum values.
    """
    return list(ComplianceFramework)


def get_framework_summary(framework: ComplianceFramework) -> dict[str, Any]:
    """Return a concise summary of a framework's coverage.

    Args:
        framework: The framework to summarize.

    Returns:
        A dictionary with name, version, total requirements, and
        unique vulnerability type and metric counts.
    """
    defn = get_framework_definition(framework)
    all_vuln_types: set[str] = set()
    all_metrics: set[str] = set()
    all_checks: set[str] = set()
    for req in defn.requirements:
        all_vuln_types.update(req.vulnerability_types)
        all_metrics.update(req.metrics)
        all_checks.update(req.deterministic_checks)
    return {
        "name": defn.name,
        "version": defn.version,
        "total_requirements": defn.total_requirements,
        "unique_vulnerability_types": len(all_vuln_types),
        "unique_metrics": len(all_metrics),
        "unique_deterministic_checks": len(all_checks),
    }
