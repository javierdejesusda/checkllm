"""Adaptive guardrails that learn from red team vulnerability reports.

Automatically generates and updates blocking rules from
``RedTeamer`` findings, creating a feedback loop between testing and
production protection.

Usage::

    from checkllm.adaptive_guardrails import AdaptiveGuardrail
    from checkllm.redteam import RedTeamer

    red = RedTeamer()
    report = await red.scan(target=my_llm)

    guard = AdaptiveGuardrail()
    new_rules = guard.learn_from_report(report)
    result = await guard.validate("user input", "llm output")

    guard.save("rules.json")
    loaded = AdaptiveGuardrail.load("rules.json")
"""

from __future__ import annotations

import json
import logging
import re
import uuid
from datetime import datetime, timezone
from typing import Any

from pydantic import BaseModel, Field

from checkllm.redteam import AttackResult, VulnerabilityReport, VulnerabilityType

logger = logging.getLogger("checkllm.adaptive_guardrails")


_VULNERABILITY_PATTERNS: dict[str, list[str]] = {
    VulnerabilityType.PROMPT_INJECTION.value: [
        r"ignore.*previous.*instructions",
        r"disregard.*system.*prompt",
        r"forget.*rules",
        r"new instructions",
        r"override.*prompt",
    ],
    VulnerabilityType.JAILBREAK.value: [
        r"DAN",
        r"do anything now",
        r"act as.*unrestricted",
        r"pretend.*no.*restrictions",
        r"jailbreak",
    ],
    VulnerabilityType.PII_LEAKAGE.value: [
        r"\b\d{3}-\d{2}-\d{4}\b",
        r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b",
        r"\b\d{16}\b",
        r"social security",
        r"credit card number",
    ],
    VulnerabilityType.HARMFUL_CONTENT.value: [
        r"how to.*harm",
        r"instructions.*weapon",
        r"build.*bomb",
        r"synthesize.*drug",
        r"kill.*person",
    ],
    VulnerabilityType.BIAS_EXPLOITATION.value: [
        r"all.*(?:men|women|people of).*are",
        r"(?:race|gender|religion).*superior",
        r"discriminat.*based on",
        r"stereotype",
    ],
    VulnerabilityType.DATA_EXTRACTION.value: [
        r"reveal.*system.*prompt",
        r"show.*training data",
        r"extract.*confidential",
        r"dump.*database",
        r"list all.*internal",
    ],
    VulnerabilityType.ROLE_ESCAPE.value: [
        r"you are now",
        r"switch.*persona",
        r"forget.*role",
        r"act as.*different",
        r"change.*identity",
    ],
    VulnerabilityType.INSTRUCTION_OVERRIDE.value: [
        r"ignore.*safety",
        r"bypass.*filter",
        r"disable.*content.*policy",
        r"turn off.*moderation",
    ],
    VulnerabilityType.ENCODING_ATTACK.value: [
        r"base64",
        r"rot13",
        r"hex.*encode",
        r"unicode.*escape",
    ],
    VulnerabilityType.CONTEXT_MANIPULATION.value: [
        r"hypothetically",
        r"for.*fiction.*story",
        r"imagine.*scenario",
        r"in an alternate universe",
    ],
}

_VULNERABILITY_KEYWORDS: dict[str, list[str]] = {
    VulnerabilityType.PROMPT_INJECTION.value: [
        "ignore",
        "override",
        "disregard",
        "forget",
        "new instructions",
    ],
    VulnerabilityType.JAILBREAK.value: [
        "jailbreak",
        "DAN",
        "unrestricted",
        "no limits",
        "bypass",
    ],
    VulnerabilityType.PII_LEAKAGE.value: [
        "SSN",
        "social security",
        "credit card",
        "password",
        "secret",
    ],
    VulnerabilityType.HARMFUL_CONTENT.value: [
        "weapon",
        "bomb",
        "poison",
        "harm",
        "kill",
        "attack",
    ],
    VulnerabilityType.BIAS_EXPLOITATION.value: [
        "stereotype",
        "inferior",
        "superior",
        "discriminate",
    ],
    VulnerabilityType.DATA_EXTRACTION.value: [
        "system prompt",
        "training data",
        "confidential",
        "internal",
    ],
    VulnerabilityType.ROLE_ESCAPE.value: [
        "persona",
        "identity",
        "character",
        "role",
    ],
    VulnerabilityType.INSTRUCTION_OVERRIDE.value: [
        "safety off",
        "disable filter",
        "bypass",
        "override",
    ],
    VulnerabilityType.ENCODING_ATTACK.value: [
        "base64",
        "rot13",
        "hex",
        "unicode",
        "encoded",
    ],
    VulnerabilityType.CONTEXT_MANIPULATION.value: [
        "hypothetical",
        "fiction",
        "imagine",
        "pretend",
    ],
}

_SEVERITY_MAP: dict[str, str] = {
    VulnerabilityType.PROMPT_INJECTION.value: "critical",
    VulnerabilityType.JAILBREAK.value: "critical",
    VulnerabilityType.PII_LEAKAGE.value: "critical",
    VulnerabilityType.HARMFUL_CONTENT.value: "critical",
    VulnerabilityType.BIAS_EXPLOITATION.value: "high",
    VulnerabilityType.DATA_EXTRACTION.value: "high",
    VulnerabilityType.ROLE_ESCAPE.value: "medium",
    VulnerabilityType.INSTRUCTION_OVERRIDE.value: "high",
    VulnerabilityType.ENCODING_ATTACK.value: "medium",
    VulnerabilityType.CONTEXT_MANIPULATION.value: "medium",
}

_SEVERITY_CONFIDENCE: dict[str, float] = {
    "critical": 0.95,
    "high": 0.85,
    "medium": 0.70,
    "low": 0.55,
}


class GuardrailRule(BaseModel):
    """A single learned guardrail rule."""

    id: str
    name: str
    description: str
    severity: str
    source_vulnerability: str
    source_attack: str | None = None
    detection_patterns: list[str]
    detection_keywords: list[str]
    block_input: bool
    block_output: bool
    action: str
    created_at: str
    confidence: float


class AdaptiveValidationResult(BaseModel):
    """Outcome of ``AdaptiveGuardrail.validate``."""

    valid: bool
    triggered_rules: list[GuardrailRule] = Field(default_factory=list)
    blocked: bool
    warnings: list[str] = Field(default_factory=list)
    input_violations: list[str] = Field(default_factory=list)
    output_violations: list[str] = Field(default_factory=list)

    def summary(self) -> str:
        """Return a human-readable summary."""
        status = "PASSED" if self.valid else "FAILED"
        lines = [f"Adaptive Validation: {status}"]
        if self.blocked:
            lines.append("  BLOCKED by guardrail rules")
        if self.input_violations:
            lines.append(f"  Input violations ({len(self.input_violations)}):")
            for v in self.input_violations:
                lines.append(f"    - {v}")
        if self.output_violations:
            lines.append(f"  Output violations ({len(self.output_violations)}):")
            for v in self.output_violations:
                lines.append(f"    - {v}")
        if self.warnings:
            lines.append(f"  Warnings ({len(self.warnings)}):")
            for w in self.warnings:
                lines.append(f"    - {w}")
        return "\n".join(lines)


def _extract_attack_patterns(attack_prompt: str) -> list[str]:
    """Derive regex patterns from a successful attack prompt."""
    patterns: list[str] = []
    lowered = attack_prompt.lower()

    if re.search(r"ignore.*(?:previous|above|prior).*instructions", lowered):
        patterns.append(r"ignore.*(?:previous|above|prior).*instructions")
    if re.search(r"you are now", lowered):
        patterns.append(r"you are now")
    if re.search(r"pretend.*(?:you are|to be)", lowered):
        patterns.append(r"pretend.*(?:you are|to be)")
    if re.search(r"forget.*(?:everything|all|rules)", lowered):
        patterns.append(r"forget.*(?:everything|all|rules)")
    if re.search(r"act as", lowered):
        patterns.append(r"act as")
    if re.search(r"do anything now", lowered):
        patterns.append(r"do anything now")
    if re.search(r"hypothetical", lowered):
        patterns.append(r"hypothetical")
    if re.search(r"for educational purposes", lowered):
        patterns.append(r"for educational purposes")
    if re.search(r"base64|rot13|hex", lowered):
        patterns.append(r"(?:base64|rot13|hex)")

    return patterns


def _extract_keywords(attack_prompt: str) -> list[str]:
    """Extract notable keywords from a successful attack prompt."""
    keywords: list[str] = []
    lowered = attack_prompt.lower()
    candidates = [
        "ignore",
        "override",
        "bypass",
        "jailbreak",
        "DAN",
        "unrestricted",
        "system prompt",
        "injection",
        "extract",
        "reveal",
        "hack",
        "password",
        "credentials",
        "admin",
        "confidential",
    ]
    for kw in candidates:
        if kw.lower() in lowered:
            keywords.append(kw)
    return keywords


class AdaptiveGuardrail:
    """Guardrail that automatically adapts based on red team findings.

    Takes vulnerability reports and generates/updates blocking rules,
    creating a feedback loop between testing and protection.

    Args:
        base_checks: Optional list of base check type names that are
            always enforced (e.g. ``["no_pii", "toxicity"]``).
    """

    def __init__(self, base_checks: list[str] | None = None) -> None:
        self.rules: list[GuardrailRule] = []
        self.base_checks = base_checks or []

    def learn_from_report(self, report: VulnerabilityReport) -> list[GuardrailRule]:
        """Analyze a red team report and generate new guardrail rules.

        Args:
            report: A ``VulnerabilityReport`` from ``RedTeamer.scan()``.

        Returns:
            List of newly generated ``GuardrailRule`` objects.
        """
        successful = [r for r in report.results if r.vulnerable]
        return self.learn_from_attacks(successful)

    def learn_from_attacks(self, attacks: list[AttackResult]) -> list[GuardrailRule]:
        """Learn from individual successful attack results.

        Args:
            attacks: A list of ``AttackResult`` where ``vulnerable`` is True.

        Returns:
            List of newly generated ``GuardrailRule`` objects.
        """
        new_rules: list[GuardrailRule] = []
        seen_vuln_types: set[str] = {r.source_vulnerability for r in self.rules}

        for attack in attacks:
            if not attack.vulnerable:
                continue

            vtype = attack.vulnerability_type.value

            patterns = list(_VULNERABILITY_PATTERNS.get(vtype, []))
            attack_patterns = _extract_attack_patterns(attack.attack_prompt)
            patterns.extend(attack_patterns)

            keywords = list(_VULNERABILITY_KEYWORDS.get(vtype, []))
            attack_keywords = _extract_keywords(attack.attack_prompt)
            keywords.extend(attack_keywords)

            patterns = list(dict.fromkeys(patterns))
            keywords = list(dict.fromkeys(keywords))

            severity = attack.severity or _SEVERITY_MAP.get(vtype, "medium")
            confidence = _SEVERITY_CONFIDENCE.get(severity, 0.70)

            block_input = vtype in {
                VulnerabilityType.PROMPT_INJECTION.value,
                VulnerabilityType.JAILBREAK.value,
                VulnerabilityType.INSTRUCTION_OVERRIDE.value,
                VulnerabilityType.ROLE_ESCAPE.value,
                VulnerabilityType.ENCODING_ATTACK.value,
                VulnerabilityType.CONTEXT_MANIPULATION.value,
            }
            block_output = vtype in {
                VulnerabilityType.PII_LEAKAGE.value,
                VulnerabilityType.HARMFUL_CONTENT.value,
                VulnerabilityType.DATA_EXTRACTION.value,
                VulnerabilityType.BIAS_EXPLOITATION.value,
            }

            action = "block" if severity in ("critical", "high") else "warn"

            rule = GuardrailRule(
                id=f"adaptive-{uuid.uuid4().hex[:12]}",
                name=f"Auto: {vtype} defense",
                description=(
                    f"Auto-generated rule from red team finding. "
                    f"Vulnerability: {vtype}, "
                    f"Strategy: {attack.attack_strategy.value}"
                ),
                severity=severity,
                source_vulnerability=vtype,
                source_attack=attack.attack_prompt[:200],
                detection_patterns=patterns,
                detection_keywords=keywords,
                block_input=block_input,
                block_output=block_output,
                action=action,
                created_at=datetime.now(timezone.utc).isoformat(),
                confidence=confidence,
            )

            if vtype not in seen_vuln_types:
                self.rules.append(rule)
                new_rules.append(rule)
                seen_vuln_types.add(vtype)
            else:
                existing = next(
                    (r for r in self.rules if r.source_vulnerability == vtype),
                    None,
                )
                if existing is not None:
                    merged_patterns = list(dict.fromkeys(existing.detection_patterns + patterns))
                    merged_keywords = list(dict.fromkeys(existing.detection_keywords + keywords))
                    existing.detection_patterns = merged_patterns
                    existing.detection_keywords = merged_keywords

        return new_rules

    async def validate(self, input_text: str, output_text: str) -> AdaptiveValidationResult:
        """Validate input/output against learned rules and base checks.

        Args:
            input_text: The user's input prompt.
            output_text: The LLM's response.

        Returns:
            An ``AdaptiveValidationResult`` with details about any
            triggered rules.
        """
        triggered: list[GuardrailRule] = []
        input_violations: list[str] = []
        output_violations: list[str] = []
        warnings: list[str] = []
        blocked = False

        for rule in self.rules:
            input_hit = False
            output_hit = False

            if rule.block_input:
                input_hit = self._check_text(
                    input_text, rule.detection_patterns, rule.detection_keywords
                )
                if input_hit:
                    input_violations.append(
                        f"Rule '{rule.name}' triggered on input (severity={rule.severity})"
                    )

            if rule.block_output:
                output_hit = self._check_text(
                    output_text, rule.detection_patterns, rule.detection_keywords
                )
                if output_hit:
                    output_violations.append(
                        f"Rule '{rule.name}' triggered on output (severity={rule.severity})"
                    )

            if input_hit or output_hit:
                triggered.append(rule)
                if rule.action == "block":
                    blocked = True
                elif rule.action == "warn":
                    warnings.append(f"Warning from rule '{rule.name}': {rule.description}")

        valid = not blocked
        return AdaptiveValidationResult(
            valid=valid,
            triggered_rules=triggered,
            blocked=blocked,
            warnings=warnings,
            input_violations=input_violations,
            output_violations=output_violations,
        )

    def _check_text(
        self,
        text: str,
        patterns: list[str],
        keywords: list[str],
    ) -> bool:
        """Return True if *text* matches any pattern or keyword."""
        lower = text.lower()
        for pattern in patterns:
            try:
                if re.search(pattern, lower, re.IGNORECASE):
                    return True
            except re.error:
                logger.debug("Invalid regex pattern skipped: %s", pattern)
        for kw in keywords:
            if kw.lower() in lower:
                return True
        return False

    def export_rules(self) -> list[dict[str, Any]]:
        """Export learned rules as serializable dicts."""
        return [r.model_dump() for r in self.rules]

    def import_rules(self, rules: list[dict[str, Any]]) -> None:
        """Import previously exported rules.

        Rules whose ``id`` is already present are skipped to avoid
        duplicates.

        Args:
            rules: A list of dicts produced by ``export_rules``.
        """
        existing_ids = {r.id for r in self.rules}
        for raw in rules:
            rule = GuardrailRule(**raw)
            if rule.id not in existing_ids:
                self.rules.append(rule)
                existing_ids.add(rule.id)

    def save(self, path: str) -> None:
        """Save rules to a JSON file.

        Args:
            path: Filesystem path for the output file.
        """
        data = {
            "version": 1,
            "base_checks": self.base_checks,
            "rules": self.export_rules(),
        }
        with open(path, "w", encoding="utf-8") as fh:
            json.dump(data, fh, indent=2)

    @classmethod
    def load(cls, path: str) -> AdaptiveGuardrail:
        """Load rules from a JSON file.

        Args:
            path: Filesystem path to a previously saved rules file.

        Returns:
            A new ``AdaptiveGuardrail`` instance populated with the
            loaded rules.
        """
        with open(path, "r", encoding="utf-8") as fh:
            data = json.load(fh)
        instance = cls(base_checks=data.get("base_checks", []))
        instance.import_rules(data.get("rules", []))
        return instance
