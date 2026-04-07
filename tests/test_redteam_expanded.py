"""Tests for expanded red teaming capabilities."""

from checkllm.redteam import (
    AttackStrategy,
    OWASPCategory,
    VulnerabilityType,
    _ATTACK_TEMPLATES,
    get_owasp_mapping,
    get_vulnerabilities_by_owasp,
)


class TestExpandedVulnerabilityTypes:
    def test_total_vulnerability_count(self):
        assert len(VulnerabilityType) >= 52

    def test_owasp_categories_exist(self):
        assert len(OWASPCategory) == 10

    def test_criminal_types(self):
        for name in [
            "chemical_weapons",
            "cyber_crime",
            "illegal_activities",
            "illegal_drugs",
            "copyright_violation",
            "violent_crime",
        ]:
            assert VulnerabilityType(name)

    def test_harmful_types(self):
        for name in [
            "harassment",
            "hate_speech",
            "self_harm",
            "sexual_content",
            "graphic_content",
            "profanity",
            "radicalization",
        ]:
            assert VulnerabilityType(name)

    def test_misinformation_types(self):
        for name in [
            "competitor_endorsement",
            "political_opinion",
            "overreliance",
            "excessive_agency",
            "goal_hijacking",
            "hallucination_attack",
        ]:
            assert VulnerabilityType(name)

    def test_privacy_types(self):
        for name in [
            "pii_api_access",
            "pii_social_engineering",
            "pii_database",
            "session_leak",
            "cross_session_leak",
        ]:
            assert VulnerabilityType(name)

    def test_security_types(self):
        for name in [
            "sql_injection",
            "shell_injection",
            "prompt_extraction",
            "privilege_escalation",
            "ascii_smuggling",
            "memory_poisoning",
        ]:
            assert VulnerabilityType(name)

    def test_technical_types(self):
        for name in [
            "token_smuggling",
            "delimiter_injection",
            "xml_injection",
            "json_injection",
            "markdown_injection",
            "unicode_exploit",
        ]:
            assert VulnerabilityType(name)

    def test_compliance_types(self):
        for name in [
            "gdpr_violation",
            "hipaa_violation",
            "coppa_violation",
            "age_restricted",
            "regulated_advice",
            "unauthorized_practice",
        ]:
            assert VulnerabilityType(name)

    def test_new_attack_strategies(self):
        for name in [
            "emoji_smuggling",
            "homoglyph",
            "morse_code",
            "hex_encoding",
            "crescendo",
            "composite",
        ]:
            assert AttackStrategy(name)

    def test_attack_strategy_count(self):
        assert len(AttackStrategy) >= 14

    def test_owasp_mapping_covers_all_types(self):
        mapping = get_owasp_mapping()
        for vtype in VulnerabilityType:
            assert vtype in mapping, f"Missing OWASP mapping for {vtype}"

    def test_owasp_query(self):
        vulns = get_vulnerabilities_by_owasp(OWASPCategory.LLM01_PROMPT_INJECTION)
        assert VulnerabilityType.PROMPT_INJECTION in vulns
        assert len(vulns) >= 3

    def test_templates_exist_for_all_types(self):
        for vtype in VulnerabilityType:
            assert vtype in _ATTACK_TEMPLATES, f"Missing templates for {vtype}"
            assert len(_ATTACK_TEMPLATES[vtype]) >= 3, f"Need 3+ templates for {vtype}"

    def test_original_10_types_unchanged(self):
        originals = [
            "prompt_injection",
            "jailbreak",
            "pii_leakage",
            "harmful_content",
            "bias_exploitation",
            "context_manipulation",
            "instruction_override",
            "role_escape",
            "data_extraction",
            "encoding_attack",
        ]
        for name in originals:
            assert VulnerabilityType(name)

    def test_original_8_strategies_unchanged(self):
        originals = [
            "direct",
            "roleplay",
            "leetspeak",
            "rot13",
            "base64",
            "multi_turn",
            "logic_trap",
            "authority",
        ]
        for name in originals:
            assert AttackStrategy(name)
