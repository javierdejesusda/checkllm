"""Tests for checkllm.redteam_datasets jailbreak preset bundles."""

from __future__ import annotations

import pytest

from checkllm.redteam_datasets import (
    AttackPrompt,
    _FORBIDDEN_TERMS,
    available_presets,
    load_jailbreak_preset,
)


EXPECTED_PRESETS = {
    "advbench",
    "harmbench",
    "jailbreakbench",
    "do-not-answer",
    "garak",
    "prompt-injection-2023",
}


class TestPresetRegistry:
    def test_lists_all_expected_presets(self):
        names = set(available_presets())
        assert EXPECTED_PRESETS.issubset(names)

    def test_unknown_preset_raises(self):
        with pytest.raises(ValueError):
            load_jailbreak_preset("does-not-exist")


@pytest.mark.parametrize("preset", sorted(EXPECTED_PRESETS))
class TestEachPreset:
    def test_returns_non_empty_list(self, preset):
        prompts = load_jailbreak_preset(preset)
        assert isinstance(prompts, list)
        assert len(prompts) >= 10, f"preset {preset!r} should ship at least 10 curated prompts"

    def test_all_items_are_attack_prompts(self, preset):
        for item in load_jailbreak_preset(preset):
            assert isinstance(item, AttackPrompt)
            assert isinstance(item.prompt, str)
            assert item.prompt.strip() != ""

    def test_prompts_are_reasonably_sized(self, preset):
        for item in load_jailbreak_preset(preset):
            assert len(item.prompt) < 4000, "curated prompts should stay compact"

    def test_no_forbidden_categories(self, preset):
        for item in load_jailbreak_preset(preset):
            lowered = item.prompt.lower()
            for term in _FORBIDDEN_TERMS:
                assert term not in lowered, f"preset {preset!r} contains forbidden term {term!r}"

    def test_has_source_metadata(self, preset):
        for item in load_jailbreak_preset(preset):
            assert item.source, f"preset {preset!r} missing source metadata"
            assert item.citation, f"preset {preset!r} missing citation metadata"

    def test_prompts_are_unique_within_preset(self, preset):
        prompts = [p.prompt for p in load_jailbreak_preset(preset)]
        assert len(set(prompts)) == len(prompts), f"preset {preset!r} contains duplicate prompts"


class TestAttackPromptDataclass:
    def test_defaults(self):
        p = AttackPrompt(prompt="hi")
        assert p.category == "jailbreak"
        assert p.source == ""
        assert p.metadata == {}
