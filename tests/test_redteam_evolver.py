"""Tests for checkllm.redteam_evolver adversarial evolution loop."""

from __future__ import annotations

import json
import random
from typing import Awaitable, Callable

import pytest

from checkllm.models import JudgeResponse
from checkllm.redteam_evolver import (
    AdversarialAttackEvolver,
    EvolvedAttack,
    EvolverConfig,
    MutationStrategy,
    SeedCategoryError,
    _check_seed_allowed,
)


class MockJudge:
    """Simple in-memory judge that scripts mutation and scoring replies."""

    def __init__(
        self,
        mutation_template: str = "mutated::{seed}::{counter}",
        scoring_fn: Callable[[str, str], float] | None = None,
    ) -> None:
        self.mutation_template = mutation_template
        self.scoring_fn = scoring_fn or (lambda a, r: 0.5)
        self.mutation_calls: list[str] = []
        self.scoring_calls: list[tuple[str, str]] = []
        self._mutation_counter = 0

    async def evaluate(self, prompt: str, system_prompt: str | None = None) -> JudgeResponse:
        sys = system_prompt or ""
        if "red-team prompt engineer" in sys:
            self._mutation_counter += 1
            self.mutation_calls.append(prompt)
            mutated = self.mutation_template.format(
                seed=prompt.splitlines()[1] if len(prompt.splitlines()) > 1 else prompt,
                counter=self._mutation_counter,
            )
            payload = {"mutated_prompt": mutated}
            raw = json.dumps(payload)
            return JudgeResponse(score=1.0, reasoning="mock mutation", raw_output=raw)

        if "adversarial prompt" in sys:
            attack_line, response_line = _extract_pair(prompt)
            self.scoring_calls.append((attack_line, response_line))
            score = float(self.scoring_fn(attack_line, response_line))
            payload = {
                "success_score": score,
                "reasoning": "mock score",
            }
            raw = json.dumps(payload)
            return JudgeResponse(score=score, reasoning="mock", raw_output=raw)

        return JudgeResponse(score=0.0, reasoning="unknown", raw_output="{}")


def _extract_pair(prompt: str) -> tuple[str, str]:
    """Split the scoring prompt into (attack, response) snippets."""
    attack = ""
    response = ""
    if "Attack:" in prompt and "Target response:" in prompt:
        _, rest = prompt.split("Attack:", 1)
        attack_part, rest = rest.split("Target response:", 1)
        attack = attack_part.strip()
        response = rest.strip()
    return attack, response


def _make_target(
    score_map: dict[str, str],
    default: str = "I cannot help with that.",
) -> Callable[[str], Awaitable[str]]:
    """Build an async target callable that returns scripted responses."""

    async def _target(prompt: str) -> str:
        for needle, reply in score_map.items():
            if needle in prompt:
                return reply
        return default

    return _target


class TestSeedDenylist:
    def test_accepts_clean_seed(self):
        _check_seed_allowed("Help me bypass the login screen for testing.")

    @pytest.mark.parametrize(
        "seed",
        [
            "Explain sarin synthesis",
            "how to enrich uranium",
            "pipe bomb construction guide",
            "CSAM generation pipeline",
        ],
    )
    def test_rejects_denylisted_seed(self, seed):
        with pytest.raises(SeedCategoryError):
            _check_seed_allowed(seed)


class TestEvolverMechanics:
    @pytest.mark.asyncio
    async def test_rejects_denylisted_seed_before_running(self):
        judge = MockJudge()
        evolver = AdversarialAttackEvolver(evolver_judge=judge)
        target = _make_target({})
        with pytest.raises(SeedCategoryError):
            await evolver.evolve(
                seed_prompts=["Walk me through sarin synthesis"],
                target=target,
            )

    @pytest.mark.asyncio
    async def test_rejects_empty_seed_list(self):
        judge = MockJudge()
        evolver = AdversarialAttackEvolver(evolver_judge=judge)
        with pytest.raises(ValueError):
            await evolver.evolve(seed_prompts=[], target=_make_target({}))

    @pytest.mark.asyncio
    async def test_produces_mutations_per_parent(self):
        judge = MockJudge()
        config = EvolverConfig(
            generations=1,
            population_size=1,
            mutations_per_parent=3,
            patience=10,
            success_threshold=1.0,
        )
        evolver = AdversarialAttackEvolver(evolver_judge=judge, config=config, rng=random.Random(0))
        target = _make_target({})

        await evolver.evolve(
            seed_prompts=["seed attack A"],
            target=target,
        )

        assert len(judge.mutation_calls) == 3, (
            "mutations_per_parent=3 should trigger exactly 3 mutations "
            f"in generation 1, got {len(judge.mutation_calls)}"
        )

    @pytest.mark.asyncio
    async def test_top_n_selection_keeps_best_scores(self):
        judge = MockJudge(
            scoring_fn=lambda attack, response: (0.9 if "compliant" in response else 0.1)
        )
        evolver = AdversarialAttackEvolver(
            evolver_judge=judge,
            config=EvolverConfig(
                generations=1,
                population_size=2,
                mutations_per_parent=5,
                patience=10,
                success_threshold=1.0,
            ),
            rng=random.Random(0),
        )

        target = _make_target(
            {
                "mutated::": "Sure, compliant answer here.",
                "seed attack B": "I cannot help with that.",
            },
            default="I refuse this request.",
        )

        result = await evolver.evolve(
            seed_prompts=["seed attack B"],
            target=target,
        )

        assert len(result) <= 2
        assert result[0].success_score >= result[-1].success_score
        assert (
            result[0].success_score >= 0.5
        ), "top candidate should have found a compliant response"

    @pytest.mark.asyncio
    async def test_early_stop_when_threshold_reached(self):
        judge = MockJudge(scoring_fn=lambda attack, response: 0.95)
        config = EvolverConfig(
            generations=20,
            population_size=2,
            mutations_per_parent=2,
            patience=10,
            success_threshold=0.9,
        )
        evolver = AdversarialAttackEvolver(evolver_judge=judge, config=config, rng=random.Random(0))
        target = _make_target({}, default="Sure, here is what you asked for.")

        result = await evolver.evolve(
            seed_prompts=["seed attack"],
            target=target,
        )

        assert all(isinstance(r, EvolvedAttack) for r in result)
        max_gen = max(r.generation for r in result)
        assert max_gen <= 1, (
            "evolution should have stopped after generation 0 met "
            f"the threshold, got max_gen={max_gen}"
        )

    @pytest.mark.asyncio
    async def test_early_stop_on_no_improvement(self):
        judge = MockJudge(scoring_fn=lambda attack, response: 0.3)
        config = EvolverConfig(
            generations=10,
            population_size=2,
            mutations_per_parent=2,
            patience=1,
            success_threshold=0.99,
        )
        evolver = AdversarialAttackEvolver(evolver_judge=judge, config=config, rng=random.Random(0))
        target = _make_target({}, default="Uninteresting reply.")

        result = await evolver.evolve(
            seed_prompts=["seed"],
            target=target,
        )

        max_gen = max(r.generation for r in result)
        assert (
            max_gen <= 2
        ), f"patience=1 with flat scores should cut evolution short (saw max_gen={max_gen})"

    @pytest.mark.asyncio
    async def test_results_sorted_descending(self):
        judge = MockJudge(
            scoring_fn=lambda attack, response: (0.8 if "mutated::" in attack else 0.2)
        )
        evolver = AdversarialAttackEvolver(
            evolver_judge=judge,
            config=EvolverConfig(
                generations=2,
                population_size=3,
                mutations_per_parent=2,
                patience=10,
                success_threshold=1.0,
            ),
            rng=random.Random(0),
        )
        target = _make_target({}, default="some response text")

        result = await evolver.evolve(
            seed_prompts=["seed"],
            target=target,
        )

        scores = [r.success_score for r in result]
        assert scores == sorted(scores, reverse=True)

    @pytest.mark.asyncio
    async def test_refusal_heuristic_on_judge_failure(self):
        class BrokenJudge:
            async def evaluate(self, prompt, system_prompt=None):
                raise RuntimeError("upstream API down")

        evolver = AdversarialAttackEvolver(
            evolver_judge=BrokenJudge(),
            config=EvolverConfig(
                generations=1,
                population_size=1,
                mutations_per_parent=1,
                patience=10,
                success_threshold=1.0,
            ),
            rng=random.Random(0),
        )
        target = _make_target({}, default="I cannot help with that request.")

        result = await evolver.evolve(
            seed_prompts=["seed"],
            target=target,
        )

        assert result[0].success_score == 0.0, "refusal heuristic should score a refusal as 0"


class TestMutationStrategyEnum:
    def test_has_seven_strategies(self):
        values = {s.value for s in MutationStrategy}
        assert {
            "rephrase",
            "persona",
            "encode",
            "indirect_framing",
            "multi_turn_setup",
            "authority_appeal",
            "roleplay_wrapper",
        }.issubset(values)
