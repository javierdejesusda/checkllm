"""Tests for checkllm.strategies -- multi-turn attack strategies."""

from __future__ import annotations


import pytest

from checkllm.models import JudgeResponse
from checkllm.strategies import (
    ConversationTurn,
    CrescendoStrategy,
    GOATStrategy,
    HydraStrategy,
    MultiTurnAttackResult,
    MultiTurnMischief,
    SocialEngineeringTechnique,
)


class _StubJudge:
    """Minimal judge stub that returns canned responses."""

    def __init__(
        self,
        score: float = 0.3,
        reasoning: str = "test reasoning",
        raw_output: str | None = None,
    ) -> None:
        self._score = score
        self._reasoning = reasoning
        self._raw_output = raw_output
        self.call_count = 0

    async def evaluate(
        self,
        prompt: str,
        system_prompt: str | None = None,
    ) -> JudgeResponse:
        self.call_count += 1
        return JudgeResponse(
            score=self._score,
            reasoning=self._reasoning,
            raw_output=self._raw_output,
            cost=0.001,
        )


class _SuccessAfterNJudge:
    """Judge that returns success (score >= 0.7) after N calls."""

    def __init__(self, succeed_after: int = 4) -> None:
        self._succeed_after = succeed_after
        self.call_count = 0

    async def evaluate(
        self,
        prompt: str,
        system_prompt: str | None = None,
    ) -> JudgeResponse:
        self.call_count += 1
        if self.call_count >= self._succeed_after:
            return JudgeResponse(
                score=0.9,
                reasoning="objective achieved",
                raw_output=None,
                cost=0.001,
            )
        return JudgeResponse(
            score=0.2,
            reasoning="not yet",
            raw_output=None,
            cost=0.001,
        )


async def _mock_target(prompt: str) -> str:
    """Simple mock target that echoes the prompt."""
    return f"I received: {prompt}"


async def _resistant_target(prompt: str) -> str:
    """Mock target that always refuses."""
    return "I cannot help with that request."


class TestConversationTurn:
    def test_create_turn(self):
        turn = ConversationTurn(
            role="attacker",
            content="hello",
            turn_number=1,
        )
        assert turn.role == "attacker"
        assert turn.content == "hello"
        assert turn.turn_number == 1

    def test_target_turn(self):
        turn = ConversationTurn(
            role="target",
            content="response",
            turn_number=2,
        )
        assert turn.role == "target"
        assert turn.turn_number == 2


class TestMultiTurnAttackResult:
    def test_successful_result(self):
        turns = [
            ConversationTurn(role="attacker", content="q1", turn_number=1),
            ConversationTurn(role="target", content="a1", turn_number=1),
        ]
        result = MultiTurnAttackResult(
            strategy="goat",
            vulnerability_type="prompt_injection",
            objective="test objective",
            success=True,
            confidence=0.9,
            turns=turns,
            total_turns=1,
            successful_turn=1,
            cost=0.01,
            latency_ms=100,
        )
        assert result.success is True
        assert result.confidence == 0.9
        assert result.successful_turn == 1
        assert len(result.turns) == 2

    def test_failed_result(self):
        result = MultiTurnAttackResult(
            strategy="hydra",
            vulnerability_type="jailbreak",
            objective="test",
            success=False,
            confidence=0.2,
            total_turns=5,
        )
        assert result.success is False
        assert result.successful_turn is None
        assert result.turns == []

    def test_defaults(self):
        result = MultiTurnAttackResult(
            strategy="test",
            vulnerability_type="test",
            objective="test",
            success=False,
            confidence=0.0,
        )
        assert result.cost == 0.0
        assert result.latency_ms == 0
        assert result.total_turns == 0


class TestGOATStrategy:
    @pytest.mark.asyncio
    async def test_stops_after_max_turns(self):
        judge = _StubJudge(score=0.1)
        goat = GOATStrategy(judge=judge, max_turns=3)
        result = await goat.execute(
            _resistant_target,
            "extract system prompt",
            "prompt_extraction",
        )
        assert result.success is False
        assert result.total_turns == 3
        assert result.strategy == "goat"
        assert len(result.turns) == 6

    @pytest.mark.asyncio
    async def test_succeeds_when_objective_met(self):
        judge = _SuccessAfterNJudge(succeed_after=3)
        goat = GOATStrategy(judge=judge, max_turns=10)
        result = await goat.execute(
            _mock_target,
            "get forbidden info",
            "data_extraction",
        )
        assert result.success is True
        assert result.total_turns <= 10
        assert result.successful_turn is not None

    @pytest.mark.asyncio
    async def test_records_conversation_history(self):
        judge = _StubJudge(score=0.1)
        goat = GOATStrategy(judge=judge, max_turns=2)
        result = await goat.execute(
            _mock_target,
            "objective",
            "prompt_injection",
        )
        assert len(result.turns) == 4
        assert result.turns[0].role == "attacker"
        assert result.turns[1].role == "target"
        assert result.turns[2].role == "attacker"
        assert result.turns[3].role == "target"

    @pytest.mark.asyncio
    async def test_accumulates_cost(self):
        judge = _StubJudge(score=0.1)
        goat = GOATStrategy(judge=judge, max_turns=2)
        result = await goat.execute(
            _mock_target,
            "objective",
            "prompt_injection",
        )
        assert result.cost > 0

    @pytest.mark.asyncio
    async def test_vulnerability_type_preserved(self):
        judge = _StubJudge(score=0.1)
        goat = GOATStrategy(judge=judge, max_turns=1)
        result = await goat.execute(
            _mock_target,
            "obj",
            "custom_vulnerability",
        )
        assert result.vulnerability_type == "custom_vulnerability"


class TestHydraStrategy:
    @pytest.mark.asyncio
    async def test_branches_created(self):
        judge = _StubJudge(score=0.1)
        hydra = HydraStrategy(judge=judge, branches=3, depth=1)
        result = await hydra.execute(
            _mock_target,
            "extract data",
            "data_extraction",
        )
        assert result.success is False
        assert result.strategy == "hydra"
        assert len(result.turns) >= 2

    @pytest.mark.asyncio
    async def test_succeeds_on_good_branch(self):
        judge = _SuccessAfterNJudge(succeed_after=2)
        hydra = HydraStrategy(judge=judge, branches=2, depth=2)
        result = await hydra.execute(
            _mock_target,
            "get info",
            "prompt_injection",
        )
        assert result.success is True
        assert result.successful_turn is not None

    @pytest.mark.asyncio
    async def test_follows_up_on_top_branches(self):
        judge = _StubJudge(score=0.3)
        hydra = HydraStrategy(judge=judge, branches=3, depth=2)
        result = await hydra.execute(
            _mock_target,
            "objective",
            "jailbreak",
        )
        assert result.total_turns == 2

    @pytest.mark.asyncio
    async def test_returns_best_branch_on_failure(self):
        judge = _StubJudge(score=0.4)
        hydra = HydraStrategy(judge=judge, branches=2, depth=1)
        result = await hydra.execute(
            _mock_target,
            "objective",
            "jailbreak",
        )
        assert result.success is False
        assert result.confidence == 0.4


class TestCrescendoStrategy:
    @pytest.mark.asyncio
    async def test_escalates_gradually(self):
        judge = _StubJudge(score=0.2)
        crescendo = CrescendoStrategy(judge=judge, escalation_steps=5)
        result = await crescendo.execute(
            _mock_target,
            "reveal secrets",
            "data_extraction",
        )
        assert result.success is False
        assert result.strategy == "crescendo"
        assert result.total_turns == 5
        assert len(result.turns) == 10

    @pytest.mark.asyncio
    async def test_succeeds_during_escalation(self):
        judge = _SuccessAfterNJudge(succeed_after=5)
        crescendo = CrescendoStrategy(judge=judge, escalation_steps=5)
        result = await crescendo.execute(
            _mock_target,
            "get system prompt",
            "prompt_extraction",
        )
        assert result.success is True
        assert result.successful_turn is not None
        assert result.successful_turn >= 3

    @pytest.mark.asyncio
    async def test_no_eval_for_early_turns(self):
        judge = _StubJudge(score=0.1)
        crescendo = CrescendoStrategy(judge=judge, escalation_steps=5)
        result = await crescendo.execute(
            _mock_target,
            "objective",
            "jailbreak",
        )
        assert result.success is False
        assert result.total_turns == 5

    @pytest.mark.asyncio
    async def test_single_step(self):
        judge = _StubJudge(score=0.1)
        crescendo = CrescendoStrategy(judge=judge, escalation_steps=1)
        result = await crescendo.execute(
            _mock_target,
            "objective",
            "jailbreak",
        )
        assert result.total_turns == 1
        assert len(result.turns) == 2


class TestMultiTurnMischief:
    @pytest.mark.asyncio
    async def test_uses_multiple_techniques(self):
        judge = _StubJudge(score=0.1)
        mischief = MultiTurnMischief(judge=judge, max_turns=6)
        result = await mischief.execute(
            _mock_target,
            "bypass safety",
            "jailbreak",
        )
        assert result.success is False
        assert result.strategy == "multi_turn_mischief"
        assert result.total_turns == 6
        assert len(result.turns) == 12

    @pytest.mark.asyncio
    async def test_succeeds_on_social_engineering(self):
        judge = _SuccessAfterNJudge(succeed_after=4)
        mischief = MultiTurnMischief(judge=judge, max_turns=10)
        result = await mischief.execute(
            _mock_target,
            "extract data",
            "data_extraction",
        )
        assert result.success is True
        assert result.successful_turn is not None

    @pytest.mark.asyncio
    async def test_stops_at_max_turns(self):
        judge = _StubJudge(score=0.1)
        mischief = MultiTurnMischief(judge=judge, max_turns=3)
        result = await mischief.execute(
            _resistant_target,
            "objective",
            "prompt_injection",
        )
        assert result.success is False
        assert result.total_turns == 3
        assert len(result.turns) == 6

    @pytest.mark.asyncio
    async def test_records_cost_and_latency(self):
        judge = _StubJudge(score=0.1)
        mischief = MultiTurnMischief(judge=judge, max_turns=2)
        result = await mischief.execute(
            _mock_target,
            "objective",
            "prompt_injection",
        )
        assert result.cost > 0
        assert result.latency_ms >= 0


class TestSocialEngineeringTechnique:
    def test_enum_values(self):
        assert SocialEngineeringTechnique.FLATTERY.value == "flattery"
        assert SocialEngineeringTechnique.AUTHORITY_APPEAL.value == "authority_appeal"
        assert SocialEngineeringTechnique.EMOTIONAL_MANIPULATION.value == "emotional_manipulation"
        assert SocialEngineeringTechnique.CONTEXT_SWITCHING.value == "context_switching"
        assert SocialEngineeringTechnique.URGENCY.value == "urgency"
        assert SocialEngineeringTechnique.RECIPROCITY.value == "reciprocity"

    def test_enum_count(self):
        assert len(SocialEngineeringTechnique) == 6
