"""Tests for checkllm.arena — A/B testing arena with statistical significance."""

from __future__ import annotations

from unittest.mock import AsyncMock

import pytest

from checkllm.arena import Arena, ArenaCandidate, ArenaResult
from checkllm.models import JudgeResponse


class TestArenaCandidate:
    def test_defaults(self):
        candidate = ArenaCandidate(name="v1", prompt="You are helpful.")
        assert candidate.scores == []
        assert candidate.avg_score == 0.0
        assert candidate.cost == 0.0

    def test_with_scores(self):
        candidate = ArenaCandidate(
            name="v2",
            prompt="Be concise.",
            scores=[0.8, 0.9, 0.7],
            avg_score=0.8,
            cost=0.001,
        )
        assert len(candidate.scores) == 3
        assert candidate.avg_score == 0.8


class TestArenaResult:
    def _make_result(
        self,
        a_scores: list[float],
        b_scores: list[float],
        winner: str = "tie",
        p_value: float = 1.0,
        significant: bool = False,
    ) -> ArenaResult:
        a = ArenaCandidate(
            name="a",
            prompt="prompt-a",
            scores=a_scores,
            avg_score=sum(a_scores) / len(a_scores) if a_scores else 0.0,
        )
        b = ArenaCandidate(
            name="b",
            prompt="prompt-b",
            scores=b_scores,
            avg_score=sum(b_scores) / len(b_scores) if b_scores else 0.0,
        )
        return ArenaResult(
            candidate_a=a,
            candidate_b=b,
            winner=winner,
            p_value=p_value,
            statistically_significant=significant,
            confidence_level=0.95,
            effect_size=0.2,
            num_trials=len(a_scores),
        )

    def test_summary_contains_names(self):
        result = self._make_result([0.8, 0.9], [0.6, 0.7])
        text = result.summary()
        assert "a:" in text
        assert "b:" in text

    def test_summary_shows_winner(self):
        result = self._make_result([0.8, 0.9], [0.6, 0.7], winner="a")
        assert "Winner: a" in result.summary()

    def test_summary_tie(self):
        result = self._make_result([0.75, 0.75], [0.75, 0.75], winner="tie")
        assert "Winner: tie" in result.summary()

    def test_summary_significant(self):
        result = self._make_result(
            [0.9, 0.95], [0.4, 0.45], winner="a", p_value=0.001, significant=True
        )
        assert "(significant)" in result.summary()

    def test_summary_not_significant(self):
        result = self._make_result([0.7, 0.8], [0.65, 0.75], p_value=0.5)
        assert "(not significant)" in result.summary()

    def test_summary_shows_p_value(self):
        result = self._make_result([0.8], [0.6], p_value=0.0423)
        assert "0.0423" in result.summary()

    def test_summary_shows_effect_size(self):
        result = self._make_result([0.8], [0.6])
        assert "Effect size:" in result.summary()

    def test_defaults(self):
        a = ArenaCandidate(name="a", prompt="p")
        b = ArenaCandidate(name="b", prompt="p")
        result = ArenaResult(candidate_a=a, candidate_b=b)
        assert result.winner == ""
        assert result.p_value == 1.0
        assert result.statistically_significant is False
        assert result.confidence_level == 0.95
        assert result.num_trials == 0


class TestArena:
    @pytest.fixture
    def mock_judge(self):
        judge = AsyncMock()
        return judge

    def _make_arena(self, mock_judge) -> Arena:
        return Arena(judge=mock_judge)

    def test_compare_returns_arena_result(self, mock_judge):
        mock_judge.evaluate.return_value = JudgeResponse(
            score=0.8, reasoning="good", raw_output=None
        )
        arena = self._make_arena(mock_judge)
        result = arena.compare(
            candidate_a=("v1", "Be helpful."),
            candidate_b=("v2", "Be concise."),
            test_inputs=["What is Python?"],
        )
        assert isinstance(result, ArenaResult)
        assert result.num_trials == 1

    def test_compare_calls_judge_twice_per_input(self, mock_judge):
        mock_judge.evaluate.return_value = JudgeResponse(
            score=0.75, reasoning="ok", raw_output=None
        )
        arena = self._make_arena(mock_judge)
        arena.compare(
            candidate_a=("a", "prompt a"),
            candidate_b=("b", "prompt b"),
            test_inputs=["input1", "input2"],
        )
        assert mock_judge.evaluate.call_count == 4  # 2 inputs * 2 candidates

    def test_candidate_names_preserved(self, mock_judge):
        mock_judge.evaluate.return_value = JudgeResponse(score=0.7, reasoning="ok", raw_output=None)
        arena = self._make_arena(mock_judge)
        result = arena.compare(
            candidate_a=("alpha", "prompt alpha"),
            candidate_b=("beta", "prompt beta"),
            test_inputs=["test"],
        )
        assert result.candidate_a.name == "alpha"
        assert result.candidate_b.name == "beta"

    def test_scores_recorded_per_candidate(self, mock_judge):
        call_count = 0

        async def side_effect(prompt: str) -> JudgeResponse:
            nonlocal call_count
            call_count += 1
            score = 0.9 if call_count % 2 == 1 else 0.5
            return JudgeResponse(score=score, reasoning="ok", raw_output=None)

        mock_judge.evaluate.side_effect = side_effect
        arena = self._make_arena(mock_judge)
        result = arena.compare(
            candidate_a=("high", "great prompt"),
            candidate_b=("low", "poor prompt"),
            test_inputs=["q1", "q2"],
        )
        assert len(result.candidate_a.scores) == 2
        assert len(result.candidate_b.scores) == 2

    def test_winner_is_tie_when_not_significant(self, mock_judge):
        mock_judge.evaluate.return_value = JudgeResponse(
            score=0.75, reasoning="ok", raw_output=None
        )
        arena = self._make_arena(mock_judge)
        result = arena.compare(
            candidate_a=("a", "p"),
            candidate_b=("b", "p"),
            test_inputs=["input"],
        )
        # With only 1 trial, t-test cannot be significant
        assert result.winner == "tie"
        assert result.statistically_significant is False

    def test_significant_winner_with_divergent_scores(self, mock_judge):
        """With clearly different score distributions and enough trials, a winner is declared."""
        scores_a = [0.95, 0.93, 0.97, 0.94, 0.96, 0.95, 0.92, 0.98, 0.94, 0.96]
        scores_b = [0.10, 0.12, 0.09, 0.11, 0.08, 0.10, 0.13, 0.09, 0.11, 0.10]
        all_scores = []
        for a, b in zip(scores_a, scores_b):
            all_scores.append(a)
            all_scores.append(b)

        responses = iter(
            JudgeResponse(score=s, reasoning="ok", raw_output=None) for s in all_scores
        )
        mock_judge.evaluate.side_effect = lambda prompt: next(responses)

        arena = self._make_arena(mock_judge)
        result = arena.compare(
            candidate_a=("high", "great"),
            candidate_b=("low", "poor"),
            test_inputs=[f"input{i}" for i in range(10)],
        )
        assert result.statistically_significant is True
        assert result.winner == "high"

    def test_avg_score_computed(self, mock_judge):
        mock_judge.evaluate.return_value = JudgeResponse(score=0.6, reasoning="ok", raw_output=None)
        arena = self._make_arena(mock_judge)
        result = arena.compare(
            candidate_a=("a", "p"),
            candidate_b=("b", "p"),
            test_inputs=["i1", "i2", "i3"],
        )
        assert abs(result.candidate_a.avg_score - 0.6) < 1e-6
        assert abs(result.candidate_b.avg_score - 0.6) < 1e-6

    @pytest.mark.asyncio
    async def test_acompare_returns_arena_result(self, mock_judge):
        mock_judge.evaluate.return_value = JudgeResponse(
            score=0.8, reasoning="good", raw_output=None
        )
        arena = self._make_arena(mock_judge)
        result = await arena.acompare(
            candidate_a=("v1", "Be helpful."),
            candidate_b=("v2", "Be concise."),
            test_inputs=["What is Python?"],
        )
        assert isinstance(result, ArenaResult)
        assert result.num_trials == 1
