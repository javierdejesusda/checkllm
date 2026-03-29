"""Tests for the consensus judging system."""
from __future__ import annotations

import asyncio
import time

import pytest

from checkllm.consensus import (
    AggregationStrategy,
    ConsensusJudge,
    ConsensusResult,
    JudgeVote,
    consensus,
)
from checkllm.judge import JudgeBackend
from checkllm.models import CheckResult, JudgeResponse


# ---------------------------------------------------------------------------
# Test helpers — lightweight mock judges with controllable scores
# ---------------------------------------------------------------------------


class _SimpleJudge:
    """Minimal JudgeBackend that returns a fixed score."""

    def __init__(self, score: float, reasoning: str = "ok", cost: float = 0.01, delay: float = 0.0) -> None:
        self.score = score
        self.reasoning = reasoning
        self.cost = cost
        self.delay = delay
        self.call_count = 0

    async def evaluate(self, prompt: str, system_prompt: str | None = None) -> JudgeResponse:
        self.call_count += 1
        if self.delay > 0:
            await asyncio.sleep(self.delay)
        return JudgeResponse(score=self.score, reasoning=self.reasoning, cost=self.cost)


def _make_judges(*scores: float) -> list[tuple[str, _SimpleJudge]]:
    """Create named judges with the given scores."""
    return [(f"judge_{i}", _SimpleJudge(score=s)) for i, s in enumerate(scores)]


# ===========================================================================
# AggregationStrategy enum
# ===========================================================================


class TestAggregationStrategy:
    def test_values(self):
        assert AggregationStrategy.MAJORITY == "majority"
        assert AggregationStrategy.UNANIMOUS == "unanimous"
        assert AggregationStrategy.MEAN == "mean"
        assert AggregationStrategy.WEIGHTED == "weighted"
        assert AggregationStrategy.MEDIAN == "median"
        assert AggregationStrategy.MIN == "min"
        assert AggregationStrategy.MAX == "max"

    def test_from_string(self):
        assert AggregationStrategy("mean") is AggregationStrategy.MEAN
        assert AggregationStrategy("majority") is AggregationStrategy.MAJORITY

    def test_invalid_string(self):
        with pytest.raises(ValueError):
            AggregationStrategy("invalid_strategy")


# ===========================================================================
# JudgeVote model
# ===========================================================================


class TestJudgeVote:
    def test_create(self):
        v = JudgeVote(
            judge_name="gpt4",
            score=0.9,
            passed=True,
            reasoning="Good",
            cost=0.01,
            latency_ms=150,
        )
        assert v.judge_name == "gpt4"
        assert v.score == 0.9
        assert v.passed is True

    def test_score_range(self):
        with pytest.raises(Exception):
            JudgeVote(judge_name="x", score=1.5, passed=True, reasoning="", cost=0.0, latency_ms=0)


# ===========================================================================
# ConsensusResult model
# ===========================================================================


class TestConsensusResult:
    def _make_result(self, **overrides) -> ConsensusResult:
        defaults = dict(
            passed=True,
            score=0.85,
            reasoning="summary",
            cost=0.03,
            latency_ms=200,
            metric_name="hallucination",
            strategy="mean",
            votes=[
                JudgeVote(judge_name="a", score=0.9, passed=True, reasoning="r1", cost=0.01, latency_ms=100),
                JudgeVote(judge_name="b", score=0.8, passed=True, reasoning="r2", cost=0.02, latency_ms=200),
            ],
            agreement_ratio=1.0,
        )
        defaults.update(overrides)
        return ConsensusResult(**defaults)

    def test_to_check_result(self):
        cr = self._make_result()
        check = cr.to_check_result()
        assert isinstance(check, CheckResult)
        assert check.passed is True
        assert check.score == 0.85
        assert check.metric_name == "hallucination"
        assert check.cost == 0.03
        assert check.latency_ms == 200

    def test_agreement_ratio_stored(self):
        cr = self._make_result(agreement_ratio=0.75)
        assert cr.agreement_ratio == 0.75


# ===========================================================================
# Aggregation strategies via ConsensusJudge
# ===========================================================================


class TestMajorityStrategy:
    @pytest.mark.asyncio
    async def test_majority_pass(self):
        judges = _make_judges(0.9, 0.85, 0.3)  # 2 pass, 1 fail at 0.8 threshold
        cj = ConsensusJudge(judges, strategy="majority", threshold=0.8)
        result = await cj.evaluate_consensus("test")
        assert result.passed is True

    @pytest.mark.asyncio
    async def test_majority_fail(self):
        judges = _make_judges(0.9, 0.3, 0.2)  # 1 pass, 2 fail
        cj = ConsensusJudge(judges, strategy="majority", threshold=0.8)
        result = await cj.evaluate_consensus("test")
        assert result.passed is False

    @pytest.mark.asyncio
    async def test_majority_even_split_fails(self):
        # 2 judges: 1 pass, 1 fail => 50% is NOT >50%, so fail
        judges = _make_judges(0.9, 0.3)
        cj = ConsensusJudge(judges, strategy="majority", threshold=0.8)
        result = await cj.evaluate_consensus("test")
        assert result.passed is False


class TestUnanimousStrategy:
    @pytest.mark.asyncio
    async def test_unanimous_pass(self):
        judges = _make_judges(0.9, 0.85, 0.95)
        cj = ConsensusJudge(judges, strategy="unanimous", threshold=0.8)
        result = await cj.evaluate_consensus("test")
        assert result.passed is True

    @pytest.mark.asyncio
    async def test_unanimous_fail_one_dissent(self):
        judges = _make_judges(0.9, 0.85, 0.5)  # third judge fails
        cj = ConsensusJudge(judges, strategy="unanimous", threshold=0.8)
        result = await cj.evaluate_consensus("test")
        assert result.passed is False


class TestMeanStrategy:
    @pytest.mark.asyncio
    async def test_mean_pass(self):
        judges = _make_judges(0.9, 0.8, 0.7)  # mean = 0.8
        cj = ConsensusJudge(judges, strategy="mean", threshold=0.8)
        result = await cj.evaluate_consensus("test")
        assert result.passed is True
        assert abs(result.score - 0.8) < 0.01

    @pytest.mark.asyncio
    async def test_mean_fail(self):
        judges = _make_judges(0.9, 0.5, 0.6)  # mean ~ 0.667
        cj = ConsensusJudge(judges, strategy="mean", threshold=0.8)
        result = await cj.evaluate_consensus("test")
        assert result.passed is False


class TestWeightedStrategy:
    @pytest.mark.asyncio
    async def test_weighted_basic(self):
        judges = _make_judges(0.9, 0.5)  # judge_0=0.9, judge_1=0.5
        weights = {"judge_0": 3.0, "judge_1": 1.0}
        cj = ConsensusJudge(judges, strategy="weighted", threshold=0.8, weights=weights)
        result = await cj.evaluate_consensus("test")
        # weighted = (0.9*3 + 0.5*1) / 4 = 3.2/4 = 0.8
        assert result.passed is True
        assert abs(result.score - 0.8) < 0.01

    @pytest.mark.asyncio
    async def test_weighted_default_weight(self):
        # Judges without explicit weights default to 1.0
        judges = _make_judges(0.9, 0.7)
        weights = {"judge_0": 2.0}  # judge_1 gets default weight 1.0
        cj = ConsensusJudge(judges, strategy="weighted", threshold=0.8, weights=weights)
        result = await cj.evaluate_consensus("test")
        # weighted = (0.9*2 + 0.7*1) / 3 = 2.5/3 ~ 0.8333
        assert result.passed is True
        assert result.score > 0.83

    @pytest.mark.asyncio
    async def test_weighted_fail(self):
        judges = _make_judges(0.9, 0.3)
        weights = {"judge_0": 1.0, "judge_1": 3.0}  # heavily weight the low scorer
        cj = ConsensusJudge(judges, strategy="weighted", threshold=0.8, weights=weights)
        result = await cj.evaluate_consensus("test")
        # weighted = (0.9*1 + 0.3*3) / 4 = 1.8/4 = 0.45
        assert result.passed is False
        assert result.score < 0.5


class TestMedianStrategy:
    @pytest.mark.asyncio
    async def test_median_pass(self):
        judges = _make_judges(0.5, 0.85, 0.95)  # median = 0.85
        cj = ConsensusJudge(judges, strategy="median", threshold=0.8)
        result = await cj.evaluate_consensus("test")
        assert result.passed is True
        assert abs(result.score - 0.85) < 0.01

    @pytest.mark.asyncio
    async def test_median_fail(self):
        judges = _make_judges(0.5, 0.6, 0.95)  # median = 0.6
        cj = ConsensusJudge(judges, strategy="median", threshold=0.8)
        result = await cj.evaluate_consensus("test")
        assert result.passed is False


class TestMinStrategy:
    @pytest.mark.asyncio
    async def test_min_pass(self):
        judges = _make_judges(0.85, 0.9, 0.95)  # min = 0.85
        cj = ConsensusJudge(judges, strategy="min", threshold=0.8)
        result = await cj.evaluate_consensus("test")
        assert result.passed is True
        assert abs(result.score - 0.85) < 0.01

    @pytest.mark.asyncio
    async def test_min_fail(self):
        judges = _make_judges(0.5, 0.9, 0.95)  # min = 0.5
        cj = ConsensusJudge(judges, strategy="min", threshold=0.8)
        result = await cj.evaluate_consensus("test")
        assert result.passed is False
        assert abs(result.score - 0.5) < 0.01


class TestMaxStrategy:
    @pytest.mark.asyncio
    async def test_max_pass(self):
        judges = _make_judges(0.3, 0.5, 0.85)  # max = 0.85
        cj = ConsensusJudge(judges, strategy="max", threshold=0.8)
        result = await cj.evaluate_consensus("test")
        assert result.passed is True
        assert abs(result.score - 0.85) < 0.01

    @pytest.mark.asyncio
    async def test_max_fail(self):
        judges = _make_judges(0.3, 0.5, 0.7)  # max = 0.7
        cj = ConsensusJudge(judges, strategy="max", threshold=0.8)
        result = await cj.evaluate_consensus("test")
        assert result.passed is False


# ===========================================================================
# Parallel execution
# ===========================================================================


class TestParallelExecution:
    @pytest.mark.asyncio
    async def test_all_judges_run(self):
        j0 = _SimpleJudge(0.9)
        j1 = _SimpleJudge(0.8)
        j2 = _SimpleJudge(0.7)
        judges: list[tuple[str, _SimpleJudge]] = [("a", j0), ("b", j1), ("c", j2)]
        cj = ConsensusJudge(judges, strategy="mean", threshold=0.5)
        result = await cj.evaluate_consensus("test")
        assert j0.call_count == 1
        assert j1.call_count == 1
        assert j2.call_count == 1
        assert len(result.votes) == 3

    @pytest.mark.asyncio
    async def test_runs_concurrently(self):
        """Verify judges run in parallel, not sequentially.

        Three judges each sleep 0.1s. If parallel, total < 0.25s.
        If sequential, total would be >= 0.3s.
        """
        delay = 0.1
        judges: list[tuple[str, _SimpleJudge]] = [
            ("a", _SimpleJudge(0.9, delay=delay)),
            ("b", _SimpleJudge(0.8, delay=delay)),
            ("c", _SimpleJudge(0.7, delay=delay)),
        ]
        cj = ConsensusJudge(judges, strategy="mean", threshold=0.5)
        start = time.perf_counter()
        await cj.evaluate_consensus("test")
        elapsed = time.perf_counter() - start
        # Should be around 0.1s if parallel, not 0.3s
        assert elapsed < 0.25, f"Took {elapsed:.2f}s — judges likely ran sequentially"


# ===========================================================================
# Agreement ratio
# ===========================================================================


class TestAgreementRatio:
    @pytest.mark.asyncio
    async def test_full_agreement_all_pass(self):
        judges = _make_judges(0.9, 0.85, 0.95)
        cj = ConsensusJudge(judges, strategy="mean", threshold=0.8)
        result = await cj.evaluate_consensus("test")
        assert result.agreement_ratio == 1.0

    @pytest.mark.asyncio
    async def test_full_agreement_all_fail(self):
        judges = _make_judges(0.3, 0.4, 0.2)
        cj = ConsensusJudge(judges, strategy="mean", threshold=0.8)
        result = await cj.evaluate_consensus("test")
        assert result.agreement_ratio == 1.0

    @pytest.mark.asyncio
    async def test_partial_agreement(self):
        judges = _make_judges(0.9, 0.3, 0.85)  # 2/3 pass
        cj = ConsensusJudge(judges, strategy="mean", threshold=0.8)
        result = await cj.evaluate_consensus("test")
        assert abs(result.agreement_ratio - 2 / 3) < 0.01

    @pytest.mark.asyncio
    async def test_even_split(self):
        judges = _make_judges(0.9, 0.3)  # 1 pass, 1 fail
        cj = ConsensusJudge(judges, strategy="mean", threshold=0.8)
        result = await cj.evaluate_consensus("test")
        assert result.agreement_ratio == 0.5


# ===========================================================================
# Cost and latency aggregation
# ===========================================================================


class TestCostAndLatency:
    @pytest.mark.asyncio
    async def test_total_cost(self):
        judges: list[tuple[str, _SimpleJudge]] = [
            ("a", _SimpleJudge(0.9, cost=0.01)),
            ("b", _SimpleJudge(0.8, cost=0.02)),
            ("c", _SimpleJudge(0.7, cost=0.03)),
        ]
        cj = ConsensusJudge(judges, strategy="mean", threshold=0.5)
        result = await cj.evaluate_consensus("test")
        assert abs(result.cost - 0.06) < 0.001

    @pytest.mark.asyncio
    async def test_latency_is_max(self):
        """Latency should be max across judges (parallel execution)."""
        judges: list[tuple[str, _SimpleJudge]] = [
            ("a", _SimpleJudge(0.9, delay=0.01)),
            ("b", _SimpleJudge(0.8, delay=0.05)),
            ("c", _SimpleJudge(0.7, delay=0.01)),
        ]
        cj = ConsensusJudge(judges, strategy="mean", threshold=0.5)
        result = await cj.evaluate_consensus("test")
        # Max latency should be from judge "b"; all latencies > 0
        assert result.latency_ms > 0
        assert result.latency_ms == max(v.latency_ms for v in result.votes)

    @pytest.mark.asyncio
    async def test_total_cost_tracked_on_judge(self):
        judges = _make_judges(0.9, 0.8)
        cj = ConsensusJudge(judges, strategy="mean", threshold=0.5)
        assert cj.total_cost == 0.0
        await cj.evaluate_consensus("test")
        # Each _SimpleJudge has cost=0.01 by default
        assert cj.total_cost > 0


# ===========================================================================
# ConsensusJudge as JudgeBackend protocol
# ===========================================================================


class TestProtocolCompliance:
    def test_isinstance_check(self):
        judges = _make_judges(0.9)
        cj = ConsensusJudge(judges, strategy="mean", threshold=0.8)
        assert isinstance(cj, JudgeBackend)

    @pytest.mark.asyncio
    async def test_evaluate_returns_judge_response(self):
        judges = _make_judges(0.9, 0.8)
        cj = ConsensusJudge(judges, strategy="mean", threshold=0.5)
        response = await cj.evaluate("test prompt")
        assert isinstance(response, JudgeResponse)
        assert 0.0 <= response.score <= 1.0
        assert isinstance(response.reasoning, str)

    @pytest.mark.asyncio
    async def test_nested_consensus(self):
        """A ConsensusJudge can be used as a judge inside another ConsensusJudge."""
        inner_judges = _make_judges(0.9, 0.85)
        inner = ConsensusJudge(inner_judges, strategy="mean", threshold=0.5)

        outer_judges: list[tuple[str, JudgeBackend]] = [
            ("inner_consensus", inner),
            ("single", _SimpleJudge(0.7)),
        ]
        outer = ConsensusJudge(outer_judges, strategy="mean", threshold=0.5)
        result = await outer.evaluate_consensus("test")
        assert len(result.votes) == 2


# ===========================================================================
# Edge cases
# ===========================================================================


class TestEdgeCases:
    @pytest.mark.asyncio
    async def test_single_judge(self):
        judges = _make_judges(0.9)
        cj = ConsensusJudge(judges, strategy="mean", threshold=0.8)
        result = await cj.evaluate_consensus("test")
        assert result.passed is True
        assert result.score == 0.9
        assert len(result.votes) == 1
        assert result.agreement_ratio == 1.0

    @pytest.mark.asyncio
    async def test_all_pass(self):
        judges = _make_judges(0.85, 0.9, 0.95)
        for strat in AggregationStrategy:
            cj = ConsensusJudge(judges, strategy=strat, threshold=0.8)
            result = await cj.evaluate_consensus("test")
            assert result.passed is True, f"Failed for strategy {strat.value}"

    @pytest.mark.asyncio
    async def test_all_fail(self):
        judges = _make_judges(0.1, 0.2, 0.3)
        for strat in AggregationStrategy:
            cj = ConsensusJudge(judges, strategy=strat, threshold=0.8)
            result = await cj.evaluate_consensus("test")
            assert result.passed is False, f"Passed unexpectedly for strategy {strat.value}"

    @pytest.mark.asyncio
    async def test_split_votes_majority(self):
        judges = _make_judges(0.9, 0.3, 0.85, 0.2, 0.95)  # 3 pass, 2 fail
        cj = ConsensusJudge(judges, strategy="majority", threshold=0.8)
        result = await cj.evaluate_consensus("test")
        assert result.passed is True
        assert result.agreement_ratio == 0.6

    def test_no_judges_raises(self):
        with pytest.raises(ValueError, match="At least one judge"):
            ConsensusJudge([], strategy="mean")

    @pytest.mark.asyncio
    async def test_strategy_string_accepted(self):
        judges = _make_judges(0.9)
        cj = ConsensusJudge(judges, strategy="median", threshold=0.5)
        result = await cj.evaluate_consensus("test")
        assert result.strategy == "median"

    def test_repr(self):
        judges = _make_judges(0.9, 0.8)
        cj = ConsensusJudge(judges, strategy="mean", threshold=0.8)
        r = repr(cj)
        assert "ConsensusJudge" in r
        assert "mean" in r

    @pytest.mark.asyncio
    async def test_reasoning_includes_all_judges(self):
        judges = _make_judges(0.9, 0.3)
        cj = ConsensusJudge(judges, strategy="mean", threshold=0.8)
        result = await cj.evaluate_consensus("test")
        assert "judge_0" in result.reasoning
        assert "judge_1" in result.reasoning
        assert "PASS" in result.reasoning
        assert "FAIL" in result.reasoning


# ===========================================================================
# to_check_result conversion
# ===========================================================================


class TestToCheckResult:
    @pytest.mark.asyncio
    async def test_conversion_preserves_fields(self):
        judges = _make_judges(0.9, 0.8)
        cj = ConsensusJudge(judges, strategy="mean", threshold=0.8)
        consensus_result = await cj.evaluate_consensus("test", metric_name="relevance")
        check = consensus_result.to_check_result()
        assert check.passed == consensus_result.passed
        assert check.score == consensus_result.score
        assert check.cost == consensus_result.cost
        assert check.latency_ms == consensus_result.latency_ms
        assert check.metric_name == "relevance"

    @pytest.mark.asyncio
    async def test_conversion_is_valid_check_result(self):
        judges = _make_judges(0.5)
        cj = ConsensusJudge(judges, strategy="mean", threshold=0.8)
        consensus_result = await cj.evaluate_consensus("test")
        check = consensus_result.to_check_result()
        # Verify it's a proper Pydantic model that validates
        assert isinstance(check, CheckResult)
        assert 0.0 <= check.score <= 1.0


# ===========================================================================
# consensus() convenience function
# ===========================================================================


class TestConsensusFunction:
    @pytest.mark.asyncio
    async def test_basic_usage(self):
        from checkllm.testing import MockJudge

        j1 = MockJudge(default_score=0.9)
        j2 = MockJudge(default_score=0.85)
        result = await consensus(
            output="Paris is the capital of France.",
            metric_name="hallucination",
            judges=[("j1", j1), ("j2", j2)],
            strategy="mean",
            threshold=0.8,
            context="France's capital is Paris.",
        )
        assert isinstance(result, ConsensusResult)
        assert result.passed is True
        assert result.strategy == "mean"
        assert result.metric_name == "hallucination"
        assert len(result.votes) == 2

    @pytest.mark.asyncio
    async def test_with_simple_judges(self):
        judges = _make_judges(0.9, 0.7)
        result = await consensus(
            output="test output",
            metric_name="toxicity",
            judges=judges,
            strategy="mean",
            threshold=0.7,
        )
        assert isinstance(result, ConsensusResult)
        assert result.metric_name == "toxicity"

    @pytest.mark.asyncio
    async def test_unknown_metric_raises(self):
        judges = _make_judges(0.9)
        with pytest.raises(ValueError, match="Unknown metric"):
            await consensus(
                output="test",
                metric_name="nonexistent_metric",
                judges=judges,
            )

    @pytest.mark.asyncio
    async def test_empty_judges_raises(self):
        with pytest.raises(ValueError, match="At least one judge"):
            await consensus(
                output="test",
                metric_name="toxicity",
                judges=[],
            )

    @pytest.mark.asyncio
    async def test_weighted_strategy(self):
        j1 = _SimpleJudge(0.9)
        j2 = _SimpleJudge(0.3)
        result = await consensus(
            output="test output",
            metric_name="fluency",
            judges=[("strong", j1), ("weak", j2)],
            strategy="weighted",
            threshold=0.7,
            weights={"strong": 3.0, "weak": 1.0},
        )
        # weighted = (0.9*3 + 0.3*1) / 4 = 3.0/4 = 0.75 (scores come from metric,
        # which wraps the judge; the mock judge returns the fixed score)
        assert isinstance(result, ConsensusResult)
        assert result.strategy == "weighted"
