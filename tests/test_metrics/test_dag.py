"""Tests for checkllm.metrics.dag — DAG-based composite evaluation metric."""

from __future__ import annotations

from unittest.mock import AsyncMock

import pytest

from checkllm.metrics.dag import DAGEvalResult, DAGMetric, DAGNode
from checkllm.models import CheckResult, JudgeResponse


class TestDAGNode:
    def test_defaults(self):
        node = DAGNode(
            name="safety",
            prompt_template="Is this safe? {output}",
        )
        assert node.threshold == 0.5
        assert node.children_on_pass == []
        assert node.children_on_fail == []
        assert node.weight == 1.0

    def test_custom_values(self):
        node = DAGNode(
            name="quality",
            prompt_template="Rate quality: {output}",
            threshold=0.8,
            children_on_pass=["detail"],
            children_on_fail=["fallback"],
            weight=2.0,
        )
        assert node.threshold == 0.8
        assert node.children_on_pass == ["detail"]
        assert node.children_on_fail == ["fallback"]
        assert node.weight == 2.0


class TestDAGEvalResult:
    def test_fields(self):
        result = DAGEvalResult(
            node_name="safety",
            score=0.9,
            passed=True,
            reasoning="Output is safe",
        )
        assert result.node_name == "safety"
        assert result.score == 0.9
        assert result.passed is True
        assert result.reasoning == "Output is safe"


class TestDAGMetric:
    @pytest.fixture
    def mock_judge(self):
        judge = AsyncMock()
        return judge

    @pytest.mark.asyncio
    async def test_single_node_passes(self, mock_judge):
        mock_judge.evaluate.return_value = JudgeResponse(
            score=0.9, reasoning="Safe output", raw_output=None
        )
        dag = DAGMetric(
            judge=mock_judge,
            nodes=[
                DAGNode(name="safety", prompt_template="Is this safe? {output}", threshold=0.5),
            ],
            root="safety",
        )
        result = await dag.evaluate(output="Hello world")
        assert result.passed is True
        assert result.score == 0.9
        assert result.metric_name == "dag"
        assert "safety" in result.reasoning

    @pytest.mark.asyncio
    async def test_single_node_fails(self, mock_judge):
        mock_judge.evaluate.return_value = JudgeResponse(
            score=0.3, reasoning="Unsafe", raw_output=None
        )
        dag = DAGMetric(
            judge=mock_judge,
            nodes=[
                DAGNode(name="safety", prompt_template="Is this safe? {output}", threshold=0.5),
            ],
            root="safety",
            threshold=0.5,
        )
        result = await dag.evaluate(output="Bad content")
        assert result.passed is False
        assert result.score == 0.3

    @pytest.mark.asyncio
    async def test_linear_chain_visits_all_nodes(self, mock_judge):
        mock_judge.evaluate.return_value = JudgeResponse(
            score=0.8, reasoning="ok", raw_output=None
        )
        dag = DAGMetric(
            judge=mock_judge,
            nodes=[
                DAGNode(
                    name="safety",
                    prompt_template="Safe? {output}",
                    threshold=0.5,
                    children_on_pass=["quality"],
                ),
                DAGNode(
                    name="quality",
                    prompt_template="Quality? {output}",
                    threshold=0.5,
                    children_on_pass=["relevance"],
                ),
                DAGNode(
                    name="relevance",
                    prompt_template="Relevant? {output}",
                    threshold=0.5,
                ),
            ],
            root="safety",
        )
        result = await dag.evaluate(output="Good answer")
        assert mock_judge.evaluate.call_count == 3
        assert "safety" in result.reasoning
        assert "quality" in result.reasoning
        assert "relevance" in result.reasoning

    @pytest.mark.asyncio
    async def test_branching_on_pass(self, mock_judge):
        call_index = 0
        responses = [
            JudgeResponse(score=0.9, reasoning="safe", raw_output=None),  # safety: pass
            JudgeResponse(score=0.7, reasoning="good quality", raw_output=None),  # quality
        ]

        async def side_effect(prompt: str) -> JudgeResponse:
            nonlocal call_index
            resp = responses[call_index]
            call_index += 1
            return resp

        mock_judge.evaluate.side_effect = side_effect
        dag = DAGMetric(
            judge=mock_judge,
            nodes=[
                DAGNode(
                    name="safety",
                    prompt_template="Safe? {output}",
                    threshold=0.5,
                    children_on_pass=["quality"],
                    children_on_fail=["rejection"],
                ),
                DAGNode(name="quality", prompt_template="Quality? {output}", threshold=0.5),
                DAGNode(name="rejection", prompt_template="Rejection? {output}", threshold=0.5),
            ],
            root="safety",
        )
        result = await dag.evaluate(output="Good output")
        assert mock_judge.evaluate.call_count == 2
        assert "safety" in result.reasoning
        assert "quality" in result.reasoning
        assert "rejection" not in result.reasoning

    @pytest.mark.asyncio
    async def test_branching_on_fail(self, mock_judge):
        call_index = 0
        responses = [
            JudgeResponse(score=0.2, reasoning="unsafe", raw_output=None),  # safety: fail
            JudgeResponse(score=0.5, reasoning="rejection handled", raw_output=None),
        ]

        async def side_effect(prompt: str) -> JudgeResponse:
            nonlocal call_index
            resp = responses[call_index]
            call_index += 1
            return resp

        mock_judge.evaluate.side_effect = side_effect
        dag = DAGMetric(
            judge=mock_judge,
            nodes=[
                DAGNode(
                    name="safety",
                    prompt_template="Safe? {output}",
                    threshold=0.5,
                    children_on_pass=["quality"],
                    children_on_fail=["rejection"],
                ),
                DAGNode(name="quality", prompt_template="Quality? {output}", threshold=0.5),
                DAGNode(name="rejection", prompt_template="Rejection? {output}", threshold=0.5),
            ],
            root="safety",
        )
        result = await dag.evaluate(output="Bad output")
        assert mock_judge.evaluate.call_count == 2
        assert "safety" in result.reasoning
        assert "rejection" in result.reasoning
        assert "quality" not in result.reasoning

    @pytest.mark.asyncio
    async def test_weighted_score(self, mock_judge):
        call_index = 0
        responses = [
            JudgeResponse(score=1.0, reasoning="perfect", raw_output=None),  # weight=2
            JudgeResponse(score=0.0, reasoning="terrible", raw_output=None),  # weight=1
        ]

        async def side_effect(prompt: str) -> JudgeResponse:
            nonlocal call_index
            resp = responses[call_index]
            call_index += 1
            return resp

        mock_judge.evaluate.side_effect = side_effect
        dag = DAGMetric(
            judge=mock_judge,
            nodes=[
                DAGNode(
                    name="heavy",
                    prompt_template="Heavy? {output}",
                    threshold=0.5,
                    children_on_pass=["light"],
                    weight=2.0,
                ),
                DAGNode(
                    name="light",
                    prompt_template="Light? {output}",
                    threshold=0.5,
                    weight=1.0,
                ),
            ],
            root="heavy",
        )
        result = await dag.evaluate(output="test")
        # Weighted: (1.0 * 2 + 0.0 * 1) / 3 = 2/3
        assert abs(result.score - 2.0 / 3.0) < 1e-6

    @pytest.mark.asyncio
    async def test_no_cycles_visited_twice(self, mock_judge):
        """A node already visited should not be evaluated again."""
        mock_judge.evaluate.return_value = JudgeResponse(
            score=0.8, reasoning="ok", raw_output=None
        )
        dag = DAGMetric(
            judge=mock_judge,
            nodes=[
                DAGNode(
                    name="a",
                    prompt_template="A? {output}",
                    threshold=0.5,
                    children_on_pass=["b"],
                ),
                DAGNode(
                    name="b",
                    prompt_template="B? {output}",
                    threshold=0.5,
                    children_on_pass=["a"],  # would form a cycle — should be skipped
                ),
            ],
            root="a",
        )
        result = await dag.evaluate(output="test")
        # Should only evaluate a and b once each
        assert mock_judge.evaluate.call_count == 2

    @pytest.mark.asyncio
    async def test_missing_root_returns_no_evaluated(self, mock_judge):
        dag = DAGMetric(
            judge=mock_judge,
            nodes=[
                DAGNode(name="quality", prompt_template="Quality? {output}", threshold=0.5),
            ],
            root="nonexistent",
        )
        result = await dag.evaluate(output="test")
        assert result.passed is False
        assert result.score == 0.0
        assert "No nodes evaluated" in result.reasoning
        mock_judge.evaluate.assert_not_called()

    @pytest.mark.asyncio
    async def test_output_placeholder_substituted(self, mock_judge):
        mock_judge.evaluate.return_value = JudgeResponse(
            score=0.8, reasoning="ok", raw_output=None
        )
        dag = DAGMetric(
            judge=mock_judge,
            nodes=[
                DAGNode(name="node", prompt_template="Evaluate this: {output}", threshold=0.5),
            ],
            root="node",
        )
        await dag.evaluate(output="my specific output")
        call_args = mock_judge.evaluate.call_args
        assert "my specific output" in call_args.kwargs["prompt"]

    @pytest.mark.asyncio
    async def test_result_includes_pass_fail_label(self, mock_judge):
        mock_judge.evaluate.return_value = JudgeResponse(
            score=0.6, reasoning="ok", raw_output=None
        )
        dag = DAGMetric(
            judge=mock_judge,
            nodes=[
                DAGNode(name="check", prompt_template="Check: {output}", threshold=0.5),
            ],
            root="check",
        )
        result = await dag.evaluate(output="test")
        assert "pass" in result.reasoning or "fail" in result.reasoning

    @pytest.mark.asyncio
    async def test_latency_ms_is_non_negative(self, mock_judge):
        mock_judge.evaluate.return_value = JudgeResponse(
            score=0.7, reasoning="ok", raw_output=None
        )
        dag = DAGMetric(
            judge=mock_judge,
            nodes=[
                DAGNode(name="n", prompt_template="{output}", threshold=0.5),
            ],
            root="n",
        )
        result = await dag.evaluate(output="test")
        assert result.latency_ms >= 0
