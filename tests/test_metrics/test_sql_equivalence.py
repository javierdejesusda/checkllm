from __future__ import annotations

import pytest

from checkllm.metrics.sql_equivalence import SQLEquivalenceMetric
from checkllm.testing import MockJudge


class TestSQLEquivalenceMetric:
    @pytest.fixture
    def judge(self):
        return MockJudge()

    @pytest.mark.asyncio
    async def test_passes_when_equivalent(self, judge):
        judge.set_default(score=0.95, reasoning="Queries are semantically equivalent")
        metric = SQLEquivalenceMetric(judge=judge, threshold=0.8)
        result = await metric.evaluate(
            output="SELECT name FROM users WHERE age > 18 ORDER BY name",
            reference="SELECT name FROM users WHERE age > 18 ORDER BY name ASC",
        )
        assert result.passed is True
        assert result.score == 0.95
        assert result.metric_name == "sql_equivalence"

    @pytest.mark.asyncio
    async def test_fails_when_not_equivalent(self, judge):
        judge.set_default(score=0.1, reasoning="Queries produce different results")
        metric = SQLEquivalenceMetric(judge=judge, threshold=0.8)
        result = await metric.evaluate(
            output="SELECT name FROM users WHERE age > 18",
            reference="SELECT name FROM users WHERE age < 18",
        )
        assert result.passed is False
        assert result.score == 0.1

    @pytest.mark.asyncio
    async def test_with_schema(self, judge):
        judge.set_default(score=0.9, reasoning="ok")
        metric = SQLEquivalenceMetric(judge=judge, threshold=0.8)
        result = await metric.evaluate(
            output="SELECT u.name FROM users u JOIN orders o ON u.id = o.user_id",
            reference="SELECT users.name FROM users INNER JOIN orders ON users.id = orders.user_id",
            schema="CREATE TABLE users (id INT, name VARCHAR); CREATE TABLE orders (id INT, user_id INT);",
        )
        assert result.passed is True
        last_call = judge.calls[-1]
        assert "CREATE TABLE" in last_call["prompt"]

    @pytest.mark.asyncio
    async def test_without_schema(self, judge):
        judge.set_default(score=0.9, reasoning="ok")
        metric = SQLEquivalenceMetric(judge=judge, threshold=0.8)
        await metric.evaluate(
            output="SELECT * FROM t",
            reference="SELECT * FROM t",
        )
        last_call = judge.calls[-1]
        assert "SELECT * FROM t" in last_call["prompt"]

    @pytest.mark.asyncio
    async def test_threshold_boundary(self, judge):
        judge.set_default(score=0.8, reasoning="Exactly at threshold")
        metric = SQLEquivalenceMetric(judge=judge, threshold=0.8)
        result = await metric.evaluate(output="SELECT 1", reference="SELECT 1")
        assert result.passed is True

        judge.set_default(score=0.79, reasoning="Just below threshold")
        result = await metric.evaluate(output="SELECT 1", reference="SELECT 1")
        assert result.passed is False

    @pytest.mark.asyncio
    async def test_latency_is_recorded(self, judge):
        judge.set_default(score=0.9, reasoning="ok")
        metric = SQLEquivalenceMetric(judge=judge, threshold=0.8)
        result = await metric.evaluate(output="SELECT 1", reference="SELECT 1")
        assert result.latency_ms >= 0
