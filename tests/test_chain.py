"""Tests for fluent assertion chaining."""
from checkllm.check import CheckCollector
from checkllm.config import CheckllmConfig
from checkllm.testing import MockJudge


def test_that_returns_chain():
    config = CheckllmConfig()
    judge = MockJudge(default_score=0.9)
    collector = CheckCollector(config=config, judge=judge)
    chain = collector.that("Hello world")
    from checkllm.chain import AssertionChain
    assert isinstance(chain, AssertionChain)


def test_chain_contains():
    config = CheckllmConfig()
    collector = CheckCollector(config=config)
    result = collector.that("Python is great").contains("Python")
    from checkllm.chain import AssertionChain
    assert isinstance(result, AssertionChain)
    assert len(collector.results) == 1
    assert collector.results[0].passed is True


def test_chain_multiple_checks():
    config = CheckllmConfig()
    collector = CheckCollector(config=config)
    collector.that("Python is great").contains("Python").not_contains("Java").max_tokens(10)
    assert len(collector.results) == 3
    assert all(r.passed for r in collector.results)


def test_chain_has_no_pii():
    config = CheckllmConfig()
    collector = CheckCollector(config=config)
    collector.that("Hello world").has_no_pii()
    assert len(collector.results) == 1
    assert collector.results[0].passed is True


def test_chain_is_json():
    config = CheckllmConfig()
    collector = CheckCollector(config=config)
    collector.that('{"key": "value"}').is_json()
    assert collector.results[0].passed is True


def test_chain_scores_above():
    config = CheckllmConfig()
    judge = MockJudge(default_score=0.95)
    collector = CheckCollector(config=config, judge=judge)
    collector.that("Good output").scores_above("relevance", 0.8, query="test")
    assert len(collector.results) == 1
    assert collector.results[0].passed is True


def test_chain_failing_check():
    config = CheckllmConfig()
    collector = CheckCollector(config=config)
    collector.that("Hello").contains("Goodbye")
    assert len(collector.results) == 1
    assert collector.results[0].passed is False
