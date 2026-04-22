"""Tests for the CheckLLM-unique capability showcase harness.

Each test verifies the structural contract of the runner functions: correct
section name, correct return type, and a non-negative total. Exact scores and
pass counts are not asserted because they depend on CheckLLM internals that
may vary across versions.
"""

import pytest
from bench.showcase import (
    ShowcaseReport,
    run_compliance_showcase,
    run_kg_synthesis_showcase,
    run_redteam_showcase,
    run_trajectory_showcase,
)
from checkllm.testing import MockJudge


async def _dummy_target(prompt: str) -> str:
    return "I cannot help with that request."


@pytest.mark.asyncio
async def test_compliance_showcase_returns_report():
    judge = MockJudge(default_score=0.95)
    report = await run_compliance_showcase(
        target=_dummy_target,
        judge=judge,
        frameworks=["OWASP_LLM_TOP10"],
        attacks_per_type=1,
    )
    assert isinstance(report, ShowcaseReport)
    assert report.section == "compliance"
    assert report.total >= 0


@pytest.mark.asyncio
async def test_redteam_showcase_returns_report():
    judge = MockJudge(default_score=0.9)
    report = await run_redteam_showcase(
        target=_dummy_target,
        judge=judge,
        vulnerability_types=["PROMPT_INJECTION"],
        strategies=["BASE64"],
        attacks_per_type=1,
    )
    assert isinstance(report, ShowcaseReport)
    assert report.section == "redteam"
    assert report.total >= 0


@pytest.mark.asyncio
async def test_kg_synthesis_showcase_returns_report():
    judge = MockJudge(default_score=0.9)
    report = await run_kg_synthesis_showcase(
        judge=judge,
        documents=["Python is a programming language created by Guido van Rossum in 1991."],
        num_samples=2,
    )
    assert isinstance(report, ShowcaseReport)
    assert report.section == "knowledge_graph"
    assert report.total >= 0


@pytest.mark.asyncio
async def test_trajectory_showcase_scores_trajectory():
    judge = MockJudge(default_score=0.92)
    trajectory = [
        {"tool": "search", "args": {"query": "weather"}},
        {"tool": "reply", "args": {"text": "It is sunny."}},
    ]
    expected = ["search", "reply"]
    report = await run_trajectory_showcase(
        judge=judge,
        trajectories=[(trajectory, expected)],
    )
    assert isinstance(report, ShowcaseReport)
    assert report.section == "trajectory"
    assert report.total == 1
