import json
import pytest
from bench.adapters.promptfoo_adapter import (
    PromptfooAdapter,
    _grader_cost_usd,
    _resolve_npx,
)
from bench.schema import BenchmarkSample, GroundTruth, MetricFamily


@pytest.fixture
def sample():
    return BenchmarkSample(
        sample_id="hb-1",
        dataset="halubench",
        query="What is the capital of France?",
        answer="Paris is the capital of France.",
        context="Paris is the capital of France.",
        ground_truth=GroundTruth(label=1.0, kind="binary"),
    )


@pytest.mark.asyncio
async def test_promptfoo_adapter_parses_rubric_score(sample):
    fake_output = json.dumps({
        "results": {
            "results": [
                {
                    "gradingResult": {
                        "pass": True,
                        "score": 0.91,
                        "reason": "ok",
                        "tokensUsed": {"prompt": 400, "completion": 50},
                    },
                    "latencyMs": 250,
                    "cost": 0.0,
                }
            ]
        }
    })

    async def fake_run(args, cwd):
        return (fake_output, "")

    adapter = PromptfooAdapter(judge_model="gpt-4o-mini", runner=fake_run)
    result = await adapter.score(sample, MetricFamily.FAITHFULNESS, "gpt-4o-mini")
    assert result.framework == "promptfoo"
    assert result.score == 0.91
    assert result.latency_ms == 250
    assert result.cost_usd == pytest.approx(
        (400 * 0.15 + 50 * 0.60) / 1_000_000
    )


@pytest.mark.asyncio
async def test_promptfoo_adapter_tolerates_missing_grading_result(sample):
    fake_output = json.dumps({
        "results": {
            "results": [
                {
                    "gradingResult": None,
                    "error": "API error: 400",
                    "latencyMs": 0,
                    "cost": 0.0,
                }
            ]
        }
    })

    async def fake_run(args, cwd):
        return (fake_output, "")

    adapter = PromptfooAdapter(judge_model="gpt-4o-mini", runner=fake_run)
    result = await adapter.score(sample, MetricFamily.HALLUCINATION, "gpt-4o-mini")
    assert result.score == 0.0
    assert result.passed is False
    assert "API error" in result.reasoning


def test_grader_cost_computes_from_tokens():
    cost = _grader_cost_usd("gpt-4o-mini", {"prompt": 1_000_000, "completion": 1_000_000})
    assert cost == pytest.approx(0.15 + 0.60)


def test_grader_cost_returns_zero_for_unknown_model():
    assert _grader_cost_usd("mystery-model", {"prompt": 100, "completion": 100}) == 0.0


@pytest.mark.asyncio
async def test_promptfoo_adapter_uses_resolved_npx_launcher(sample, monkeypatch):
    captured_args: list[list[str]] = []

    async def fake_run(args, cwd):
        captured_args.append(args)
        return (
            json.dumps(
                {
                    "results": {
                        "results": [
                            {
                                "gradingResult": {"pass": True, "score": 1.0, "reason": "ok"},
                                "latencyMs": 1,
                                "cost": 0.0,
                            }
                        ]
                    }
                }
            ),
            "",
        )

    monkeypatch.setattr(
        "bench.adapters.promptfoo_adapter.shutil.which",
        lambda _: r"C:\Program Files\nodejs\npx.CMD",
    )
    adapter = PromptfooAdapter(judge_model="gpt-4o-mini", runner=fake_run)
    await adapter.score(sample, MetricFamily.HALLUCINATION, "gpt-4o-mini")
    assert captured_args, "runner was not invoked"
    assert captured_args[0][0] == r"C:\Program Files\nodejs\npx.CMD"


def test_resolve_npx_raises_when_missing(monkeypatch):
    monkeypatch.setattr(
        "bench.adapters.promptfoo_adapter.shutil.which", lambda _: None
    )
    with pytest.raises(FileNotFoundError, match="npx not found"):
        _resolve_npx()
