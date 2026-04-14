import json
import pytest
from bench.adapters.promptfoo_adapter import PromptfooAdapter
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
                    "gradingResult": {"pass": True, "score": 0.91, "reason": "ok"},
                    "latencyMs": 250,
                    "cost": 0.0002,
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
    assert result.cost_usd == 0.0002
