import pytest
from bench.adapters.deepeval_adapter import DeepEvalAdapter
from bench.schema import BenchmarkSample, GroundTruth, MetricFamily


class FakeDeepEvalModel:
    """Implements the DeepEvalBaseLLM interface DeepEval metrics need."""

    def __init__(self, score_to_return: float = 0.15) -> None:
        self.score_to_return = score_to_return

    def get_model_name(self) -> str:
        return "fake-gpt-4o-mini"

    def load_model(self):
        return self

    def generate(self, prompt: str, schema=None):
        import json
        return json.dumps({"score": self.score_to_return, "reason": "fake"})

    async def a_generate(self, prompt: str, schema=None):
        return self.generate(prompt, schema)


@pytest.fixture
def sample():
    return BenchmarkSample(
        sample_id="hb-2",
        dataset="halubench",
        query="When was the Eiffel Tower built?",
        answer="The Eiffel Tower was completed in 1925.",
        context="The Eiffel Tower opened in 1889.",
        ground_truth=GroundTruth(label=0.0, kind="binary"),
    )


@pytest.mark.asyncio
async def test_deepeval_adapter_wraps_hallucination_metric(sample):
    adapter = DeepEvalAdapter(model=FakeDeepEvalModel(score_to_return=0.9))
    result = await adapter.score(sample, MetricFamily.HALLUCINATION, "fake-gpt-4o-mini")
    assert result.framework == "deepeval"
    assert result.metric_family is MetricFamily.HALLUCINATION
    assert 0.0 <= result.score <= 1.0
    assert result.judge_model == "fake-gpt-4o-mini"
