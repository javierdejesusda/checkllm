import pytest
from bench.adapters.ragas_adapter import RagasAdapter
from bench.schema import BenchmarkSample, GroundTruth, MetricFamily


class FakeLangChainLLM:
    """Minimal stub exposing the ainvoke / invoke contract Ragas expects."""

    def __init__(self, payload: str = '{"score": 0.88, "reason": "fake"}') -> None:
        self.payload = payload

    async def ainvoke(self, prompt, **kwargs):
        from types import SimpleNamespace
        return SimpleNamespace(content=self.payload)

    def invoke(self, prompt, **kwargs):
        from types import SimpleNamespace
        return SimpleNamespace(content=self.payload)


@pytest.fixture
def sample():
    return BenchmarkSample(
        sample_id="rt-1",
        dataset="ragtruth",
        query="Who wrote Hamlet?",
        answer="William Shakespeare wrote Hamlet.",
        context="Hamlet was written by William Shakespeare around 1600.",
        ground_truth=GroundTruth(label=1.0, kind="binary"),
    )


@pytest.mark.asyncio
async def test_ragas_adapter_returns_faithfulness_score(sample):
    adapter = RagasAdapter(llm=FakeLangChainLLM())
    result = await adapter.score(sample, MetricFamily.FAITHFULNESS, "fake-gpt-4o-mini")
    assert result.framework == "ragas"
    assert result.metric_family is MetricFamily.FAITHFULNESS
    assert 0.0 <= result.score <= 1.0
