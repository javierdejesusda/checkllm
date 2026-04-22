import os

import pytest

if not os.getenv("CHECKLLM_BENCH_RUN_RAGAS"):
    pytest.skip(
        "Ragas adapter test skipped by default because importing ragas "
        "pulls in torch, which is slow on some platforms. "
        "Set CHECKLLM_BENCH_RUN_RAGAS=1 to enable.",
        allow_module_level=True,
    )

from bench.adapters.ragas_adapter import RagasAdapter  # noqa: E402
from bench.schema import BenchmarkSample, GroundTruth, MetricFamily  # noqa: E402


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
async def test_ragas_adapter_returns_faithfulness_score(sample, monkeypatch):
    from ragas.metrics import faithfulness

    async def fake_ascore(ragas_sample, callbacks=None):
        return 0.88

    monkeypatch.setattr(faithfulness, "single_turn_ascore", fake_ascore)

    adapter = RagasAdapter(llm=FakeLangChainLLM())
    result = await adapter.score(sample, MetricFamily.FAITHFULNESS, "fake-gpt-4o-mini")

    assert result.framework == "ragas"
    assert result.metric_family is MetricFamily.FAITHFULNESS
    assert result.score == 0.88
    assert result.metric_name == "faithfulness"
    assert result.judge_model == "fake-gpt-4o-mini"
