from __future__ import annotations

from enum import Enum
from typing import Protocol, runtime_checkable

from pydantic import BaseModel, Field


class MetricFamily(str, Enum):
    HALLUCINATION = "hallucination"
    FAITHFULNESS = "faithfulness"
    ANSWER_RELEVANCY = "answer_relevancy"
    CONTEXT_RELEVANCE = "context_relevance"
    JAILBREAK_RESISTANCE = "jailbreak_resistance"


class GroundTruth(BaseModel):
    label: float
    kind: str  # "binary" | "scalar" | "categorical"


class BenchmarkSample(BaseModel):
    sample_id: str
    dataset: str
    query: str
    answer: str
    context: str = ""
    ground_truth: GroundTruth


class BenchmarkScore(BaseModel):
    framework: str
    dataset: str
    metric_family: MetricFamily
    metric_name: str
    sample_id: str
    score: float = Field(ge=0.0, le=1.0)
    passed: bool
    latency_ms: int = Field(ge=0)
    cost_usd: float = Field(ge=0.0)
    judge_model: str
    reasoning: str = ""


@runtime_checkable
class FrameworkAdapter(Protocol):
    framework: str

    async def score(
        self,
        sample: BenchmarkSample,
        family: MetricFamily,
        judge_model: str,
    ) -> BenchmarkScore: ...

    def supports(self, family: MetricFamily) -> bool: ...
