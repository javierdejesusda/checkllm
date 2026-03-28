from __future__ import annotations

from pydantic import BaseModel, Field, field_validator


class CheckResult(BaseModel):
    """Result of a single check evaluation."""

    passed: bool
    score: float = Field(ge=0.0, le=1.0)
    reasoning: str
    cost: float = Field(ge=0.0)
    latency_ms: int = Field(ge=0)
    metric_name: str

    @field_validator("score")
    @classmethod
    def validate_score_range(cls, v: float) -> float:
        if not 0.0 <= v <= 1.0:
            raise ValueError(f"score must be between 0.0 and 1.0, got {v}")
        return v


class JudgeResponse(BaseModel):
    """Raw response from an LLM judge."""

    score: float = Field(ge=0.0, le=1.0)
    reasoning: str
    raw_output: str | None = None


class CheckFailedError(Exception):
    """Raised when one or more checks fail during a test."""

    def __init__(self, results: list[CheckResult]) -> None:
        self.results = results
        self.failed_results = [r for r in results if not r.passed]
        count = len(self.failed_results)
        names = ", ".join(r.metric_name for r in self.failed_results)
        super().__init__(f"{count} check(s) failed: {names}")
