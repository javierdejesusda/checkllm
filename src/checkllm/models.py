from __future__ import annotations

from typing import Any

from pydantic import BaseModel, Field, field_validator


class CheckResult(BaseModel):
    """Result of a single check evaluation."""

    passed: bool
    score: float = Field(ge=0.0, le=1.0)
    reasoning: str
    cost: float = Field(ge=0.0)
    latency_ms: int = Field(ge=0)
    metric_name: str
    threshold: float | None = Field(default=None, ge=0.0, le=1.0)
    input_preview: str | None = Field(default=None)
    # Granular cost attribution for dashboard rollups (by provider, metric,
    # test, timestamp).  Populated by judge-backed checks and by the engine
    # run loop.  Optional so deterministic checks and legacy callers stay
    # backward-compatible.  Stored as a plain dict so Pydantic can
    # round-trip it via ``model_dump() / json.loads``; the canonical shape
    # is :class:`checkllm.pricing.CostBreakdown`.
    cost_breakdown: dict[str, Any] | None = Field(default=None)

    @field_validator("score")
    @classmethod
    def validate_score_range(cls, v: float) -> float:
        if not 0.0 <= v <= 1.0:
            raise ValueError(f"score must be between 0.0 and 1.0, got {v}")
        return v

    def format_failure(self) -> str:
        """Format a human-readable failure description for terminal output."""
        lines = [f"  FAILED: {self.metric_name}"]
        if self.threshold is not None:
            lines.append(f"    Score: {self.score:.2f} (threshold: {self.threshold:.2f})")
        else:
            lines.append(f"    Score: {self.score:.2f}")
        lines.append(f"    Reason: {self.reasoning}")
        if self.input_preview:
            preview = self.input_preview[:120]
            if len(self.input_preview) > 120:
                preview += "..."
            lines.append(f"    Input: {preview}")
        if self.cost > 0:
            lines.append(f"    Cost: ${self.cost:.4f} | Latency: {self.latency_ms}ms")
        return "\n".join(lines)


class JudgeResponse(BaseModel):
    """Raw response from an LLM judge."""

    score: float = Field(ge=0.0, le=1.0)
    reasoning: str
    raw_output: str | None = None
    cost: float = Field(default=0.0, ge=0.0)
    # Token accounting -- populated by judge backends when available so the
    # caller can build a granular cost breakdown.  Absent values default
    # to 0 to keep backward compatibility.
    input_tokens: int = Field(default=0, ge=0)
    output_tokens: int = Field(default=0, ge=0)
    model: str | None = Field(default=None)
    provider: str | None = Field(default=None)


class CheckFailedError(Exception):
    """Raised when one or more checks fail during a test."""

    def __init__(self, results: list[CheckResult]) -> None:
        self.results = results
        self.failed_results = [r for r in results if not r.passed]
        count = len(self.failed_results)
        names = ", ".join(r.metric_name for r in self.failed_results)
        summary = f"{count} check(s) failed: {names}"

        details = [summary, ""]
        for r in self.failed_results:
            details.append(r.format_failure())
            details.append("")

        super().__init__("\n".join(details))
