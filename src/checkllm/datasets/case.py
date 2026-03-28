from __future__ import annotations

from typing import Any

from pydantic import BaseModel, Field


class Case(BaseModel):
    """A single test case for dataset-driven evaluation."""

    input: str
    query: str | None = None
    criteria: str | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)

    @property
    def expected_criteria(self) -> str | None:
        """Alias for criteria, for convenience in test code."""
        return self.criteria
