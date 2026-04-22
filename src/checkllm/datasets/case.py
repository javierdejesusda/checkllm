from __future__ import annotations

from typing import Any

from pydantic import BaseModel, Field


class Case(BaseModel):
    """A single test case for dataset-driven evaluation."""

    input: str
    expected: str | None = None
    query: str | None = None
    context: str | None = None
    criteria: str | None = None
    image: str | None = None
    images: list[str] | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)

    @property
    def expected_criteria(self) -> str | None:
        """Alias for criteria, for convenience in test code."""
        return self.criteria

    @property
    def image_sources(self) -> list[str]:
        """Return all image sources as a list (single ``image`` or ``images``).

        Returns:
            A possibly-empty list of image source strings (paths, URLs, or
            base64 strings). ``images`` takes precedence over ``image`` when
            both are present.
        """
        if self.images:
            return list(self.images)
        if self.image:
            return [self.image]
        return []
