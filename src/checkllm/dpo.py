"""DPO (Direct Preference Optimization) export for converting evaluation results.

Converts evaluation results into DPO training data with multiple export
formats. Supports manual pair addition, automatic pair generation from
scored comparisons, and extraction from arena A/B test results.

Usage::

    from checkllm.dpo import DPOExporter

    exporter = DPOExporter()
    exporter.add_from_comparisons(
        prompt="What is Python?",
        responses=[
            {"output": "Python is a high-level language...", "score": 0.95},
            {"output": "Python is a snake...", "score": 0.2},
        ],
        min_score_gap=0.2,
    )
    dataset = exporter.build()
    dataset.to_json("training_data.json")
"""

from __future__ import annotations

import json
import random
import statistics
from datetime import datetime, timezone
from enum import Enum
from pathlib import Path
from typing import Any

from pydantic import BaseModel, Field, computed_field


class ExportFormat(str, Enum):
    """Supported DPO export formats."""

    JSON = "json"
    JSONL = "jsonl"
    HUGGINGFACE = "huggingface"
    OPENAI = "openai"


class DPOPair(BaseModel):
    """A single preference pair for DPO training.

    Attributes:
        prompt: The input prompt.
        chosen: The preferred response.
        rejected: The dispreferred response.
        chosen_score: Score of the chosen response (0.0 to 1.0).
        rejected_score: Score of the rejected response (0.0 to 1.0).
        metadata: Optional metadata about the pair source.
    """

    prompt: str
    chosen: str
    rejected: str
    chosen_score: float = Field(ge=0.0, le=1.0)
    rejected_score: float = Field(ge=0.0, le=1.0)
    metadata: dict[str, Any] = Field(default_factory=dict)

    @computed_field  # type: ignore[prop-decorator]
    @property
    def score_gap(self) -> float:
        """Absolute score difference between chosen and rejected."""
        return abs(self.chosen_score - self.rejected_score)


class DPOStats(BaseModel):
    """Aggregate statistics for a DPO dataset.

    Attributes:
        total_pairs: Number of preference pairs.
        avg_score_gap: Mean absolute score difference across pairs.
        max_score_gap: Largest score gap.
        min_score_gap: Smallest score gap.
        unique_prompts: Number of distinct prompts.
        avg_chosen_score: Mean score of chosen responses.
        avg_rejected_score: Mean score of rejected responses.
    """

    total_pairs: int
    avg_score_gap: float
    max_score_gap: float
    min_score_gap: float
    unique_prompts: int
    avg_chosen_score: float
    avg_rejected_score: float


class DPODataset(BaseModel):
    """A collection of DPO preference pairs with export capabilities.

    Attributes:
        pairs: The preference pairs.
        created_at: ISO-8601 timestamp of dataset creation.
        source: Identifier for the data source.
    """

    pairs: list[DPOPair]
    created_at: str
    source: str = "checkllm"

    def to_json(self, path: str) -> None:
        """Export pairs as a JSON array.

        Each element contains ``prompt``, ``chosen``, and ``rejected`` keys.

        Args:
            path: Destination file path.
        """
        records = [
            {"prompt": p.prompt, "chosen": p.chosen, "rejected": p.rejected}
            for p in self.pairs
        ]
        Path(path).write_text(json.dumps(records, indent=2), encoding="utf-8")

    def to_jsonl(self, path: str) -> None:
        """Export pairs as newline-delimited JSON (JSONL).

        Args:
            path: Destination file path.
        """
        lines = [
            json.dumps({"prompt": p.prompt, "chosen": p.chosen, "rejected": p.rejected})
            for p in self.pairs
        ]
        Path(path).write_text("\n".join(lines) + "\n" if lines else "", encoding="utf-8")

    def to_huggingface(self, path: str) -> None:
        """Export in HuggingFace datasets columnar format.

        Produces a JSON object with ``prompt``, ``chosen``, and ``rejected``
        lists suitable for ``datasets.Dataset.from_dict``.

        Args:
            path: Destination file path.
        """
        data = {
            "prompt": [p.prompt for p in self.pairs],
            "chosen": [p.chosen for p in self.pairs],
            "rejected": [p.rejected for p in self.pairs],
        }
        Path(path).write_text(json.dumps(data, indent=2), encoding="utf-8")

    def to_openai(self, path: str) -> None:
        """Export in OpenAI fine-tuning JSONL format.

        Each line contains a ``messages`` array with user/assistant roles and
        a ``weight`` field set to the score gap.

        Args:
            path: Destination file path.
        """
        lines: list[str] = []
        for p in self.pairs:
            record = {
                "messages": [
                    {"role": "user", "content": p.prompt},
                    {"role": "assistant", "content": p.chosen},
                ],
                "weight": round(p.score_gap, 4),
            }
            lines.append(json.dumps(record))
        Path(path).write_text("\n".join(lines) + "\n" if lines else "", encoding="utf-8")

    def stats(self) -> DPOStats:
        """Compute aggregate statistics for this dataset.

        Returns:
            A DPOStats summarising the pairs in this dataset.
        """
        if not self.pairs:
            return DPOStats(
                total_pairs=0,
                avg_score_gap=0.0,
                max_score_gap=0.0,
                min_score_gap=0.0,
                unique_prompts=0,
                avg_chosen_score=0.0,
                avg_rejected_score=0.0,
            )

        gaps = [p.score_gap for p in self.pairs]
        return DPOStats(
            total_pairs=len(self.pairs),
            avg_score_gap=statistics.mean(gaps),
            max_score_gap=max(gaps),
            min_score_gap=min(gaps),
            unique_prompts=len({p.prompt for p in self.pairs}),
            avg_chosen_score=statistics.mean(p.chosen_score for p in self.pairs),
            avg_rejected_score=statistics.mean(p.rejected_score for p in self.pairs),
        )

    def filter(self, min_score_gap: float = 0.0) -> DPODataset:
        """Return a new dataset containing only pairs that meet the gap threshold.

        Args:
            min_score_gap: Minimum absolute score difference to keep a pair.

        Returns:
            A filtered DPODataset.
        """
        return DPODataset(
            pairs=[p for p in self.pairs if p.score_gap >= min_score_gap],
            created_at=self.created_at,
            source=self.source,
        )

    def sample(self, n: int, seed: int = 42) -> DPODataset:
        """Return a random sample of pairs.

        Args:
            n: Number of pairs to sample. Clamped to dataset size.
            seed: Random seed for reproducibility.

        Returns:
            A sampled DPODataset.
        """
        rng = random.Random(seed)
        k = min(n, len(self.pairs))
        sampled = rng.sample(self.pairs, k)
        return DPODataset(
            pairs=sampled,
            created_at=self.created_at,
            source=self.source,
        )


class DPOExporter:
    """Collects preference pairs and builds DPO training datasets.

    Supports manual pair addition, automatic pair generation from scored
    comparisons, and extraction from arena A/B test results.
    """

    def __init__(self) -> None:
        self._pairs: list[DPOPair] = []

    def add_pair(self, pair: DPOPair) -> None:
        """Add a single preference pair.

        Args:
            pair: A DPOPair to add.
        """
        self._pairs.append(pair)

    def add_from_comparisons(
        self,
        prompt: str,
        responses: list[dict[str, Any]],
        min_score_gap: float = 0.0,
    ) -> None:
        """Generate preference pairs from scored responses.

        For *N* responses, generates up to N*(N-1)/2 pairs where the score
        gap meets the threshold. Pairs are sorted so the largest score gaps
        are added first.

        Args:
            prompt: The shared input prompt.
            responses: List of dicts each containing ``output`` (str) and
                ``score`` (float 0-1).
            min_score_gap: Minimum absolute score difference to form a pair.
        """
        candidates: list[tuple[str, float]] = []
        for r in responses:
            candidates.append((r["output"], float(r["score"])))

        candidates.sort(key=lambda x: x[1], reverse=True)

        raw_pairs: list[tuple[str, float, str, float, float]] = []
        for i in range(len(candidates)):
            for j in range(i + 1, len(candidates)):
                chosen_out, chosen_score = candidates[i]
                rejected_out, rejected_score = candidates[j]
                gap = chosen_score - rejected_score
                if gap >= min_score_gap:
                    raw_pairs.append(
                        (chosen_out, chosen_score, rejected_out, rejected_score, gap)
                    )

        raw_pairs.sort(key=lambda x: x[4], reverse=True)

        for chosen_out, chosen_score, rejected_out, rejected_score, _gap in raw_pairs:
            self._pairs.append(
                DPOPair(
                    prompt=prompt,
                    chosen=chosen_out,
                    rejected=rejected_out,
                    chosen_score=chosen_score,
                    rejected_score=rejected_score,
                )
            )

    def add_from_arena(self, arena_results: list[dict[str, Any]]) -> None:
        """Extract preference pairs from arena A/B test results.

        Each arena result dict must contain:
        - ``prompt``: the shared input prompt.
        - ``contestants``: a list of dicts with ``output`` and ``score`` keys.

        The highest-scoring contestant is chosen, the lowest is rejected.
        Ties are skipped.

        Args:
            arena_results: List of arena result dicts.
        """
        for result in arena_results:
            prompt = result["prompt"]
            contestants = sorted(
                result["contestants"], key=lambda c: c["score"], reverse=True
            )
            if len(contestants) < 2:
                continue
            best = contestants[0]
            worst = contestants[-1]
            if best["score"] == worst["score"]:
                continue
            self._pairs.append(
                DPOPair(
                    prompt=prompt,
                    chosen=best["output"],
                    rejected=worst["output"],
                    chosen_score=float(best["score"]),
                    rejected_score=float(worst["score"]),
                    metadata={"source": "arena"},
                )
            )

    def build(self) -> DPODataset:
        """Build a DPODataset from all collected pairs.

        Returns:
            A DPODataset ready for export.
        """
        return DPODataset(
            pairs=list(self._pairs),
            created_at=datetime.now(timezone.utc).isoformat(),
        )

    def clear(self) -> None:
        """Remove all collected pairs."""
        self._pairs.clear()
