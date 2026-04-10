"""Pre-run cost estimation for checkllm checks."""
from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path

from checkllm.judge import _OPENAI_PRICES, _ANTHROPIC_PRICES, _DEFAULT_PRICE

# Deterministic checks — zero cost
_DETERMINISTIC_CHECKS = {
    "contains", "not_contains", "exact_match", "starts_with", "ends_with",
    "regex", "similarity", "max_tokens", "min_tokens", "word_count",
    "char_count", "sentence_count", "is_json", "is_valid_python",
    "json_schema", "json_field", "is_valid_sql", "is_valid_markdown",
    "readability", "language", "bleu", "rouge_l", "all_of", "any_of",
    "none_of", "no_pii", "greater_than", "less_than", "between",
    "latency", "cost", "meteor", "gleu", "chrf", "perplexity_check",
    "latency_check", "cost_check", "string_distance", "exact_match_strict",
}

# Average tokens per judge call (prompt + completion estimate)
_AVG_PROMPT_TOKENS = 500
_AVG_COMPLETION_TOKENS = 150


@dataclass
class CostEstimate:
    """Result of a cost estimation."""
    deterministic_count: int = 0
    judge_count: int = 0
    total_cost: float = 0.0
    model: str = "gpt-4o"

    def summary(self) -> str:
        return (
            f"Estimated: {self.deterministic_count} deterministic, "
            f"{self.judge_count} judge checks — ~${self.total_cost:.2f} ({self.model})"
        )


@dataclass
class SingleCheckEstimate:
    """Cost estimate for a single check type."""
    cost: float
    is_deterministic: bool


def _get_price(model: str) -> tuple[float, float]:
    """Look up per-token price for a model."""
    all_prices = {**_OPENAI_PRICES, **_ANTHROPIC_PRICES}
    return all_prices.get(model, _DEFAULT_PRICE)


def estimate_check_cost(check_name: str, model: str = "gpt-4o") -> SingleCheckEstimate:
    """Estimate cost of a single check."""
    if check_name in _DETERMINISTIC_CHECKS:
        return SingleCheckEstimate(cost=0.0, is_deterministic=True)

    input_price, output_price = _get_price(model)
    cost = _AVG_PROMPT_TOKENS * input_price + _AVG_COMPLETION_TOKENS * output_price
    return SingleCheckEstimate(cost=cost, is_deterministic=False)


def estimate_from_test_file(file_path: str, model: str = "gpt-4o") -> CostEstimate:
    """Scan a test file and estimate total cost."""
    content = Path(file_path).read_text()
    # Match check.<method_name>( patterns
    check_calls = re.findall(r"check\.(\w+)\(", content)

    det_count = 0
    judge_count = 0
    total_cost = 0.0

    for call in check_calls:
        # Skip non-check methods
        if call in ("expect", "that"):
            continue
        est = estimate_check_cost(call, model)
        if est.is_deterministic:
            det_count += 1
        else:
            judge_count += 1
            total_cost += est.cost

    return CostEstimate(
        deterministic_count=det_count,
        judge_count=judge_count,
        total_cost=round(total_cost, 4),
        model=model,
    )
