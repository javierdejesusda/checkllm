from __future__ import annotations

import json
import re
from typing import Any, Type

import tiktoken
from pydantic import BaseModel, ValidationError

from checkllm.models import CheckResult


class DeterministicChecks:
    """Deterministic checks that run locally with zero API calls."""

    def contains(self, output: str, substring: str) -> CheckResult:
        found = substring in output
        return CheckResult(
            passed=found,
            score=1.0 if found else 0.0,
            reasoning=f"Substring '{substring}' {'found' if found else 'not found'} in output",
            cost=0.0,
            latency_ms=0,
            metric_name="contains",
        )

    def not_contains(self, output: str, substring: str) -> CheckResult:
        absent = substring not in output
        return CheckResult(
            passed=absent,
            score=1.0 if absent else 0.0,
            reasoning=f"Substring '{substring}' {'not found' if absent else 'found'} in output",
            cost=0.0,
            latency_ms=0,
            metric_name="not_contains",
        )

    def max_tokens(self, output: str, limit: int) -> CheckResult:
        enc = tiktoken.get_encoding("cl100k_base")
        token_count = len(enc.encode(output))
        passed = token_count <= limit
        return CheckResult(
            passed=passed,
            score=min(1.0, limit / max(token_count, 1)),
            reasoning=f"Token count: {token_count}, limit: {limit}",
            cost=0.0,
            latency_ms=0,
            metric_name="max_tokens",
        )

    def latency(self, actual_ms: int | float, max_ms: int | float) -> CheckResult:
        passed = actual_ms <= max_ms
        return CheckResult(
            passed=passed,
            score=min(1.0, max_ms / max(actual_ms, 1)),
            reasoning=f"Latency: {actual_ms}ms, limit: {max_ms}ms",
            cost=0.0,
            latency_ms=0,
            metric_name="latency",
        )

    def cost(self, actual_usd: float, max_usd: float) -> CheckResult:
        passed = actual_usd <= max_usd
        return CheckResult(
            passed=passed,
            score=min(1.0, max_usd / max(actual_usd, 0.0001)),
            reasoning=f"Cost: ${actual_usd:.4f}, limit: ${max_usd:.4f}",
            cost=0.0,
            latency_ms=0,
            metric_name="cost",
        )

    def json_schema(self, output: str, schema: Type[BaseModel]) -> CheckResult:
        try:
            parsed = json.loads(output)
            schema.model_validate(parsed)
            return CheckResult(
                passed=True,
                score=1.0,
                reasoning=f"Output is valid {schema.__name__} JSON",
                cost=0.0,
                latency_ms=0,
                metric_name="json_schema",
            )
        except (json.JSONDecodeError, ValidationError) as e:
            return CheckResult(
                passed=False,
                score=0.0,
                reasoning=f"JSON schema validation failed: {e}",
                cost=0.0,
                latency_ms=0,
                metric_name="json_schema",
            )

    def regex(self, output: str, pattern: str) -> CheckResult:
        match = re.search(pattern, output)
        passed = match is not None
        return CheckResult(
            passed=passed,
            score=1.0 if passed else 0.0,
            reasoning=f"Pattern '{pattern}' {'matched' if passed else 'not matched'} in output",
            cost=0.0,
            latency_ms=0,
            metric_name="regex",
        )
