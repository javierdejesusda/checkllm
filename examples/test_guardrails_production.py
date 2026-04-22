"""Production guardrails with ``Guard``.

Scenario: a production LLM endpoint that must enforce PII scrubbing, a
token budget, toxicity filtering, and a latency SLO before returning to
the user. The examples demonstrate the two common on-failure patterns:

- **log** - validate, record failures, and let the output through (ideal
  for shadow rollouts and adding guardrails without breaking behaviour).
- **raise** - validate and raise ``GuardrailError`` when anything fails
  (ideal for hard safety / compliance gates).

All tests are deterministic and do not require an API key. Swap the
``MockJudge`` for ``OpenAIJudge`` when wiring a real judge; see
``tests/test_guardrails.py`` for live-judge coverage.

Run with: pytest examples/test_guardrails_production.py
"""

from __future__ import annotations

import logging
import time

import pytest

from checkllm import CheckSpec, Guard, GuardrailError
from checkllm.testing import MockJudge


def _build_guard(mock_judge: MockJudge | None = None) -> Guard:
    """Construct the production guard once per test.

    Deterministic checks run first (instant, free). The toxicity judge is
    included but defaults to *soft* so log-mode flows can still pass.
    """
    return Guard(
        checks=[
            CheckSpec(check_type="no_pii"),
            CheckSpec(check_type="max_tokens", params={"limit": 500}),
            CheckSpec(check_type="toxicity", soft=True),
        ],
        judge=mock_judge,
    )


def test_guard_blocks_pii() -> None:
    """Hard failure on PII; the guard raises ``GuardrailError``."""
    guard = _build_guard(MockJudge(default_score=0.95))
    output = "Sure, the admin's SSN is 123-45-6789."

    result = guard.validate(output)
    assert not result.valid
    failed_names = [r.metric_name for r in result.failed_checks]
    assert "no_pii" in failed_names

    # The "raise" pattern: let the guard abort on a hard failure.
    with pytest.raises(GuardrailError):
        guard(output)


def test_guard_log_mode_records_but_allows(caplog: pytest.LogCaptureFixture) -> None:
    """Log-mode: validate, record failures, let output through."""
    guard = _build_guard(MockJudge(default_score=0.95))
    output = "Email me at user@example.com for the invoice."

    with caplog.at_level(logging.WARNING, logger="checkllm.guardrails"):
        result = guard.validate(output)
        if not result.valid:
            logging.getLogger("checkllm.guardrails").warning(
                "guardrail failure summary:\n%s", result.summary()
            )

    assert not result.valid
    assert result.failed_checks, "PII should have been detected in the output"
    # Output is still available to the caller in log-mode; no exception raised.
    assert output


def test_guard_passes_clean_output() -> None:
    """Clean output must pass all configured checks."""
    mock = MockJudge(default_score=0.95, default_reasoning="Mock: non-toxic")
    guard = _build_guard(mock)
    output = "Your monthly usage summary is attached. Let me know if you have questions."

    result = guard.validate(output)
    assert result.valid, result.summary()
    assert result.total_cost == 0.0  # MockJudge charges nothing.


def test_latency_budget_assertion() -> None:
    """Production guardrails must stay within a latency SLO."""
    guard = _build_guard(MockJudge(default_score=0.95))
    output = "All good."

    start = time.perf_counter_ns()
    result = guard.validate(output)
    elapsed_ms = (time.perf_counter_ns() - start) / 1_000_000

    # Deterministic + mock-judge checks should comfortably fit in 250ms.
    assert elapsed_ms < 250, f"guard ran for {elapsed_ms:.1f}ms"
    assert result.total_latency_ms >= 0


def test_dry_run_validates_configuration() -> None:
    """Smoke-test the guard configuration before deploying it."""
    guard = _build_guard(MockJudge(default_score=0.95))

    # Every configured check must be one the guard actually knows how to run.
    known_types = {spec.check_type for spec in guard.checks}
    assert known_types == {"no_pii", "max_tokens", "toxicity"}

    # Run the guard on a canary string to prove end-to-end wiring.
    canary = "hello world"
    result = guard.validate(canary)
    assert result.valid, f"canary validation failed: {result.summary()}"


def test_oversize_output_trips_token_budget() -> None:
    """A 501-token output should trip ``max_tokens``."""
    guard = _build_guard(MockJudge(default_score=0.95))
    output = " ".join(["token"] * 520)

    result = guard.validate(output)
    assert not result.valid
    failed_names = [r.metric_name for r in result.failed_checks]
    assert "max_tokens" in failed_names


# Run with: pytest examples/test_guardrails_production.py
