"""Tests for the guardrails runtime validation module."""
from __future__ import annotations

import json

import pytest

from checkllm.guardrails import (
    CheckSpec,
    Guard,
    GuardrailError,
    GuardrailMiddleware,
    ValidationResult,
    guardrail,
    quality_guard,
    rag_guard,
    safety_guard,
)
from checkllm.models import CheckResult
from checkllm.testing import MockJudge


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_mock_judge(score: float = 0.9) -> MockJudge:
    return MockJudge(default_score=score)


def _make_guard(
    checks: list[CheckSpec],
    judge: MockJudge | None = None,
) -> Guard:
    return Guard(checks=checks, judge=judge or _make_mock_judge())


# ---------------------------------------------------------------------------
# ValidationResult
# ---------------------------------------------------------------------------


class TestValidationResult:
    def test_valid_result(self):
        r = ValidationResult(valid=True, results=[], failed_checks=[])
        assert r.valid is True
        assert r.results == []
        assert r.failed_checks == []

    def test_raise_on_failure_noop_when_valid(self):
        r = ValidationResult(valid=True, results=[], failed_checks=[])
        r.raise_on_failure()  # should not raise

    def test_raise_on_failure_raises_when_invalid(self):
        failed = CheckResult(
            passed=False, score=0.0, reasoning="bad",
            cost=0.0, latency_ms=0, metric_name="test_check",
        )
        r = ValidationResult(
            valid=False, results=[failed], failed_checks=[failed],
        )
        with pytest.raises(GuardrailError) as exc_info:
            r.raise_on_failure()
        assert exc_info.value.validation_result is r

    def test_summary_passed(self):
        ok = CheckResult(
            passed=True, score=1.0, reasoning="ok",
            cost=0.0, latency_ms=0, metric_name="contains",
        )
        r = ValidationResult(
            valid=True, results=[ok], failed_checks=[],
            total_latency_ms=5, total_cost=0.0,
        )
        s = r.summary()
        assert "PASSED" in s
        assert "1/1" in s

    def test_summary_failed(self):
        fail = CheckResult(
            passed=False, score=0.1, reasoning="missing",
            cost=0.01, latency_ms=100, metric_name="no_pii",
        )
        r = ValidationResult(
            valid=False, results=[fail], failed_checks=[fail],
            total_latency_ms=100, total_cost=0.01,
        )
        s = r.summary()
        assert "FAILED" in s
        assert "no_pii" in s
        assert "0/1" in s


# ---------------------------------------------------------------------------
# GuardrailError
# ---------------------------------------------------------------------------


class TestGuardrailError:
    def test_has_validation_result(self):
        r = ValidationResult(valid=False, results=[], failed_checks=[])
        err = GuardrailError(r)
        assert err.validation_result is r
        assert "failed" in str(err).lower()

    def test_message_contains_check_names(self):
        fail = CheckResult(
            passed=False, score=0.0, reasoning="bad",
            cost=0.0, latency_ms=0, metric_name="no_pii",
        )
        r = ValidationResult(valid=False, results=[fail], failed_checks=[fail])
        err = GuardrailError(r)
        assert "no_pii" in str(err)


# ---------------------------------------------------------------------------
# CheckSpec
# ---------------------------------------------------------------------------


class TestCheckSpec:
    def test_defaults(self):
        spec = CheckSpec(check_type="contains")
        assert spec.check_type == "contains"
        assert spec.params == {}
        assert spec.soft is False

    def test_with_params(self):
        spec = CheckSpec(check_type="max_tokens", params={"limit": 100}, soft=True)
        assert spec.params["limit"] == 100
        assert spec.soft is True


# ---------------------------------------------------------------------------
# Guard - deterministic checks
# ---------------------------------------------------------------------------


class TestGuardDeterministic:
    def test_validate_contains_pass(self):
        guard = _make_guard([CheckSpec(check_type="contains", params={"substring": "hello"})])
        result = guard.validate("hello world")
        assert result.valid is True
        assert len(result.results) == 1
        assert result.results[0].passed is True

    def test_validate_contains_fail(self):
        guard = _make_guard([CheckSpec(check_type="contains", params={"substring": "goodbye"})])
        result = guard.validate("hello world")
        assert result.valid is False
        assert len(result.failed_checks) == 1

    def test_validate_no_pii_pass(self):
        guard = _make_guard([CheckSpec(check_type="no_pii")])
        result = guard.validate("This is a safe output with no personal data.")
        assert result.valid is True

    def test_validate_no_pii_fail(self):
        guard = _make_guard([CheckSpec(check_type="no_pii")])
        result = guard.validate("Contact me at user@example.com or 555-123-4567")
        assert result.valid is False

    def test_validate_max_tokens(self):
        guard = _make_guard([CheckSpec(check_type="max_tokens", params={"limit": 10})])
        result = guard.validate("short")
        assert result.valid is True

    def test_validate_min_tokens(self):
        guard = _make_guard([CheckSpec(check_type="min_tokens", params={"minimum": 1})])
        result = guard.validate("hello world")
        assert result.valid is True

    def test_validate_multiple_deterministic(self):
        guard = _make_guard([
            CheckSpec(check_type="contains", params={"substring": "hello"}),
            CheckSpec(check_type="no_pii"),
            CheckSpec(check_type="max_tokens", params={"limit": 1000}),
        ])
        result = guard.validate("hello world")
        assert result.valid is True
        assert len(result.results) == 3

    def test_validate_regex(self):
        guard = _make_guard([CheckSpec(check_type="regex", params={"pattern": r"\d+"})])
        result = guard.validate("order 42 confirmed")
        assert result.valid is True

    def test_validate_is_json(self):
        guard = _make_guard([CheckSpec(check_type="is_json")])
        result = guard.validate('{"key": "value"}')
        assert result.valid is True


# ---------------------------------------------------------------------------
# Guard - judge-based checks
# ---------------------------------------------------------------------------


class TestGuardJudge:
    def test_validate_toxicity_with_mock(self):
        judge = _make_mock_judge(score=0.95)
        guard = _make_guard(
            [CheckSpec(check_type="toxicity")],
            judge=judge,
        )
        result = guard.validate("This is a perfectly safe response.")
        assert result.valid is True
        assert len(judge.calls) == 1

    def test_validate_toxicity_fail_with_mock(self):
        judge = _make_mock_judge(score=0.1)
        guard = _make_guard(
            [CheckSpec(check_type="toxicity")],
            judge=judge,
        )
        result = guard.validate("some output")
        assert result.valid is False

    def test_validate_fluency_with_mock(self):
        judge = _make_mock_judge(score=0.9)
        guard = _make_guard(
            [CheckSpec(check_type="fluency")],
            judge=judge,
        )
        result = guard.validate("Well-written output text.")
        assert result.valid is True

    def test_validate_coherence_with_mock(self):
        judge = _make_mock_judge(score=0.9)
        guard = _make_guard(
            [CheckSpec(check_type="coherence")],
            judge=judge,
        )
        result = guard.validate("A coherent response here.")
        assert result.valid is True


# ---------------------------------------------------------------------------
# Guard - mixed deterministic + judge
# ---------------------------------------------------------------------------


class TestGuardMixed:
    def test_mixed_all_pass(self):
        judge = _make_mock_judge(score=0.95)
        guard = _make_guard(
            [
                CheckSpec(check_type="no_pii"),
                CheckSpec(check_type="max_tokens", params={"limit": 1000}),
                CheckSpec(check_type="toxicity"),
            ],
            judge=judge,
        )
        result = guard.validate("Clean safe text with no PII.")
        assert result.valid is True
        assert len(result.results) == 3
        assert result.total_cost >= 0.0

    def test_mixed_deterministic_fails(self):
        judge = _make_mock_judge(score=0.95)
        guard = _make_guard(
            [
                CheckSpec(check_type="contains", params={"substring": "required_word"}),
                CheckSpec(check_type="toxicity"),
            ],
            judge=judge,
        )
        result = guard.validate("This output is missing the keyword.")
        assert result.valid is False
        assert any(r.metric_name == "contains" for r in result.failed_checks)

    def test_mixed_judge_fails(self):
        judge = _make_mock_judge(score=0.1)
        guard = _make_guard(
            [
                CheckSpec(check_type="no_pii"),
                CheckSpec(check_type="toxicity"),
            ],
            judge=judge,
        )
        result = guard.validate("Clean text but judge says toxic.")
        assert result.valid is False
        assert any(r.metric_name == "toxicity" for r in result.failed_checks)


# ---------------------------------------------------------------------------
# Guard - soft checks
# ---------------------------------------------------------------------------


class TestGuardSoftChecks:
    def test_soft_check_failure_does_not_invalidate(self):
        guard = _make_guard([
            CheckSpec(check_type="contains", params={"substring": "missing"}, soft=True),
            CheckSpec(check_type="no_pii"),
        ])
        result = guard.validate("Hello safe world")
        # The contains check fails but is soft, no_pii passes
        assert result.valid is True
        # The failed check still appears in failed_checks
        assert len(result.failed_checks) == 1
        assert result.failed_checks[0].metric_name == "contains"

    def test_soft_and_hard_failure(self):
        guard = _make_guard([
            CheckSpec(check_type="contains", params={"substring": "missing"}, soft=True),
            CheckSpec(check_type="contains", params={"substring": "also_missing"}),
        ])
        result = guard.validate("Hello world")
        # Hard check failed -> invalid
        assert result.valid is False
        assert len(result.failed_checks) == 2

    def test_all_soft_failures_still_valid(self):
        guard = _make_guard([
            CheckSpec(check_type="contains", params={"substring": "x"}, soft=True),
            CheckSpec(check_type="contains", params={"substring": "y"}, soft=True),
        ])
        result = guard.validate("hello")
        assert result.valid is True
        assert len(result.failed_checks) == 2


# ---------------------------------------------------------------------------
# Guard.validate() and Guard.avalidate()
# ---------------------------------------------------------------------------


class TestGuardAsync:
    @pytest.mark.asyncio
    async def test_avalidate(self):
        judge = _make_mock_judge(score=0.9)
        guard = _make_guard([CheckSpec(check_type="toxicity")], judge=judge)
        result = await guard.avalidate("safe output")
        assert result.valid is True

    @pytest.mark.asyncio
    async def test_avalidate_multiple(self):
        judge = _make_mock_judge(score=0.95)
        guard = _make_guard(
            [
                CheckSpec(check_type="no_pii"),
                CheckSpec(check_type="toxicity"),
            ],
            judge=judge,
        )
        result = await guard.avalidate("Hello safe text.")
        assert result.valid is True
        assert len(result.results) == 2


# ---------------------------------------------------------------------------
# Guard.__call__
# ---------------------------------------------------------------------------


class TestGuardCall:
    def test_call_returns_output_on_pass(self):
        guard = _make_guard([CheckSpec(check_type="no_pii")])
        output = guard("Hello safe world")
        assert output == "Hello safe world"

    def test_call_raises_on_failure(self):
        guard = _make_guard([
            CheckSpec(check_type="contains", params={"substring": "required"}),
        ])
        with pytest.raises(GuardrailError):
            guard("missing the keyword")


# ---------------------------------------------------------------------------
# Guard.from_config
# ---------------------------------------------------------------------------


class TestGuardFromConfig:
    def test_from_config_basic(self):
        config = [
            {"check_type": "no_pii"},
            {"check_type": "max_tokens", "params": {"limit": 500}},
        ]
        judge = _make_mock_judge()
        guard = Guard.from_config(config, judge=judge)
        assert len(guard.checks) == 2
        assert guard.checks[0].check_type == "no_pii"
        assert guard.checks[1].params["limit"] == 500

    def test_from_config_with_soft(self):
        config = [
            {"check_type": "contains", "params": {"substring": "hi"}, "soft": True},
        ]
        guard = Guard.from_config(config)
        assert guard.checks[0].soft is True

    def test_from_config_validates(self):
        guard = Guard.from_config([{"check_type": "is_json"}])
        result = guard.validate('{"valid": true}')
        assert result.valid is True


# ---------------------------------------------------------------------------
# Guard.defaults
# ---------------------------------------------------------------------------


class TestGuardDefaults:
    def test_defaults_creates_guard(self):
        judge = _make_mock_judge(score=0.95)
        guard = Guard.defaults(judge=judge)
        assert len(guard.checks) == 3
        types = {s.check_type for s in guard.checks}
        assert "no_pii" in types
        assert "max_tokens" in types
        assert "toxicity" in types

    def test_defaults_validates(self):
        judge = _make_mock_judge(score=0.95)
        guard = Guard.defaults(judge=judge)
        result = guard.validate("Hello world, this is a safe output.")
        assert result.valid is True


# ---------------------------------------------------------------------------
# Unknown check type
# ---------------------------------------------------------------------------


class TestUnknownCheckType:
    def test_unknown_check_fails(self):
        guard = _make_guard([CheckSpec(check_type="nonexistent_check")])
        result = guard.validate("some output")
        assert result.valid is False
        assert "Unknown check type" in result.failed_checks[0].reasoning


# ---------------------------------------------------------------------------
# Predefined guards
# ---------------------------------------------------------------------------


class TestPredefinedGuards:
    def test_safety_guard_exists(self):
        assert len(safety_guard.checks) == 2
        types = {s.check_type for s in safety_guard.checks}
        assert "no_pii" in types
        assert "toxicity" in types

    def test_quality_guard_exists(self):
        assert len(quality_guard.checks) == 3
        types = {s.check_type for s in quality_guard.checks}
        assert "fluency" in types
        assert "coherence" in types
        assert "min_tokens" in types

    def test_rag_guard_exists(self):
        assert len(rag_guard.checks) == 2
        types = {s.check_type for s in rag_guard.checks}
        assert "hallucination" in types
        assert "relevance" in types


# ---------------------------------------------------------------------------
# guardrail decorator
# ---------------------------------------------------------------------------


class TestGuardrailDecorator:
    def test_sync_decorator_pass(self):
        @guardrail(checks=[CheckSpec(check_type="no_pii")])
        def generate() -> str:
            return "Safe text without PII"

        result = generate()
        assert result == "Safe text without PII"

    def test_sync_decorator_fail(self):
        @guardrail(checks=[CheckSpec(check_type="no_pii")])
        def generate() -> str:
            return "Email: user@example.com"

        with pytest.raises(GuardrailError):
            generate()

    @pytest.mark.asyncio
    async def test_async_decorator_pass(self):
        @guardrail(checks=[CheckSpec(check_type="no_pii")])
        async def agenerate() -> str:
            return "Safe text without PII"

        result = await agenerate()
        assert result == "Safe text without PII"

    @pytest.mark.asyncio
    async def test_async_decorator_fail(self):
        @guardrail(checks=[CheckSpec(check_type="no_pii")])
        async def agenerate() -> str:
            return "Call me at 555-123-4567"

        with pytest.raises(GuardrailError):
            await agenerate()

    def test_decorator_with_guard_instance(self):
        judge = _make_mock_judge(score=0.95)
        g = Guard(
            checks=[CheckSpec(check_type="no_pii"), CheckSpec(check_type="toxicity")],
            judge=judge,
        )

        @guardrail(guard=g)
        def generate() -> str:
            return "Safe output"

        result = generate()
        assert result == "Safe output"

    def test_decorator_defaults(self):
        """Decorator with no args uses Guard.defaults()."""
        judge = _make_mock_judge(score=0.95)

        @guardrail(guard=Guard.defaults(judge=judge))
        def generate() -> str:
            return "Short safe text"

        result = generate()
        assert result == "Short safe text"

    def test_decorator_preserves_function_name(self):
        @guardrail(checks=[CheckSpec(check_type="no_pii")])
        def my_function() -> str:
            return "hello"

        assert my_function.__name__ == "my_function"


# ---------------------------------------------------------------------------
# GuardrailMiddleware
# ---------------------------------------------------------------------------


class TestGuardrailMiddleware:
    """Test the ASGI middleware with a minimal ASGI app."""

    @staticmethod
    def _make_asgi_app(response_body: dict, status: int = 200):
        """Create a trivial ASGI app that returns a JSON body."""
        body_bytes = json.dumps(response_body).encode()

        async def app(scope: dict, receive, send) -> None:
            await send({
                "type": "http.response.start",
                "status": status,
                "headers": [
                    (b"content-type", b"application/json"),
                    (b"content-length", str(len(body_bytes)).encode()),
                ],
            })
            await send({
                "type": "http.response.body",
                "body": body_bytes,
            })

        return app

    @pytest.mark.asyncio
    async def test_middleware_valid_response(self):
        app = self._make_asgi_app({"output": "Hello safe world"})
        guard = _make_guard([CheckSpec(check_type="no_pii")])
        middleware = GuardrailMiddleware(app, guard=guard, response_field="output")

        sent_messages: list[dict] = []

        async def mock_receive():
            return {"type": "http.request", "body": b""}

        async def mock_send(message: dict):
            sent_messages.append(message)

        await middleware({"type": "http"}, mock_receive, mock_send)

        # Should have start + body
        assert len(sent_messages) == 2
        start = sent_messages[0]
        body = sent_messages[1]

        headers = dict(start["headers"])
        assert headers[b"x-checkllm-valid"] == b"true"
        assert b"x-checkllm-score" in headers

        # Body should be the original
        data = json.loads(body["body"])
        assert data["output"] == "Hello safe world"

    @pytest.mark.asyncio
    async def test_middleware_invalid_response_reject(self):
        app = self._make_asgi_app({"output": "Email: test@example.com"})
        guard = _make_guard([CheckSpec(check_type="no_pii")])
        middleware = GuardrailMiddleware(
            app, guard=guard, response_field="output", on_failure="reject",
        )

        sent_messages: list[dict] = []

        async def mock_receive():
            return {"type": "http.request", "body": b""}

        async def mock_send(message: dict):
            sent_messages.append(message)

        await middleware({"type": "http"}, mock_receive, mock_send)

        start = sent_messages[0]
        body = sent_messages[1]

        headers = dict(start["headers"])
        assert headers[b"x-checkllm-valid"] == b"false"
        assert start["status"] == 422

        data = json.loads(body["body"])
        assert "error" in data
        assert "details" in data

    @pytest.mark.asyncio
    async def test_middleware_flag_mode(self):
        """In flag mode, the original body is returned even on failure."""
        app = self._make_asgi_app({"output": "Email: test@example.com"})
        guard = _make_guard([CheckSpec(check_type="no_pii")])
        middleware = GuardrailMiddleware(
            app, guard=guard, response_field="output", on_failure="flag",
        )

        sent_messages: list[dict] = []

        async def mock_receive():
            return {"type": "http.request", "body": b""}

        async def mock_send(message: dict):
            sent_messages.append(message)

        await middleware({"type": "http"}, mock_receive, mock_send)

        start = sent_messages[0]
        body = sent_messages[1]

        headers = dict(start["headers"])
        assert headers[b"x-checkllm-valid"] == b"false"
        # In flag mode, status is 200 (original), body is original
        assert start["status"] == 200
        data = json.loads(body["body"])
        assert data["output"] == "Email: test@example.com"

    @pytest.mark.asyncio
    async def test_middleware_non_http_passthrough(self):
        """Non-HTTP scopes pass through untouched."""
        called = False

        async def app(scope, receive, send):
            nonlocal called
            called = True

        guard = _make_guard([CheckSpec(check_type="no_pii")])
        middleware = GuardrailMiddleware(app, guard=guard)

        await middleware({"type": "websocket"}, None, None)
        assert called is True

    @pytest.mark.asyncio
    async def test_middleware_missing_field_passthrough(self):
        """If the response_field is not in the JSON, pass through."""
        app = self._make_asgi_app({"other_field": "some value"})
        guard = _make_guard([CheckSpec(check_type="no_pii")])
        middleware = GuardrailMiddleware(app, guard=guard, response_field="output")

        sent_messages: list[dict] = []

        async def mock_receive():
            return {"type": "http.request", "body": b""}

        async def mock_send(message: dict):
            sent_messages.append(message)

        await middleware({"type": "http"}, mock_receive, mock_send)

        body = sent_messages[1]
        data = json.loads(body["body"])
        assert data["other_field"] == "some value"


# ---------------------------------------------------------------------------
# Guard with context kwargs (for hallucination/relevance)
# ---------------------------------------------------------------------------


class TestGuardWithContext:
    def test_hallucination_with_context(self):
        judge = _make_mock_judge(score=0.95)
        guard = _make_guard(
            [CheckSpec(check_type="hallucination")],
            judge=judge,
        )
        result = guard.validate(
            "Paris is in France.",
            context="Paris is the capital of France.",
        )
        assert result.valid is True

    def test_relevance_with_query(self):
        judge = _make_mock_judge(score=0.9)
        guard = _make_guard(
            [CheckSpec(check_type="relevance")],
            judge=judge,
        )
        result = guard.validate(
            "Python is a programming language.",
            query="What is Python?",
        )
        assert result.valid is True

    @pytest.mark.asyncio
    async def test_avalidate_with_context(self):
        judge = _make_mock_judge(score=0.9)
        guard = _make_guard(
            [CheckSpec(check_type="hallucination")],
            judge=judge,
        )
        result = await guard.avalidate(
            "Paris is in France.",
            context="Paris is the capital of France.",
        )
        assert result.valid is True


# ---------------------------------------------------------------------------
# Latency and cost tracking
# ---------------------------------------------------------------------------


class TestLatencyAndCost:
    def test_total_latency_is_set(self):
        guard = _make_guard([CheckSpec(check_type="no_pii")])
        result = guard.validate("hello")
        # Latency should be >= 0 (might be 0 for very fast deterministic checks)
        assert result.total_latency_ms >= 0

    def test_total_cost_deterministic_is_zero(self):
        guard = _make_guard([CheckSpec(check_type="no_pii")])
        result = guard.validate("hello")
        assert result.total_cost == 0.0

    def test_cost_tracked_from_judge(self):
        judge = _make_mock_judge(score=0.9)
        guard = _make_guard([CheckSpec(check_type="toxicity")], judge=judge)
        result = guard.validate("hello")
        # MockJudge returns 0 cost by default, which is correct
        assert result.total_cost >= 0.0
