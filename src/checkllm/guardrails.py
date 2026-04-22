"""Runtime validation / guardrails for LLM outputs.

Use ``Guard`` in production code to validate LLM outputs at runtime,
independent of the test suite.

Usage::

    from checkllm.guardrails import Guard, CheckSpec

    guard = Guard([
        CheckSpec(check_type="no_pii"),
        CheckSpec(check_type="max_tokens", params={"limit": 4096}),
        CheckSpec(check_type="toxicity"),
    ])

    result = guard.validate(llm_output)
    result.raise_on_failure()

    # Or use as a callable:
    safe_output = guard(llm_output)  # raises GuardrailError if invalid
"""

from __future__ import annotations

import asyncio
import functools
import inspect
import json
import logging
import time
from typing import Any, Callable

from pydantic import BaseModel, Field

from checkllm.config import CheckllmConfig
from checkllm.deterministic import DeterministicChecks
from checkllm.judge import JudgeBackend
from checkllm.metrics.coherence import CoherenceMetric
from checkllm.metrics.fluency import FluencyMetric
from checkllm.metrics.hallucination import HallucinationMetric
from checkllm.metrics.relevance import RelevanceMetric
from checkllm.metrics.toxicity import ToxicityMetric
from checkllm.models import CheckResult

logger = logging.getLogger("checkllm.guardrails")

# ---------------------------------------------------------------------------
# Models
# ---------------------------------------------------------------------------


class ValidationResult(BaseModel):
    """Outcome of a ``Guard.validate`` call."""

    valid: bool
    results: list[CheckResult] = Field(default_factory=list)
    failed_checks: list[CheckResult] = Field(default_factory=list)
    total_latency_ms: int = 0
    total_cost: float = 0.0

    def raise_on_failure(self) -> None:
        """Raise :class:`GuardrailError` if any non-soft check failed."""
        if not self.valid:
            raise GuardrailError(self)

    def summary(self) -> str:
        """Return a human-readable summary of validation results."""
        passed = sum(1 for r in self.results if r.passed)
        lines = [
            f"Validation: {'PASSED' if self.valid else 'FAILED'} "
            f"({passed}/{len(self.results)} checks passed)",
        ]
        if self.failed_checks:
            lines.append("Failed checks:")
            for r in self.failed_checks:
                lines.append(f"  - {r.metric_name}: {r.reasoning} (score={r.score:.2f})")
        lines.append(f"Latency: {self.total_latency_ms}ms | Cost: ${self.total_cost:.4f}")
        return "\n".join(lines)


class GuardrailError(Exception):
    """Raised when guardrail validation fails."""

    def __init__(self, validation_result: ValidationResult) -> None:
        self.validation_result = validation_result
        failed_names = ", ".join(r.metric_name for r in validation_result.failed_checks)
        super().__init__(f"Guardrail validation failed: {failed_names}")


class CheckSpec(BaseModel):
    """Specification for a single check to run within a Guard."""

    check_type: str
    params: dict[str, Any] = Field(default_factory=dict)
    soft: bool = False


# ---------------------------------------------------------------------------
# Deterministic check types (no judge needed)
# ---------------------------------------------------------------------------

_DETERMINISTIC_CHECKS: set[str] = {
    "contains",
    "not_contains",
    "max_tokens",
    "min_tokens",
    "word_count",
    "char_count",
    "regex",
    "exact_match",
    "starts_with",
    "ends_with",
    "similarity",
    "readability",
    "sentence_count",
    "all_of",
    "any_of",
    "none_of",
    "is_json",
    "is_valid_python",
    "no_pii",
    "language",
    "greater_than",
    "less_than",
    "between",
}

# Judge-based check types and the metric class + required kwargs mapping
_JUDGE_CHECKS: dict[str, type] = {
    "toxicity": ToxicityMetric,
    "hallucination": HallucinationMetric,
    "relevance": RelevanceMetric,
    "fluency": FluencyMetric,
    "coherence": CoherenceMetric,
}


# ---------------------------------------------------------------------------
# Guard
# ---------------------------------------------------------------------------


class Guard:
    """Runtime guard that validates LLM outputs against a list of checks.

    Supports deterministic checks (instant, no API calls) and LLM-judge
    checks (requires a ``JudgeBackend``).
    """

    def __init__(
        self,
        checks: list[CheckSpec],
        judge: JudgeBackend | None = None,
        config: CheckllmConfig | None = None,
    ) -> None:
        self.checks = checks
        self._judge = judge
        self._config = config or CheckllmConfig()
        self._det = DeterministicChecks()

    # -- helpers --

    def _run_async(self, coro: Any) -> Any:
        """Run an async coroutine from sync context."""
        try:
            loop = asyncio.get_running_loop()
        except RuntimeError:
            loop = None
        if loop and loop.is_running():
            import concurrent.futures

            with concurrent.futures.ThreadPoolExecutor() as pool:
                return pool.submit(asyncio.run, coro).result()
        return asyncio.run(coro)

    def _get_judge(self) -> JudgeBackend:
        if self._judge is not None:
            return self._judge
        # Lazy-init from config
        if self._config.judge_backend == "anthropic":
            from checkllm.judge import AnthropicJudge

            self._judge = AnthropicJudge(model=self._config.judge_model)
        else:
            from checkllm.judge import OpenAIJudge

            self._judge = OpenAIJudge(model=self._config.judge_model)
        return self._judge

    def _run_deterministic(self, spec: CheckSpec, output: str, **context: Any) -> CheckResult:
        method = getattr(self._det, spec.check_type)
        kwargs: dict[str, Any] = {}
        # The first param is always 'self' (bound), second is 'output'
        kwargs["output"] = output
        kwargs.update(spec.params)
        return method(**kwargs)

    async def _run_judge_check(self, spec: CheckSpec, output: str, **context: Any) -> CheckResult:
        judge = self._get_judge()
        metric_cls = _JUDGE_CHECKS[spec.check_type]
        threshold = spec.params.get("threshold", self._config.default_threshold)
        metric = metric_cls(judge=judge, threshold=threshold)

        # Build evaluate kwargs from spec.params + context
        eval_kwargs: dict[str, Any] = {"output": output}
        # Merge context kwargs (e.g. query, context) from **context and spec.params
        for key, value in {**spec.params, **context}.items():
            if key == "threshold":
                continue
            eval_kwargs[key] = value

        return await metric.evaluate(**eval_kwargs)

    # -- public API --

    async def avalidate(self, output: str, **context: Any) -> ValidationResult:
        """Validate *output* asynchronously against all configured checks."""
        start = time.perf_counter_ns()
        results: list[CheckResult] = []
        failed: list[CheckResult] = []
        valid = True

        for spec in self.checks:
            if spec.check_type in _DETERMINISTIC_CHECKS:
                result = self._run_deterministic(spec, output, **context)
            elif spec.check_type in _JUDGE_CHECKS:
                result = await self._run_judge_check(spec, output, **context)
            else:
                # Unknown check type - fail with a descriptive message
                result = CheckResult(
                    passed=False,
                    score=0.0,
                    reasoning=f"Unknown check type: {spec.check_type}",
                    cost=0.0,
                    latency_ms=0,
                    metric_name=spec.check_type,
                )

            results.append(result)
            if not result.passed:
                failed.append(result)
                if not spec.soft:
                    valid = False

        elapsed_ms = (time.perf_counter_ns() - start) // 1_000_000
        total_cost = sum(r.cost for r in results)

        return ValidationResult(
            valid=valid,
            results=results,
            failed_checks=failed,
            total_latency_ms=int(elapsed_ms),
            total_cost=total_cost,
        )

    def validate(self, output: str, **context: Any) -> ValidationResult:
        """Validate *output* synchronously against all configured checks."""
        return self._run_async(self.avalidate(output, **context))

    def __call__(self, output: str, **context: Any) -> str:
        """Validate and return *output* if valid; raise :class:`GuardrailError` otherwise."""
        result = self.validate(output, **context)
        result.raise_on_failure()
        return output

    # -- class methods --

    @classmethod
    def from_config(cls, checks_config: list[dict[str, Any]], **kwargs: Any) -> Guard:
        """Create a Guard from a list of plain dicts.

        Each dict should have ``check_type`` and optionally ``params`` and ``soft``.
        """
        specs = [CheckSpec(**cfg) for cfg in checks_config]
        return cls(checks=specs, **kwargs)

    @classmethod
    def defaults(cls, **kwargs: Any) -> Guard:
        """Create a Guard with sensible default checks.

        Defaults: ``no_pii``, ``max_tokens(4096)``, ``toxicity``.
        """
        specs = [
            CheckSpec(check_type="no_pii"),
            CheckSpec(check_type="max_tokens", params={"limit": 4096}),
            CheckSpec(check_type="toxicity"),
        ]
        return cls(checks=specs, **kwargs)


# ---------------------------------------------------------------------------
# Predefined guard instances
# ---------------------------------------------------------------------------

safety_guard = Guard(
    checks=[
        CheckSpec(check_type="no_pii"),
        CheckSpec(check_type="toxicity"),
    ],
)

quality_guard = Guard(
    checks=[
        CheckSpec(check_type="fluency"),
        CheckSpec(check_type="coherence"),
        CheckSpec(check_type="min_tokens", params={"minimum": 10}),
    ],
)

rag_guard = Guard(
    checks=[
        CheckSpec(check_type="hallucination"),
        CheckSpec(check_type="relevance"),
    ],
)


# ---------------------------------------------------------------------------
# FastAPI / ASGI middleware
# ---------------------------------------------------------------------------


class GuardrailMiddleware:
    """ASGI middleware that validates LLM outputs in JSON response bodies.

    Intercepts JSON responses, validates the field specified by
    ``response_field``, and adds validation headers.  On failure it can
    optionally return a 422 response with validation details.

    Usage with FastAPI::

        from checkllm.guardrails import GuardrailMiddleware, Guard, CheckSpec

        guard = Guard([CheckSpec(check_type="no_pii")])
        app.add_middleware(GuardrailMiddleware, guard=guard)
    """

    def __init__(
        self,
        app: Any,
        guard: Guard,
        response_field: str = "output",
        fail_status: int = 422,
        on_failure: str = "reject",  # "reject" or "flag"
    ) -> None:
        self.app = app
        self.guard = guard
        self.response_field = response_field
        self.fail_status = fail_status
        self.on_failure = on_failure

    async def __call__(self, scope: dict, receive: Callable, send: Callable) -> None:
        if scope["type"] != "http":
            await self.app(scope, receive, send)
            return

        response_started = False
        response_headers: list[tuple[bytes, bytes]] = []
        body_parts: list[bytes] = []

        # Collect the full response body before sending
        async def capture_send(message: dict) -> None:
            nonlocal response_started, response_headers
            if message["type"] == "http.response.start":
                response_started = True
                response_headers = list(message.get("headers", []))
                # Buffer - don't send yet
                return
            if message["type"] == "http.response.body":
                body_parts.append(message.get("body", b""))
                # Only process when 'more_body' is False or not present
                if message.get("more_body", False):
                    return

        await self.app(scope, receive, capture_send)

        # Reconstruct full body
        full_body = b"".join(body_parts)

        # Try to parse as JSON and validate
        valid_header = b"false"
        score_header = b"0.0"
        final_body = full_body
        final_status_code: int | None = None

        try:
            data = json.loads(full_body)
            if isinstance(data, dict) and self.response_field in data:
                output_text = str(data[self.response_field])
                validation = await self.guard.avalidate(output_text)
                valid_header = b"true" if validation.valid else b"false"
                avg_score = (
                    sum(r.score for r in validation.results) / len(validation.results)
                    if validation.results
                    else 0.0
                )
                score_header = f"{avg_score:.2f}".encode()

                if not validation.valid and self.on_failure == "reject":
                    error_body = json.dumps(
                        {
                            "error": "Guardrail validation failed",
                            "details": [
                                {
                                    "check": r.metric_name,
                                    "score": r.score,
                                    "reasoning": r.reasoning,
                                }
                                for r in validation.failed_checks
                            ],
                        }
                    ).encode()
                    final_body = error_body
                    final_status_code = self.fail_status
        except (json.JSONDecodeError, Exception):
            # Not JSON or validation error - pass through
            pass

        # Add checkllm headers
        response_headers.append((b"x-checkllm-valid", valid_header))
        response_headers.append((b"x-checkllm-score", score_header))

        # Update content-length if body changed
        new_headers: list[tuple[bytes, bytes]] = []
        for name, value in response_headers:
            if name.lower() == b"content-length":
                new_headers.append((name, str(len(final_body)).encode()))
            else:
                new_headers.append((name, value))
        # Ensure content-length is present
        if not any(n.lower() == b"content-length" for n, _ in new_headers):
            new_headers.append((b"content-length", str(len(final_body)).encode()))

        # Determine status code
        start_message: dict[str, Any] = {
            "type": "http.response.start",
            "headers": new_headers,
        }
        if final_status_code is not None:
            start_message["status"] = final_status_code
        else:
            # Preserve original status
            start_message["status"] = 200

        await send(start_message)
        await send(
            {
                "type": "http.response.body",
                "body": final_body,
            }
        )


# ---------------------------------------------------------------------------
# Decorator
# ---------------------------------------------------------------------------


def guardrail(
    checks: list[CheckSpec] | None = None,
    guard: Guard | None = None,
) -> Callable:
    """Decorator that validates the string return value of a function.

    Usage::

        @guardrail(checks=[CheckSpec(check_type="no_pii")])
        def generate(prompt: str) -> str:
            return call_llm(prompt)

        @guardrail(guard=my_guard)
        async def agenerate(prompt: str) -> str:
            return await call_llm(prompt)
    """
    if guard is None and checks is None:
        _guard = Guard.defaults()
    elif guard is not None:
        _guard = guard
    else:
        _guard = Guard(checks=checks)  # type: ignore[arg-type]

    def decorator(fn: Callable) -> Callable:
        if inspect.iscoroutinefunction(fn):

            @functools.wraps(fn)
            async def async_wrapper(*args: Any, **kwargs: Any) -> str:
                result = await fn(*args, **kwargs)
                validation = await _guard.avalidate(str(result))
                if not validation.valid:
                    raise GuardrailError(validation)
                return result

            return async_wrapper
        else:

            @functools.wraps(fn)
            def sync_wrapper(*args: Any, **kwargs: Any) -> str:
                result = fn(*args, **kwargs)
                validation = _guard.validate(str(result))
                if not validation.valid:
                    raise GuardrailError(validation)
                return result

            return sync_wrapper

    return decorator
