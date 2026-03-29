"""Programmatic API for using checkllm outside of pytest.

Provides a standalone evaluation interface with sync/async support,
a builder pattern, and shorthand check parsing.

Usage::

    from checkllm.api import evaluate, check_output, Evaluator

    # Simple one-liner
    result = check_output("LLM output here", checks=["no_pii", "max_tokens:200"])

    # Full async evaluation
    results = await evaluate(
        output="The answer is 42.",
        checks=[
            {"type": "contains", "params": {"substring": "42"}},
            {"type": "no_pii"},
        ],
        threshold=0.8,
    )

    # Builder pattern
    evaluator = (
        Evaluator()
        .with_judge("openai", model="gpt-4o-mini")
        .with_threshold(0.8)
        .with_budget(5.0)
        .add_check("contains", substring="expected")
        .add_check("no_pii")
    )
    result = evaluator.run("LLM output here")
"""
from __future__ import annotations

import asyncio
from typing import Any

from checkllm.config import CheckllmConfig
from checkllm.guardrails import CheckSpec, Guard, ValidationResult
from checkllm.judge import JudgeBackend


# ---------------------------------------------------------------------------
# Shorthand parser
# ---------------------------------------------------------------------------


def parse_check_shorthand(s: str) -> dict[str, Any]:
    """Parse a shorthand check string into a check spec dict.

    Examples::

        parse_check_shorthand("no_pii")
        # -> {"type": "no_pii", "params": {}}

        parse_check_shorthand("max_tokens:200")
        # -> {"type": "max_tokens", "params": {"limit": 200}}

        parse_check_shorthand("contains:hello")
        # -> {"type": "contains", "params": {"substring": "hello"}}

        parse_check_shorthand("min_tokens:50")
        # -> {"type": "min_tokens", "params": {"minimum": 50}}
    """
    if ":" not in s:
        return {"type": s, "params": {}}

    check_type, raw_value = s.split(":", 1)

    # Map check types to their expected parameter names
    param_key_map: dict[str, str] = {
        "max_tokens": "limit",
        "min_tokens": "minimum",
        "contains": "substring",
        "not_contains": "substring",
        "starts_with": "prefix",
        "ends_with": "suffix",
        "regex": "pattern",
        "word_count": "maximum",
        "char_count": "maximum",
        "sentence_count": "maximum",
        "similarity": "reference",
        "language": "expected",
        "greater_than": "value",
        "less_than": "value",
    }

    param_name = param_key_map.get(check_type, "value")

    # Try to parse the value as a number
    value: Any
    try:
        value = int(raw_value)
    except ValueError:
        try:
            value = float(raw_value)
        except ValueError:
            value = raw_value

    return {"type": check_type, "params": {param_name: value}}


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _normalise_check(spec: dict[str, Any] | str) -> CheckSpec:
    """Convert a dict or shorthand string into a :class:`CheckSpec`."""
    if isinstance(spec, str):
        parsed = parse_check_shorthand(spec)
    else:
        parsed = spec
    return CheckSpec(
        check_type=parsed["type"],
        params=parsed.get("params", {}),
    )


_UNSET: object = object()


def _build_guard(
    checks: list[dict[str, Any] | str],
    judge: str | JudgeBackend | None = None,
    threshold: float | object = _UNSET,
    budget: float | None = None,
    config: CheckllmConfig | None = None,
    judge_kwargs: dict[str, Any] | None = None,
) -> Guard:
    """Construct a :class:`Guard` from the given parameters."""
    specs = [_normalise_check(c) for c in checks]

    # Build config
    cfg = config or CheckllmConfig()
    if threshold is not _UNSET:
        cfg = cfg.model_copy(update={"default_threshold": threshold})
    if budget is not None:
        cfg = cfg.model_copy(update={"budget": budget})

    # Resolve judge
    resolved_judge: JudgeBackend | None = None
    if isinstance(judge, str):
        cfg = cfg.model_copy(update={"judge_backend": judge})
        extra = judge_kwargs or {}
        if "model" in extra:
            cfg = cfg.model_copy(update={"judge_model": extra["model"]})
        # Judge will be lazily initialised by Guard._get_judge()
    elif judge is not None:
        resolved_judge = judge

    return Guard(checks=specs, judge=resolved_judge, config=cfg)


def _run_async(coro: Any) -> Any:
    """Run a coroutine from sync context, handling existing event loops."""
    try:
        loop = asyncio.get_running_loop()
    except RuntimeError:
        loop = None
    if loop and loop.is_running():
        import concurrent.futures

        with concurrent.futures.ThreadPoolExecutor() as pool:
            return pool.submit(asyncio.run, coro).result()
    return asyncio.run(coro)


# ---------------------------------------------------------------------------
# evaluate() - async
# ---------------------------------------------------------------------------


async def evaluate(
    output: str,
    checks: list[dict[str, Any] | str],
    judge: str | JudgeBackend = "openai",
    threshold: float = 0.8,
    budget: float | None = None,
) -> ValidationResult:
    """Evaluate *output* against a list of checks asynchronously.

    Parameters
    ----------
    output:
        The LLM output text to validate.
    checks:
        List of check specifications.  Each element is either a dict with
        ``type`` and optional ``params`` keys, or a shorthand string like
        ``"no_pii"`` or ``"max_tokens:200"``.
    judge:
        Judge backend name (``"openai"``, ``"anthropic"``) or a
        :class:`JudgeBackend` instance.
    threshold:
        Default pass/fail threshold for judge-based checks.
    budget:
        Maximum USD budget for judge API calls.  ``None`` means unlimited.

    Returns
    -------
    ValidationResult
        The combined validation result for all checks.
    """
    guard = _build_guard(
        checks=checks,
        judge=judge,
        threshold=threshold,
        budget=budget,
    )
    return await guard.avalidate(output)


# ---------------------------------------------------------------------------
# check_output() - sync wrapper
# ---------------------------------------------------------------------------


def check_output(
    output: str,
    checks: list[dict[str, Any] | str],
    judge: str | JudgeBackend = "openai",
    threshold: float = 0.8,
    budget: float | None = None,
) -> ValidationResult:
    """Synchronous wrapper around :func:`evaluate`.

    Accepts the same parameters and returns the same result.
    """
    return _run_async(
        evaluate(
            output=output,
            checks=checks,
            judge=judge,
            threshold=threshold,
            budget=budget,
        )
    )


# ---------------------------------------------------------------------------
# Evaluator - builder pattern
# ---------------------------------------------------------------------------


class Evaluator:
    """Builder-pattern evaluator for fluent configuration.

    Usage::

        evaluator = (
            Evaluator()
            .with_judge("openai", model="gpt-4o-mini")
            .with_threshold(0.8)
            .with_budget(5.0)
            .add_check("contains", substring="expected")
            .add_check("no_pii")
        )
        result = evaluator.run("LLM output here")
    """

    def __init__(self) -> None:
        self._judge: str | JudgeBackend | None = None
        self._judge_kwargs: dict[str, Any] = {}
        self._threshold: float | object = _UNSET
        self._budget: float | None = None
        self._config: CheckllmConfig | None = None
        self._checks: list[dict[str, Any]] = []

    # -- builder methods (return self for chaining) --

    def with_judge(self, backend: str | JudgeBackend, **kwargs: Any) -> Evaluator:
        """Set the judge backend.

        Parameters
        ----------
        backend:
            A backend name like ``"openai"`` / ``"anthropic"`` or a
            :class:`JudgeBackend` instance (including :class:`MockJudge`).
        **kwargs:
            Extra keyword arguments forwarded when the backend is a string
            (e.g. ``model="gpt-4o-mini"``).
        """
        self._judge = backend
        self._judge_kwargs = kwargs
        return self

    def with_threshold(self, threshold: float) -> Evaluator:
        """Set the default pass/fail threshold."""
        self._threshold = threshold
        return self

    def with_budget(self, budget: float) -> Evaluator:
        """Set the maximum USD budget for judge API calls."""
        self._budget = budget
        return self

    def with_config(self, config: CheckllmConfig) -> Evaluator:
        """Provide a full :class:`CheckllmConfig` instance."""
        self._config = config
        return self

    def add_check(self, check_type: str, **params: Any) -> Evaluator:
        """Add a check to the evaluation pipeline.

        Parameters
        ----------
        check_type:
            The check type name (e.g. ``"contains"``, ``"no_pii"``).
        **params:
            Parameters for the check (e.g. ``substring="hello"``).
        """
        self._checks.append({"type": check_type, "params": params})
        return self

    # -- internal --

    def _build_guard(self) -> Guard:
        """Construct a :class:`Guard` from the accumulated builder state."""
        return _build_guard(
            checks=self._checks,
            judge=self._judge,
            threshold=self._threshold,
            budget=self._budget,
            config=self._config,
            judge_kwargs=self._judge_kwargs,
        )

    # -- execution methods --

    def run(self, output: str) -> ValidationResult:
        """Run all checks synchronously and return the result."""
        guard = self._build_guard()
        return _run_async(guard.avalidate(output))

    async def arun(self, output: str) -> ValidationResult:
        """Run all checks asynchronously and return the result."""
        guard = self._build_guard()
        return await guard.avalidate(output)

    def batch_run(self, outputs: list[str]) -> list[ValidationResult]:
        """Run all checks against multiple outputs in parallel.

        Returns a list of :class:`ValidationResult`, one per output, in order.
        """

        async def _batch() -> list[ValidationResult]:
            guard = self._build_guard()
            tasks = [guard.avalidate(o) for o in outputs]
            return list(await asyncio.gather(*tasks))

        return _run_async(_batch())
