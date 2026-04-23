"""Check registry and ``@check`` decorator.

This module provides the ``@check`` decorator (the check-side equivalent of
``@metric`` in :mod:`checkllm.metrics`). Decorated callables are stored in a
process-global :data:`CHECK_REGISTRY` together with metadata (``name``,
``description``, ``tags``) and can be invoked by name via
:func:`run_check`.

Composition primitives — :class:`AllOf`, :class:`AnyOf`, :class:`Not` —
combine registered or ad-hoc check callables into new check objects with
aggregated rationale. Composites are themselves callables, which means they
can be nested (``AnyOf(Not(c1), AllOf(c2, c3))``).

The decorator is additive: the 39 deterministic checks on
:class:`checkllm.deterministic.DeterministicChecks` continue to work with
their original signatures; they are registered into :data:`CHECK_REGISTRY`
on import so they are discoverable by name.
"""

from __future__ import annotations

import logging
from collections.abc import Callable, Iterable
from dataclasses import dataclass, field
from typing import Any

from checkllm.models import CheckResult

logger = logging.getLogger("checkllm.checks")


CheckCallable = Callable[..., CheckResult]


@dataclass
class RegisteredCheck:
    """Registry entry for a single check.

    Named ``RegisteredCheck`` to avoid colliding with
    :class:`checkllm.guardrails.CheckSpec`, which is a different concept
    (guardrail configuration).
    """

    name: str
    func: CheckCallable
    description: str = ""
    tags: tuple[str, ...] = field(default_factory=tuple)
    source: str = "builtin"

    def __call__(self, *args: Any, **kwargs: Any) -> CheckResult:
        return self.func(*args, **kwargs)


class CheckRegistry:
    """Process-global registry for check callables."""

    def __init__(self) -> None:
        self._specs: dict[str, RegisteredCheck] = {}

    def register(
        self,
        name: str,
        func: CheckCallable,
        *,
        description: str = "",
        tags: Iterable[str] = (),
        source: str = "builtin",
        overwrite: bool = False,
    ) -> RegisteredCheck:
        """Register a check by name.

        Args:
            name: Unique name used to look up the check.
            func: The callable returning a :class:`CheckResult`.
            description: Human-readable blurb (defaults to ``func.__doc__``).
            tags: Free-form string tags for filtering.
            source: Origin string (``builtin``, ``user``, ``plugin:...``).
            overwrite: Allow replacing an existing entry. Defaults to False
                so accidental collisions fail loudly.
        """
        if not overwrite and name in self._specs:
            raise ValueError(f"Check '{name}' is already registered.")
        if description:
            desc = description
        elif func.__doc__:
            desc = func.__doc__.strip().splitlines()[0]
        else:
            desc = ""
        spec = RegisteredCheck(
            name=name,
            func=func,
            description=desc,
            tags=tuple(tags),
            source=source,
        )
        self._specs[name] = spec
        return spec

    def unregister(self, name: str) -> None:
        """Remove a check from the registry. No-op if absent."""
        self._specs.pop(name, None)

    def get(self, name: str) -> RegisteredCheck:
        """Return the spec for ``name``. Raises :class:`KeyError` if missing."""
        if name not in self._specs:
            raise KeyError(f"Check '{name}' is not registered.")
        return self._specs[name]

    def __contains__(self, name: object) -> bool:
        return isinstance(name, str) and name in self._specs

    def __iter__(self):
        return iter(self._specs.values())

    def __len__(self) -> int:
        return len(self._specs)

    def names(self) -> list[str]:
        return sorted(self._specs)

    def list_detailed(self) -> list[dict[str, Any]]:
        return [
            {
                "name": s.name,
                "description": s.description,
                "tags": list(s.tags),
                "source": s.source,
            }
            for s in self._specs.values()
        ]

    def filter_by_tag(self, tag: str) -> list[RegisteredCheck]:
        return [s for s in self._specs.values() if tag in s.tags]


CHECK_REGISTRY = CheckRegistry()


def check(
    name: str | None = None,
    *,
    description: str = "",
    tags: Iterable[str] = (),
    source: str = "user",
) -> Callable[[CheckCallable], CheckCallable]:
    """Decorator that registers a check callable in :data:`CHECK_REGISTRY`.

    Usage::

        @check("has_greeting", tags=("format",))
        def has_greeting(output: str) -> CheckResult:
            passed = output.lower().startswith("hello")
            return CheckResult(passed=passed, score=1.0 if passed else 0.0,
                               reasoning="...", cost=0.0, latency_ms=0,
                               metric_name="has_greeting")

    When ``name`` is omitted, the function's ``__name__`` is used.
    """

    def decorator(func: CheckCallable) -> CheckCallable:
        final_name = name or func.__name__
        CHECK_REGISTRY.register(
            final_name,
            func,
            description=description,
            tags=tags,
            source=source,
        )
        # Stamp metadata onto the function for introspection.
        func.__checkllm_check_name__ = final_name  # type: ignore[attr-defined]
        func.__checkllm_check_tags__ = tuple(tags)  # type: ignore[attr-defined]
        return func

    return decorator


def run_check(name: str, *args: Any, **kwargs: Any) -> CheckResult:
    """Invoke a registered check by name."""
    return CHECK_REGISTRY.get(name)(*args, **kwargs)


def _resolve(check_obj: CheckCallable | str) -> CheckCallable:
    """Resolve a check reference (string name or callable) to a callable."""
    if callable(check_obj):
        return check_obj
    if isinstance(check_obj, str):  # type: ignore[unreachable]
        return CHECK_REGISTRY.get(check_obj).func
    raise TypeError(f"Expected callable or check name, got {type(check_obj).__name__}")


def _describe(check_obj: CheckCallable | str) -> str:
    if isinstance(check_obj, str):
        return check_obj
    name = getattr(check_obj, "__checkllm_check_name__", None)
    if name:
        return name
    return getattr(check_obj, "__name__", repr(check_obj))


@dataclass
class _Composite:
    """Base class for composite checks. Not instantiated directly."""

    members: tuple[CheckCallable | str, ...]
    metric_name: str = "composite"

    def _run_members(self, *args: Any, **kwargs: Any) -> list[CheckResult]:
        results: list[CheckResult] = []
        for member in self.members:
            func = _resolve(member)
            results.append(func(*args, **kwargs))
        return results


class AllOf(_Composite):
    """Composite check that passes iff every member passes.

    Example::

        is_safe_and_formatted = AllOf(my_no_pii_check, my_is_json_check)
        result = is_safe_and_formatted(output)
    """

    def __init__(
        self,
        *checks: CheckCallable | str,
        name: str = "all_of",
    ) -> None:
        super().__init__(members=checks, metric_name=name)

    def __call__(self, *args: Any, **kwargs: Any) -> CheckResult:
        results = self._run_members(*args, **kwargs)
        passed = all(r.passed for r in results)
        scores = [r.score for r in results]
        score = min(scores) if scores else 0.0
        failed_names = [r.metric_name for r in results if not r.passed]
        if passed:
            reasoning = f"AllOf({', '.join(_describe(m) for m in self.members)}): all passed"
        else:
            reasoning = "AllOf failed on: " + ", ".join(failed_names)
        # Pass-through cost/latency accumulation.
        cost = sum(r.cost for r in results)
        latency = sum(r.latency_ms for r in results)
        return CheckResult(
            passed=passed,
            score=score,
            reasoning=reasoning,
            cost=cost,
            latency_ms=latency,
            metric_name=self.metric_name,
        )


class AnyOf(_Composite):
    """Composite check that passes iff any member passes."""

    def __init__(
        self,
        *checks: CheckCallable | str,
        name: str = "any_of",
    ) -> None:
        super().__init__(members=checks, metric_name=name)

    def __call__(self, *args: Any, **kwargs: Any) -> CheckResult:
        results = self._run_members(*args, **kwargs)
        passed = any(r.passed for r in results)
        scores = [r.score for r in results]
        score = max(scores) if scores else 0.0
        passing_names = [r.metric_name for r in results if r.passed]
        if passed:
            reasoning = "AnyOf passed via: " + ", ".join(passing_names)
        else:
            reasoning = f"AnyOf({', '.join(_describe(m) for m in self.members)}): none passed"
        cost = sum(r.cost for r in results)
        latency = sum(r.latency_ms for r in results)
        return CheckResult(
            passed=passed,
            score=score,
            reasoning=reasoning,
            cost=cost,
            latency_ms=latency,
            metric_name=self.metric_name,
        )


class Not:
    """Composite check that inverts another check's pass/fail and score."""

    def __init__(self, check_obj: CheckCallable | str, *, name: str = "not") -> None:
        self._check = check_obj
        self.metric_name = name

    def __call__(self, *args: Any, **kwargs: Any) -> CheckResult:
        inner = _resolve(self._check)(*args, **kwargs)
        passed = not inner.passed
        score = 1.0 - inner.score
        reasoning = f"Not({_describe(self._check)}): " + (
            "inner failed -> pass" if passed else "inner passed -> fail"
        )
        if inner.reasoning:
            reasoning += f" | inner: {inner.reasoning}"
        return CheckResult(
            passed=passed,
            score=max(0.0, min(1.0, score)),
            reasoning=reasoning,
            cost=inner.cost,
            latency_ms=inner.latency_ms,
            metric_name=self.metric_name,
        )


__all__ = [
    "AllOf",
    "AnyOf",
    "CHECK_REGISTRY",
    "CheckCallable",
    "CheckRegistry",
    "Not",
    "RegisteredCheck",
    "check",
    "run_check",
]
