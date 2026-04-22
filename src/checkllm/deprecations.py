"""Deprecation warning infrastructure for checkllm.

checkllm follows a staged deprecation policy:

1. A feature is marked deprecated with a warning that names the
   removal version (e.g., CheckllmRemovedIn5Warning).
2. The warning is emitted for at least two minor releases.
3. In the named major version, the feature is removed.
"""

from __future__ import annotations

import functools
import warnings
from typing import Any, Callable, TypeVar

F = TypeVar("F", bound=Callable[..., Any])


class CheckllmDeprecationWarning(DeprecationWarning):
    """Base class for all checkllm deprecation warnings."""


class CheckllmRemovedIn5Warning(CheckllmDeprecationWarning):
    """Warning for features that will be removed in checkllm 5.0."""


class CheckllmRemovedIn6Warning(CheckllmDeprecationWarning):
    """Warning for features that will be removed in checkllm 6.0."""


_REMOVAL_VERSION_MAP: dict[str, type[CheckllmDeprecationWarning]] = {
    "5.0": CheckllmRemovedIn5Warning,
    "6.0": CheckllmRemovedIn6Warning,
}


def deprecated(
    reason: str,
    removal_version: str,
    stacklevel: int = 2,
) -> Callable[[F], F]:
    """Decorator to mark a function or method as deprecated.

    Args:
        reason: Human-readable migration guidance.
        removal_version: The major version where this will be removed.
        stacklevel: How many frames to skip in the warning.
    """
    warning_cls = _REMOVAL_VERSION_MAP.get(
        removal_version,
        CheckllmDeprecationWarning,
    )

    def decorator(func: F) -> F:
        message = (
            f"{func.__qualname__} is deprecated and will be removed in "
            f"checkllm {removal_version}. {reason}"
        )

        @functools.wraps(func)
        def wrapper(*args: Any, **kwargs: Any) -> Any:
            warnings.warn(message, warning_cls, stacklevel=stacklevel)
            return func(*args, **kwargs)

        if func.__doc__:
            wrapper.__doc__ = f".. deprecated:: {removal_version}\n   {reason}\n\n{func.__doc__}"

        return wrapper  # type: ignore[return-value]

    return decorator
