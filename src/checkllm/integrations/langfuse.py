"""LangFuse tracer integration for checkllm.

Exports checkllm evaluation spans and judge calls to LangFuse via the
``langfuse`` SDK. Compatible with :class:`checkllm.tracing.Tracer`.

Usage::

    from checkllm.integrations.langfuse import LangFuseTracer

    tracer = LangFuseTracer()
    with tracer.span("evaluate", {"model": "gpt-4o"}):
        ...

Environment variables:
    ``LANGFUSE_PUBLIC_KEY`` — public API key.
    ``LANGFUSE_SECRET_KEY`` — secret API key.
    ``LANGFUSE_HOST`` — optional self-hosted endpoint URL.

Install with ``pip install checkllm[langfuse]``.
"""
from __future__ import annotations

import logging
import os
import uuid
from contextlib import contextmanager
from typing import Any, Generator

from checkllm.models import CheckResult
from checkllm.tracing import Span, Tracer

logger = logging.getLogger("checkllm.integrations.langfuse")

_LANGFUSE_INSTALL_HINT = (
    "LangFuse integration requires the 'langfuse' package. "
    "Install with: pip install checkllm[langfuse]"
)


def _import_langfuse() -> Any:
    """Import and return the langfuse Langfuse client class.

    Returns:
        The ``langfuse.Langfuse`` class.

    Raises:
        ImportError: If ``langfuse`` is not installed.
    """
    try:
        from langfuse import Langfuse

        return Langfuse
    except ImportError as exc:
        raise ImportError(_LANGFUSE_INSTALL_HINT) from exc


class LangFuseTracer(Tracer):
    """Tracer that exports spans to LangFuse.

    Args:
        public_key: LangFuse public key. Falls back to
            ``LANGFUSE_PUBLIC_KEY`` in the environment.
        secret_key: LangFuse secret key. Falls back to
            ``LANGFUSE_SECRET_KEY``.
        host: Optional LangFuse host URL for self-hosted deployments.
            Falls back to ``LANGFUSE_HOST``.
        client: Pre-configured ``langfuse.Langfuse`` instance, primarily
            for testing.
        enable_otel: Forwarded to the base tracer.
    """

    def __init__(
        self,
        public_key: str | None = None,
        secret_key: str | None = None,
        host: str | None = None,
        client: Any | None = None,
        enable_otel: bool = False,
    ) -> None:
        super().__init__(service_name="checkllm", enable_otel=enable_otel)

        if client is not None:
            self._client = client
        else:
            client_cls = _import_langfuse()
            kwargs: dict[str, Any] = {}
            pub = public_key or os.getenv("LANGFUSE_PUBLIC_KEY")
            sec = secret_key or os.getenv("LANGFUSE_SECRET_KEY")
            h = host or os.getenv("LANGFUSE_HOST")
            if pub:
                kwargs["public_key"] = pub
            if sec:
                kwargs["secret_key"] = sec
            if h:
                kwargs["host"] = h
            self._client = client_cls(**kwargs)

        self._trace: Any = None
        self._span_stack_ext: list[Any] = []

    def _ensure_trace(self) -> Any:
        """Lazily create the top-level LangFuse trace object."""
        if self._trace is None:
            try:
                self._trace = self._client.trace(
                    id=uuid.uuid4().hex, name="checkllm.evaluation"
                )
            except Exception as exc:  # pragma: no cover - network path
                logger.debug("langfuse trace creation failed: %s", exc)
                self._trace = None
        return self._trace

    @contextmanager
    def span(
        self, name: str, attributes: dict[str, Any] | None = None
    ) -> Generator[Span, None, None]:
        """Open a traced span and mirror it as a LangFuse span."""
        trace_obj = self._ensure_trace()
        parent = self._span_stack_ext[-1] if self._span_stack_ext else trace_obj

        ext_span: Any = None
        if parent is not None:
            try:
                ext_span = parent.span(
                    name=name, metadata=dict(attributes or {})
                )
            except Exception as exc:  # pragma: no cover - network path
                logger.debug("langfuse span start failed: %s", exc)
                ext_span = None

        if ext_span is not None:
            self._span_stack_ext.append(ext_span)

        error: BaseException | None = None
        try:
            with super().span(name, attributes) as local_span:
                yield local_span
        except BaseException as exc:
            error = exc
            raise
        finally:
            if ext_span is not None:
                self._span_stack_ext.pop()
                try:
                    if error is not None:
                        ext_span.end(
                            level="ERROR",
                            status_message=str(error),
                        )
                    else:
                        ext_span.end()
                except Exception as exc:  # pragma: no cover - network path
                    logger.debug("langfuse span end failed: %s", exc)

    def record_check(self, result: CheckResult) -> None:
        """Record a check result and mirror it as a LangFuse score."""
        super().record_check(result)

        target = (
            self._span_stack_ext[-1] if self._span_stack_ext else self._trace
        )
        if target is None:
            return
        try:
            target.score(
                name=result.metric_name,
                value=result.score,
                comment=result.reasoning,
            )
        except Exception as exc:  # pragma: no cover - network path
            logger.debug("langfuse score failed: %s", exc)

    def flush(self) -> None:
        """Flush pending events to the LangFuse backend."""
        try:
            self._client.flush()
        except Exception as exc:  # pragma: no cover - network path
            logger.debug("langfuse flush failed: %s", exc)
