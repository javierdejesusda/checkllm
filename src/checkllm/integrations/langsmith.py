"""LangSmith tracer integration for checkllm.

Exports checkllm evaluation spans and judge runs to LangSmith via the
``langsmith`` SDK. Compatible with the :class:`checkllm.tracing.Tracer`
API and usable as a lightweight wrapper around judge calls.

Usage::

    from checkllm.integrations.langsmith import LangSmithTracer

    tracer = LangSmithTracer(project_name="my-evals")
    with tracer.span("evaluate", {"model": "gpt-4o"}) as span:
        ...

Environment variables:
    ``LANGSMITH_API_KEY`` (or ``LANGCHAIN_API_KEY``) — authentication.
    ``LANGSMITH_PROJECT`` — default project name.

Install with ``pip install checkllm[langsmith]``.
"""

from __future__ import annotations

import logging
import os
import time
import uuid
from contextlib import contextmanager
from typing import Any, Generator

from checkllm.models import CheckResult
from checkllm.tracing import Span, Tracer

logger = logging.getLogger("checkllm.integrations.langsmith")

_LANGSMITH_INSTALL_HINT = (
    "LangSmith integration requires the 'langsmith' package. "
    "Install with: pip install checkllm[langsmith]"
)


def _import_langsmith() -> Any:
    """Import and return the langsmith Client class.

    Returns:
        The ``langsmith.Client`` class.

    Raises:
        ImportError: If ``langsmith`` is not installed.
    """
    try:
        from langsmith import Client

        return Client
    except ImportError as exc:
        raise ImportError(_LANGSMITH_INSTALL_HINT) from exc


class LangSmithTracer(Tracer):
    """Tracer that exports spans to LangSmith.

    Wraps the base :class:`checkllm.tracing.Tracer` so existing code
    keeps working while runs are streamed to LangSmith.

    Args:
        project_name: Target LangSmith project. Falls back to the
            ``LANGSMITH_PROJECT`` environment variable.
        api_key: Optional explicit API key. Falls back to
            ``LANGSMITH_API_KEY``.
        api_url: Optional LangSmith endpoint override.
        client: Optional pre-configured ``langsmith.Client`` instance,
            primarily used for testing.
        enable_otel: Forwarded to the base tracer.
    """

    def __init__(
        self,
        project_name: str | None = None,
        api_key: str | None = None,
        api_url: str | None = None,
        client: Any | None = None,
        enable_otel: bool = False,
    ) -> None:
        super().__init__(service_name="checkllm", enable_otel=enable_otel)

        self.project_name = project_name or os.getenv("LANGSMITH_PROJECT") or "checkllm"

        if client is not None:
            self._client = client
        else:
            client_cls = _import_langsmith()
            kwargs: dict[str, Any] = {}
            resolved_key = (
                api_key or os.getenv("LANGSMITH_API_KEY") or os.getenv("LANGCHAIN_API_KEY")
            )
            if resolved_key:
                kwargs["api_key"] = resolved_key
            if api_url:
                kwargs["api_url"] = api_url
            self._client = client_cls(**kwargs)

        self._run_stack: list[dict[str, Any]] = []

    @contextmanager
    def span(
        self, name: str, attributes: dict[str, Any] | None = None
    ) -> Generator[Span, None, None]:
        """Open a traced span and mirror it as a LangSmith run."""
        run_id = uuid.uuid4()
        parent_run_id = self._run_stack[-1]["run_id"] if self._run_stack else None
        start_time = time.time()

        run_info: dict[str, Any] = {
            "run_id": run_id,
            "parent_run_id": parent_run_id,
            "name": name,
        }
        self._run_stack.append(run_info)

        try:
            self._client.create_run(
                id=run_id,
                name=name,
                run_type="chain",
                inputs=dict(attributes or {}),
                project_name=self.project_name,
                parent_run_id=parent_run_id,
                start_time=start_time,
            )
        except Exception as exc:  # pragma: no cover - network error path
            logger.debug("langsmith create_run failed: %s", exc)

        error: BaseException | None = None
        try:
            with super().span(name, attributes) as local_span:
                yield local_span
        except BaseException as exc:
            error = exc
            raise
        finally:
            self._run_stack.pop()
            end_time = time.time()
            outputs: dict[str, Any] = {"status": "error" if error else "ok"}
            try:
                self._client.update_run(
                    run_id=run_id,
                    end_time=end_time,
                    outputs=outputs,
                    error=str(error) if error else None,
                )
            except Exception as exc:  # pragma: no cover - network error path
                logger.debug("langsmith update_run failed: %s", exc)

    def record_check(self, result: CheckResult) -> None:
        """Record a check result and mirror it as a LangSmith feedback entry."""
        super().record_check(result)

        current_run = self._run_stack[-1]["run_id"] if self._run_stack else None
        if current_run is None:
            return
        try:
            self._client.create_feedback(
                run_id=current_run,
                key=result.metric_name,
                score=result.score,
                comment=result.reasoning,
                value=result.passed,
            )
        except Exception as exc:  # pragma: no cover - network error path
            logger.debug("langsmith create_feedback failed: %s", exc)
