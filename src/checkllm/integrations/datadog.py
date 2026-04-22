"""Datadog OTLP exporter for checkllm.

Sends checkllm evaluation spans to a Datadog Agent over OTLP. By default
it targets ``http://localhost:4318/v1/traces`` (HTTP/protobuf) which is
the standard agent ingest endpoint.

Two transport modes are supported:

1. ``otlp_http`` — uses the OpenTelemetry HTTP exporter (default).
2. ``otlp_grpc`` — uses the OpenTelemetry gRPC exporter.
3. ``ddtrace`` — falls back to the Datadog ``ddtrace`` SDK when
   available. The SDK is initialised but not automatically patched; the
   caller owns integration patching.

Usage::

    from checkllm.integrations.datadog import DatadogTracer

    tracer = DatadogTracer(service="checkllm-evals", env="prod")
    with tracer.span("evaluate", {"model": "gpt-4o"}):
        ...

Environment variables:
    ``DD_AGENT_URL`` or ``DD_OTLP_ENDPOINT`` — agent ingest URL.
    ``DD_SERVICE``, ``DD_ENV``, ``DD_VERSION`` — standard Datadog tags.

Install with ``pip install checkllm[datadog]``.
"""
from __future__ import annotations

import logging
import os
from typing import Any

from checkllm.tracing import Tracer

logger = logging.getLogger("checkllm.integrations.datadog")

_DATADOG_INSTALL_HINT = (
    "Datadog integration requires OpenTelemetry exporters or 'ddtrace'. "
    "Install with: pip install checkllm[datadog]"
)


def _configure_otlp_exporter(
    endpoint: str,
    transport: str,
    service: str,
    env: str | None,
    version: str | None,
    headers: dict[str, str] | None,
) -> None:
    """Wire an OTLP exporter into the global OpenTelemetry provider.

    Args:
        endpoint: Full OTLP endpoint URL (e.g.
            ``http://localhost:4318/v1/traces``).
        transport: One of ``"otlp_http"`` or ``"otlp_grpc"``.
        service: Service name to tag spans with.
        env: Optional ``deployment.environment`` resource attribute.
        version: Optional ``service.version`` resource attribute.
        headers: Optional extra headers for the exporter.

    Raises:
        ImportError: If the required OpenTelemetry packages are missing.
    """
    try:
        from opentelemetry import trace as otel_trace
        from opentelemetry.sdk.resources import Resource
        from opentelemetry.sdk.trace import TracerProvider
        from opentelemetry.sdk.trace.export import BatchSpanProcessor
    except ImportError as exc:
        raise ImportError(_DATADOG_INSTALL_HINT) from exc

    if transport == "otlp_grpc":
        try:
            from opentelemetry.exporter.otlp.proto.grpc.trace_exporter import (
                OTLPSpanExporter as GrpcSpanExporter,
            )
        except ImportError as exc:
            raise ImportError(_DATADOG_INSTALL_HINT) from exc
        exporter: Any = GrpcSpanExporter(endpoint=endpoint, headers=headers)
    else:
        try:
            from opentelemetry.exporter.otlp.proto.http.trace_exporter import (
                OTLPSpanExporter as HttpSpanExporter,
            )
        except ImportError as exc:
            raise ImportError(_DATADOG_INSTALL_HINT) from exc
        exporter = HttpSpanExporter(endpoint=endpoint, headers=headers)

    attributes: dict[str, str] = {"service.name": service}
    if env:
        attributes["deployment.environment"] = env
    if version:
        attributes["service.version"] = version

    resource = Resource.create(attributes)
    provider = TracerProvider(resource=resource)
    provider.add_span_processor(BatchSpanProcessor(exporter))
    otel_trace.set_tracer_provider(provider)


def _configure_ddtrace(
    service: str, env: str | None, version: str | None, agent_url: str | None
) -> Any:
    """Configure the ``ddtrace`` SDK and return its tracer.

    Args:
        service: Datadog service name.
        env: Optional Datadog environment tag.
        version: Optional Datadog version tag.
        agent_url: Optional Datadog Agent URL.

    Returns:
        The configured ``ddtrace.tracer``.

    Raises:
        ImportError: If ``ddtrace`` is not installed.
    """
    try:
        from ddtrace import tracer as dd_tracer
    except ImportError as exc:
        raise ImportError(_DATADOG_INSTALL_HINT) from exc

    configure_kwargs: dict[str, Any] = {}
    if agent_url:
        configure_kwargs["agent_url"] = agent_url
    if configure_kwargs:
        dd_tracer.configure(**configure_kwargs)
    dd_tracer.set_tags(
        {
            k: v
            for k, v in {"service": service, "env": env, "version": version}.items()
            if v is not None
        }
    )
    return dd_tracer


class DatadogTracer(Tracer):
    """Tracer that exports spans to Datadog.

    Args:
        service: Datadog service tag for every emitted span.
        env: Optional deployment environment tag.
        version: Optional service version tag.
        transport: ``"otlp_http"`` (default), ``"otlp_grpc"``, or
            ``"ddtrace"``.
        endpoint: OTLP endpoint. Defaults to
            ``http://localhost:4318/v1/traces`` for HTTP or
            ``http://localhost:4317`` for gRPC.
        headers: Optional extra headers for OTLP transports.
    """

    def __init__(
        self,
        service: str = "checkllm",
        env: str | None = None,
        version: str | None = None,
        transport: str = "otlp_http",
        endpoint: str | None = None,
        headers: dict[str, str] | None = None,
    ) -> None:
        resolved_env = env or os.getenv("DD_ENV")
        resolved_version = version or os.getenv("DD_VERSION")
        resolved_service = service or os.getenv("DD_SERVICE") or "checkllm"

        if transport == "ddtrace":
            resolved_endpoint = endpoint or os.getenv("DD_AGENT_URL")
            self._dd_tracer = _configure_ddtrace(
                service=resolved_service,
                env=resolved_env,
                version=resolved_version,
                agent_url=resolved_endpoint,
            )
            super().__init__(service_name=resolved_service, enable_otel=False)
        else:
            default_endpoint = (
                "http://localhost:4318/v1/traces"
                if transport == "otlp_http"
                else "http://localhost:4317"
            )
            resolved_endpoint = (
                endpoint
                or os.getenv("DD_OTLP_ENDPOINT")
                or os.getenv("DD_AGENT_URL")
                or default_endpoint
            )
            _configure_otlp_exporter(
                endpoint=resolved_endpoint,
                transport=transport,
                service=resolved_service,
                env=resolved_env,
                version=resolved_version,
                headers=headers,
            )
            self._dd_tracer = None
            super().__init__(service_name=resolved_service, enable_otel=True)

        self.service = resolved_service
        self.env = resolved_env
        self.version = resolved_version
        self.transport = transport
        self.endpoint = resolved_endpoint
