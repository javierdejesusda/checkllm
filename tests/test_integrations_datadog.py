"""Tests for the Datadog tracer integration.

OpenTelemetry and ``ddtrace`` are stubbed so no network or SDK init
actually happens.
"""
from __future__ import annotations

import sys
import types
from unittest.mock import MagicMock

import pytest


@pytest.fixture
def patched_configure(monkeypatch):
    """Replace the real exporter configuration with capturing stubs."""
    import checkllm.integrations.datadog as dd

    calls: dict = {}

    def fake_otlp(
        endpoint, transport, service, env, version, headers
    ):
        calls["otlp"] = {
            "endpoint": endpoint,
            "transport": transport,
            "service": service,
            "env": env,
            "version": version,
            "headers": headers,
        }

    fake_dd_tracer = MagicMock(name="ddtrace.tracer")

    def fake_ddtrace(service, env, version, agent_url):
        calls["dd"] = {
            "service": service,
            "env": env,
            "version": version,
            "agent_url": agent_url,
        }
        return fake_dd_tracer

    monkeypatch.setattr(dd, "_configure_otlp_exporter", fake_otlp)
    monkeypatch.setattr(dd, "_configure_ddtrace", fake_ddtrace)
    return calls, fake_dd_tracer


def test_otlp_http_is_default(patched_configure):
    calls, _ = patched_configure
    from checkllm.integrations.datadog import DatadogTracer

    tracer = DatadogTracer(service="svc", env="prod", version="1.2.3")

    assert tracer.transport == "otlp_http"
    assert calls["otlp"]["endpoint"].endswith("/v1/traces")
    assert calls["otlp"]["service"] == "svc"
    assert calls["otlp"]["env"] == "prod"
    assert calls["otlp"]["version"] == "1.2.3"


def test_otlp_grpc_uses_gprc_port(patched_configure):
    calls, _ = patched_configure
    from checkllm.integrations.datadog import DatadogTracer

    DatadogTracer(transport="otlp_grpc")

    assert calls["otlp"]["transport"] == "otlp_grpc"
    assert calls["otlp"]["endpoint"].endswith(":4317")


def test_ddtrace_transport_sets_up_sdk(patched_configure):
    calls, fake_dd = patched_configure
    from checkllm.integrations.datadog import DatadogTracer

    tracer = DatadogTracer(transport="ddtrace", service="svc")

    assert tracer.transport == "ddtrace"
    assert tracer._dd_tracer is fake_dd
    assert calls["dd"]["service"] == "svc"


def test_endpoint_override_wins(patched_configure, monkeypatch):
    calls, _ = patched_configure
    monkeypatch.setenv("DD_OTLP_ENDPOINT", "http://env:4318/v1/traces")
    from checkllm.integrations.datadog import DatadogTracer

    DatadogTracer(endpoint="http://custom:9999/v1/traces")
    assert calls["otlp"]["endpoint"] == "http://custom:9999/v1/traces"


def test_env_var_fallback(patched_configure, monkeypatch):
    calls, _ = patched_configure
    monkeypatch.setenv("DD_OTLP_ENDPOINT", "http://env-agent:4318/v1/traces")
    monkeypatch.setenv("DD_ENV", "staging")
    monkeypatch.setenv("DD_VERSION", "9.9.9")
    from checkllm.integrations.datadog import DatadogTracer

    DatadogTracer()
    assert calls["otlp"]["endpoint"] == "http://env-agent:4318/v1/traces"
    assert calls["otlp"]["env"] == "staging"
    assert calls["otlp"]["version"] == "9.9.9"


def test_import_error_without_otel(monkeypatch):
    """Real ``_configure_otlp_exporter`` should raise a helpful error."""
    from checkllm.integrations.datadog import _configure_otlp_exporter

    monkeypatch.setitem(sys.modules, "opentelemetry", None)
    with pytest.raises(ImportError, match="Datadog integration"):
        _configure_otlp_exporter(
            endpoint="http://localhost:4318/v1/traces",
            transport="otlp_http",
            service="svc",
            env=None,
            version=None,
            headers=None,
        )


def test_import_error_without_ddtrace(monkeypatch):
    from checkllm.integrations.datadog import _configure_ddtrace

    monkeypatch.setitem(sys.modules, "ddtrace", None)
    with pytest.raises(ImportError, match="Datadog integration"):
        _configure_ddtrace(
            service="svc", env=None, version=None, agent_url=None
        )


def test_factory_returns_datadog_tracer(patched_configure):
    from checkllm.integrations import get_tracer
    from checkllm.integrations.datadog import DatadogTracer

    tracer = get_tracer("datadog", transport="otlp_http")
    assert isinstance(tracer, DatadogTracer)


def test_span_works_with_local_fallback(patched_configure):
    from checkllm.integrations.datadog import DatadogTracer

    tracer = DatadogTracer()
    with tracer.span("evaluate", {"k": "v"}) as span:
        assert span.name == "evaluate"


def _install_fake_opentelemetry(monkeypatch):
    """Install fake OpenTelemetry modules so _configure_otlp_exporter runs."""
    fake_trace = MagicMock(name="otel.trace")
    fake_resource_cls = MagicMock(name="Resource")
    fake_provider_cls = MagicMock(name="TracerProvider")
    fake_batch_cls = MagicMock(name="BatchSpanProcessor")
    fake_http_exporter_cls = MagicMock(name="HttpOTLPSpanExporter")
    fake_grpc_exporter_cls = MagicMock(name="GrpcOTLPSpanExporter")

    otel_mod = types.ModuleType("opentelemetry")
    otel_mod.trace = fake_trace
    sdk_mod = types.ModuleType("opentelemetry.sdk")
    resources_mod = types.ModuleType("opentelemetry.sdk.resources")
    resources_mod.Resource = fake_resource_cls
    trace_mod = types.ModuleType("opentelemetry.sdk.trace")
    trace_mod.TracerProvider = fake_provider_cls
    export_mod = types.ModuleType("opentelemetry.sdk.trace.export")
    export_mod.BatchSpanProcessor = fake_batch_cls
    exporter_http = types.ModuleType(
        "opentelemetry.exporter.otlp.proto.http.trace_exporter"
    )
    exporter_http.OTLPSpanExporter = fake_http_exporter_cls
    exporter_grpc = types.ModuleType(
        "opentelemetry.exporter.otlp.proto.grpc.trace_exporter"
    )
    exporter_grpc.OTLPSpanExporter = fake_grpc_exporter_cls

    for name, mod in {
        "opentelemetry": otel_mod,
        "opentelemetry.sdk": sdk_mod,
        "opentelemetry.sdk.resources": resources_mod,
        "opentelemetry.sdk.trace": trace_mod,
        "opentelemetry.sdk.trace.export": export_mod,
        "opentelemetry.exporter": types.ModuleType("opentelemetry.exporter"),
        "opentelemetry.exporter.otlp": types.ModuleType(
            "opentelemetry.exporter.otlp"
        ),
        "opentelemetry.exporter.otlp.proto": types.ModuleType(
            "opentelemetry.exporter.otlp.proto"
        ),
        "opentelemetry.exporter.otlp.proto.http": types.ModuleType(
            "opentelemetry.exporter.otlp.proto.http"
        ),
        "opentelemetry.exporter.otlp.proto.grpc": types.ModuleType(
            "opentelemetry.exporter.otlp.proto.grpc"
        ),
        "opentelemetry.exporter.otlp.proto.http.trace_exporter": exporter_http,
        "opentelemetry.exporter.otlp.proto.grpc.trace_exporter": exporter_grpc,
    }.items():
        monkeypatch.setitem(sys.modules, name, mod)

    return {
        "trace": fake_trace,
        "resource": fake_resource_cls,
        "provider": fake_provider_cls,
        "batch": fake_batch_cls,
        "http_exporter": fake_http_exporter_cls,
        "grpc_exporter": fake_grpc_exporter_cls,
    }


def test_configure_otlp_http_wires_exporter(monkeypatch):
    """_configure_otlp_exporter should register an HTTP exporter."""
    fakes = _install_fake_opentelemetry(monkeypatch)
    from checkllm.integrations.datadog import _configure_otlp_exporter

    _configure_otlp_exporter(
        endpoint="http://dd:4318/v1/traces",
        transport="otlp_http",
        service="svc",
        env="prod",
        version="2.0",
        headers={"X-DD": "1"},
    )
    fakes["http_exporter"].assert_called_once()
    fakes["resource"].create.assert_called_once()
    resource_attrs = fakes["resource"].create.call_args.args[0]
    assert resource_attrs["service.name"] == "svc"
    assert resource_attrs["deployment.environment"] == "prod"
    assert resource_attrs["service.version"] == "2.0"
    fakes["trace"].set_tracer_provider.assert_called_once()


def test_configure_otlp_grpc_wires_exporter(monkeypatch):
    fakes = _install_fake_opentelemetry(monkeypatch)
    from checkllm.integrations.datadog import _configure_otlp_exporter

    _configure_otlp_exporter(
        endpoint="http://dd:4317",
        transport="otlp_grpc",
        service="svc",
        env=None,
        version=None,
        headers=None,
    )
    fakes["grpc_exporter"].assert_called_once()
    fakes["http_exporter"].assert_not_called()


def test_configure_ddtrace_sets_tags(monkeypatch):
    """_configure_ddtrace should set service/env/version tags on the tracer."""
    fake_tracer = MagicMock(name="ddtrace.tracer")
    fake_mod = types.ModuleType("ddtrace")
    fake_mod.tracer = fake_tracer
    monkeypatch.setitem(sys.modules, "ddtrace", fake_mod)

    from checkllm.integrations.datadog import _configure_ddtrace

    result = _configure_ddtrace(
        service="svc", env="prod", version="1.0", agent_url="http://agent:8126"
    )
    assert result is fake_tracer
    fake_tracer.configure.assert_called_once_with(
        agent_url="http://agent:8126"
    )
    fake_tracer.set_tags.assert_called_once()
    tags = fake_tracer.set_tags.call_args.args[0]
    assert tags == {"service": "svc", "env": "prod", "version": "1.0"}


def test_configure_ddtrace_skips_configure_without_agent_url(monkeypatch):
    fake_tracer = MagicMock(name="ddtrace.tracer")
    fake_mod = types.ModuleType("ddtrace")
    fake_mod.tracer = fake_tracer
    monkeypatch.setitem(sys.modules, "ddtrace", fake_mod)

    from checkllm.integrations.datadog import _configure_ddtrace

    _configure_ddtrace(service="svc", env=None, version=None, agent_url=None)
    fake_tracer.configure.assert_not_called()
    # set_tags should still be called (even if with an empty dict).
    fake_tracer.set_tags.assert_called_once()


def test_dd_service_env_var_fallback(patched_configure, monkeypatch):
    calls, _ = patched_configure
    monkeypatch.setenv("DD_SERVICE", "svc-from-env")
    monkeypatch.delenv("DD_ENV", raising=False)
    from checkllm.integrations.datadog import DatadogTracer

    # Explicit service overrides env; passing "" should fall through.
    tracer = DatadogTracer(service="")
    assert tracer.service == "svc-from-env"
    assert calls["otlp"]["service"] == "svc-from-env"
