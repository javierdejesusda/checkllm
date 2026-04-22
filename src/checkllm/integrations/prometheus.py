"""Prometheus / Grafana exporter for checkllm metrics.

Exposes standard evaluation telemetry as Prometheus counters and
histograms. The exporter can run its own HTTP endpoint or register
metrics against an externally supplied ``CollectorRegistry`` so the app
can scrape them from its own ``/metrics`` route.

Metrics emitted:

- ``checkllm_evaluations_total`` (Counter, labels: ``metric``, ``status``)
- ``checkllm_evaluation_duration_seconds`` (Histogram, labels: ``metric``)
- ``checkllm_judge_cost_usd_total`` (Counter, labels: ``metric``, ``judge``)
- ``checkllm_judge_tokens_total`` (Counter, labels: ``metric``, ``judge``,
  ``kind``)

Usage::

    from checkllm.integrations.prometheus import PrometheusExporter

    exporter = PrometheusExporter(port=9464)
    exporter.start_http_server()
    exporter.record_check(check_result, judge="gpt-4o")

Install with ``pip install checkllm[prometheus]``.
"""
from __future__ import annotations

import logging
from typing import Any

from checkllm.models import CheckResult

logger = logging.getLogger("checkllm.integrations.prometheus")

_PROMETHEUS_INSTALL_HINT = (
    "Prometheus integration requires the 'prometheus-client' package. "
    "Install with: pip install checkllm[prometheus]"
)


def _import_prometheus() -> Any:
    """Import and return the ``prometheus_client`` module.

    Returns:
        The imported ``prometheus_client`` module.

    Raises:
        ImportError: If ``prometheus_client`` is not installed.
    """
    try:
        import prometheus_client

        return prometheus_client
    except ImportError as exc:
        raise ImportError(_PROMETHEUS_INSTALL_HINT) from exc


class PrometheusExporter:
    """Prometheus metrics exporter for checkllm.

    Args:
        namespace: Metric name prefix. Defaults to ``"checkllm"``.
        registry: Optional pre-existing ``CollectorRegistry``. When
            omitted a fresh registry is created so the exporter does not
            pollute the global default.
        port: HTTP port used by :meth:`start_http_server`.
        histogram_buckets: Optional explicit duration histogram buckets
            in seconds.
    """

    def __init__(
        self,
        namespace: str = "checkllm",
        registry: Any | None = None,
        port: int = 9464,
        histogram_buckets: tuple[float, ...] | None = None,
    ) -> None:
        prom = _import_prometheus()

        self._prom = prom
        self.namespace = namespace
        self.port = port
        self.registry = registry or prom.CollectorRegistry()

        buckets = histogram_buckets or (
            0.005,
            0.01,
            0.025,
            0.05,
            0.1,
            0.25,
            0.5,
            1.0,
            2.5,
            5.0,
            10.0,
            30.0,
            60.0,
        )

        self.evaluations_total = prom.Counter(
            f"{namespace}_evaluations_total",
            "Total number of checkllm evaluations executed.",
            labelnames=("metric", "status"),
            registry=self.registry,
        )
        self.evaluation_duration_seconds = prom.Histogram(
            f"{namespace}_evaluation_duration_seconds",
            "Duration of checkllm evaluations in seconds.",
            labelnames=("metric",),
            buckets=buckets,
            registry=self.registry,
        )
        self.judge_cost_usd_total = prom.Counter(
            f"{namespace}_judge_cost_usd_total",
            "Total judge cost in USD accumulated by checkllm evaluations.",
            labelnames=("metric", "judge"),
            registry=self.registry,
        )
        self.judge_tokens_total = prom.Counter(
            f"{namespace}_judge_tokens_total",
            "Total tokens consumed by checkllm judge calls.",
            labelnames=("metric", "judge", "kind"),
            registry=self.registry,
        )

        self._http_server: Any = None

    def start_http_server(self, port: int | None = None) -> None:
        """Start a built-in HTTP server exposing the metrics.

        Args:
            port: Optional port override. Defaults to the constructor value.
        """
        effective_port = port if port is not None else self.port
        if hasattr(self._prom, "start_http_server"):
            self._http_server = self._prom.start_http_server(
                effective_port, registry=self.registry
            )
            logger.info(
                "Prometheus exporter listening on :%d", effective_port
            )
        else:  # pragma: no cover - defensive branch
            raise RuntimeError("prometheus_client has no start_http_server")

    def generate_latest(self) -> bytes:
        """Return the current metrics snapshot in Prometheus text format."""
        result: bytes = self._prom.generate_latest(self.registry)
        return result

    def record_check(
        self,
        result: CheckResult,
        judge: str = "unknown",
        prompt_tokens: int | None = None,
        completion_tokens: int | None = None,
    ) -> None:
        """Record a single check evaluation result.

        Args:
            result: The ``CheckResult`` produced by a checkllm metric.
            judge: Name of the judge model used (e.g. ``"gpt-4o"``).
            prompt_tokens: Optional prompt token count to credit.
            completion_tokens: Optional completion token count to credit.
        """
        status = "passed" if result.passed else "failed"
        self.evaluations_total.labels(
            metric=result.metric_name, status=status
        ).inc()
        self.evaluation_duration_seconds.labels(
            metric=result.metric_name
        ).observe(result.latency_ms / 1000.0)
        if result.cost:
            self.judge_cost_usd_total.labels(
                metric=result.metric_name, judge=judge
            ).inc(result.cost)
        if prompt_tokens:
            self.judge_tokens_total.labels(
                metric=result.metric_name, judge=judge, kind="prompt"
            ).inc(prompt_tokens)
        if completion_tokens:
            self.judge_tokens_total.labels(
                metric=result.metric_name,
                judge=judge,
                kind="completion",
            ).inc(completion_tokens)
