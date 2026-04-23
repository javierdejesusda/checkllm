"""Tests for per-provider/metric/test cost rollup endpoints on the dashboard."""

from __future__ import annotations

import io
import json
import time
from http.client import HTTPResponse
from pathlib import Path
from typing import Any

import pytest

from checkllm.dashboard import DashboardHandler
from checkllm.history import RunHistory
from checkllm.models import CheckResult
from checkllm.pricing import build_cost_breakdown


class _FakeSocket(io.BytesIO):
    """File-like stand-in so HTTPResponse can parse a raw HTTP byte stream."""

    def makefile(self, *args: Any, **kwargs: Any) -> "_FakeSocket":
        return self


class _CapturedResponse:
    """Minimal HTTPResponse wrapper used for assertions."""

    def __init__(self, raw: bytes) -> None:
        self.raw = raw

    @property
    def status(self) -> int:
        first = self.raw.split(b"\r\n", 1)[0]
        return int(first.split(b" ")[1])

    @property
    def body(self) -> bytes:
        _, body = self.raw.split(b"\r\n\r\n", 1)
        return body

    def json(self) -> Any:
        return json.loads(self.body.decode("utf-8"))


class _StubHandler(DashboardHandler):
    """Instantiate :class:`DashboardHandler` without a live socket."""

    # pylint: disable=super-init-not-called
    def __init__(self) -> None:  # type: ignore[no-untyped-def]
        self._raw_out = io.BytesIO()
        self.wfile = self._raw_out  # type: ignore[assignment]
        self.rfile = io.BytesIO()  # type: ignore[assignment]
        self.headers: dict[str, str] = {}
        self.path = "/"
        self.command = "GET"
        self.request_version = "HTTP/1.1"
        self.requestline = "GET / HTTP/1.1"
        self.client_address = ("test", 0)
        self.server = None  # type: ignore[assignment]

    def log_message(self, *args: Any, **kwargs: Any) -> None:
        return

    def get_captured(self) -> _CapturedResponse:
        return _CapturedResponse(self._raw_out.getvalue())


def _make_check(
    metric: str,
    *,
    cost: float,
    model: str,
    test_id: str,
    timestamp: float,
    input_tokens: int = 100,
    output_tokens: int = 50,
    passed: bool = True,
) -> CheckResult:
    """Build a :class:`CheckResult` with a fully populated cost breakdown."""
    bd = build_cost_breakdown(
        model=model,
        input_tokens=input_tokens,
        output_tokens=output_tokens,
        metric=metric,
        test_id=test_id,
        timestamp=timestamp,
    )
    # Override total_cost so tests control expected aggregates.
    bd.total_cost = cost
    return CheckResult(
        passed=passed,
        score=0.9 if passed else 0.1,
        reasoning="synthetic",
        cost=cost,
        latency_ms=10,
        metric_name=metric,
        cost_breakdown=bd.to_dict(),
    )


@pytest.fixture()
def seeded_db(tmp_path: Path) -> Path:
    """Populate a SQLite history with two runs spanning multiple providers."""
    db_path = tmp_path / "history.db"
    history = RunHistory(db_path=db_path)

    base_ts = 1_700_000_000.0  # fixed timestamp inside breakdowns
    run_a = {
        "tests/test_alpha.py::test_a": [
            _make_check(
                "hallucination",
                cost=0.001,
                model="gpt-4o",
                test_id="tests/test_alpha.py::test_a",
                timestamp=base_ts,
            ),
            _make_check(
                "relevance",
                cost=0.002,
                model="claude-sonnet-4-6",
                test_id="tests/test_alpha.py::test_a",
                timestamp=base_ts + 30,
            ),
        ],
        "tests/test_beta.py::test_b": [
            _make_check(
                "toxicity",
                cost=0.003,
                model="gpt-4o",
                test_id="tests/test_beta.py::test_b",
                timestamp=base_ts + 60,
            ),
        ],
    }
    history.record_run(run_a, label="rollup-fixture")

    # Second run, one day later, single claude call.
    run_b = {
        "tests/test_alpha.py::test_a": [
            _make_check(
                "hallucination",
                cost=0.004,
                model="claude-opus-4-6",
                test_id="tests/test_alpha.py::test_a",
                timestamp=base_ts + 86_400,
            ),
        ],
    }
    history.record_run(run_b, label="rollup-fixture-2")
    history.close()
    return db_path


def _invoke(seeded_db: Path, path: str) -> _CapturedResponse:
    DashboardHandler._db_path = str(seeded_db)
    handler = _StubHandler()
    handler.path = path
    handler.do_GET()
    return handler.get_captured()


def test_cost_by_provider_rollup(seeded_db: Path) -> None:
    resp = _invoke(seeded_db, "/api/cost/by-provider")
    assert resp.status == 200
    data = resp.json()
    assert data["dimension"] == "provider"
    providers = {b["provider"]: b for b in data["buckets"]}
    assert pytest.approx(providers["openai"]["cost"], rel=1e-6) == 0.001 + 0.003
    assert pytest.approx(providers["anthropic"]["cost"], rel=1e-6) == 0.002 + 0.004
    assert providers["openai"]["calls"] == 2
    assert providers["anthropic"]["calls"] == 2
    assert pytest.approx(data["total_cost"], rel=1e-6) == 0.010


def test_cost_by_metric_rollup(seeded_db: Path) -> None:
    resp = _invoke(seeded_db, "/api/cost/by-metric")
    assert resp.status == 200
    metrics = {b["metric"]: b for b in resp.json()["buckets"]}
    assert pytest.approx(metrics["hallucination"]["cost"], rel=1e-6) == 0.005
    assert pytest.approx(metrics["relevance"]["cost"], rel=1e-6) == 0.002
    assert pytest.approx(metrics["toxicity"]["cost"], rel=1e-6) == 0.003


def test_cost_by_test_rollup(seeded_db: Path) -> None:
    resp = _invoke(seeded_db, "/api/cost/by-test")
    assert resp.status == 200
    buckets = {b["test_id"]: b for b in resp.json()["buckets"]}
    # test_alpha has three checks across runs (0.001 + 0.002 + 0.004).
    assert pytest.approx(buckets["tests/test_alpha.py::test_a"]["cost"], rel=1e-6) == 0.007
    assert pytest.approx(buckets["tests/test_beta.py::test_b"]["cost"], rel=1e-6) == 0.003


def test_cost_timeseries_hour_bucket(seeded_db: Path) -> None:
    resp = _invoke(seeded_db, "/api/cost/timeseries?bucket=hour")
    assert resp.status == 200
    data = resp.json()
    assert data["bucket"] == "hour"
    assert data["bucket_seconds"] == 3600
    # First run spans <1h so it falls in a single bucket; second run +1 day.
    assert len(data["points"]) >= 2
    total = sum(p["cost"] for p in data["points"])
    assert pytest.approx(total, rel=1e-6) == 0.010


def test_cost_timeseries_day_bucket(seeded_db: Path) -> None:
    resp = _invoke(seeded_db, "/api/cost/timeseries?bucket=day")
    assert resp.status == 200
    data = resp.json()
    assert data["bucket_seconds"] == 86_400
    # Two distinct days.
    assert len(data["points"]) == 2
    first, second = data["points"]
    assert second["bucket_start"] - first["bucket_start"] == 86_400


def test_cost_timeseries_bad_bucket_returns_400(seeded_db: Path) -> None:
    resp = _invoke(seeded_db, "/api/cost/timeseries?bucket=week")
    assert resp.status == 400
    assert "bucket" in resp.json()["error"]


def test_cost_rollup_scoped_to_single_run(seeded_db: Path) -> None:
    # Run id 1 is the first record; only openai + anthropic with 3 calls total.
    resp = _invoke(seeded_db, "/api/cost/by-provider?run=1")
    assert resp.status == 200
    data = resp.json()
    providers = {b["provider"]: b["cost"] for b in data["buckets"]}
    assert pytest.approx(providers["openai"], rel=1e-6) == 0.001 + 0.003
    assert pytest.approx(providers["anthropic"], rel=1e-6) == 0.002
    assert "anthropic" in providers
    assert pytest.approx(data["total_cost"], rel=1e-6) == 0.006


def test_pricing_lookup_covers_requested_models() -> None:
    """Sanity check: every advertised model resolves to non-default pricing."""
    from checkllm.pricing import DEFAULT_PRICING, lookup_price

    sample = [
        "gpt-4o",
        "gpt-4o-mini",
        "gpt-4.1",
        "o1",
        "o3",
        "claude-sonnet-4-5",
        "claude-sonnet-4-6",
        "claude-sonnet-4-7",
        "claude-opus-4-7",
        "claude-haiku-4-5",
        "gemini-1.5-pro",
        "gemini-2.0-flash",
        "gemini-2.5-pro",
        "deepseek-chat",
        "deepseek-reasoner",
        "anthropic.claude-sonnet-4-v1:0",
    ]
    for model in sample:
        price = lookup_price(model)
        assert price != DEFAULT_PRICING, f"{model} falls back to default pricing"
        assert price[0] > 0 and price[1] > 0


def test_cost_breakdown_serialization_round_trip(tmp_path: Path) -> None:
    """A persisted run preserves cost_breakdown metadata end-to-end."""
    from checkllm.pricing import breakdown_from_dict

    db = tmp_path / "rt.db"
    history = RunHistory(db_path=db)
    check = _make_check(
        "relevance",
        cost=0.0025,
        model="gpt-4o",
        test_id="tests/roundtrip.py::test_x",
        timestamp=time.time(),
        input_tokens=123,
        output_tokens=45,
    )
    history.record_run({"tests/roundtrip.py::test_x": [check]})
    run = history.list_runs()[0]
    record = history.get_run(run.run_id)
    assert record is not None
    serialized = record.results["tests/roundtrip.py::test_x"][0]
    bd = breakdown_from_dict(serialized.get("cost_breakdown"))
    assert bd is not None
    assert bd.provider == "openai"
    assert bd.model == "gpt-4o"
    assert bd.metric == "relevance"
    assert bd.input_tokens == 123
    assert bd.output_tokens == 45
    history.close()
