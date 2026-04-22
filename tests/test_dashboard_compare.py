"""Tests for the dashboard comparison view (build_comparison + routes)."""

from __future__ import annotations

import http.client
import json
import threading
import time
from http.server import HTTPServer

import pytest

from checkllm.dashboard import (
    DashboardHandler,
    build_comparison,
    render_comparison_html,
)
from checkllm.history import RunHistory
from checkllm.models import CheckResult


def _cr(metric: str, score: float, passed: bool = True) -> CheckResult:
    return CheckResult(
        passed=passed,
        score=score,
        reasoning="ok",
        cost=0.0,
        latency_ms=1,
        metric_name=metric,
    )


@pytest.fixture()
def history_with_two_runs(tmp_path):
    db = tmp_path / "history.db"
    history = RunHistory(db_path=str(db))
    run_a = history.record_run(
        {
            "t1": [_cr("relevance", 0.5), _cr("toxicity", 0.2)],
        },
        label="baseline",
    )
    run_b = history.record_run(
        {
            "t1": [_cr("relevance", 0.9), _cr("toxicity", 0.4)],
        },
        label="candidate",
    )
    yield history, run_a, run_b, str(db)
    history.close()


def test_build_comparison_identifies_improved_and_regressed(history_with_two_runs):
    history, run_a, run_b, _ = history_with_two_runs
    view = build_comparison(history, run_a, run_b)
    assert view.snapshot_a == str(run_a)
    assert view.snapshot_b == str(run_b)
    # relevance improved (0.5 -> 0.9)
    assert "relevance" in view.improved
    # toxicity regressed if higher is worse... but build_comparison_view only
    # knows raw deltas (higher = improved), so 0.2 -> 0.4 is "improved" here.
    assert view.metrics_diff["relevance"] == pytest.approx(0.4, abs=1e-6)


def test_build_comparison_missing_run_raises(history_with_two_runs):
    history, run_a, _, _ = history_with_two_runs
    with pytest.raises(ValueError):
        build_comparison(history, run_a, 99999)


def test_build_comparison_non_integer_raises(history_with_two_runs):
    history, run_a, _, _ = history_with_two_runs
    with pytest.raises(ValueError):
        build_comparison(history, "not-an-int", run_a)


def test_render_comparison_html_contains_snapshot_ids_and_table(
    history_with_two_runs,
):
    history, run_a, run_b, _ = history_with_two_runs
    view = build_comparison(history, run_a, run_b)
    html = render_comparison_html(view)
    assert "<!DOCTYPE html>" in html
    assert str(run_a) in html
    assert str(run_b) in html
    assert "<table>" in html
    assert "relevance" in html
    # Summary headers present.
    assert "Improved" in html
    assert "Regressed" in html
    # Direction symbols are '+' / '-' / '=' (no emoji arrows).
    assert "+" in html
    # Row classes applied.
    assert "improved" in html


def _start_server(db_path: str) -> tuple[HTTPServer, threading.Thread, int]:
    """Spin up a DashboardHandler server on a random port."""
    DashboardHandler._db_path = db_path
    server = HTTPServer(("127.0.0.1", 0), DashboardHandler)
    port = server.server_address[1]
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    # Tiny warmup so the first request doesn't race the listen.
    time.sleep(0.05)
    return server, thread, port


def test_compare_html_endpoint_returns_200(history_with_two_runs):
    _, run_a, run_b, db_path = history_with_two_runs
    server, thread, port = _start_server(db_path)
    try:
        conn = http.client.HTTPConnection("127.0.0.1", port, timeout=5)
        conn.request("GET", f"/compare?a={run_a}&b={run_b}")
        resp = conn.getresponse()
        body = resp.read().decode("utf-8")
        assert resp.status == 200
        assert "text/html" in resp.getheader("Content-Type", "")
        assert str(run_a) in body
        assert str(run_b) in body
        assert "<table>" in body
    finally:
        server.shutdown()
        thread.join(timeout=2)
        server.server_close()


def test_compare_html_endpoint_missing_params_returns_400(
    history_with_two_runs,
):
    _, _, _, db_path = history_with_two_runs
    server, thread, port = _start_server(db_path)
    try:
        conn = http.client.HTTPConnection("127.0.0.1", port, timeout=5)
        conn.request("GET", "/compare")
        resp = conn.getresponse()
        _ = resp.read()
        assert resp.status == 400
    finally:
        server.shutdown()
        thread.join(timeout=2)
        server.server_close()


def test_api_compare_post_returns_valid_json(history_with_two_runs):
    _, run_a, run_b, db_path = history_with_two_runs
    server, thread, port = _start_server(db_path)
    try:
        conn = http.client.HTTPConnection("127.0.0.1", port, timeout=5)
        payload = json.dumps({"a": run_a, "b": run_b})
        conn.request(
            "POST",
            "/api/compare",
            body=payload,
            headers={"Content-Type": "application/json"},
        )
        resp = conn.getresponse()
        body = resp.read().decode("utf-8")
        assert resp.status == 200
        data = json.loads(body)
        assert data["snapshot_a"] == str(run_a)
        assert data["snapshot_b"] == str(run_b)
        assert "metrics_diff" in data
        assert "summary" in data
        assert data["summary"]["total"] >= 1
    finally:
        server.shutdown()
        thread.join(timeout=2)
        server.server_close()


def test_api_compare_post_missing_ids_returns_400(history_with_two_runs):
    _, _, _, db_path = history_with_two_runs
    server, thread, port = _start_server(db_path)
    try:
        conn = http.client.HTTPConnection("127.0.0.1", port, timeout=5)
        conn.request(
            "POST",
            "/api/compare",
            body="{}",
            headers={"Content-Type": "application/json"},
        )
        resp = conn.getresponse()
        _ = resp.read()
        assert resp.status == 400
    finally:
        server.shutdown()
        thread.join(timeout=2)
        server.server_close()
