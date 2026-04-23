"""Tests for the live-progress WebSocket dashboard and progress broker."""

from __future__ import annotations

import asyncio
import json

import pytest

from checkllm.progress import (
    ProgressBroker,
    emit_check_completed,
    emit_run_completed,
    emit_test_completed,
    emit_test_started,
    get_broker,
    reset_broker,
)


@pytest.fixture(autouse=True)
def _isolated_broker():
    """Reset the module-global broker so each test sees a clean history."""
    reset_broker()
    yield
    reset_broker()


def _starlette_available() -> bool:
    try:
        import starlette  # noqa: F401
        from starlette.testclient import TestClient  # noqa: F401
    except ImportError:
        return False
    return True


pytestmark = pytest.mark.skipif(
    not _starlette_available(), reason="starlette is required for WebSocket tests"
)


def test_live_html_served() -> None:
    """``GET /live`` returns the self-contained subscription page."""
    from starlette.testclient import TestClient

    from checkllm.dashboard_ws import create_app

    app = create_app()
    with TestClient(app) as client:
        resp = client.get("/live")
    assert resp.status_code == 200
    body = resp.text
    assert "<title>checkllm live progress</title>" in body
    assert "/ws/progress" in body
    # Sanity: no innerHTML injection anywhere in the static page.
    assert "innerHTML" not in body


def test_ws_streams_emitted_events() -> None:
    """Events emitted after connection are received via the WebSocket."""
    from starlette.testclient import TestClient

    from checkllm.dashboard_ws import create_app

    broker = get_broker()
    app = create_app(broker=broker)

    with TestClient(app) as client:
        with client.websocket_connect("/ws/progress") as ws:
            # Give the server a moment to drain the (empty) history buffer.
            emit_test_started("tests/demo.py::test_live")
            ev = json.loads(ws.receive_text())
            assert ev["type"] == "test_started"
            assert ev["test_id"] == "tests/demo.py::test_live"

            emit_check_completed(
                test_id="tests/demo.py::test_live",
                metric="hallucination",
                passed=True,
                score=0.95,
                cost=0.0012,
                duration_ms=42.0,
                provider="openai",
                model="gpt-4o",
            )
            ev = json.loads(ws.receive_text())
            assert ev["type"] == "check_completed"
            assert ev["metric"] == "hallucination"
            assert ev["provider"] == "openai"
            assert ev["cost"] == pytest.approx(0.0012)

            emit_test_completed(
                "tests/demo.py::test_live",
                passed=True,
                duration_ms=60.0,
                checks=1,
                cost=0.0012,
            )
            ev = json.loads(ws.receive_text())
            assert ev["type"] == "test_completed"
            assert ev["passed"] is True

            emit_run_completed(
                total_tests=1,
                total_checks=1,
                passed=1,
                failed=0,
                total_cost=0.0012,
                duration_ms=80.0,
            )
            ev = json.loads(ws.receive_text())
            assert ev["type"] == "run_completed"
            assert ev["total_tests"] == 1


def test_ws_replays_history_on_subscribe() -> None:
    """Events emitted before subscription are replayed from the broker history."""
    from starlette.testclient import TestClient

    from checkllm.dashboard_ws import create_app

    broker = get_broker()
    emit_test_started("tests/replay.py::test_early")
    emit_check_completed(
        test_id="tests/replay.py::test_early",
        metric="relevance",
        passed=True,
        score=0.8,
        cost=0.0,
    )
    app = create_app(broker=broker)

    with TestClient(app) as client:
        with client.websocket_connect("/ws/progress") as ws:
            first = json.loads(ws.receive_text())
            second = json.loads(ws.receive_text())
    types = [first["type"], second["type"]]
    assert types == ["test_started", "check_completed"]


def test_ws_multiple_subscribers_get_fanout() -> None:
    """Two clients each receive every emitted event."""
    from starlette.testclient import TestClient

    from checkllm.dashboard_ws import create_app

    broker = get_broker()
    app = create_app(broker=broker)

    with TestClient(app) as client:
        with (
            client.websocket_connect("/ws/progress") as ws_a,
            client.websocket_connect("/ws/progress") as ws_b,
        ):
            emit_test_started("tests/fanout.py::test_x")
            evs = [
                json.loads(ws_a.receive_text()),
                json.loads(ws_b.receive_text()),
            ]
    for ev in evs:
        assert ev["type"] == "test_started"
        assert ev["test_id"] == "tests/fanout.py::test_x"


def test_broker_emit_noop_without_subscribers() -> None:
    """Emission without subscribers is safe and records history."""
    broker = ProgressBroker()
    broker.emit("test_started", test_id="tests/silent.py::test_y")
    hist = broker.history()
    assert len(hist) == 1
    assert hist[0].type == "test_started"
    assert hist[0].payload["test_id"] == "tests/silent.py::test_y"


def test_broker_subscribe_isolated_per_queue() -> None:
    """Two subscribers get independent queues."""

    async def runner() -> tuple[str, str]:
        broker = ProgressBroker()
        q1 = broker.subscribe()
        q2 = broker.subscribe()
        broker.emit("check_completed", metric="toxicity", passed=True, score=0.99)
        ev1 = await q1.get()
        ev2 = await q2.get()
        return ev1.payload["metric"], ev2.payload["metric"]

    a, b = asyncio.run(runner())
    assert a == "toxicity"
    assert b == "toxicity"


def test_check_collector_emits_events_end_to_end() -> None:
    """Running a deterministic check via the collector surfaces ws events."""
    from checkllm.check import CheckCollector
    from checkllm.config import CheckllmConfig

    reset_broker()
    broker = get_broker()
    collector = CheckCollector(config=CheckllmConfig(cache_enabled=False))
    collector.bind_test("tests/smoke.py::test_deterministic")
    collector.contains("hello world", "hello")
    try:
        collector.teardown()
    except Exception:
        pass

    types = [ev.type for ev in broker.history()]
    assert "test_started" in types
    assert "check_completed" in types
    assert "test_completed" in types
