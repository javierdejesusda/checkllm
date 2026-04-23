"""WebSocket-based live progress dashboard.

Builds a Starlette application that exposes:

* ``GET /live`` -- minimal static HTML+JS page that subscribes to
  ``/ws/progress`` and renders a running event list in-browser.
* ``WS /ws/progress`` -- pushes :class:`checkllm.progress.ProgressEvent`
  instances to every connected client as they arrive.  Subscribers receive
  any events that were emitted while they were connecting via the broker's
  retained scrollback.

The Starlette dependency is intentionally optional.  The module function
:func:`create_app` raises a clear ``ImportError`` when Starlette is not
installed so the main HTTP dashboard keeps working with zero extra deps.
"""

from __future__ import annotations

import asyncio
import json
import logging
from typing import TYPE_CHECKING, Any

from checkllm.progress import ProgressBroker, ProgressEvent, get_broker

if TYPE_CHECKING:
    from starlette.applications import Starlette
    from starlette.requests import Request
    from starlette.websockets import WebSocket

logger = logging.getLogger("checkllm.dashboard_ws")

# Self-contained live page.  All dynamic strings are inserted via
# ``textContent`` / ``createElement`` -- no ``innerHTML`` anywhere -- so
# server-emitted payloads cannot inject markup.
LIVE_HTML = """<!DOCTYPE html>
<html lang=\"en\">
<head>
<meta charset=\"utf-8\">
<meta name=\"viewport\" content=\"width=device-width, initial-scale=1\">
<title>checkllm live progress</title>
<style>
:root {
  --bg: #0d1117; --bg2: #161b22; --bg3: #21262d; --bg4: #30363d;
  --fg: #c9d1d9; --fg2: #8b949e; --green: #3fb950; --red: #f85149;
  --accent: #58a6ff; --yellow: #d29922;
  --mono: "SFMono-Regular", Consolas, "Liberation Mono", Menlo, monospace;
}
* { box-sizing: border-box; margin: 0; padding: 0; }
body {
  font: 14px -apple-system, BlinkMacSystemFont, "Segoe UI", Helvetica, Arial, sans-serif;
  background: var(--bg); color: var(--fg); min-height: 100vh;
}
header {
  background: var(--bg2); padding: 12px 20px;
  border-bottom: 1px solid var(--bg4);
  display: flex; align-items: center; gap: 16px;
}
header h1 { font-size: 1.1rem; font-weight: 600; letter-spacing: -.3px; }
header .status {
  margin-left: auto; display: flex; gap: 10px; align-items: center;
  font-size: .85rem; color: var(--fg2);
}
header .dot {
  width: 10px; height: 10px; border-radius: 50%;
  background: var(--yellow); box-shadow: 0 0 6px currentColor;
}
header .dot.ok { background: var(--green); }
header .dot.err { background: var(--red); }
main { max-width: 1100px; margin: 20px auto; padding: 0 20px; }
.summary {
  display: flex; gap: 12px; flex-wrap: wrap; margin-bottom: 18px;
}
.card {
  background: var(--bg2); border: 1px solid var(--bg4); border-radius: 6px;
  padding: 12px 16px; min-width: 120px;
}
.card .v { font-size: 1.4rem; font-weight: 700; color: var(--accent); }
.card .l { font-size: .72rem; text-transform: uppercase; color: var(--fg2);
           letter-spacing: .5px; margin-top: 2px; }
.card.green .v { color: var(--green); }
.card.red .v { color: var(--red); }
.card.yellow .v { color: var(--yellow); }
.log {
  background: var(--bg2); border: 1px solid var(--bg4); border-radius: 6px;
  padding: 0; overflow: hidden;
}
.log-head {
  padding: 10px 14px; border-bottom: 1px solid var(--bg4);
  font-weight: 600;
}
.log-body { max-height: 60vh; overflow-y: auto; }
.row {
  padding: 8px 14px; border-bottom: 1px solid var(--bg3);
  display: grid; grid-template-columns: 90px 140px 1fr auto;
  gap: 10px; align-items: center; font-size: .88rem;
}
.row:last-child { border-bottom: none; }
.row .t { color: var(--fg2); font-family: var(--mono); font-size: .78rem; }
.row .type { font-weight: 600; }
.row.test_started .type { color: var(--accent); }
.row.check_completed .type { color: var(--yellow); }
.row.test_completed .type { color: var(--green); }
.row.run_completed .type { color: var(--green); }
.row.check_completed.failed .type { color: var(--red); }
.row .detail { color: var(--fg); font-family: var(--mono); font-size: .78rem;
               overflow: hidden; text-overflow: ellipsis; white-space: nowrap; }
.row .cost { color: var(--fg2); font-family: var(--mono); font-size: .78rem; }
.empty { padding: 40px; text-align: center; color: var(--fg2); }
</style>
</head>
<body>
<header>
  <h1>checkllm -- live progress</h1>
  <div class=\"status\">
    <span class=\"dot\" id=\"dot\"></span>
    <span id=\"status\">connecting...</span>
  </div>
</header>
<main>
  <section class=\"summary\">
    <div class=\"card\"><div class=\"v\" id=\"nEvents\">0</div><div class=\"l\">Events</div></div>
    <div class=\"card\"><div class=\"v\" id=\"nTests\">0</div><div class=\"l\">Tests</div></div>
    <div class=\"card green\"><div class=\"v\" id=\"nPass\">0</div><div class=\"l\">Passed</div></div>
    <div class=\"card red\"><div class=\"v\" id=\"nFail\">0</div><div class=\"l\">Failed</div></div>
    <div class=\"card yellow\"><div class=\"v\" id=\"nCost\">$0.0000</div><div class=\"l\">Cost</div></div>
  </section>
  <section class=\"log\">
    <div class=\"log-head\">Events</div>
    <div class=\"log-body\" id=\"log\"><div class=\"empty\">Waiting for events...</div></div>
  </section>
</main>
<script>
(function() {
  var logEl = document.getElementById('log');
  var statusEl = document.getElementById('status');
  var dotEl = document.getElementById('dot');
  var counts = { events: 0, tests: new Set(), pass: 0, fail: 0, cost: 0 };
  var firstEvent = true;

  function fmtTime(ts) {
    try {
      var d = new Date(Number(ts) * 1000);
      var hh = String(d.getHours()).padStart(2, '0');
      var mm = String(d.getMinutes()).padStart(2, '0');
      var ss = String(d.getSeconds()).padStart(2, '0');
      return hh + ':' + mm + ':' + ss;
    } catch (e) {
      return '';
    }
  }

  function refreshCounters() {
    document.getElementById('nEvents').textContent = counts.events;
    document.getElementById('nTests').textContent = counts.tests.size;
    document.getElementById('nPass').textContent = counts.pass;
    document.getElementById('nFail').textContent = counts.fail;
    document.getElementById('nCost').textContent = '$' + counts.cost.toFixed(4);
  }

  function makeSpan(cls, text) {
    var span = document.createElement('span');
    span.className = cls;
    span.textContent = String(text == null ? '' : text);
    return span;
  }

  function addRow(ev) {
    if (firstEvent) {
      // Remove the 'waiting' placeholder only once.
      logEl.textContent = '';
      firstEvent = false;
    }
    var row = document.createElement('div');
    var failed = ev.type === 'check_completed' && ev.passed === false;
    row.className = 'row ' + String(ev.type || '') + (failed ? ' failed' : '');
    var detail = '';
    if (ev.type === 'test_started') {
      detail = ev.test_id || '';
    } else if (ev.type === 'check_completed') {
      var score = (typeof ev.score === 'number') ? ev.score.toFixed(3) : '-';
      detail = (ev.metric || '') + ' -> score=' + score +
               (ev.provider ? ' (' + ev.provider + ')' : '') +
               ' in ' + (ev.test_id || '');
    } else if (ev.type === 'test_completed') {
      detail = (ev.test_id || '') + ' checks=' + (ev.checks || 0);
    } else if (ev.type === 'run_completed') {
      detail = 'tests=' + (ev.total_tests || 0) +
               ' passed=' + (ev.passed || 0) +
               ' failed=' + (ev.failed || 0);
    }
    var costText = (typeof ev.cost === 'number' && ev.cost > 0)
      ? ('$' + ev.cost.toFixed(4)) : '';
    row.appendChild(makeSpan('t', fmtTime(ev.timestamp)));
    row.appendChild(makeSpan('type', ev.type || ''));
    row.appendChild(makeSpan('detail', detail));
    row.appendChild(makeSpan('cost', costText));
    logEl.prepend(row);
    while (logEl.childNodes.length > 500) {
      logEl.removeChild(logEl.lastChild);
    }
  }

  function handle(ev) {
    counts.events += 1;
    if (ev.type === 'test_started' && ev.test_id) { counts.tests.add(ev.test_id); }
    if (ev.type === 'test_completed') {
      if (ev.passed) { counts.pass += 1; } else { counts.fail += 1; }
    }
    if (ev.type === 'check_completed' && typeof ev.cost === 'number') {
      counts.cost += ev.cost;
    }
    if (ev.type === 'run_completed' && typeof ev.total_cost === 'number') {
      counts.cost = ev.total_cost;
    }
    refreshCounters();
    addRow(ev);
  }

  function connect() {
    var proto = location.protocol === 'https:' ? 'wss:' : 'ws:';
    var url = proto + '//' + location.host + '/ws/progress';
    var ws = new WebSocket(url);
    ws.onopen = function() {
      statusEl.textContent = 'connected';
      dotEl.className = 'dot ok';
    };
    ws.onmessage = function(msg) {
      try { handle(JSON.parse(msg.data)); } catch (e) { /* ignore */ }
    };
    ws.onclose = function() {
      statusEl.textContent = 'disconnected -- retrying';
      dotEl.className = 'dot err';
      setTimeout(connect, 1500);
    };
    ws.onerror = function() {
      statusEl.textContent = 'error';
      dotEl.className = 'dot err';
    };
  }
  connect();
})();
</script>
</body>
</html>
"""


def create_app(broker: ProgressBroker | None = None) -> "Starlette":
    """Build the Starlette app exposing ``/live`` and ``/ws/progress``.

    Args:
        broker: Optional :class:`ProgressBroker` override (used in tests).
            Defaults to the process-wide broker.

    Returns:
        A configured :class:`starlette.applications.Starlette` instance.

    Raises:
        ImportError: If Starlette is not installed.
    """
    try:
        from starlette.applications import Starlette
        from starlette.responses import HTMLResponse
        from starlette.routing import Route, WebSocketRoute
        from starlette.websockets import WebSocketDisconnect
    except ImportError as exc:  # pragma: no cover - documented optional dep
        raise ImportError(
            "checkllm.dashboard_ws requires starlette. "
            "Install with 'pip install starlette' (or the [dashboard] extra)."
        ) from exc

    pb: ProgressBroker = broker or get_broker()

    async def live_page(request: "Request") -> "HTMLResponse":
        return HTMLResponse(LIVE_HTML)

    async def progress_socket(websocket: "WebSocket") -> None:
        await websocket.accept()
        queue = pb.subscribe()
        try:
            # Drain any retained history so late subscribers see prior
            # events without needing the run to restart.
            for ev in pb.history():
                await _send_event(websocket, ev)
            while True:
                ev = await queue.get()
                await _send_event(websocket, ev)
        except WebSocketDisconnect:
            return
        except Exception:  # noqa: BLE001
            logger.debug("/ws/progress handler error", exc_info=True)
        finally:
            pb.unsubscribe(queue)
            try:
                await websocket.close()
            except Exception:  # noqa: BLE001
                pass

    return Starlette(
        debug=False,
        routes=[
            Route("/live", endpoint=live_page),
            WebSocketRoute("/ws/progress", endpoint=progress_socket),
        ],
    )


async def _send_event(websocket: "WebSocket", event: ProgressEvent) -> None:
    """Serialize and push a single :class:`ProgressEvent` over ``websocket``."""
    payload: dict[str, Any] = event.to_dict()
    try:
        await websocket.send_text(json.dumps(payload))
    except Exception:  # noqa: BLE001
        logger.debug("failed to send progress event", exc_info=True)


def run(
    host: str = "127.0.0.1",
    port: int = 8485,
    *,
    broker: ProgressBroker | None = None,
) -> None:  # pragma: no cover - convenience entry point
    """Run the live dashboard with uvicorn.

    Args:
        host: Bind address.
        port: TCP port.
        broker: Optional broker override.
    """
    try:
        import uvicorn
    except ImportError as exc:
        raise ImportError(
            "checkllm.dashboard_ws.run requires uvicorn. Install with 'pip install uvicorn'."
        ) from exc
    app = create_app(broker=broker)
    uvicorn.run(app, host=host, port=port)


# Silence unused-import warnings for type-only imports when mypy is off.
_ = asyncio
