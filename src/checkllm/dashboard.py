"""Interactive web dashboard for viewing checkllm experiment history.

Serves a self-contained HTML page backed by the RunHistory SQLite database.
No external JS/CSS dependencies required.

Usage::

    from checkllm.dashboard import start_dashboard
    start_dashboard(port=8484)

Or via the CLI::

    checkllm dashboard
"""
from __future__ import annotations

import http.server
import json
import logging
import threading
import webbrowser
from pathlib import Path
from typing import Any

from checkllm.history import RunHistory, RunRecord, RunSummary

logger = logging.getLogger("checkllm.dashboard")

# ---------------------------------------------------------------------------
# HTML template — complete self-contained page
#
# Security note: this dashboard is a **local-only** developer tool served on
# localhost.  All dynamic data originates from the user's own SQLite database.
# The ``esc()`` helper in the JS below escapes every dynamic string through
# ``document.createElement / textContent`` before insertion, preventing any
# injection via stored data.  ``innerHTML`` usage is intentional for building
# the local UI from fully-escaped fragments.
# ---------------------------------------------------------------------------

DASHBOARD_HTML = r"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>checkllm Dashboard</title>
<style>
/* ---- Reset & Base ---- */
*, *::before, *::after { box-sizing: border-box; margin: 0; padding: 0; }
:root {
  --bg: #0d1117; --bg2: #161b22; --bg3: #21262d; --bg4: #30363d;
  --fg: #c9d1d9; --fg2: #8b949e; --fg3: #6e7681;
  --accent: #58a6ff; --accent2: #388bfd; --green: #3fb950; --red: #f85149;
  --yellow: #d29922; --purple: #bc8cff; --orange: #f0883e;
  --radius: 8px; --shadow: 0 1px 3px rgba(0,0,0,.4);
  --font: -apple-system, BlinkMacSystemFont, "Segoe UI", Helvetica, Arial, sans-serif;
  --mono: "SFMono-Regular", Consolas, "Liberation Mono", Menlo, monospace;
}
html { font-size: 14px; }
body { font-family: var(--font); background: var(--bg); color: var(--fg); min-height: 100vh; }
a { color: var(--accent); text-decoration: none; }
a:hover { text-decoration: underline; }

/* ---- Layout ---- */
.topbar {
  background: var(--bg2); border-bottom: 1px solid var(--bg4);
  padding: 12px 24px; display: flex; align-items: center; gap: 16px;
  position: sticky; top: 0; z-index: 100;
}
.topbar h1 { font-size: 1.2rem; font-weight: 600; color: var(--fg); letter-spacing: -.3px; }
.topbar .subtitle { color: var(--fg2); font-size: .85rem; }
.topbar nav { margin-left: auto; display: flex; gap: 8px; }
.topbar nav button {
  background: var(--bg3); color: var(--fg); border: 1px solid var(--bg4);
  padding: 6px 14px; border-radius: var(--radius); cursor: pointer;
  font-size: .85rem; transition: background .15s;
}
.topbar nav button:hover, .topbar nav button.active { background: var(--accent2); border-color: var(--accent); }

.container { max-width: 1280px; margin: 0 auto; padding: 24px; }

/* ---- Cards ---- */
.card {
  background: var(--bg2); border: 1px solid var(--bg4); border-radius: var(--radius);
  box-shadow: var(--shadow); margin-bottom: 20px; overflow: hidden;
}
.card-header {
  padding: 14px 18px; border-bottom: 1px solid var(--bg4);
  display: flex; align-items: center; gap: 12px;
}
.card-header h2 { font-size: 1rem; font-weight: 600; }
.card-body { padding: 18px; }

/* ---- Summary stats ---- */
.stats-row { display: flex; gap: 16px; flex-wrap: wrap; margin-bottom: 20px; }
.stat-card {
  flex: 1; min-width: 140px; background: var(--bg2); border: 1px solid var(--bg4);
  border-radius: var(--radius); padding: 16px; text-align: center;
}
.stat-card .value { font-size: 1.8rem; font-weight: 700; color: var(--accent); }
.stat-card .label { font-size: .78rem; color: var(--fg2); margin-top: 4px; text-transform: uppercase; letter-spacing: .5px; }
.stat-card.green .value { color: var(--green); }
.stat-card.red .value { color: var(--red); }
.stat-card.yellow .value { color: var(--yellow); }
.stat-card.purple .value { color: var(--purple); }

/* ---- Table ---- */
table { width: 100%; border-collapse: collapse; }
th { text-align: left; color: var(--fg2); font-weight: 500; font-size: .78rem;
     text-transform: uppercase; letter-spacing: .5px; padding: 10px 14px;
     border-bottom: 1px solid var(--bg4); }
td { padding: 10px 14px; border-bottom: 1px solid var(--bg3); font-size: .9rem; }
tr:hover td { background: var(--bg3); }
tr.clickable { cursor: pointer; }

/* ---- Badges ---- */
.badge {
  display: inline-block; padding: 2px 8px; border-radius: 12px;
  font-size: .75rem; font-weight: 600;
}
.badge.pass { background: rgba(63,185,80,.15); color: var(--green); }
.badge.fail { background: rgba(248,81,73,.15); color: var(--red); }
.badge.skip { background: rgba(139,148,158,.15); color: var(--fg2); }

/* ---- Bar charts (pure CSS) ---- */
.bar-chart { display: flex; flex-direction: column; gap: 8px; }
.bar-row { display: flex; align-items: center; gap: 10px; }
.bar-label { width: 140px; font-size: .82rem; color: var(--fg2); text-align: right; overflow: hidden; text-overflow: ellipsis; white-space: nowrap; }
.bar-track { flex: 1; height: 22px; background: var(--bg3); border-radius: 4px; overflow: hidden; position: relative; }
.bar-fill { height: 100%; border-radius: 4px; transition: width .4s ease; min-width: 2px; }
.bar-fill.good { background: linear-gradient(90deg, var(--green), #2ea043); }
.bar-fill.ok { background: linear-gradient(90deg, var(--yellow), #c69026); }
.bar-fill.bad { background: linear-gradient(90deg, var(--red), #da3633); }
.bar-value { width: 50px; font-size: .82rem; font-family: var(--mono); color: var(--fg2); }

/* ---- Compare ---- */
.compare-grid { display: grid; grid-template-columns: 1fr 1fr; gap: 16px; }
.compare-grid .card { margin-bottom: 0; }
@media (max-width: 800px) { .compare-grid { grid-template-columns: 1fr; } }

/* ---- Filter ---- */
.filter-bar { display: flex; gap: 10px; margin-bottom: 16px; align-items: center; flex-wrap: wrap; }
.filter-bar input, .filter-bar select {
  background: var(--bg3); color: var(--fg); border: 1px solid var(--bg4);
  padding: 7px 12px; border-radius: var(--radius); font-size: .85rem;
  outline: none; transition: border-color .15s;
}
.filter-bar input:focus, .filter-bar select:focus { border-color: var(--accent); }
.filter-bar input { flex: 1; min-width: 200px; }

/* ---- Misc ---- */
.empty { text-align: center; padding: 48px; color: var(--fg3); font-size: .95rem; }
.back-link { display: inline-block; margin-bottom: 16px; font-size: .85rem; }
.mono { font-family: var(--mono); }
.text-green { color: var(--green); }
.text-red { color: var(--red); }
.text-yellow { color: var(--yellow); }
.text-muted { color: var(--fg2); }
.mt-1 { margin-top: 8px; }
.mt-2 { margin-top: 16px; }
.mb-1 { margin-bottom: 8px; }
.compare-select {
  display: flex; gap: 8px; align-items: center; margin-bottom: 16px;
}
.compare-select select { min-width: 200px; }
.compare-select button {
  background: var(--accent2); color: #fff; border: none; padding: 7px 16px;
  border-radius: var(--radius); cursor: pointer; font-size: .85rem;
}
.compare-select button:hover { background: var(--accent); }
.hidden { display: none; }
.delta.positive { color: var(--green); }
.delta.negative { color: var(--red); }
.delta.neutral { color: var(--fg3); }
.loading { text-align: center; padding: 48px; color: var(--fg3); }
</style>
</head>
<body>

<div class="topbar">
  <h1>checkllm</h1>
  <span class="subtitle">Dashboard</span>
  <nav>
    <button id="nav-runs" class="active" onclick="showView('runs')">Runs</button>
    <button id="nav-compare" onclick="showView('compare')">Compare</button>
  </nav>
</div>

<div class="container">
  <!-- RUNS LIST VIEW -->
  <div id="view-runs">
    <div class="stats-row" id="global-stats"></div>
    <div class="card">
      <div class="card-header"><h2>Experiment Runs</h2></div>
      <div class="card-body">
        <div class="filter-bar">
          <input type="text" id="filter-search" placeholder="Search by label, commit..." oninput="filterRuns()">
          <select id="filter-status" onchange="filterRuns()">
            <option value="">All statuses</option>
            <option value="pass">All passed</option>
            <option value="fail">Has failures</option>
          </select>
        </div>
        <div id="runs-table"></div>
      </div>
    </div>
  </div>

  <!-- RUN DETAIL VIEW -->
  <div id="view-detail" class="hidden">
    <a href="#" class="back-link" onclick="showView('runs'); return false;">&larr; Back to runs</a>
    <div id="detail-header"></div>
    <div class="stats-row" id="detail-stats"></div>
    <div class="card">
      <div class="card-header"><h2>Score Distribution</h2></div>
      <div class="card-body" id="detail-chart"></div>
    </div>
    <div class="card">
      <div class="card-header"><h2>Assertion Results</h2></div>
      <div class="card-body" id="detail-table"></div>
    </div>
  </div>

  <!-- COMPARE VIEW -->
  <div id="view-compare" class="hidden">
    <div class="card">
      <div class="card-header"><h2>Compare Runs</h2></div>
      <div class="card-body">
        <div class="compare-select">
          <select id="compare-a"></select>
          <span class="text-muted">vs</span>
          <select id="compare-b"></select>
          <button onclick="doCompare()">Compare</button>
        </div>
        <div id="compare-result"></div>
      </div>
    </div>
  </div>
</div>

<script>
/* ================================================================
   checkllm dashboard — local-only developer tool.
   All dynamic strings are escaped via esc() (textContent-based)
   before DOM insertion to prevent any stored-data injection.
   ================================================================ */

/* ---- State ---- */
var allRuns = [];
var currentView = 'runs';

/* ---- API ---- */
function api(path) {
  return fetch(path).then(function(resp) {
    if (!resp.ok) return null;
    return resp.json();
  });
}

/* ---- Escaping: every dynamic value passes through this ---- */
function esc(s) {
  var d = document.createElement('div');
  d.textContent = String(s);
  return d.innerHTML;  // safe: textContent encodes all entities
}

/* ---- Safe DOM setter (wraps escaped HTML fragments) ---- */
function setContent(el, safeHtml) {
  // All callers build safeHtml from esc()-wrapped dynamic values
  // and static HTML structure only.  This is a local dev tool
  // reading from the user's own SQLite database.
  el.textContent = '';
  var tmp = document.createElement('template');
  tmp.innerHTML = safeHtml;
  el.appendChild(tmp.content);
}

/* ---- Init ---- */
function init() {
  api('/api/runs').then(function(data) {
    allRuns = data || [];
    renderGlobalStats();
    renderRunsTable(allRuns);
    populateCompareSelects();
  });
}

/* ---- Navigation ---- */
function showView(v) {
  currentView = v;
  ['runs', 'detail', 'compare'].forEach(function(id) {
    var el = document.getElementById('view-' + id);
    if (id === v) el.classList.remove('hidden');
    else el.classList.add('hidden');
  });
  document.querySelectorAll('.topbar nav button').forEach(function(b) { b.classList.remove('active'); });
  var navBtn = document.getElementById('nav-' + v);
  if (navBtn) navBtn.classList.add('active');
}

/* ---- Global Stats ---- */
function renderGlobalStats() {
  var el = document.getElementById('global-stats');
  if (!allRuns.length) { el.textContent = ''; return; }
  var total = allRuns.length;
  var totalChecks = 0, totalPassed = 0, totalFailed = 0, totalCost = 0;
  for (var i = 0; i < allRuns.length; i++) {
    totalChecks += allRuns[i].total_checks;
    totalPassed += allRuns[i].passed_checks;
    totalFailed += allRuns[i].failed_checks;
    totalCost += allRuns[i].total_cost;
  }
  setContent(el,
    '<div class="stat-card"><div class="value">' + esc(total) + '</div><div class="label">Runs</div></div>' +
    '<div class="stat-card"><div class="value">' + esc(totalChecks) + '</div><div class="label">Total Checks</div></div>' +
    '<div class="stat-card green"><div class="value">' + esc(totalPassed) + '</div><div class="label">Passed</div></div>' +
    '<div class="stat-card red"><div class="value">' + esc(totalFailed) + '</div><div class="label">Failed</div></div>' +
    '<div class="stat-card purple"><div class="value">$' + esc(totalCost.toFixed(4)) + '</div><div class="label">Total Cost</div></div>'
  );
}

/* ---- Runs Table ---- */
function renderRunsTable(runs) {
  var el = document.getElementById('runs-table');
  if (!runs.length) { el.textContent = 'No runs recorded yet.'; return; }
  var html = '<table><thead><tr>' +
    '<th>ID</th><th>Time</th><th>Label</th><th>Commit</th>' +
    '<th>Checks</th><th>Passed</th><th>Failed</th><th>Cost</th><th>Status</th>' +
    '</tr></thead><tbody>';
  for (var i = 0; i < runs.length; i++) {
    var r = runs[i];
    var ts = new Date(r.timestamp * 1000).toLocaleString();
    var status = r.failed_checks > 0
      ? '<span class="badge fail">FAIL</span>'
      : '<span class="badge pass">PASS</span>';
    html += '<tr class="clickable" data-run-id="' + esc(r.run_id) + '">' +
      '<td class="mono">' + esc(r.run_id) + '</td>' +
      '<td>' + esc(ts) + '</td>' +
      '<td>' + esc(r.label || '') + '</td>' +
      '<td class="mono">' + esc(r.git_commit || '-') + '</td>' +
      '<td>' + esc(r.total_checks) + '</td>' +
      '<td class="text-green">' + esc(r.passed_checks) + '</td>' +
      '<td class="' + (r.failed_checks > 0 ? 'text-red' : '') + '">' + esc(r.failed_checks) + '</td>' +
      '<td class="mono">$' + esc(r.total_cost.toFixed(4)) + '</td>' +
      '<td>' + status + '</td>' +
      '</tr>';
  }
  html += '</tbody></table>';
  setContent(el, html);
  // Attach click handlers via event delegation
  el.addEventListener('click', function(e) {
    var row = e.target.closest('tr[data-run-id]');
    if (row) showRunDetail(parseInt(row.getAttribute('data-run-id'), 10));
  });
}

/* ---- Filter ---- */
function filterRuns() {
  var q = document.getElementById('filter-search').value.toLowerCase();
  var status = document.getElementById('filter-status').value;
  var filtered = allRuns;
  if (q) {
    filtered = filtered.filter(function(r) {
      return (r.label || '').toLowerCase().indexOf(q) !== -1 ||
             (r.git_commit || '').toLowerCase().indexOf(q) !== -1 ||
             String(r.run_id).indexOf(q) !== -1;
    });
  }
  if (status === 'pass') filtered = filtered.filter(function(r) { return r.failed_checks === 0; });
  if (status === 'fail') filtered = filtered.filter(function(r) { return r.failed_checks > 0; });
  renderRunsTable(filtered);
}

/* ---- Run Detail ---- */
function showRunDetail(runId) {
  showView('detail');
  var headerEl = document.getElementById('detail-header');
  headerEl.textContent = 'Loading...';

  api('/api/runs/' + runId).then(function(run) {
    if (!run) { headerEl.textContent = 'Run not found.'; return; }

    var ts = new Date(run.timestamp * 1000).toLocaleString();
    var badge = run.failed_checks > 0 ? '<span class="badge fail">FAIL</span>' : '<span class="badge pass">PASS</span>';
    setContent(headerEl,
      '<h2 style="margin-bottom:8px;">Run #' + esc(run.run_id) + ' ' + badge + '</h2>' +
      '<div class="text-muted mb-1">' + esc(ts) + ' &middot; ' + esc(run.label || 'unlabeled') + ' &middot; ' + esc(run.git_commit || 'no commit') + '</div>'
    );

    // Stats
    var passRate = run.total_checks > 0 ? ((run.passed_checks / run.total_checks) * 100).toFixed(1) : '0.0';
    setContent(document.getElementById('detail-stats'),
      '<div class="stat-card"><div class="value">' + esc(run.total_checks) + '</div><div class="label">Total Checks</div></div>' +
      '<div class="stat-card green"><div class="value">' + esc(run.passed_checks) + '</div><div class="label">Passed</div></div>' +
      '<div class="stat-card red"><div class="value">' + esc(run.failed_checks) + '</div><div class="label">Failed</div></div>' +
      '<div class="stat-card yellow"><div class="value">' + esc(passRate) + '%</div><div class="label">Pass Rate</div></div>' +
      '<div class="stat-card purple"><div class="value">$' + esc(run.total_cost.toFixed(4)) + '</div><div class="label">Cost</div></div>'
    );

    // Chart: group scores by metric
    var metricScores = {};
    var results = run.results || {};
    var testNames = Object.keys(results);
    for (var ti = 0; ti < testNames.length; ti++) {
      var checks = results[testNames[ti]];
      for (var ci = 0; ci < checks.length; ci++) {
        var c = checks[ci];
        var name = c.metric_name || 'unknown';
        if (!metricScores[name]) metricScores[name] = [];
        metricScores[name].push(c.score);
      }
    }
    var chartHtml = '<div class="bar-chart">';
    var metricNames = Object.keys(metricScores);
    for (var mi = 0; mi < metricNames.length; mi++) {
      var mn = metricNames[mi];
      var scores = metricScores[mn];
      var avg = 0;
      for (var si = 0; si < scores.length; si++) avg += scores[si];
      avg = avg / scores.length;
      var pct = (avg * 100).toFixed(1);
      var cls = avg >= 0.8 ? 'good' : avg >= 0.5 ? 'ok' : 'bad';
      chartHtml += '<div class="bar-row">' +
        '<span class="bar-label">' + esc(mn) + '</span>' +
        '<div class="bar-track"><div class="bar-fill ' + cls + '" style="width:' + esc(pct) + '%"></div></div>' +
        '<span class="bar-value">' + esc(pct) + '%</span>' +
        '</div>';
    }
    chartHtml += '</div>';
    setContent(document.getElementById('detail-chart'), chartHtml);

    // Detailed table
    var tblHtml = '<table><thead><tr>' +
      '<th>Test</th><th>Metric</th><th>Status</th><th>Score</th><th>Cost</th><th>Reasoning</th>' +
      '</tr></thead><tbody>';
    for (var ti2 = 0; ti2 < testNames.length; ti2++) {
      var tName = testNames[ti2];
      var chks = results[tName];
      for (var ci2 = 0; ci2 < chks.length; ci2++) {
        var ch = chks[ci2];
        var chBadge = ch.passed ? '<span class="badge pass">PASS</span>' : '<span class="badge fail">FAIL</span>';
        tblHtml += '<tr>' +
          '<td>' + esc(tName) + '</td>' +
          '<td class="mono">' + esc(ch.metric_name || '') + '</td>' +
          '<td>' + chBadge + '</td>' +
          '<td class="mono">' + esc((ch.score || 0).toFixed(3)) + '</td>' +
          '<td class="mono">$' + esc((ch.cost || 0).toFixed(4)) + '</td>' +
          '<td class="text-muted" style="max-width:320px;overflow:hidden;text-overflow:ellipsis;white-space:nowrap;">' + esc((ch.reasoning || '').substring(0, 120)) + '</td>' +
          '</tr>';
      }
    }
    tblHtml += '</tbody></table>';
    setContent(document.getElementById('detail-table'), tblHtml);
  });
}

/* ---- Compare ---- */
function populateCompareSelects() {
  var optHtml = '';
  for (var i = 0; i < allRuns.length; i++) {
    var r = allRuns[i];
    var ts = new Date(r.timestamp * 1000).toLocaleDateString();
    optHtml += '<option value="' + esc(r.run_id) + '">#' + esc(r.run_id) + ' - ' + esc(ts) + ' ' + esc(r.label || '') + '</option>';
  }
  setContent(document.getElementById('compare-a'), optHtml);
  setContent(document.getElementById('compare-b'), optHtml);
  // Default: select last two
  if (allRuns.length >= 2) {
    document.getElementById('compare-a').value = allRuns[1].run_id;
    document.getElementById('compare-b').value = allRuns[0].run_id;
  }
}

function doCompare() {
  var a = document.getElementById('compare-a').value;
  var b = document.getElementById('compare-b').value;
  if (!a || !b) return;
  var el = document.getElementById('compare-result');
  el.textContent = 'Loading...';

  api('/api/compare?a=' + a + '&b=' + b).then(function(data) {
    if (!data) { el.textContent = 'Could not load comparison data.'; return; }

    var runA = data.run_a;
    var runB = data.run_b;

    // Summary cards
    var html = '<div class="compare-grid mt-2">';
    html += renderCompareCard(runA, 'A');
    html += renderCompareCard(runB, 'B');
    html += '</div>';

    // Diff table
    html += '<div class="card mt-2"><div class="card-header"><h2>Score Comparison</h2></div><div class="card-body">';
    html += '<table><thead><tr>' +
      '<th>Test :: Metric</th><th>#' + esc(runA.run_id) + '</th><th>#' + esc(runB.run_id) + '</th><th>Delta</th><th>Status</th>' +
      '</tr></thead><tbody>';

    var scoresA = buildScoreMap(runA.results || {});
    var scoresB = buildScoreMap(runB.results || {});
    var keySet = {};
    var k;
    for (k in scoresA) keySet[k] = true;
    for (k in scoresB) keySet[k] = true;
    var allKeys = Object.keys(keySet).sort();

    for (var ki = 0; ki < allKeys.length; ki++) {
      var key = allKeys[ki];
      var cA = scoresA[key];
      var cB = scoresB[key];
      var sA = cA != null ? cA.score : null;
      var sB = cB != null ? cB.score : null;
      var sAStr = sA != null ? sA.toFixed(3) : '-';
      var sBStr = sB != null ? sB.toFixed(3) : '-';
      var deltaStr, statusBadge;
      if (sA != null && sB != null) {
        var d = sB - sA;
        deltaStr = (d >= 0 ? '+' : '') + d.toFixed(3);
        var dcls = d > 0.01 ? 'positive' : d < -0.01 ? 'negative' : 'neutral';
        var dlabel = d > 0.01 ? 'IMPROVED' : d < -0.01 ? 'REGRESSED' : 'SAME';
        var dbadge = d > 0.01 ? 'pass' : d < -0.01 ? 'fail' : 'skip';
        statusBadge = '<span class="badge ' + dbadge + '">' + dlabel + '</span>';
        deltaStr = '<span class="delta ' + dcls + '">' + esc(deltaStr) + '</span>';
      } else if (sA == null) {
        deltaStr = '<span class="text-muted">new</span>';
        statusBadge = '<span class="badge skip">NEW</span>';
      } else {
        deltaStr = '<span class="text-muted">removed</span>';
        statusBadge = '<span class="badge skip">REMOVED</span>';
      }
      html += '<tr>' +
        '<td>' + esc(key) + '</td><td class="mono">' + esc(sAStr) + '</td><td class="mono">' + esc(sBStr) + '</td>' +
        '<td class="mono">' + deltaStr + '</td><td>' + statusBadge + '</td>' +
        '</tr>';
    }

    html += '</tbody></table></div></div>';

    // Cost comparison
    var costDelta = runB.total_cost - runA.total_cost;
    var passDelta = runB.passed_checks - runA.passed_checks;
    var costCls = costDelta > 0 ? 'negative' : costDelta < 0 ? 'positive' : 'neutral';
    var passCls = passDelta > 0 ? 'positive' : passDelta < 0 ? 'negative' : 'neutral';
    html += '<div class="text-muted mt-2" style="font-size:.85rem;">' +
      'Cost: $' + esc(runA.total_cost.toFixed(4)) + ' &rarr; $' + esc(runB.total_cost.toFixed(4)) +
      ' (<span class="delta ' + costCls + '">' + esc((costDelta >= 0 ? '+' : '') + costDelta.toFixed(4)) + '</span>)' +
      ' &middot; ' +
      'Passed: ' + esc(runA.passed_checks) + '/' + esc(runA.total_checks) + ' &rarr; ' + esc(runB.passed_checks) + '/' + esc(runB.total_checks) +
      ' (<span class="delta ' + passCls + '">' + esc((passDelta >= 0 ? '+' : '') + passDelta) + '</span>)' +
      '</div>';

    setContent(el, html);
  });
}

function renderCompareCard(run, label) {
  var ts = new Date(run.timestamp * 1000).toLocaleString();
  var passRate = run.total_checks > 0 ? ((run.passed_checks / run.total_checks) * 100).toFixed(1) : '0.0';
  return '<div class="card">' +
    '<div class="card-header"><h2>Run #' + esc(run.run_id) + ' (' + esc(label) + ')</h2></div>' +
    '<div class="card-body">' +
    '<div class="text-muted mb-1">' + esc(ts) + ' &middot; ' + esc(run.label || 'unlabeled') + '</div>' +
    '<div style="display:flex;gap:24px;flex-wrap:wrap;">' +
    '<div><span class="text-green" style="font-size:1.3rem;font-weight:700;">' + esc(run.passed_checks) + '</span> <span class="text-muted">passed</span></div>' +
    '<div><span class="text-red" style="font-size:1.3rem;font-weight:700;">' + esc(run.failed_checks) + '</span> <span class="text-muted">failed</span></div>' +
    '<div><span class="text-yellow" style="font-size:1.3rem;font-weight:700;">' + esc(passRate) + '%</span> <span class="text-muted">pass rate</span></div>' +
    '<div><span style="font-size:1.3rem;font-weight:700;color:var(--purple);">$' + esc(run.total_cost.toFixed(4)) + '</span> <span class="text-muted">cost</span></div>' +
    '</div></div></div>';
}

function buildScoreMap(results) {
  var m = {};
  var testNames = Object.keys(results);
  for (var i = 0; i < testNames.length; i++) {
    var checks = results[testNames[i]];
    for (var j = 0; j < checks.length; j++) {
      m[testNames[i] + '::' + (checks[j].metric_name || '')] = checks[j];
    }
  }
  return m;
}

/* ---- Trend Chart (all dynamic values escaped via esc()) ---- */
function renderTrendChart(container, points, label) {
  if (!points || points.length === 0) {
    setContent(container, '<p class="empty">No trend data available</p>');
    return;
  }
  var w = 600, h = 200, pad = 40;
  var scores = points.map(function(p) { return p.score; });
  var minS = Math.min.apply(null, scores.concat([0]));
  var maxS = Math.max.apply(null, scores.concat([1]));
  var range = maxS - minS || 1;

  var xStep = (w - 2 * pad) / Math.max(points.length - 1, 1);
  var pts = points.map(function(p, i) {
    var x = pad + i * xStep;
    var y = h - pad - ((p.score - minS) / range) * (h - 2 * pad);
    return { x: x, y: y, score: p.score, passed: p.passed };
  });

  var pathD = pts.map(function(p, i) { return (i === 0 ? 'M' : 'L') + p.x + ',' + p.y; }).join(' ');

  var svg = '<svg viewBox="0 0 ' + w + ' ' + h + '" style="width:100%;max-width:' + w + 'px">';
  // Grid lines
  for (var i = 0; i <= 4; i++) {
    var gy = pad + i * (h - 2 * pad) / 4;
    var val = (maxS - (i / 4) * range).toFixed(2);
    svg += '<line x1="' + pad + '" y1="' + gy + '" x2="' + (w - pad) + '" y2="' + gy + '" stroke="var(--bg4)" stroke-dasharray="4"/>';
    svg += '<text x="' + (pad - 5) + '" y="' + (gy + 4) + '" text-anchor="end" fill="var(--fg3)" font-size="11">' + esc(val) + '</text>';
  }
  // Line
  svg += '<path d="' + pathD + '" fill="none" stroke="var(--accent)" stroke-width="2"/>';
  // Dots — colors are hardcoded CSS vars (not dynamic data)
  pts.forEach(function(p) {
    var color = p.passed ? 'var(--green)' : 'var(--red)';
    svg += '<circle cx="' + p.x + '" cy="' + p.y + '" r="4" fill="' + color + '"/>';
  });
  svg += '</svg>';
  // All dynamic content (label, val) is escaped via esc(); SVG coords are numbers
  setContent(container, '<h3 style="color:var(--fg2);margin-bottom:8px">' + esc(label) + '</h3>' + svg);
}

/* ---- Cost Breakdown (all dynamic values escaped via esc()) ---- */
function renderCostBreakdown(container, data) {
  if (!data || data.total_cost === 0) {
    setContent(container, '<p class="empty">No cost data for this run</p>');
    return;
  }
  var html = '<div class="stats-row">' +
    '<div class="stat-card yellow"><div class="value">$' + esc(data.total_cost.toFixed(4)) + '</div><div class="label">Total Cost</div></div>' +
    '</div>';

  html += '<h3 style="color:var(--fg2);margin:16px 0 8px">By Metric</h3><div class="bar-chart">';
  for (var name in data.by_metric) {
    var cost = data.by_metric[name];
    if (cost <= 0) continue;
    var pct = Math.min(100, (cost / data.total_cost) * 100);
    html += '<div class="bar-row">' +
      '<div class="bar-label">' + esc(name) + '</div>' +
      '<div class="bar-track"><div class="bar-fill ok" style="width:' + esc(pct.toFixed(2)) + '%"></div></div>' +
      '<div class="bar-value">$' + esc(cost.toFixed(4)) + '</div>' +
      '</div>';
  }
  html += '</div>';

  html += '<h3 style="color:var(--fg2);margin:16px 0 8px">By Test</h3><div class="bar-chart">';
  for (var tname in data.by_test) {
    var tcost = data.by_test[tname];
    if (tcost <= 0) continue;
    var tpct = Math.min(100, (tcost / data.total_cost) * 100);
    html += '<div class="bar-row">' +
      '<div class="bar-label">' + esc(tname) + '</div>' +
      '<div class="bar-track"><div class="bar-fill ok" style="width:' + esc(tpct.toFixed(2)) + '%"></div></div>' +
      '<div class="bar-value">$' + esc(tcost.toFixed(4)) + '</div>' +
      '</div>';
  }
  html += '</div>';

  setContent(container, html);
}

/* ---- Boot ---- */
init();
</script>
</body>
</html>"""

# ---------------------------------------------------------------------------
# HTTP handler
# ---------------------------------------------------------------------------


class DashboardHandler(http.server.BaseHTTPRequestHandler):
    """HTTP request handler serving the dashboard UI and JSON API."""

    # Class-level state shared across requests (set before server starts).
    # We store the *path* rather than an open RunHistory so each handler
    # thread can create its own SQLite connection (SQLite objects cannot
    # cross threads by default).
    _db_path: str | None = None
    _results_dir: Path | None = None

    def do_GET(self) -> None:  # noqa: N802
        path = self.path.split("?")[0]
        query = self._parse_query()

        if path == "/" or path == "/index.html":
            self._serve_html()
        elif path == "/api/runs":
            self._serve_runs()
        elif path.startswith("/api/runs/"):
            run_id = path.split("/")[-1]
            self._serve_run_detail(run_id)
        elif path == "/api/compare":
            self._serve_compare(query)
        elif path == "/api/metrics":
            self._serve_metrics()
        elif path == "/api/trends":
            self._serve_trends(query)
        elif path == "/api/cost-breakdown":
            self._serve_cost_breakdown(query)
        else:
            self._send_json({"error": "not found"}, status=404)

    def log_message(self, format: str, *args: Any) -> None:
        """Suppress default request logging to keep terminal clean."""
        logger.debug(format, *args)

    # -- helpers ------------------------------------------------------------

    def _parse_query(self) -> dict[str, str]:
        """Parse query string parameters from the request path."""
        qs: dict[str, str] = {}
        if "?" in self.path:
            raw = self.path.split("?", 1)[1]
            for part in raw.split("&"):
                if "=" in part:
                    k, v = part.split("=", 1)
                    qs[k] = v
        return qs

    def _serve_html(self) -> None:
        body = DASHBOARD_HTML.encode("utf-8")
        self.send_response(200)
        self.send_header("Content-Type", "text/html; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-cache")
        self.end_headers()
        self.wfile.write(body)

    def _send_json(self, data: Any, status: int = 200) -> None:
        body = json.dumps(data, default=str).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-cache")
        self.end_headers()
        self.wfile.write(body)

    def _get_history(self) -> RunHistory:
        """Create a RunHistory instance for this request.

        A new connection is created per request to avoid SQLite's
        cross-thread restriction.  The caller should close it when done,
        or use it within a short-lived scope.
        """
        return RunHistory(db_path=DashboardHandler._db_path)

    def _serve_runs(self) -> None:
        """GET /api/runs -- list all runs as lightweight summaries."""
        history = self._get_history()
        try:
            runs = history.list_runs(limit=200)
            self._send_json([
                {
                    "run_id": r.run_id,
                    "timestamp": r.timestamp,
                    "label": r.label,
                    "git_commit": r.git_commit,
                    "total_cost": r.total_cost,
                    "total_checks": r.total_checks,
                    "passed_checks": r.passed_checks,
                    "failed_checks": r.failed_checks,
                }
                for r in runs
            ])
        finally:
            history.close()

    def _serve_run_detail(self, run_id_str: str) -> None:
        """GET /api/runs/<id> -- full detail for one run."""
        try:
            run_id = int(run_id_str)
        except ValueError:
            self._send_json({"error": "invalid run id"}, status=400)
            return

        history = self._get_history()
        try:
            record = history.get_run(run_id)
            if record is None:
                self._send_json({"error": "run not found"}, status=404)
                return

            self._send_json({
                "run_id": record.run_id,
                "timestamp": record.timestamp,
                "label": record.label,
                "git_commit": record.git_commit,
                "total_cost": record.total_cost,
                "total_checks": record.total_checks,
                "passed_checks": record.passed_checks,
                "failed_checks": record.failed_checks,
                "results": record.results,
            })
        finally:
            history.close()

    def _serve_compare(self, query: dict[str, str]) -> None:
        """GET /api/compare?a=<id>&b=<id> -- compare two runs."""
        a_str = query.get("a", "")
        b_str = query.get("b", "")
        try:
            a_id, b_id = int(a_str), int(b_str)
        except ValueError:
            self._send_json({"error": "invalid run ids"}, status=400)
            return

        history = self._get_history()
        try:
            run_a = history.get_run(a_id)
            run_b = history.get_run(b_id)
            if run_a is None or run_b is None:
                self._send_json({"error": "one or both runs not found"}, status=404)
                return

            def _serialize_run(r: RunRecord) -> dict[str, Any]:
                return {
                    "run_id": r.run_id,
                    "timestamp": r.timestamp,
                    "label": r.label,
                    "git_commit": r.git_commit,
                    "total_cost": r.total_cost,
                    "total_checks": r.total_checks,
                    "passed_checks": r.passed_checks,
                    "failed_checks": r.failed_checks,
                    "results": r.results,
                }

            self._send_json({
                "run_a": _serialize_run(run_a),
                "run_b": _serialize_run(run_b),
            })
        finally:
            history.close()

    def _serve_trends(self, query: dict[str, str]) -> None:
        """GET /api/trends?metric=<name>&test=<test>&limit=<n> -- score trend over time."""
        metric_name = query.get("metric", "")
        test_name = query.get("test", "")
        limit = int(query.get("limit", "20"))

        if not metric_name or not test_name:
            self._send_json({"error": "metric and test params required"}, status=400)
            return

        history = self._get_history()
        try:
            data = history.get_metric_trend(test_name, metric_name, limit=limit)
            self._send_json({"points": data})
        finally:
            history.close()

    def _serve_cost_breakdown(self, query: dict[str, str]) -> None:
        """GET /api/cost-breakdown?run=<id> -- cost per metric and test."""
        run_id = query.get("run", "")
        try:
            rid = int(run_id)
        except ValueError:
            self._send_json({"error": "invalid run id"}, status=400)
            return

        history = self._get_history()
        try:
            record = history.get_run(rid)
            if record is None:
                self._send_json({"error": "run not found"}, status=404)
                return

            by_metric: dict[str, float] = {}
            by_test: dict[str, float] = {}
            for test_name, checks in record.results.items():
                test_cost = 0.0
                for c in checks:
                    cost = c.get("cost", 0.0)
                    metric = c.get("metric_name", "unknown")
                    by_metric[metric] = by_metric.get(metric, 0.0) + cost
                    test_cost += cost
                by_test[test_name] = test_cost

            self._send_json({
                "total_cost": record.total_cost,
                "by_metric": dict(sorted(by_metric.items(), key=lambda x: -x[1])),
                "by_test": dict(sorted(by_test.items(), key=lambda x: -x[1])),
            })
        finally:
            history.close()

    def _serve_metrics(self) -> None:
        """GET /api/metrics -- list available metric names."""
        judge_metrics = [
            "hallucination", "relevance", "toxicity", "rubric", "fluency",
            "coherence", "sentiment", "correctness", "faithfulness",
            "context_relevance", "answer_completeness", "instruction_following",
            "summarization", "bias", "consistency", "groundedness",
        ]
        deterministic_metrics = [
            "contains", "not_contains", "exact_match", "starts_with",
            "ends_with", "regex", "max_tokens", "min_tokens", "word_count",
            "char_count", "sentence_count", "similarity", "readability",
            "latency", "cost", "json_schema", "is_json", "is_valid_python",
            "all_of", "any_of", "none_of", "bleu", "rouge_l", "no_pii",
        ]
        self._send_json({
            "judge": judge_metrics,
            "deterministic": deterministic_metrics,
        })


# ---------------------------------------------------------------------------
# Server entry point
# ---------------------------------------------------------------------------


def start_dashboard(
    port: int = 8484,
    host: str = "localhost",
    open_browser: bool = True,
    db_path: str | None = None,
    results_dir: str | None = None,
) -> None:
    """Start the dashboard HTTP server.

    Parameters
    ----------
    port:
        TCP port to listen on (default ``8484``).
    host:
        Bind address (default ``localhost``).
    open_browser:
        If ``True``, open the dashboard URL in the default browser.
    db_path:
        Path to the RunHistory SQLite database. Defaults to
        ``.checkllm/history.db``.
    results_dir:
        Optional directory containing JSON result files.
    """
    DashboardHandler._db_path = db_path

    if results_dir:
        DashboardHandler._results_dir = Path(results_dir)

    server = http.server.HTTPServer((host, port), DashboardHandler)
    url = f"http://{host}:{port}"

    # Rich banner
    try:
        from rich.console import Console
        console = Console()
        console.print(f"\n  [bold green]checkllm dashboard[/] running at [bold cyan]{url}[/]")
        console.print(f"  [dim]Press Ctrl+C to stop.[/]\n")
    except ImportError:
        print(f"\n  checkllm dashboard running at {url}")
        print(f"  Press Ctrl+C to stop.\n")

    if open_browser:
        threading.Timer(0.5, lambda: webbrowser.open(url)).start()

    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()
        logger.info("Dashboard server stopped.")
