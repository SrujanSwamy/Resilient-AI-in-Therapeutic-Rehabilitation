"""
Clinician Dashboard
--------------------
End-Semester Enhancement: Web-based dashboard for physiotherapists to
monitor patient progress, review session details, and compare against
population benchmarks.

Runs as a standalone Flask web app. Communicates with the REST API
(or directly with the database if API is not running).

Launch:
    python subteam2_cloud/dashboard.py
    # Opens at http://localhost:5000
"""

import os
import sys
import json
import logging
from datetime import datetime
from typing import Dict, List

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from subteam2_cloud.database import RehabDatabase
from subteam2_cloud.benchmarking import PopulationBenchmark

logger = logging.getLogger(__name__)

DB_PATH = os.path.join(ROOT, "rehab_data")

# -----------------------------------------------------------------------
# HTML Template (single-file, no external files needed)
# -----------------------------------------------------------------------

DASHBOARD_HTML = """<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>RehabAI Clinician Dashboard</title>
<style>
  :root {
    --bg: #0f1117;
    --surface: #1a1d2e;
    --card: #222537;
    --border: #2e3250;
    --accent: #6c63ff;
    --accent2: #00d4aa;
    --danger: #ff5c7a;
    --warn: #ffb547;
    --text: #e2e8f0;
    --sub: #8892b0;
    --green: #22d3a4;
    --red: #ff5c7a;
  }
  * { margin:0; padding:0; box-sizing:border-box; }
  body {
    font-family: 'Segoe UI', system-ui, sans-serif;
    background: var(--bg);
    color: var(--text);
    min-height: 100vh;
  }

  /* ---- Sidebar ---- */
  .sidebar {
    position: fixed; top:0; left:0; height:100vh; width:220px;
    background: var(--surface);
    border-right: 1px solid var(--border);
    display:flex; flex-direction:column;
    padding: 24px 0;
    z-index: 100;
  }
  .logo {
    padding: 0 20px 24px;
    font-size: 1.1rem; font-weight: 700;
    color: var(--accent);
    border-bottom: 1px solid var(--border);
    display:flex; align-items:center; gap:10px;
  }
  .logo::before { content:"🩺"; font-size:1.3rem; }
  nav a {
    display:flex; align-items:center; gap:10px;
    padding: 12px 20px;
    color: var(--sub);
    text-decoration:none;
    font-size:0.9rem;
    transition: all 0.2s;
    border-left: 3px solid transparent;
  }
  nav a:hover, nav a.active {
    color: var(--text);
    background: rgba(108,99,255,0.12);
    border-left-color: var(--accent);
  }
  .nav-icon { font-size:1.1rem; width:20px; text-align:center; }

  /* ---- Main ---- */
  .main { margin-left:220px; padding: 28px 32px; }
  .header {
    display:flex; justify-content:space-between; align-items:center;
    margin-bottom:28px;
  }
  h1 { font-size:1.5rem; font-weight:700; }
  .badge {
    background:var(--accent); color:#fff;
    padding:4px 12px; border-radius:20px; font-size:0.78rem; font-weight:600;
  }

  /* ---- KPI Cards ---- */
  .kpi-grid { display:grid; grid-template-columns:repeat(4,1fr); gap:16px; margin-bottom:28px; }
  .kpi {
    background:var(--card);
    border:1px solid var(--border);
    border-radius:12px;
    padding:20px;
    transition: transform 0.2s;
  }
  .kpi:hover { transform:translateY(-2px); }
  .kpi-label { font-size:0.75rem; color:var(--sub); text-transform:uppercase; letter-spacing:0.5px; margin-bottom:8px; }
  .kpi-value { font-size:2rem; font-weight:700; }
  .kpi-sub { font-size:0.78rem; color:var(--sub); margin-top:4px; }
  .kpi-green .kpi-value { color:var(--green); }
  .kpi-red .kpi-value { color:var(--red); }
  .kpi-accent .kpi-value { color:var(--accent); }
  .kpi-warn .kpi-value { color:var(--warn); }

  /* ---- Panels ---- */
  .panel-grid { display:grid; grid-template-columns:1fr 1fr; gap:20px; margin-bottom:24px; }
  .panel {
    background:var(--card);
    border:1px solid var(--border);
    border-radius:12px;
    padding:20px;
  }
  .panel-full { grid-column:1/-1; }
  .panel h2 { font-size:0.95rem; font-weight:600; margin-bottom:16px; color:var(--sub); text-transform:uppercase; letter-spacing:0.5px;}

  /* ---- Table ---- */
  table { width:100%; border-collapse:collapse; font-size:0.85rem; }
  th { text-align:left; padding:8px 12px; color:var(--sub); font-weight:500; border-bottom:1px solid var(--border); font-size:0.78rem; text-transform:uppercase; letter-spacing:0.5px; }
  td { padding:10px 12px; border-bottom:1px solid rgba(255,255,255,0.04); }
  tr:last-child td { border-bottom:none; }
  tr:hover td { background:rgba(255,255,255,0.03); }

  /* ---- Pill badges ---- */
  .pill {
    display:inline-block; padding:3px 10px; border-radius:20px;
    font-size:0.72rem; font-weight:600;
  }
  .pill-correct { background:rgba(34,211,164,0.15); color:var(--green);}
  .pill-incorrect { background:rgba(255,92,122,0.15); color:var(--red);}
  .pill-warn { background:rgba(255,181,71,0.15); color:var(--warn);}
  .pill-info { background:rgba(108,99,255,0.15); color:var(--accent);}

  /* ---- Bar chart ---- */
  .bar-row { display:flex; align-items:center; gap:12px; margin-bottom:10px; font-size:0.82rem; }
  .bar-label { width:100px; color:var(--sub); }
  .bar-outer { flex:1; background:rgba(255,255,255,0.06); border-radius:4px; height:10px; overflow:hidden; }
  .bar-inner { height:100%; border-radius:4px; transition:width 0.6s ease; }
  .bar-val { width:50px; text-align:right; color:var(--text); }

  /* ---- Progress ring ---- */
  .ring-wrap { display:flex; align-items:center; gap:20px; }
  .ring-info p { font-size:0.82rem; color:var(--sub); margin-bottom:6px; }
  .ring-info strong { font-size:1.4rem; font-weight:700; }

  /* ---- Alerts ---- */
  .alert { background:rgba(255,181,71,0.08); border:1px solid rgba(255,181,71,0.3); border-radius:8px; padding:12px 16px; margin-bottom:10px; font-size:0.85rem; }
  .alert-title { font-weight:600; color:var(--warn); margin-bottom:4px; }
  .alert-body { color:var(--sub); }

  /* ---- Responsive ---- */
  @media(max-width:900px) {
    .kpi-grid { grid-template-columns:repeat(2,1fr); }
    .panel-grid { grid-template-columns:1fr; }
    .sidebar { width:180px; }
    .main { margin-left:180px; }
  }

  /* ---- Section toggle ---- */
  section { display:none; }
  section.active { display:block; }
  .refresh-btn {
    background:var(--accent); color:#fff; border:none; border-radius:8px;
    padding:8px 18px; cursor:pointer; font-size:0.85rem; font-weight:600;
    transition:background 0.2s;
  }
  .refresh-btn:hover { background:#5753e0; }
</style>
</head>
<body>

<div class="sidebar">
  <div class="logo">RehabAI</div>
  <nav>
    <a href="#" class="active nav-link" data-section="overview">
      <span class="nav-icon">📊</span> Overview
    </a>
    <a href="#" class="nav-link" data-section="patients">
      <span class="nav-icon">👤</span> Patients
    </a>
    <a href="#" class="nav-link" data-section="sessions">
      <span class="nav-icon">📋</span> Sessions
    </a>
    <a href="#" class="nav-link" data-section="benchmark">
      <span class="nav-icon">📈</span> Benchmark
    </a>
    <a href="#" class="nav-link" data-section="alerts">
      <span class="nav-icon">⚠️</span> Alerts
    </a>
  </nav>
</div>

<div class="main">
  <div class="header">
    <div>
      <h1>Clinician Dashboard</h1>
      <p style="color:var(--sub);font-size:0.82rem;margin-top:4px;">Resilient AI in Therapeutic Rehabilitation</p>
    </div>
    <div style="display:flex;gap:12px;align-items:center;">
      <span class="badge">LIVE</span>
      <button class="refresh-btn" onclick="loadAllData()">↻ Refresh</button>
    </div>
  </div>

  <!-- OVERVIEW -->
  <section id="overview" class="active">
    <div class="kpi-grid" id="kpi-grid">
      <div class="kpi kpi-accent"><div class="kpi-label">Total Sessions</div><div class="kpi-value" id="kpi-total">—</div></div>
      <div class="kpi kpi-green"><div class="kpi-label">Accuracy Rate</div><div class="kpi-value" id="kpi-acc">—</div></div>
      <div class="kpi kpi-warn"><div class="kpi-label">Avg Fluidity</div><div class="kpi-value" id="kpi-flu">—</div></div>
      <div class="kpi kpi-red"><div class="kpi-label">Compensation Rate</div><div class="kpi-value" id="kpi-comp">—</div></div>
    </div>

    <div class="panel-grid">
      <div class="panel">
        <h2>Accuracy by Exercise</h2>
        <div id="acc-bars">Loading…</div>
      </div>
      <div class="panel">
        <h2>Fluidity by Exercise</h2>
        <div id="flu-bars">Loading…</div>
      </div>
      <div class="panel panel-full">
        <h2>Recent Sessions</h2>
        <table>
          <thead><tr>
            <th>Patient</th><th>Exercise</th><th>Result</th>
            <th>RMSE</th><th>Confidence</th><th>Fluidity</th><th>Compensation</th><th>Time</th>
          </tr></thead>
          <tbody id="recent-sessions">
            <tr><td colspan="8" style="color:var(--sub);text-align:center;">Loading…</td></tr>
          </tbody>
        </table>
      </div>
    </div>
  </section>

  <!-- PATIENTS -->
  <section id="patients">
    <div class="panel">
      <h2>All Patients</h2>
      <table>
        <thead><tr>
          <th>Patient ID</th><th>Sessions</th><th>Accuracy</th>
          <th>Mean RMSE</th><th>Avg Fluidity</th><th>Compensation Rate</th><th>Last Session</th>
        </tr></thead>
        <tbody id="patients-table">
          <tr><td colspan="7" style="color:var(--sub);text-align:center;">Loading…</td></tr>
        </tbody>
      </table>
    </div>
  </section>

  <!-- SESSIONS -->
  <section id="sessions">
    <div class="panel">
      <h2>All Sessions</h2>
      <table>
        <thead><tr>
          <th>Session ID</th><th>Patient</th><th>Exercise</th><th>Result</th>
          <th>RMSE</th><th>Confidence</th><th>Fluidity</th><th>Compensation</th><th>Timestamp</th>
        </tr></thead>
        <tbody id="sessions-table">
          <tr><td colspan="9" style="color:var(--sub);text-align:center;">Loading…</td></tr>
        </tbody>
      </table>
    </div>
  </section>

  <!-- BENCHMARK -->
  <section id="benchmark">
    <div id="benchmark-panels">
      <p style="color:var(--sub)">Loading benchmark data…</p>
    </div>
  </section>

  <!-- ALERTS -->
  <section id="alerts">
    <div class="panel">
      <h2>Compensation & Low-Confidence Alerts</h2>
      <div id="alerts-list"><p style="color:var(--sub)">Loading alerts…</p></div>
    </div>
  </section>
</div>

<script>
const API = '/api';
let _sessions = [];

async function apiFetch(path) {
  try {
    const r = await fetch(API + path);
    if (!r.ok) throw new Error(r.statusText);
    return await r.json();
  } catch(e) {
    console.error('API error:', path, e);
    return null;
  }
}

function pct(val) { return val !== null && val !== undefined ? (val * 100).toFixed(1) + '%' : '—'; }
function num(val, d=3) { return val !== null && val !== undefined ? parseFloat(val).toFixed(d) : '—'; }
function ts(val) { return val ? val.substring(0,19).replace('T',' ') : '—'; }
function pillResult(c) {
  if (c === 1 || c === true) return '<span class="pill pill-correct">Correct</span>';
  if (c === 0 || c === false) return '<span class="pill pill-incorrect">Incorrect</span>';
  return '<span class="pill pill-info">—</span>';
}
function pillComp(found, types) {
  if (!found) return '<span class="pill pill-correct">None</span>';
  return `<span class="pill pill-warn">${(types||[]).join(', ') || 'Yes'}</span>`;
}

async function loadOverview() {
  const data = await apiFetch('/sessions?limit=200');
  if (!data) return;
  _sessions = data.sessions || [];

  const total = data.total;
  document.getElementById('kpi-total').textContent = total;

  const correct_vals = _sessions.filter(s=>s.correctness!==null).map(s=>s.correctness);
  const flu_vals = _sessions.filter(s=>s.fluidity_score!==null).map(s=>s.fluidity_score);
  const comp_count = _sessions.filter(s=>s.compensation_found).length;

  const acc = correct_vals.length ? correct_vals.reduce((a,b)=>a+b,0)/correct_vals.length : null;
  const flu = flu_vals.length ? flu_vals.reduce((a,b)=>a+b,0)/flu_vals.length : null;

  document.getElementById('kpi-acc').textContent = acc !== null ? (acc*100).toFixed(1)+'%' : '—';
  document.getElementById('kpi-flu').textContent = flu !== null ? (flu*100).toFixed(1)+'%' : '—';
  document.getElementById('kpi-comp').textContent = total>0 ? (comp_count/total*100).toFixed(1)+'%' : '—';

  // Recent sessions (last 10)
  const recent = _sessions.slice(0, 10);
  const tbody = document.getElementById('recent-sessions');
  tbody.innerHTML = recent.map(s => `<tr>
    <td>${s.patient_id||'—'}</td>
    <td><span class="pill pill-info">${s.exercise_id||'—'}</span></td>
    <td>${pillResult(s.correctness)}</td>
    <td>${num(s.rmse)}</td>
    <td>${num(s.confidence,2)}</td>
    <td>${num(s.fluidity_score,2)}</td>
    <td>${pillComp(s.compensation_found, s.compensation_types)}</td>
    <td style="color:var(--sub);font-size:0.78rem">${ts(s.timestamp)}</td>
  </tr>`).join('') || '<tr><td colspan="8" style="color:var(--sub);text-align:center">No sessions</td></tr>';

  // Per-exercise bars
  const exercises = ['Ex1','Ex2','Ex3','Ex4','Ex5','Ex6'];
  const exColors = ['#6c63ff','#00d4aa','#ffb547','#ff5c7a','#60a5fa','#a78bfa'];

  ['acc-bars','flu-bars'].forEach(id => {
    const el = document.getElementById(id);
    el.innerHTML = exercises.map((ex, i) => {
      const exSessions = _sessions.filter(s=>s.exercise_id===ex);
      let val = 0;
      if (id==='acc-bars') {
        const cv = exSessions.filter(s=>s.correctness!==null).map(s=>s.correctness);
        val = cv.length ? cv.reduce((a,b)=>a+b,0)/cv.length : 0;
      } else {
        const fv = exSessions.filter(s=>s.fluidity_score!==null).map(s=>s.fluidity_score);
        val = fv.length ? fv.reduce((a,b)=>a+b,0)/fv.length : 0;
      }
      return `<div class="bar-row">
        <span class="bar-label">${ex}</span>
        <div class="bar-outer"><div class="bar-inner" style="width:${(val*100).toFixed(1)}%;background:${exColors[i]}"></div></div>
        <span class="bar-val">${(val*100).toFixed(1)}%</span>
      </div>`;
    }).join('');
  });
}

async function loadPatients() {
  if (!_sessions.length) await loadOverview();
  const patientMap = {};
  _sessions.forEach(s => {
    const pid = s.patient_id;
    if(!pid) return;
    if(!patientMap[pid]) patientMap[pid] = [];
    patientMap[pid].push(s);
  });

  const tbody = document.getElementById('patients-table');
  tbody.innerHTML = Object.entries(patientMap).map(([pid, ss]) => {
    const cv = ss.filter(s=>s.correctness!==null).map(s=>s.correctness);
    const rv = ss.filter(s=>s.rmse!==null).map(s=>s.rmse);
    const fv = ss.filter(s=>s.fluidity_score!==null).map(s=>s.fluidity_score);
    const cn = ss.filter(s=>s.compensation_found).length;
    const acc = cv.length ? cv.reduce((a,b)=>a+b,0)/cv.length : null;
    const rmse = rv.length ? rv.reduce((a,b)=>a+b,0)/rv.length : null;
    const flu = fv.length ? fv.reduce((a,b)=>a+b,0)/fv.length : null;
    const last = ss[0]?.timestamp;
    return `<tr>
      <td>${pid}</td>
      <td>${ss.length}</td>
      <td>${acc!==null?`<span class="pill ${acc>=0.5?'pill-correct':'pill-incorrect'}">${(acc*100).toFixed(1)}%</span>`:'—'}</td>
      <td>${num(rmse)}</td>
      <td>${flu!==null?(flu*100).toFixed(1)+'%':'—'}</td>
      <td>${cn>0?`<span class="pill pill-warn">${(cn/ss.length*100).toFixed(0)}%</span>`:'<span class="pill pill-correct">0%</span>'}</td>
      <td style="color:var(--sub);font-size:0.78rem">${ts(last)}</td>
    </tr>`;
  }).join('') || '<tr><td colspan="7" style="color:var(--sub);text-align:center">No patients</td></tr>';
}

async function loadSessions() {
  if (!_sessions.length) await loadOverview();
  const tbody = document.getElementById('sessions-table');
  tbody.innerHTML = _sessions.map(s => `<tr>
    <td style="font-family:monospace;font-size:0.72rem;color:var(--sub)">${(s.session_id||'').substring(0,12)}…</td>
    <td>${s.patient_id||'—'}</td>
    <td><span class="pill pill-info">${s.exercise_id||'—'}</span></td>
    <td>${pillResult(s.correctness)}</td>
    <td>${num(s.rmse)}</td>
    <td>${num(s.confidence,2)}</td>
    <td>${num(s.fluidity_score,2)}</td>
    <td>${pillComp(s.compensation_found,s.compensation_types)}</td>
    <td style="color:var(--sub);font-size:0.78rem">${ts(s.timestamp)}</td>
  </tr>`).join('') || '<tr><td colspan="9" style="color:var(--sub);text-align:center">No sessions</td></tr>';
}

async function loadBenchmark() {
  const exercises = ['Ex1','Ex2','Ex3','Ex4','Ex5','Ex6'];
  const panels = [];
  for(const ex of exercises) {
    const data = await apiFetch(`/benchmark/${ex}`);
    if(!data || data.n === 0) continue;
    const p = data.rmse_percentiles || {};
    panels.push(`<div class="panel" style="margin-bottom:16px">
      <h2>${ex} — Population Benchmark (n=${data.n})</h2>
      <div class="panel-grid" style="grid-template-columns:repeat(3,1fr);gap:12px;margin-top:12px">
        <div><span style="color:var(--sub);font-size:0.78rem">Mean RMSE</span><br><strong>${num(data.mean_rmse)}</strong></div>
        <div><span style="color:var(--sub);font-size:0.78rem">Accuracy</span><br><strong>${data.accuracy_rate!==null?pct(data.accuracy_rate):'—'}</strong></div>
        <div><span style="color:var(--sub);font-size:0.78rem">RMSE p50</span><br><strong>${num(p.p50)}</strong></div>
        <div><span style="color:var(--sub);font-size:0.78rem">RMSE p25</span><br><strong>${num(p.p25)}</strong></div>
        <div><span style="color:var(--sub);font-size:0.78rem">RMSE p75</span><br><strong>${num(p.p75)}</strong></div>
        <div><span style="color:var(--sub);font-size:0.78rem">Mean Fluidity</span><br><strong>${data.mean_fluidity!==null?pct(data.mean_fluidity):'—'}</strong></div>
      </div>
    </div>`);
  }
  document.getElementById('benchmark-panels').innerHTML = panels.join('') || '<p style="color:var(--sub)">No benchmark data available yet.</p>';
}

async function loadAlerts() {
  if (!_sessions.length) await loadOverview();
  const alerts = _sessions.filter(s => s.compensation_found || (s.confidence!==null && s.confidence < 0.8));
  const el = document.getElementById('alerts-list');
  el.innerHTML = alerts.length
    ? alerts.map(s => `<div class="alert">
        <div class="alert-title">${s.compensation_found?'⚠ Compensation Detected':'⚡ Low Confidence Alert'} — ${s.patient_id} / ${s.exercise_id}</div>
        <div class="alert-body">
          ${s.compensation_found?'Types: '+(s.compensation_types||[]).join(', '):''}
          ${s.confidence!==null&&s.confidence<0.8?'Confidence: '+(s.confidence*100).toFixed(0)+'%':''}
          &nbsp;·&nbsp; ${ts(s.timestamp)}
        </div>
      </div>`).join('')
    : '<p style="color:var(--sub)">No alerts — all sessions within normal parameters.</p>';
}

async function loadAllData() {
  await loadOverview();
  await loadPatients();
  await loadSessions();
  await loadBenchmark();
  await loadAlerts();
}

// Nav switching
document.querySelectorAll('.nav-link').forEach(link => {
  link.addEventListener('click', e => {
    e.preventDefault();
    const sec = link.dataset.section;
    document.querySelectorAll('.nav-link').forEach(l=>l.classList.remove('active'));
    document.querySelectorAll('section').forEach(s=>s.classList.remove('active'));
    link.classList.add('active');
    document.getElementById(sec).classList.add('active');
  });
});

loadAllData();
setInterval(loadAllData, 30000);  // auto-refresh every 30s
</script>
</body>
</html>
"""


# -----------------------------------------------------------------------
# Flask app
# -----------------------------------------------------------------------

try:
    from flask import Flask, jsonify, request, Response
    FLASK_AVAILABLE = True
except ImportError:
    FLASK_AVAILABLE = False

if FLASK_AVAILABLE:
    flask_app = Flask(__name__)
    _db = RehabDatabase(db_path=DB_PATH)
    _bench = PopulationBenchmark(_db)

    @flask_app.route('/')
    def index():
        return DASHBOARD_HTML

    # Proxy API routes (same as FastAPI but in Flask)
    @flask_app.route('/api/sessions')
    def api_sessions():
        limit = int(request.args.get('limit', 200))
        offset = int(request.args.get('offset', 0))
        all_s = _db.get_all_sessions()
        all_s.sort(key=lambda s: s.get('timestamp', ''), reverse=True)
        return jsonify({'total': len(all_s), 'sessions': all_s[offset:offset+limit]})

    @flask_app.route('/api/sessions', methods=['POST'])
    def api_upload_session():
        data = request.get_json() or {}
        data.setdefault('timestamp', datetime.now().isoformat())
        session_id = _db.save_session(data)
        return jsonify({'session_id': session_id, 'message': 'stored'})

    @flask_app.route('/api/sessions/<session_id>')
    def api_get_session(session_id):
        s = _db.get_session(session_id)
        if not s:
            return jsonify({'error': 'not found'}), 404
        return jsonify(s)

    @flask_app.route('/api/sessions/<session_id>/xml')
    def api_session_xml(session_id):
        xml_str = _db.export_session_xml(session_id)
        return Response(xml_str, mimetype='application/xml')

    @flask_app.route('/api/patients/<patient_id>/history')
    def api_patient_history(patient_id):
        history = _db.get_patient_history(patient_id)
        return jsonify({'patient_id': patient_id, 'sessions': history})

    @flask_app.route('/api/exercises/<exercise_id>/stats')
    def api_exercise_stats(exercise_id):
        return jsonify(_db.get_exercise_stats(exercise_id))

    @flask_app.route('/api/benchmark/<exercise_id>')
    def api_benchmark(exercise_id):
        return jsonify(_bench.compute_population_stats(exercise_id))

    @flask_app.route('/api/benchmark/<exercise_id>/patient/<patient_id>')
    def api_patient_percentile(exercise_id, patient_id):
        pct = _bench.get_patient_percentile(patient_id, exercise_id, 'rmse')
        return jsonify({'patient_id': patient_id, 'exercise_id': exercise_id, 'percentile': pct})

    @flask_app.route('/api/patients/<patient_id>/report')
    def api_patient_report(patient_id):
        return jsonify(_bench.generate_patient_report(patient_id))

    @flask_app.route('/api/export/xml')
    def api_export_xml():
        xml_str = _db.export_all_xml()
        return Response(xml_str, mimetype='application/xml')

    @flask_app.route('/api/health')
    def api_health():
        return jsonify({'status': 'healthy', 'sessions': _db.total_sessions()})


def run_dashboard(host: str = "0.0.0.0", port: int = 5000, debug: bool = False):
    """Launch the Flask dashboard."""
    if not FLASK_AVAILABLE:
        print("[Dashboard] Flask not installed. Install with: pip install flask")
        return
    print(f"\n{'='*60}")
    print("  RehabAI Clinician Dashboard")
    print(f"  http://localhost:{port}")
    print(f"{'='*60}\n")
    flask_app.run(host=host, port=port, debug=debug, use_reloader=False)


if __name__ == "__main__":
    run_dashboard(debug=True)
