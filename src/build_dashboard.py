#!/usr/bin/env python3
"""Build a static interactive HTML dashboard with lazy-loaded biomarker series."""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import time
from pathlib import Path

import numpy as np
import pandas as pd

from nhanes_common import ensure_dir


HTML_TEMPLATE = """<!DOCTYPE html>
<html lang=\"en\">
<head>
  <meta charset=\"utf-8\" />
  <meta name=\"viewport\" content=\"width=device-width, initial-scale=1\" />
  <title>NHANES Biomarker CV vs Age</title>
  <script src=\"https://cdn.plot.ly/plotly-2.35.2.min.js\"></script>
  <style>
    :root {
      --bg: #f6f3eb;
      --card: #ffffff;
      --ink: #1f2937;
      --muted: #5f6b7a;
      --accent: #0f766e;
      --accent-soft: #c9ebe5;
      --warn: #b45309;
      --line: #ddd6c8;
      --chip: #f4efe5;
      --tab: #efe7d8;
      --tab-active: #0f766e;
    }
    * { box-sizing: border-box; }
    body {
      margin: 0;
      font-family: Georgia, 'Times New Roman', serif;
      color: var(--ink);
      background: radial-gradient(circle at 8% 10%, #fff9e8, var(--bg));
    }
    .wrap { max-width: 1320px; margin: 0 auto; padding: 20px; }
    .hero {
      display: flex;
      justify-content: space-between;
      align-items: flex-end;
      gap: 16px;
      margin-bottom: 14px;
    }
    h1 { margin: 0; font-size: 34px; letter-spacing: 0.2px; }
    .sub { color: var(--muted); margin-top: 6px; }
    .status-chip {
      background: var(--chip);
      border: 1px solid var(--line);
      border-radius: 999px;
      padding: 8px 12px;
      font-size: 13px;
      color: var(--muted);
      white-space: nowrap;
    }
    .top-tabs {
      display: flex;
      gap: 8px;
      margin-bottom: 14px;
    }
    .tab-btn {
      border: 1px solid var(--line);
      background: var(--tab);
      border-radius: 10px;
      padding: 8px 12px;
      cursor: pointer;
      font-size: 14px;
      font-weight: 600;
    }
    .tab-btn.active {
      background: var(--tab-active);
      border-color: var(--tab-active);
      color: #fff;
    }
    .panel { display: none; }
    .panel.active { display: block; }

    .grid { display: grid; grid-template-columns: 330px 1fr; gap: 16px; }
    .card {
      background: var(--card);
      border: 1px solid var(--line);
      border-radius: 12px;
      padding: 12px;
      box-shadow: 0 1px 2px rgba(0,0,0,.04);
    }
    .sticky { position: sticky; top: 12px; }

    input, select {
      width: 100%;
      padding: 8px;
      border: 1px solid var(--line);
      border-radius: 8px;
      font-size: 14px;
      margin: 6px 0 10px 0;
      background: #fff;
    }
    label {
      font-size: 12px;
      color: var(--muted);
      text-transform: uppercase;
      letter-spacing: 0.6px;
    }
    .mode-buttons { display: flex; gap: 8px; margin-bottom: 10px; }
    .mode-btn {
      border: 1px solid var(--line);
      background: #f8f5ef;
      border-radius: 8px;
      padding: 7px 10px;
      cursor: pointer;
      font-size: 13px;
    }
    .mode-btn.active {
      background: var(--accent);
      color: #fff;
      border-color: var(--accent);
    }

    #plot { width: 100%; height: 540px; }
    .metric { font-size: 14px; margin: 6px 0; }
    .flag-true { color: var(--accent); font-weight: 700; }
    .flag-false { color: var(--warn); }

    .table-wrap {
      max-height: 320px;
      overflow: auto;
      border: 1px solid var(--line);
      border-radius: 8px;
      margin-top: 10px;
    }
    table { border-collapse: collapse; width: 100%; font-size: 13px; }
    th, td { border-bottom: 1px solid #eee7da; padding: 6px; text-align: left; }
    th { position: sticky; top: 0; background: #fffaf0; z-index: 1; }

    .info-grid { display: grid; grid-template-columns: 1fr 1fr; gap: 16px; }
    .info-card h3 { margin: 2px 0 8px 0; font-size: 18px; }
    .info-card p { margin: 0 0 8px 0; color: var(--muted); }
    .info-card ul { margin: 8px 0 0 18px; padding: 0; }
    .info-card li { margin: 6px 0; }
    .mono { font-family: Menlo, Monaco, 'Courier New', monospace; font-size: 12px; color: var(--muted); }
    .compare-controls { display: flex; gap: 10px; flex-wrap: wrap; align-items: end; margin-bottom: 10px; }
    .compare-controls label { text-transform: none; letter-spacing: 0; font-size: 13px; }
    .compare-controls select, .compare-controls input { margin: 4px 0 0 0; width: 220px; }
    #compare-plot { width: 100%; height: 640px; }

    @media (max-width: 980px) {
      .grid { grid-template-columns: 1fr; }
      .sticky { position: static; }
      #plot { height: 430px; }
      .info-grid { grid-template-columns: 1fr; }
      .hero { flex-direction: column; align-items: flex-start; }
    }
  </style>
</head>
<body>
  <div class=\"wrap\">
    <div class=\"hero\">
      <div>
        <h1>NHANES Blood Biomarker Variability</h1>
        <div class=\"sub\">Explore cross-sectional aging trajectories across blood biomarkers.</div>
      </div>
      <div id=\"status-chip\" class=\"status-chip\">Loading metadata…</div>
    </div>

    <div class=\"top-tabs\">
      <button id=\"tab-dashboard\" class=\"tab-btn active\" type=\"button\">Dashboard</button>
      <button id=\"tab-compare\" class=\"tab-btn\" type=\"button\">Compare Rankings</button>
      <button id=\"tab-info\" class=\"tab-btn\" type=\"button\">Info & Methods</button>
    </div>

    <div id=\"panel-dashboard\" class=\"panel active\">
      <div class=\"grid\">
        <div class=\"card sticky\">
          <div class=\"mode-buttons\">
            <button id=\"mode-cv\" class=\"mode-btn active\" type=\"button\">Plot CV</button>
            <button id=\"mode-mean\" class=\"mode-btn\" type=\"button\">Plot Mean</button>
          </div>

          <label for=\"search\">Search Biomarker</label>
          <input id=\"search\" list=\"biomarker-options\" placeholder=\"Type name, code, file...\" />
          <datalist id=\"biomarker-options\"></datalist>

          <label for=\"biomarker-select\">Select Biomarker</label>
          <select id=\"biomarker-select\"></select>

          <label><input id=\"show-low-n\" type=\"checkbox\" /> Show low-n bins (&lt;30)</label>
          <label><input id=\"show-sd-band\" type=\"checkbox\" /> Show ±1 SD band in Mean mode</label>

          <div id=\"metrics\" class=\"card\" style=\"margin-top:10px;\"></div>
        </div>

        <div class=\"card\">
          <div id=\"plot\"></div>
          <h3>Biomarkers Ranked by Most Negative Spearman Rho</h3>
          <div class=\"table-wrap\"><table id=\"rank-table\"></table></div>
        </div>
      </div>
    </div>

    <div id=\"panel-compare\" class=\"panel\">
      <div class=\"card\">
        <div class=\"compare-controls\">
          <label>Sort
            <select id=\"compare-sort\">
              <option value=\"negative\" selected>Most Negative Spearman</option>
              <option value=\"positive\">Most Positive Spearman</option>
              <option value=\"absolute\">Largest Absolute Spearman</option>
            </select>
          </label>
          <label>Top N
            <input id=\"compare-topn\" type=\"number\" min=\"10\" max=\"200\" step=\"5\" value=\"40\" />
          </label>
        </div>
        <div id=\"compare-plot\"></div>
      </div>
    </div>

    <div id=\"panel-info\" class=\"panel\">
      <div class=\"info-grid\">
        <div class=\"card info-card\">
          <h3>What This Analysis Does</h3>
          <p>For each blood biomarker test, this dashboard pools all NHANES cycles/files into one trajectory.</p>
          <ul>
            <li>Population: adults age 20+.</li>
            <li>Primary filter: non-pathological (pregnancy + major disease exclusions).</li>
            <li>Age aggregation: 5-year bins with minimum n=30 for primary trend metrics.</li>
            <li>Main metric: CV = SD / |Mean| per age bin.</li>
            <li>Pooling is done by normalized test name (not by NHANES variable code).</li>
            <li>Compatible unit variants are converted and pooled; incompatible unit systems remain separate entries.</li>
          </ul>
        </div>

        <div class=\"card info-card\">
          <h3>Healthy Filter</h3>
          <p>Participants are excluded when available fields indicate:</p>
          <ul>
            <li>Pregnancy (<span class=\"mono\">RIDEXPRG==1</span>)</li>
            <li>Diagnosed diabetes (<span class=\"mono\">DIQ010==1</span>)</li>
            <li>CVD history (<span class=\"mono\">MCQ160b/c/d/e/f</span> or legacy uppercase equivalents)</li>
            <li>Cancer history (<span class=\"mono\">MCQ220==1</span>)</li>
            <li>Weak/failing kidneys (<span class=\"mono\">KIQ022==1</span>)</li>
          </ul>
        </div>

        <div class=\"card info-card\">
          <h3>Plot Modes</h3>
          <ul>
            <li><b>Plot CV</b>: age-binned CV trend.</li>
            <li><b>Plot Mean</b>: age-binned mean with 95% CI band and raw scatter sample.</li>
            <li>95% CI quantifies uncertainty in the mean (SE-based), not raw spread.</li>
            <li>Raw scatter is sampled for performance and readability.</li>
          </ul>
        </div>

        <div class=\"card info-card\">
          <h3>Decline Criteria</h3>
          <p>A biomarker is flagged as declining variability when all conditions hold:</p>
          <ul>
            <li><span class=\"mono\">n_bins &gt;= 5</span></li>
            <li><span class=\"mono\">Spearman rho &lt; 0</span></li>
            <li><span class=\"mono\">Spearman p &lt; 0.05</span></li>
            <li><span class=\"mono\">linear_slope_cv_per_year &lt; 0</span></li>
          </ul>
        </div>
      </div>
    </div>
  </div>

  <script>
    const DATA_BASE = './data';
    const DATA_VERSION = '__DATA_VERSION__';

    const selectEl = document.getElementById('biomarker-select');
    const searchEl = document.getElementById('search');
    const optionsEl = document.getElementById('biomarker-options');
    const showLowNEl = document.getElementById('show-low-n');
    const showSdBandEl = document.getElementById('show-sd-band');
    const modeCvBtn = document.getElementById('mode-cv');
    const modeMeanBtn = document.getElementById('mode-mean');
    const statusChip = document.getElementById('status-chip');

    const tabDashboardBtn = document.getElementById('tab-dashboard');
    const tabCompareBtn = document.getElementById('tab-compare');
    const tabInfoBtn = document.getElementById('tab-info');
    const panelDashboard = document.getElementById('panel-dashboard');
    const panelCompare = document.getElementById('panel-compare');
    const panelInfo = document.getElementById('panel-info');
    const compareSortEl = document.getElementById('compare-sort');
    const compareTopNEl = document.getElementById('compare-topn');

    const state = {
      metadata: [],
      metrics: [],
      seriesIndex: {},
      metricsById: new Map(),
      cache: new Map(),
      mode: 'cv',
      currentId: null,
    };

    function formatNum(v, d=4) {
      if (v === null || v === undefined || Number.isNaN(v)) return 'NA';
      return Number(v).toFixed(d);
    }

    function setTopTab(tabName) {
      const isDash = tabName === 'dashboard';
      const isCompare = tabName === 'compare';
      const isInfo = tabName === 'info';
      tabDashboardBtn.classList.toggle('active', isDash);
      tabCompareBtn.classList.toggle('active', isCompare);
      tabInfoBtn.classList.toggle('active', isInfo);
      panelDashboard.classList.toggle('active', isDash);
      panelCompare.classList.toggle('active', isCompare);
      panelInfo.classList.toggle('active', isInfo);
    }

    async function fetchJson(path) {
      const sep = path.includes('?') ? '&' : '?';
      const r = await fetch(`${path}${sep}v=${DATA_VERSION}`, { cache: 'no-store' });
      if (!r.ok) throw new Error(`Failed to fetch ${path}: ${r.status}`);
      return await r.json();
    }

    async function loadSeries(biomarkerId) {
      if (state.cache.has(biomarkerId)) return state.cache.get(biomarkerId);
      const rel = state.seriesIndex[biomarkerId];
      if (!rel) return null;
      statusChip.textContent = `Loading series… ${biomarkerId}`;
      const series = await fetchJson(`${DATA_BASE}/${rel}`);
      state.cache.set(biomarkerId, series);
      statusChip.textContent = `Loaded ${state.cache.size} series in local cache`;
      return series;
    }

    function renderOptions() {
      const opts = state.metadata.slice().sort((a, b) => String(a.display_name || a.biomarker_name || '').localeCompare(String(b.display_name || b.biomarker_name || '')));
      selectEl.innerHTML = '';
      optionsEl.innerHTML = '';
      for (const o of opts) {
        const label = `${o.display_name || o.biomarker_name}`;
        const opt = document.createElement('option');
        opt.value = o.biomarker_id;
        opt.textContent = label;
        selectEl.appendChild(opt);

        const dopt = document.createElement('option');
        dopt.value = label;
        optionsEl.appendChild(dopt);
      }
    }

    function renderMetrics(id) {
      const m = state.metricsById.get(id);
      const box = document.getElementById('metrics');
      if (!m) {
        box.innerHTML = '<div class="metric">No metrics available.</div>';
        return;
      }
      const flagCls = m.decline_flag ? 'flag-true' : 'flag-false';
      box.innerHTML = `
        <div class="metric"><b>Bins used:</b> ${m.n_bins}</div>
        <div class="metric"><b>Spearman rho:</b> ${formatNum(m.spearman_rho, 4)}</div>
        <div class="metric"><b>Spearman p:</b> ${formatNum(m.spearman_p, 5)}</div>
        <div class="metric"><b>Slope CV/year:</b> ${formatNum(m.linear_slope_cv_per_year, 6)}</div>
        <div class="metric"><b>Slope log(CV)/year:</b> ${formatNum(m.linear_slope_logcv_per_year, 6)}</div>
        <div class="metric"><b>Decline flag:</b> <span class="${flagCls}">${m.decline_flag}</span></div>
      `;
    }

    function setMode(mode) {
      state.mode = mode;
      modeCvBtn.classList.toggle('active', mode === 'cv');
      modeMeanBtn.classList.toggle('active', mode === 'mean');
    }

    async function renderPlot(id) {
      const s = await loadSeries(id);
      if (!s) return;
      state.currentId = id;
      const showLow = showLowNEl.checked;
      const points = showLow ? s.points : s.points.filter(p => p.passes_n_threshold);

      const traces = [];
      const title = `${s.display_name || s.biomarker_name}`;

      if (state.mode === 'cv') {
        traces.push({
          x: points.map(p => p.age_mid),
          y: points.map(p => p.cv),
          text: points.map(p => `age_bin=${p.age_bin}<br>n=${p.n}<br>mean=${formatNum(p.mean, 4)}<br>std=${formatNum(p.std, 4)}<br>cv=${formatNum(p.cv, 4)}`),
          mode: 'lines+markers',
          type: 'scatter',
          marker: {
            size: points.map(p => p.passes_n_threshold ? 9 : 6),
            color: points.map(p => p.passes_n_threshold ? '#0f766e' : '#b45309')
          },
          line: { color: '#0f766e', width: 2 },
          hovertemplate: '%{text}<extra></extra>',
          name: 'CV'
        });
      } else {
        if (showSdBandEl.checked && points.length > 1) {
          traces.push({
            x: points.map(p => p.age_mid).concat(points.map(p => p.age_mid).reverse()),
            y: points.map(p => p.mean + (p.std || 0)).concat(points.map(p => p.mean - (p.std || 0)).reverse()),
            type: 'scatter',
            fill: 'toself',
            fillcolor: 'rgba(15,118,110,0.08)',
            line: { color: 'rgba(0,0,0,0)' },
            hoverinfo: 'skip',
            name: '±1 SD'
          });
        }
        const ciPoints = points.filter(p => p.ci95_low !== null && p.ci95_high !== null);
        if (ciPoints.length > 1) {
          traces.push({
            x: ciPoints.map(p => p.age_mid).concat(ciPoints.map(p => p.age_mid).reverse()),
            y: ciPoints.map(p => p.ci95_high).concat(ciPoints.map(p => p.ci95_low).reverse()),
            type: 'scatter',
            fill: 'toself',
            fillcolor: 'rgba(15,118,110,0.16)',
            line: { color: 'rgba(0,0,0,0)' },
            hoverinfo: 'skip',
            name: '95% CI'
          });
        }
        traces.push({
          x: points.map(p => p.age_mid),
          y: points.map(p => p.mean),
          text: points.map(p => `age_bin=${p.age_bin}<br>n=${p.n}<br>mean=${formatNum(p.mean, 4)}<br>95%CI=[${formatNum(p.ci95_low, 4)}, ${formatNum(p.ci95_high, 4)}]`),
          mode: 'lines+markers',
          type: 'scatter',
          marker: {
            size: points.map(p => p.passes_n_threshold ? 9 : 6),
            color: points.map(p => p.passes_n_threshold ? '#0f766e' : '#b45309')
          },
          line: { color: '#0f766e', width: 2 },
          hovertemplate: '%{text}<extra></extra>',
          name: 'Mean (binned)'
        });
        if (s.raw_sample && s.raw_sample.length > 0) {
          traces.push({
            x: s.raw_sample.map(p => p.age_years),
            y: s.raw_sample.map(p => p.value),
            mode: 'markers',
            type: 'scatter',
            marker: { color: 'rgba(71,85,105,0.25)', size: 4 },
            hovertemplate: 'age=%{x}<br>value=%{y:.4f}<extra>Raw sample</extra>',
            name: 'Raw sample'
          });
        }
      }

      Plotly.newPlot('plot', traces, {
        title,
        xaxis: { title: 'Age (years)' },
        yaxis: { title: state.mode === 'cv' ? 'Coefficient of Variation (CV)' : 'Mean Biomarker Value' },
        margin: { t: 56, l: 64, r: 18, b: 54 },
        paper_bgcolor: '#ffffff',
        plot_bgcolor: '#ffffff',
        legend: { orientation: 'h', y: 1.08 }
      }, { responsive: true, displaylogo: false });
    }

    function renderRankTable() {
      const tbl = document.getElementById('rank-table');
      const ranked = state.metrics.slice().sort((a, b) => (a.spearman_rho ?? 999) - (b.spearman_rho ?? 999));
      const displayById = new Map(state.metadata.map(m => [m.biomarker_id, m.display_name || m.biomarker_name || m.biomarker_id]));
      const top = ranked.slice(0, 200);
      let html = '<thead><tr><th>Biomarker</th><th>Spearman rho</th><th>p</th><th>Decline</th></tr></thead><tbody>';
      for (const r of top) {
        const dname = displayById.get(r.biomarker_id) || r.biomarker_name || r.biomarker_id;
        html += `<tr data-id="${r.biomarker_id}"><td>${dname}</td><td>${formatNum(r.spearman_rho, 4)}</td><td>${formatNum(r.spearman_p, 5)}</td><td>${r.decline_flag}</td></tr>`;
      }
      html += '</tbody>';
      tbl.innerHTML = html;

      for (const tr of tbl.querySelectorAll('tbody tr')) {
        tr.style.cursor = 'pointer';
        tr.onclick = async () => {
          const id = tr.getAttribute('data-id');
          selectEl.value = id;
          renderMetrics(id);
          await renderPlot(id);
        };
      }
    }

    function renderComparePlot() {
      const mode = compareSortEl.value;
      const topN = Math.max(10, Math.min(200, Number(compareTopNEl.value || 40)));
      compareTopNEl.value = String(topN);

      const merged = state.metrics
        .map(m => {
          const md = state.metadata.find(x => x.biomarker_id === m.biomarker_id) || {};
          return {
            ...m,
            display_name: md.display_name || m.biomarker_name || m.biomarker_id,
          };
        })
        .filter(m => m.spearman_rho !== null && m.spearman_rho !== undefined && !Number.isNaN(Number(m.spearman_rho)));

      let ranked = merged.slice();
      if (mode === 'negative') ranked.sort((a, b) => a.spearman_rho - b.spearman_rho);
      if (mode === 'positive') ranked.sort((a, b) => b.spearman_rho - a.spearman_rho);
      if (mode === 'absolute') ranked.sort((a, b) => Math.abs(b.spearman_rho) - Math.abs(a.spearman_rho));
      ranked = ranked.slice(0, topN);

      const y = ranked.map(r => r.display_name).reverse();
      const x = ranked.map(r => Number(r.spearman_rho)).reverse();
      const custom = ranked.map(r => [r.spearman_p, r.n_bins, r.decline_flag, r.biomarker_id]).reverse();

      const colors = x.map(v => (v < 0 ? '#0f766e' : '#b45309'));
      const trace = {
        type: 'bar',
        orientation: 'h',
        y,
        x,
        marker: { color: colors },
        customdata: custom,
        hovertemplate: 'rho=%{x:.4f}<br>p=%{customdata[0]:.5f}<br>n_bins=%{customdata[1]}<br>decline=%{customdata[2]}<br>id=%{customdata[3]}<extra></extra>',
      };

      Plotly.newPlot('compare-plot', [trace], {
        title: mode === 'negative' ? `Top ${topN} Most Negative Spearman Biomarkers` :
               mode === 'positive' ? `Top ${topN} Most Positive Spearman Biomarkers` :
               `Top ${topN} Largest |Spearman| Biomarkers`,
        xaxis: { title: 'Spearman rho (Age vs CV)' },
        yaxis: { automargin: true },
        margin: { t: 56, l: 260, r: 16, b: 54 },
        paper_bgcolor: '#ffffff',
        plot_bgcolor: '#ffffff',
      }, { responsive: true, displaylogo: false });
    }

    async function applySearch() {
      const q = searchEl.value.toLowerCase().trim();
      if (!q) return;
      const hit = state.metadata.find(m => `${m.display_name || ''} ${m.biomarker_name || ''} ${m.variable_name || ''} ${m.source_files || ''} ${m.source_variables || ''}`.toLowerCase().includes(q));
      if (hit) {
        selectEl.value = hit.biomarker_id;
        renderMetrics(hit.biomarker_id);
        await renderPlot(hit.biomarker_id);
      }
    }

    async function init() {
      const [metadata, metrics, index] = await Promise.all([
        fetchJson(`${DATA_BASE}/metadata.json`),
        fetchJson(`${DATA_BASE}/metrics.json`),
        fetchJson(`${DATA_BASE}/series_index.json`),
      ]);

      state.metadata = metadata;
      state.metrics = metrics;
      state.seriesIndex = index;
      state.metricsById = new Map(metrics.map(m => [m.biomarker_id, m]));

      renderOptions();
      renderRankTable();
      renderComparePlot();
      showLowNEl.checked = false;

      const first = state.metadata[0]?.biomarker_id;
      if (first) {
        selectEl.value = first;
        renderMetrics(first);
        await renderPlot(first);
      }

      statusChip.textContent = `Ready: ${state.metadata.length} biomarkers indexed`;

      tabDashboardBtn.addEventListener('click', () => setTopTab('dashboard'));
      tabCompareBtn.addEventListener('click', () => {
        setTopTab('compare');
        renderComparePlot();
      });
      tabInfoBtn.addEventListener('click', () => setTopTab('info'));
      compareSortEl.addEventListener('change', renderComparePlot);
      compareTopNEl.addEventListener('change', renderComparePlot);
      selectEl.addEventListener('change', async () => {
        const id = selectEl.value;
        renderMetrics(id);
        await renderPlot(id);
      });
      searchEl.addEventListener('change', applySearch);
      searchEl.addEventListener('keyup', (e) => { if (e.key === 'Enter') applySearch(); });
      showLowNEl.addEventListener('change', async () => {
        if (state.currentId) await renderPlot(state.currentId);
      });
      showSdBandEl.addEventListener('change', async () => {
        if (state.currentId) await renderPlot(state.currentId);
      });
      modeCvBtn.addEventListener('click', async () => {
        setMode('cv');
        if (state.currentId) await renderPlot(state.currentId);
      });
      modeMeanBtn.addEventListener('click', async () => {
        setMode('mean');
        if (state.currentId) await renderPlot(state.currentId);
      });
    }

    init().catch(err => {
      console.error(err);
      const plot = document.getElementById('plot');
      plot.innerHTML = `<div style=\"padding:16px;color:#b45309;\">Failed to load dashboard data. Open via local server (not file://). Error: ${err.message}</div>`;
      statusChip.textContent = 'Load failed';
    });
  </script>
</body>
</html>
"""


def safe_series_filename(biomarker_id: str) -> str:
    slug = re.sub(r"[^A-Za-z0-9_-]+", "_", biomarker_id)[:80].strip("_")
    h = hashlib.sha1(biomarker_id.encode("utf-8")).hexdigest()[:10]
    return f"series/{slug}__{h}.json"


def clean_display_base(name: str) -> str:
    s = str(name or "").strip()
    # Remove leading isomer-position locants (e.g., 1,2,3,4-)
    s = re.sub(r"^\s*(?:\d+[’']?)(?:,\s*\d+[’']?)+\s*,?-?\s*", "", s)
    # Remove short all-lower/number acronym parentheses that are not units (e.g., (ocdd), (hpcdf))
    s = re.sub(r"\s*\(([a-z0-9_-]{2,12})\)", lambda m: "" if "/" not in m.group(1) else m.group(0), s)
    s = re.sub(r"\s+", " ", s).strip()
    return s


def make_display_name(name: str, unit: str) -> str:
    base = clean_display_base(name)
    u = str(unit or "").strip()
    if u:
        if not re.search(rf"\(\s*{re.escape(u)}\s*\)\s*$", base, flags=re.IGNORECASE):
            base = f"{base} ({u})"
    else:
        base = f"{base} (unit not reported)"
    return base


def build_outputs(
    cv_df: pd.DataFrame,
    metrics_df: pd.DataFrame,
    catalog_df: pd.DataFrame | None,
    long_df: pd.DataFrame | None,
    raw_sample_n: int,
    random_seed: int,
) -> tuple[pd.DataFrame, list[dict], dict[str, str], dict[str, dict]]:
    cv_df = cv_df.copy()
    if "variable_name" not in cv_df.columns:
        cv_df["variable_name"] = cv_df["biomarker_id"]
    if "unit" not in cv_df.columns:
        cv_df["unit"] = ""

    ci_se = 1.96 * (cv_df["std"] / np.sqrt(cv_df["n"].clip(lower=1)))
    cv_df["ci95_low"] = cv_df["mean"] - ci_se
    cv_df["ci95_high"] = cv_df["mean"] + ci_se
    cv_df.loc[cv_df["std"].isna(), ["ci95_low", "ci95_high"]] = np.nan

    if catalog_df is not None and not catalog_df.empty:
        need = [
            "biomarker_id",
            "variable_name",
            "biomarker_name",
            "unit",
            "source_file_count",
            "source_files",
            "source_variable_count",
            "source_variables",
        ]
        metadata = catalog_df[need].drop_duplicates().sort_values(["biomarker_name", "biomarker_id"]).copy()
    else:
        metadata = (
            cv_df[["biomarker_id", "biomarker_name", "variable_name", "unit"]]
            .drop_duplicates()
            .sort_values(["biomarker_name", "biomarker_id"])
        )
        metadata["source_file_count"] = np.nan
        metadata["source_files"] = ""
        metadata["source_variable_count"] = np.nan
        metadata["source_variables"] = ""

    metadata["variable_name"] = metadata["variable_name"].fillna(metadata["biomarker_id"])
    metadata["biomarker_name"] = metadata["biomarker_name"].fillna(metadata["variable_name"])
    metadata["unit"] = metadata["unit"].fillna("")
    metadata["source_files"] = metadata["source_files"].fillna("")
    metadata["source_file_count"] = pd.to_numeric(metadata["source_file_count"], errors="coerce").fillna(0).astype(int)
    metadata["source_variables"] = metadata["source_variables"].fillna("")
    metadata["source_variable_count"] = pd.to_numeric(metadata["source_variable_count"], errors="coerce").fillna(0).astype(int)
    metadata["display_name"] = [
        make_display_name(n, u) for n, u in zip(metadata["biomarker_name"].tolist(), metadata["unit"].tolist())
    ]

    metrics_cols = [
        "biomarker_id",
        "biomarker_name",
        "n_bins",
        "spearman_rho",
        "spearman_p",
        "linear_slope_cv_per_year",
        "linear_slope_logcv_per_year",
        "decline_flag",
    ]
    metrics_clean = metrics_df[metrics_cols].copy()
    metrics_clean = metrics_clean.replace([np.inf, -np.inf], np.nan)
    metrics_clean = metrics_clean.astype(object).where(pd.notna(metrics_clean), None)
    metrics = metrics_clean.to_dict(orient="records")

    raw_samples: dict[str, list[dict]] = {}
    if long_df is not None and not long_df.empty:
        use = long_df[["biomarker_id", "age_years", "value"]].dropna()
        rng = np.random.default_rng(random_seed)
        for bid, g in use.groupby("biomarker_id", observed=True):
            if len(g) > raw_sample_n:
                idx = rng.choice(len(g), size=raw_sample_n, replace=False)
                g = g.iloc[idx]
            raw_samples[bid] = [
                {"age_years": float(r.age_years), "value": float(r.value)} for r in g.itertuples(index=False)
            ]

    series_index: dict[str, str] = {}
    series_payloads: dict[str, dict] = {}

    for bid, g in cv_df.groupby("biomarker_id", observed=True):
        g = g.sort_values("age_mid")
        rel_path = safe_series_filename(bid)
        series_index[bid] = rel_path
        series_payloads[rel_path] = {
            "biomarker_id": bid,
            "biomarker_name": g["biomarker_name"].iloc[0],
            "display_name": make_display_name(
                str(g["biomarker_name"].iloc[0]),
                str(g["unit"].iloc[0] if "unit" in g.columns else ""),
            ),
            "variable_name": g["variable_name"].iloc[0],
            "unit": g["unit"].iloc[0] if "unit" in g.columns else "",
            "points": [
                {
                    "age_bin": str(r.age_bin),
                    "age_mid": float(r.age_mid),
                    "n": int(r.n),
                    "mean": float(r.mean),
                    "std": float(r.std) if pd.notna(r.std) else None,
                    "cv": float(r.cv),
                    "ci95_low": float(r.ci95_low) if pd.notna(r.ci95_low) else None,
                    "ci95_high": float(r.ci95_high) if pd.notna(r.ci95_high) else None,
                    "passes_n_threshold": bool(r.passes_n_threshold),
                }
                for r in g.itertuples(index=False)
            ],
            "raw_sample": raw_samples.get(bid, []),
        }

    return metadata, metrics, series_index, series_payloads


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--cv", default="data/processed/cv_by_age.parquet")
    ap.add_argument("--cv-all", default="data/processed/cv_by_age_all.parquet")
    ap.add_argument("--metrics", default="data/processed/cv_trend_metrics.parquet")
    ap.add_argument("--catalog", default="data/processed/biomarker_catalog.parquet")
    ap.add_argument("--long", default="data/processed/biomarker_long.parquet")
    ap.add_argument("--raw-sample-n", type=int, default=1200)
    ap.add_argument("--random-seed", type=int, default=42)
    ap.add_argument("--out", default="dashboard/index.html")
    ap.add_argument("--json-out", default="dashboard/dashboard_data.json")
    args = ap.parse_args()

    cv_path = Path(args.cv_all)
    if not cv_path.exists():
        cv_path = Path(args.cv)

    cv_df = pd.read_parquet(cv_path)
    metrics_df = pd.read_parquet(args.metrics)
    catalog_path = Path(args.catalog)
    catalog_df = pd.read_parquet(catalog_path) if catalog_path.exists() else None
    long_path = Path(args.long)
    long_df = None
    if long_path.exists():
        long_df = pd.read_parquet(long_path, columns=["biomarker_id", "age_years", "value"])

    metadata, metrics, series_index, series_payloads = build_outputs(
        cv_df=cv_df,
        metrics_df=metrics_df,
        catalog_df=catalog_df,
        long_df=long_df,
        raw_sample_n=args.raw_sample_n,
        random_seed=args.random_seed,
    )

    out_html = Path(args.out)
    out_json = Path(args.json_out)
    data_dir = out_html.parent / "data"
    series_dir = data_dir / "series"

    ensure_dir(out_html.parent)
    ensure_dir(data_dir)
    ensure_dir(series_dir)
    ensure_dir(out_json.parent)

    # Remove old per-series files so output always matches current dataset.
    for old in series_dir.glob("*.json"):
        old.unlink()

    (data_dir / "metadata.json").write_text(
        json.dumps(metadata.to_dict(orient="records"), ensure_ascii=True, allow_nan=False), encoding="utf-8"
    )
    (data_dir / "metrics.json").write_text(json.dumps(metrics, ensure_ascii=True, allow_nan=False), encoding="utf-8")
    (data_dir / "series_index.json").write_text(
        json.dumps(series_index, ensure_ascii=True, allow_nan=False), encoding="utf-8"
    )

    for rel, payload in series_payloads.items():
        p = data_dir / rel
        ensure_dir(p.parent)
        p.write_text(json.dumps(payload, ensure_ascii=True, allow_nan=False), encoding="utf-8")

    summary_payload = {
        "metadata_count": len(metadata),
        "metrics_count": len(metrics),
        "series_count": len(series_payloads),
        "raw_sample_n": args.raw_sample_n,
        "data_dir": str(data_dir),
    }
    out_json.write_text(json.dumps(summary_payload, ensure_ascii=True, indent=2, allow_nan=False), encoding="utf-8")

    data_version = str(int(time.time()))
    out_html.write_text(HTML_TEMPLATE.replace("__DATA_VERSION__", data_version), encoding="utf-8")

    print(f"Wrote dashboard HTML: {out_html}")
    print(f"Wrote metadata: {data_dir / 'metadata.json'}")
    print(f"Wrote metrics: {data_dir / 'metrics.json'}")
    print(f"Wrote series index: {data_dir / 'series_index.json'}")
    print(f"Wrote {len(series_payloads)} series files under: {series_dir}")
    print(f"Wrote dashboard summary JSON: {out_json}")


if __name__ == "__main__":
    main()
