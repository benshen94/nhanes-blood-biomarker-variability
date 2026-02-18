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
from scipy.stats import spearmanr

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
      flex-wrap: wrap;
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

    input[type="text"],
    input[type="search"],
    input[type="number"],
    select {
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
    input[type="checkbox"] {
      width: auto;
      margin: 0;
      accent-color: var(--accent);
    }
    .check-label {
      display: flex;
      align-items: center;
      gap: 8px;
      margin: 6px 0 10px 0;
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
      .wrap { padding: 14px; }
      h1 { font-size: 28px; }
      .sub { font-size: 15px; }
      #compare-plot { height: 560px; }
    }
    @media (max-width: 760px) {
      .tab-btn {
        flex: 1 1 calc(50% - 8px);
        text-align: center;
      }
      .table-wrap {
        max-height: none;
        overflow-x: auto;
      }
      table { font-size: 12px; min-width: 560px; }
      #plot { height: 380px; }
      #compare-plot { height: 500px; }
      .compare-controls label { width: 100%; }
      .compare-controls select,
      .compare-controls input { width: 100%; }
    }
    @media (max-width: 520px) {
      .tab-btn { flex: 1 1 100%; }
      .mode-buttons { flex-wrap: wrap; }
      .mode-btn { flex: 1 1 calc(50% - 8px); }
      #plot { height: 340px; }
      #compare-plot { height: 440px; }
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

          <label for=\"category-filter\">Clinical Category</label>
          <select id=\"category-filter\"></select>
          <label class=\"check-label\"><input id=\"include-env\" type=\"checkbox\" /> Include environmental/toxicant assays</label>

          <label for=\"biomarker-select\">Select Biomarker</label>
          <select id=\"biomarker-select\"></select>

          <label for=\"cohort-filter\">Sex Group</label>
          <select id=\"cohort-filter\">
            <option value=\"pooled\" selected>Pooled</option>
            <option value=\"female\">Female</option>
            <option value=\"male\">Male</option>
            <option value=\"both\">Both (Female + Male)</option>
          </select>

          <label class=\"check-label\"><input id=\"show-low-n\" type=\"checkbox\" checked /> Show low-n bins (&lt;30)</label>
          <label class=\"check-label\"><input id=\"show-sd-band\" type=\"checkbox\" /> Show ±1 SD band in Mean mode</label>

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
          <label>Category
            <select id=\"compare-category\"></select>
          </label>
          <label class=\"check-label\"><input id=\"compare-include-env\" type=\"checkbox\" /> Include environmental/toxicant</label>
          <label>Cohort
            <select id=\"compare-cohort\">
              <option value=\"pooled\" selected>Pooled</option>
              <option value=\"female\">Female</option>
              <option value=\"male\">Male</option>
              <option value=\"both\">Both (Female + Male)</option>
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
            <li><b>Plot Mean</b>: age-binned mean with central 95% range band and raw scatter sample.</li>
            <li>Sex view: pooled, female, male, or both on the same chart (female red, male blue).</li>
            <li>95% range is the empirical 2.5th-97.5th percentile interval of observed values in each age bin.</li>
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
    const categoryFilterEl = document.getElementById('category-filter');
    const includeEnvEl = document.getElementById('include-env');
    const compareCategoryEl = document.getElementById('compare-category');
    const compareIncludeEnvEl = document.getElementById('compare-include-env');
    const cohortFilterEl = document.getElementById('cohort-filter');
    const compareCohortEl = document.getElementById('compare-cohort');

    const CATEGORY_PRIORITY = {
      'Routine - CBC': 1,
      'Routine - CMP': 2,
      'Cardiometabolic - Lipid': 3,
      'Cardiometabolic - Glycemic': 4,
      'Organ - Thyroid': 5,
      'Organ - Renal': 6,
      'Organ - Hepatic': 7,
      'Specialized - Coagulation': 8,
      'Specialized - Nutritional/Vitamin': 9,
      'Specialized - Inflammatory': 10,
      'Hormones/Reproductive': 11,
      'Infectious/Serology': 12,
      'Other Clinical': 13,
      'Environmental/Toxicant': 14,
    };

    const COHORT_COLORS = {
      pooled: '#0f766e',
      female: '#d1495b',
      male: '#2563eb',
    };

    const state = {
      metadata: [],
      metrics: [],
      seriesIndex: {},
      metricsById: new Map(),
      metadataById: new Map(),
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

    function sortedCategories(metadata, includeEnv) {
      const cats = new Set();
      for (const m of metadata) {
        if (!includeEnv && m.is_environmental) continue;
        cats.add(m.category || 'Other Clinical');
      }
      return Array.from(cats).sort((a, b) => (CATEGORY_PRIORITY[a] ?? 999) - (CATEGORY_PRIORITY[b] ?? 999) || a.localeCompare(b));
    }

    function renderCategorySelect(selectNode, includeEnv, selectedValue) {
      const cats = sortedCategories(state.metadata, includeEnv);
      const options = [
        { value: 'all_core', label: 'Clinical/core tests first' },
        { value: 'all_non_env', label: 'All non-environmental blood tests' },
        { value: 'all', label: 'All visible categories' },
        ...cats.map(c => ({ value: `cat:${c}`, label: c })),
      ];
      selectNode.innerHTML = '';
      for (const opt of options) {
        const el = document.createElement('option');
        el.value = opt.value;
        el.textContent = opt.label;
        selectNode.appendChild(el);
      }
      const keep = options.some(o => o.value === selectedValue) ? selectedValue : 'all_core';
      selectNode.value = keep;
    }

    function metadataPasses(m, categoryValue, includeEnv) {
      const isEnv = Boolean(m.is_environmental);
      const isCore = Boolean(m.is_core_clinical);
      const cat = m.category || 'Other Clinical';
      if (!includeEnv && isEnv) return false;
      if (categoryValue === 'all_core') return isCore && !isEnv;
      if (categoryValue === 'all_non_env') return !isEnv;
      if (categoryValue === 'all') return includeEnv ? true : !isEnv;
      if (String(categoryValue || '').startsWith('cat:')) return cat === categoryValue.slice(4);
      return includeEnv ? true : !isEnv;
    }

    function getDashboardMetadata() {
      return state.metadata.filter(m => metadataPasses(m, categoryFilterEl.value, includeEnvEl.checked));
    }

    function getCompareMetrics() {
      const byId = state.metadataById;
      return state.metrics
        .map(m => {
          const md = byId.get(m.biomarker_id) || {};
          const sexMetrics = m.sex_metrics || {};
          return {
            ...m,
            display_name: md.display_name || m.biomarker_name || m.biomarker_id,
            category: md.category || 'Other Clinical',
            is_environmental: Boolean(md.is_environmental),
            is_core_clinical: Boolean(md.is_core_clinical),
            female_metric: sexMetrics.female || null,
            male_metric: sexMetrics.male || null,
          };
        })
        .filter(m => metadataPasses(m, compareCategoryEl.value, compareIncludeEnvEl.checked));
    }

    function renderOptions() {
      const opts = getDashboardMetadata().slice().sort(
        (a, b) => String(a.display_name || a.biomarker_name || '').localeCompare(String(b.display_name || b.biomarker_name || ''))
      );
      const previousId = state.currentId || selectEl.value;
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
      if (opts.length === 0) {
        state.currentId = null;
        return null;
      }
      const next = opts.some(o => o.biomarker_id === previousId) ? previousId : opts[0].biomarker_id;
      selectEl.value = next;
      state.currentId = next;
      return next;
    }

    function renderMetricRows(title, m, rawTotal, rawCap) {
      if (!m) return `<div class="metric"><b>${title}:</b> no metrics</div>`;
      const flagCls = m.decline_flag ? 'flag-true' : 'flag-false';
      return `
        <div class="metric"><b>${title} bins:</b> ${m.n_bins ?? 'NA'}</div>
        <div class="metric"><b>${title} Spearman rho:</b> ${formatNum(m.spearman_rho, 4)}</div>
        <div class="metric"><b>${title} Spearman p:</b> ${formatNum(m.spearman_p, 5)}</div>
        <div class="metric"><b>${title} Slope CV/year:</b> ${formatNum(m.linear_slope_cv_per_year, 6)}</div>
        <div class="metric"><b>${title} Slope log(CV)/year:</b> ${formatNum(m.linear_slope_logcv_per_year, 6)}</div>
        <div class="metric"><b>${title} Raw points:</b> up to ${rawCap ?? 'NA'} sampled of ${rawTotal ?? 'NA'} total</div>
        <div class="metric"><b>${title} Decline flag:</b> <span class="${flagCls}">${m.decline_flag}</span></div>
      `;
    }

    function renderMetrics(id, series=null) {
      const pooled = state.metricsById.get(id);
      const md = state.metadataById.get(id) || {};
      const box = document.getElementById('metrics');
      if (!pooled) {
        box.innerHTML = '<div class="metric">No metrics available.</div>';
        return;
      }

      const cohort = cohortFilterEl.value || 'pooled';
      const sexMetrics = (series && series.sex_metrics) ? series.sex_metrics : (pooled.sex_metrics || {});
      const rawBySex = (series && series.raw_total_n_by_sex) ? series.raw_total_n_by_sex : {};
      const rawCap = md.raw_sample_cap ?? 'NA';

      let html = `<div class="metric"><b>Category:</b> ${md.category || 'Other Clinical'}</div>`;
      if (cohort === 'both') {
        html += renderMetricRows('Female', sexMetrics.female || null, rawBySex.female ?? 'NA', rawCap);
        html += renderMetricRows('Male', sexMetrics.male || null, rawBySex.male ?? 'NA', rawCap);
      } else if (cohort === 'female' || cohort === 'male') {
        const m = sexMetrics[cohort] || null;
        html += renderMetricRows(cohort === 'female' ? 'Female' : 'Male', m, rawBySex[cohort] ?? 'NA', rawCap);
      } else {
        html += renderMetricRows('Pooled', pooled, md.raw_total_n ?? 'NA', rawCap);
      }
      box.innerHTML = html;
    }

    function setMode(mode) {
      state.mode = mode;
      modeCvBtn.classList.toggle('active', mode === 'cv');
      modeMeanBtn.classList.toggle('active', mode === 'mean');
    }

    function pickPointsByCohort(s, cohort) {
      if (cohort === 'female') return (s.sex_points && s.sex_points.female) ? s.sex_points.female : [];
      if (cohort === 'male') return (s.sex_points && s.sex_points.male) ? s.sex_points.male : [];
      return s.points || [];
    }

    function pickRawByCohort(s, cohort) {
      if (cohort === 'female') return (s.raw_sample_by_sex && s.raw_sample_by_sex.female) ? s.raw_sample_by_sex.female : [];
      if (cohort === 'male') return (s.raw_sample_by_sex && s.raw_sample_by_sex.male) ? s.raw_sample_by_sex.male : [];
      return s.raw_sample || [];
    }

    function lineTrace(points, color, label, modeName) {
      return {
        x: points.map(p => p.age_mid),
        y: points.map(p => modeName === 'cv' ? p.cv : p.mean),
        text: points.map(p => `age_bin=${p.age_bin}<br>n=${p.n}<br>mean=${formatNum(p.mean, 4)}<br>std=${formatNum(p.std, 4)}<br>cv=${formatNum(p.cv, 4)}`),
        mode: 'lines+markers',
        type: 'scatter',
        marker: { size: points.map(p => p.passes_n_threshold ? 8 : 5), color },
        line: { color, width: 2 },
        hovertemplate: '%{text}<extra></extra>',
        name: label
      };
    }

    function ciBandTrace(points, color, label) {
      const ciPoints = points.filter(p => p.range95_low !== null && p.range95_high !== null);
      if (ciPoints.length < 2) return null;
      return {
        x: ciPoints.map(p => p.age_mid).concat(ciPoints.map(p => p.age_mid).reverse()),
        y: ciPoints.map(p => p.range95_high).concat(ciPoints.map(p => p.range95_low).reverse()),
        type: 'scatter',
        fill: 'toself',
        fillcolor: color,
        line: { color: 'rgba(0,0,0,0)' },
        hoverinfo: 'skip',
        name: label
      };
    }

    function sdBandTrace(points, color, label) {
      if (points.length < 2) return null;
      return {
        x: points.map(p => p.age_mid).concat(points.map(p => p.age_mid).reverse()),
        y: points.map(p => p.mean + (p.std || 0)).concat(points.map(p => p.mean - (p.std || 0)).reverse()),
        type: 'scatter',
        fill: 'toself',
        fillcolor: color,
        line: { color: 'rgba(0,0,0,0)' },
        hoverinfo: 'skip',
        name: label
      };
    }

    async function renderPlot(id) {
      const s = await loadSeries(id);
      if (!s) return;
      state.currentId = id;
      const showLow = showLowNEl.checked;
      const cohort = cohortFilterEl.value || 'pooled';

      const traces = [];
      const title = `${s.display_name || s.biomarker_name}`;
      const selectedCohorts = cohort === 'both' ? ['female', 'male'] : [cohort];
      const cohortLabel = { pooled: 'Pooled', female: 'Female', male: 'Male' };
      const band95 = {
        pooled: 'rgba(15,118,110,0.16)',
        female: 'rgba(209,73,91,0.18)',
        male: 'rgba(37,99,235,0.18)',
      };
      const bandSd = {
        pooled: 'rgba(15,118,110,0.08)',
        female: 'rgba(209,73,91,0.09)',
        male: 'rgba(37,99,235,0.09)',
      };

      for (const c of selectedCohorts) {
        const pointsRaw = pickPointsByCohort(s, c);
        const points = showLow ? pointsRaw : pointsRaw.filter(p => p.passes_n_threshold);
        if (!points || points.length === 0) continue;

        if (state.mode === 'cv') {
          traces.push(lineTrace(points, COHORT_COLORS[c], `${cohortLabel[c]} CV`, 'cv'));
          continue;
        }

        if (showSdBandEl.checked) {
          const sd = sdBandTrace(points, bandSd[c], `${cohortLabel[c]} ±1 SD`);
          if (sd) traces.push(sd);
        }
        const ci = ciBandTrace(points, band95[c], `${cohortLabel[c]} 95% range`);
        if (ci) traces.push(ci);
        traces.push({
          ...lineTrace(points, COHORT_COLORS[c], `${cohortLabel[c]} Mean (binned)`, 'mean'),
          text: points.map(p => `age_bin=${p.age_bin}<br>n=${p.n}<br>mean=${formatNum(p.mean, 4)}<br>95%range=[${formatNum(p.range95_low, 4)}, ${formatNum(p.range95_high, 4)}]`),
        });

        const raw = pickRawByCohort(s, c);
        if (raw && raw.length > 0) {
          traces.push({
            x: raw.map(p => p.age_years),
            y: raw.map(p => p.value),
            mode: 'markers',
            type: 'scatter',
            marker: { color: c === 'female' ? 'rgba(209,73,91,0.23)' : c === 'male' ? 'rgba(37,99,235,0.23)' : 'rgba(71,85,105,0.25)', size: 4 },
            hovertemplate: 'age=%{x}<br>value=%{y:.4f}<extra>' + `${cohortLabel[c]} raw sample` + '</extra>',
            name: `${cohortLabel[c]} Raw sample`
          });
        }
      }

      renderMetrics(id, s);
      const mobile = window.matchMedia('(max-width: 760px)').matches;
      Plotly.newPlot('plot', traces, {
        title,
        xaxis: { title: 'Age (years)', tickfont: { size: mobile ? 10 : 12 } },
        yaxis: { title: state.mode === 'cv' ? 'Coefficient of Variation (CV)' : 'Mean Biomarker Value', tickfont: { size: mobile ? 10 : 12 } },
        margin: mobile ? { t: 52, l: 46, r: 10, b: 44 } : { t: 56, l: 64, r: 18, b: 54 },
        paper_bgcolor: '#ffffff',
        plot_bgcolor: '#ffffff',
        legend: {
          orientation: 'h',
          y: 1.08,
          font: { size: mobile ? 10 : 12 },
          itemwidth: mobile ? 38 : undefined
        }
      }, { responsive: true, displaylogo: false });
    }

    function renderRankTable() {
      const tbl = document.getElementById('rank-table');
      const visible = new Set(getDashboardMetadata().map(m => m.biomarker_id));
      const ranked = state.metrics
        .filter(r => visible.has(r.biomarker_id))
        .slice()
        .sort((a, b) => (a.spearman_rho ?? 999) - (b.spearman_rho ?? 999));
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
      const cohort = compareCohortEl.value || 'pooled';
      const topN = Math.max(10, Math.min(200, Number(compareTopNEl.value || 40)));
      compareTopNEl.value = String(topN);
      let merged = getCompareMetrics();

      if (cohort === 'female') {
        merged = merged
          .map(m => ({ ...m, metric: m.female_metric }))
          .filter(m => m.metric && m.metric.spearman_rho !== null && m.metric.spearman_rho !== undefined && !Number.isNaN(Number(m.metric.spearman_rho)));
      } else if (cohort === 'male') {
        merged = merged
          .map(m => ({ ...m, metric: m.male_metric }))
          .filter(m => m.metric && m.metric.spearman_rho !== null && m.metric.spearman_rho !== undefined && !Number.isNaN(Number(m.metric.spearman_rho)));
      } else if (cohort === 'both') {
        merged = merged
          .filter(m => m.female_metric && m.male_metric)
          .filter(m => m.female_metric.spearman_rho !== null && m.male_metric.spearman_rho !== null)
          .map(m => ({
            ...m,
            rho_female: Number(m.female_metric.spearman_rho),
            rho_male: Number(m.male_metric.spearman_rho),
          }));
      } else {
        merged = merged.filter(m => m.spearman_rho !== null && m.spearman_rho !== undefined && !Number.isNaN(Number(m.spearman_rho)));
      }

      let ranked = merged.slice();
      const rankVal = (m) => {
        if (cohort === 'both') {
          const avg = (m.rho_female + m.rho_male) / 2;
          if (mode === 'absolute') return Math.abs(avg);
          return avg;
        }
        const rho = cohort === 'female' || cohort === 'male' ? Number(m.metric.spearman_rho) : Number(m.spearman_rho);
        return mode === 'absolute' ? Math.abs(rho) : rho;
      };
      if (mode === 'negative') ranked.sort((a, b) => rankVal(a) - rankVal(b));
      if (mode === 'positive') ranked.sort((a, b) => rankVal(b) - rankVal(a));
      if (mode === 'absolute') ranked.sort((a, b) => rankVal(b) - rankVal(a));
      ranked = ranked.slice(0, topN);

      const y = ranked.map(r => r.display_name).reverse();
      const categoryLabel = compareCategoryEl.options[compareCategoryEl.selectedIndex]?.textContent || 'All';
      let traces = [];
      let xTitle = 'Spearman rho (Age vs CV)';

      if (cohort === 'both') {
        const xF = ranked.map(r => Number(r.rho_female)).reverse();
        const xM = ranked.map(r => Number(r.rho_male)).reverse();
        traces = [
          {
            type: 'bar',
            orientation: 'h',
            y,
            x: xF,
            marker: { color: COHORT_COLORS.female },
            name: 'Female',
            customdata: ranked.map(r => [r.female_metric?.spearman_p, r.female_metric?.n_bins, r.biomarker_id, r.category]).reverse(),
            hovertemplate: 'Female rho=%{x:.4f}<br>p=%{customdata[0]:.5f}<br>n_bins=%{customdata[1]}<br>id=%{customdata[2]}<br>category=%{customdata[3]}<extra></extra>',
          },
          {
            type: 'bar',
            orientation: 'h',
            y,
            x: xM,
            marker: { color: COHORT_COLORS.male },
            name: 'Male',
            customdata: ranked.map(r => [r.male_metric?.spearman_p, r.male_metric?.n_bins, r.biomarker_id, r.category]).reverse(),
            hovertemplate: 'Male rho=%{x:.4f}<br>p=%{customdata[0]:.5f}<br>n_bins=%{customdata[1]}<br>id=%{customdata[2]}<br>category=%{customdata[3]}<extra></extra>',
          }
        ];
        xTitle = 'Spearman rho (female vs male, Age vs CV)';
      } else {
        const x = ranked.map(r => cohort === 'female' || cohort === 'male' ? Number(r.metric.spearman_rho) : Number(r.spearman_rho)).reverse();
        const custom = ranked.map(r => {
          if (cohort === 'female' || cohort === 'male') {
            return [r.metric.spearman_p, r.metric.n_bins, r.metric.decline_flag, r.biomarker_id, r.category];
          }
          return [r.spearman_p, r.n_bins, r.decline_flag, r.biomarker_id, r.category];
        }).reverse();
        const colors = x.map(v => (v < 0 ? '#0f766e' : '#b45309'));
        traces = [{
          type: 'bar',
          orientation: 'h',
          y,
          x,
          marker: { color: colors },
          customdata: custom,
          hovertemplate: 'rho=%{x:.4f}<br>p=%{customdata[0]:.5f}<br>n_bins=%{customdata[1]}<br>decline=%{customdata[2]}<br>id=%{customdata[3]}<br>category=%{customdata[4]}<extra></extra>',
          name: cohort === 'female' ? 'Female' : cohort === 'male' ? 'Male' : 'Pooled',
        }];
      }

      const mobile = window.matchMedia('(max-width: 760px)').matches;
      Plotly.newPlot('compare-plot', traces, {
        title: mode === 'negative' ? `Top ${topN} Most Negative Spearman Biomarkers` :
               mode === 'positive' ? `Top ${topN} Most Positive Spearman Biomarkers` :
               `Top ${topN} Largest |Spearman| Biomarkers`,
        annotations: [{
          xref: 'paper',
          yref: 'paper',
          x: 1,
          y: 1.12,
          showarrow: false,
          text: `Filter: ${categoryLabel}${compareIncludeEnvEl.checked ? ' (env included)' : ''}, cohort: ${cohort}`,
          font: { size: mobile ? 10 : 12, color: '#5f6b7a' },
        }],
        barmode: cohort === 'both' ? 'group' : 'relative',
        xaxis: { title: xTitle, tickfont: { size: mobile ? 10 : 12 } },
        yaxis: { automargin: true, tickfont: { size: mobile ? 10 : 12 } },
        margin: mobile ? { t: 64, l: 150, r: 10, b: 44 } : { t: 56, l: 260, r: 16, b: 54 },
        paper_bgcolor: '#ffffff',
        plot_bgcolor: '#ffffff',
      }, { responsive: true, displaylogo: false });
    }

    async function applySearch() {
      const q = searchEl.value.toLowerCase().trim();
      if (!q) return;
      const hit = getDashboardMetadata().find(
        m => `${m.display_name || ''} ${m.biomarker_name || ''} ${m.variable_name || ''} ${m.source_files || ''} ${m.source_variables || ''}`
          .toLowerCase()
          .includes(q)
      );
      if (hit) {
        selectEl.value = hit.biomarker_id;
        renderMetrics(hit.biomarker_id);
        await renderPlot(hit.biomarker_id);
      }
    }

    async function refreshDashboardFromFilters() {
      const id = renderOptions();
      renderRankTable();
      if (!id) {
        document.getElementById('metrics').innerHTML = '<div class="metric">No biomarkers match current filters.</div>';
        Plotly.newPlot('plot', [], { title: 'No biomarkers match current filters' }, { responsive: true, displaylogo: false });
        return;
      }
      renderMetrics(id);
      await renderPlot(id);
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
      state.metadataById = new Map(metadata.map(m => [m.biomarker_id, m]));

      showLowNEl.checked = true;
      includeEnvEl.checked = false;
      compareIncludeEnvEl.checked = false;
      renderCategorySelect(categoryFilterEl, includeEnvEl.checked, 'all_core');
      renderCategorySelect(compareCategoryEl, compareIncludeEnvEl.checked, 'all_core');

      await refreshDashboardFromFilters();
      renderComparePlot();

      statusChip.textContent = `Ready: ${state.metadata.length} biomarkers indexed`;

      tabDashboardBtn.addEventListener('click', () => setTopTab('dashboard'));
      tabCompareBtn.addEventListener('click', () => {
        setTopTab('compare');
        renderComparePlot();
      });
      tabInfoBtn.addEventListener('click', () => setTopTab('info'));
      compareSortEl.addEventListener('change', renderComparePlot);
      compareTopNEl.addEventListener('change', renderComparePlot);
      compareCategoryEl.addEventListener('change', renderComparePlot);
      compareCohortEl.addEventListener('change', renderComparePlot);
      compareIncludeEnvEl.addEventListener('change', () => {
        renderCategorySelect(compareCategoryEl, compareIncludeEnvEl.checked, compareCategoryEl.value);
        renderComparePlot();
      });
      selectEl.addEventListener('change', async () => {
        const id = selectEl.value;
        state.currentId = id;
        renderMetrics(id);
        await renderPlot(id);
      });
      searchEl.addEventListener('change', applySearch);
      searchEl.addEventListener('keyup', (e) => { if (e.key === 'Enter') applySearch(); });
      categoryFilterEl.addEventListener('change', refreshDashboardFromFilters);
      includeEnvEl.addEventListener('change', async () => {
        renderCategorySelect(categoryFilterEl, includeEnvEl.checked, categoryFilterEl.value);
        await refreshDashboardFromFilters();
      });
      cohortFilterEl.addEventListener('change', async () => {
        if (state.currentId) await renderPlot(state.currentId);
      });
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
      window.addEventListener('resize', () => {
        const plotEl = document.getElementById('plot');
        const compareEl = document.getElementById('compare-plot');
        if (plotEl) Plotly.Plots.resize(plotEl);
        if (compareEl) Plotly.Plots.resize(compareEl);
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
    s = re.sub(r"^\s*(?:\d+[a-z]?[’']?(?:,\s*\d+[a-z]?[’']?){1,20})\s*,?\s*-\s*", "", s)
    s = re.sub(
        r"\s*\(([a-z0-9_-]{2,16})\)",
        lambda m: "" if "/" not in m.group(1) and "%" not in m.group(1) else m.group(0),
        s,
    )
    s = re.sub(r"\s+", " ", s).strip()
    return s


def parse_terminal_unit(label: str) -> tuple[str, str]:
    s = str(label or "").strip()
    m = re.search(r"\(([^()]*)\)\s*$", s)
    if not m:
        return s, ""
    unit = m.group(1).strip()
    base = s[: m.start()].strip().rstrip(",")
    return base, unit


def make_display_name(name: str, unit: str) -> str:
    base = clean_display_base(name)
    u = str(unit or "").strip()
    if not u:
        _, parsed_unit = parse_terminal_unit(name)
        if parsed_unit and ("/" in parsed_unit or "%" in parsed_unit):
            u = parsed_unit
    if u and not re.search(rf"\(\s*{re.escape(u)}\s*\)\s*$", base, flags=re.IGNORECASE):
        base = f"{base} ({u})"
    return base


def normalize_text(s: str) -> str:
    x = str(s or "").lower()
    x = x.replace("μ", "u").replace("µ", "u")
    x = re.sub(r"[^a-z0-9]+", " ", x)
    return re.sub(r"\s+", " ", x).strip()


CORE_CATEGORY_SET = {
    "Routine - CBC",
    "Routine - CMP",
    "Cardiometabolic - Lipid",
    "Cardiometabolic - Glycemic",
    "Organ - Thyroid",
    "Organ - Renal",
    "Organ - Hepatic",
    "Specialized - Coagulation",
    "Specialized - Nutritional/Vitamin",
    "Specialized - Inflammatory",
}

CATEGORY_ORDER = {
    "Routine - CBC": 1,
    "Routine - CMP": 2,
    "Cardiometabolic - Lipid": 3,
    "Cardiometabolic - Glycemic": 4,
    "Organ - Thyroid": 5,
    "Organ - Renal": 6,
    "Organ - Hepatic": 7,
    "Specialized - Coagulation": 8,
    "Specialized - Nutritional/Vitamin": 9,
    "Specialized - Inflammatory": 10,
    "Hormones/Reproductive": 11,
    "Infectious/Serology": 12,
    "Other Clinical": 13,
    "Environmental/Toxicant": 14,
}


def is_environmental_marker(name: str, variable_name: str, source_files: str) -> bool:
    txt = normalize_text(f"{name} {variable_name} {source_files}")
    patterns = [
        r"\bdioxin\b",
        r"\bdibenzofuran\b",
        r"\bpolychlorinated biphenyl\b",
        r"\bpcb\d*\b",
        r"\bperfluoro\b",
        r"\bpfos\b|\bpfoa\b|\bpfna\b|\bpfda\b|\bpfua\b|\bpfhx\b",
        r"\bbromodiphenyl\b",
        r"\bheptachlor\b|\bendrin\b|\baldrin\b|\bmirex\b|\bnonachlor\b|\bchlordane\b|\bdieldrin\b",
        r"\bbenzene\b|\btoluene\b|\bxylene\b|\bchloroform\b|\bbromoform\b",
        r"\btrichloroethene\b|\btetrachloroethene\b|\btrichloroethane\b",
        r"\bdichloroethane\b|\bdichlorobenzene\b",
        r"\bcarbon tetrachloride\b|\bstyrene\b|\bethylbenzene\b|\bmtbe\b|\bmethyl tert butyl ether\b",
        r"\bperchlorate\b|\bcotinine\b|\bhydroxycotinine\b",
        r"\bcadmium\b|\blead\b|\bmercury\b",
        r"\bacrylamide\b|\bglycideamide\b|\bcrotonaldehyde\b",
        r"\bpesticide\b|\btoxicant\b|\bvolatile organic\b|\bvoc\b",
    ]
    return any(re.search(p, txt) is not None for p in patterns)


def classify_biomarker(name: str, variable_name: str, source_files: str) -> tuple[str, bool, bool]:
    txt = normalize_text(f"{name} {variable_name} {source_files}")
    is_env = is_environmental_marker(name, variable_name, source_files)
    if is_env:
        return "Environmental/Toxicant", True, False

    def has_any(keys: list[str]) -> bool:
        return any(k in txt for k in keys)

    if has_any(["a1c", "glycohemoglobin", "hemoglobin a1", "glucose", "insulin", "c peptide"]):
        return "Cardiometabolic - Glycemic", False, True

    if has_any(
        [
            "hemoglobin",
            "hematocrit",
            "platelet",
            "lymphocyte",
            "neutrophil",
            "eosinophil",
            "basophil",
            "monocyte",
            "white blood cell",
            "red blood cell",
            "reticulocyte",
            "mcv",
            "mch",
            "mchc",
            "rdw",
        ]
    ):
        return "Routine - CBC", False, True

    if has_any(["cholesterol", "triglyceride", "lipoprotein", "apolipoprotein", "hdl", "ldl"]):
        return "Cardiometabolic - Lipid", False, True

    if has_any(["thyroid", "tsh", "thyroxine", "triiodothyronine", "free t4", "t4", "t3", "thyroglobulin"]):
        return "Organ - Thyroid", False, True

    if has_any(["creatinine", "blood urea nitrogen", " bun ", "cystatin", "uric acid", "egfr", "kidney"]):
        return "Organ - Renal", False, True

    if has_any(
        [
            "alanine aminotransferase",
            "aspartate aminotransferase",
            "alkaline phosphatase",
            "gamma glutamyl",
            "bilirubin",
            "albumin",
            "total protein",
            "globulin",
            "lactate dehydrogenase",
            " alt ",
            " ast ",
            " ggt ",
            " ldh ",
            "hepatic",
            "liver",
        ]
    ):
        return "Organ - Hepatic", False, True

    if has_any(["prothrombin", "pt inr", "inr", "fibrinogen", "coag", "aptt", "ptt", "d dimer"]):
        return "Specialized - Coagulation", False, True

    if has_any(
        [
            "vitamin",
            "folate",
            "ferritin",
            "transferrin",
            "iron",
            "retinol",
            "tocopherol",
            "carotene",
            "selenium",
            "zinc",
            "copper",
            "b12",
            "b6",
        ]
    ):
        return "Specialized - Nutritional/Vitamin", False, True

    if has_any(["c reactive protein", " crp ", "hs crp", "sedimentation", "inflamm", "alpha 1 acid glycoprotein"]):
        return "Specialized - Inflammatory", False, True

    if has_any(
        [
            "testosterone",
            "estradiol",
            "progesterone",
            "anti mullerian",
            "inhibin",
            "luteinizing hormone",
            "follicle stimulating",
            "shbg",
            "prolactin",
            "cortisol",
            "androstenedione",
        ]
    ):
        return "Hormones/Reproductive", False, False

    if has_any(
        [
            "antibody",
            "igg",
            "igm",
            "ige",
            "measles",
            "mumps",
            "rubella",
            "varicella",
            "toxoplasma",
            "chlamydia",
            "pertussis",
            "polio",
            "tb ",
            "cryptosporidium",
        ]
    ):
        return "Infectious/Serology", False, False

    if has_any(
        [
            "sodium",
            "potassium",
            "chloride",
            "bicarbonate",
            "co2",
            "calcium",
            "phosphorus",
            "anion gap",
            "osmolality",
            "electrolyte",
            "metabolic panel",
        ]
    ):
        return "Routine - CMP", False, True

    return "Other Clinical", False, False


AGE_BINS = list(np.arange(20, 90, 5)) + [200]
AGE_LABELS = [f"{a}-{a+4}" for a in range(20, 85, 5)] + ["85+"]
AGE_MIDS = {lab: mid for lab, mid in zip(AGE_LABELS, [a + 2.5 for a in range(20, 85, 5)] + [87.5])}


def slope(x: np.ndarray, y: np.ndarray) -> float:
    if len(x) < 2:
        return np.nan
    return float(np.polyfit(x, y, 1)[0])


def compute_binned_long(df: pd.DataFrame, group_cols: list[str]) -> pd.DataFrame:
    tmp = df.copy()
    tmp["age_bin"] = pd.cut(tmp["age_years"], bins=AGE_BINS, labels=AGE_LABELS, right=False, include_lowest=True)
    tmp["age_mid"] = tmp["age_bin"].map(AGE_MIDS).astype(float)
    tmp = tmp.dropna(subset=["age_bin", "value"])
    grouped = (
        tmp.groupby(group_cols + ["age_bin", "age_mid"], observed=True)["value"]
        .agg(
            n="count",
            mean="mean",
            std="std",
            p025=lambda s: float(np.nanpercentile(s.to_numpy(dtype=float), 2.5)),
            p975=lambda s: float(np.nanpercentile(s.to_numpy(dtype=float), 97.5)),
        )
        .reset_index()
    )
    grouped["cv"] = grouped["std"] / grouped["mean"].abs()
    grouped.loc[grouped["mean"].abs() < 1e-8, "cv"] = np.nan
    grouped["passes_n_threshold"] = grouped["n"] >= 30
    grouped = grouped.dropna(subset=["cv"]).reset_index(drop=True)
    return grouped


def trend_from_points(points: list[dict]) -> dict:
    eligible = [p for p in points if bool(p.get("passes_n_threshold")) and p.get("cv") is not None]
    x = np.asarray([float(p["age_mid"]) for p in eligible], dtype=float)
    y = np.asarray([float(p["cv"]) for p in eligible], dtype=float)
    rho = np.nan
    pval = np.nan
    if len(y) >= 2:
        rho, pval = spearmanr(x, y)
    pos = y > 0
    out = {
        "n_bins": int(len(eligible)),
        "spearman_rho": float(rho) if pd.notna(rho) else None,
        "spearman_p": float(pval) if pd.notna(pval) else None,
        "linear_slope_cv_per_year": float(slope(x, y)) if len(y) >= 2 else None,
        "linear_slope_logcv_per_year": float(slope(x[pos], np.log(y[pos]))) if int(pos.sum()) >= 2 else None,
    }
    out["decline_flag"] = bool(
        out["n_bins"] >= 5
        and out["spearman_rho"] is not None
        and out["spearman_p"] is not None
        and out["linear_slope_cv_per_year"] is not None
        and out["spearman_rho"] < 0
        and out["spearman_p"] < 0.05
        and out["linear_slope_cv_per_year"] < 0
    )
    return out


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

    raw_samples: dict[str, list[dict]] = {}
    raw_samples_by_sex: dict[str, dict[str, list[dict]]] = {}
    raw_counts: dict[str, int] = {}
    raw_counts_by_sex: dict[str, dict[str, int]] = {}
    sex_points_by_id: dict[str, dict[str, list[dict]]] = {}
    sex_metrics_by_id: dict[str, dict[str, dict]] = {}
    if long_df is not None and not long_df.empty:
        use = long_df[["biomarker_id", "age_years", "value", "sex"]].dropna(subset=["biomarker_id", "age_years", "value"])
        use["sex_norm"] = use["sex"].astype(str).str.strip().str.lower()
        use.loc[~use["sex_norm"].isin(["male", "female"]), "sex_norm"] = "unknown"

        pooled_binned = compute_binned_long(use[["biomarker_id", "age_years", "value"]], group_cols=["biomarker_id"])
        pooled_q = pooled_binned[["biomarker_id", "age_bin", "age_mid", "p025", "p975"]].copy()
        pooled_q["age_bin"] = pooled_q["age_bin"].astype(str)
        cv_df["age_bin"] = cv_df["age_bin"].astype(str)
        cv_df = cv_df.merge(pooled_q, on=["biomarker_id", "age_bin", "age_mid"], how="left")

        raw_counts = use.groupby("biomarker_id", observed=True).size().astype(int).to_dict()
        sex_counts_tbl = (
            use[use["sex_norm"].isin(["male", "female"])]
            .groupby(["biomarker_id", "sex_norm"], observed=True)
            .size()
            .reset_index(name="n")
        )
        for r in sex_counts_tbl.itertuples(index=False):
            raw_counts_by_sex.setdefault(str(r.biomarker_id), {})[str(r.sex_norm)] = int(r.n)

        rng = np.random.default_rng(random_seed)
        for bid, g in use.groupby("biomarker_id", observed=True):
            g_pool = g[["age_years", "value"]].dropna()
            if len(g_pool) > raw_sample_n:
                idx = rng.choice(len(g_pool), size=raw_sample_n, replace=False)
                g_pool = g_pool.iloc[idx]
            raw_samples[str(bid)] = [{"age_years": float(r.age_years), "value": float(r.value)} for r in g_pool.itertuples(index=False)]

        for (bid, sex_norm), g in use[use["sex_norm"].isin(["male", "female"])].groupby(["biomarker_id", "sex_norm"], observed=True):
            g2 = g[["age_years", "value"]].dropna()
            if len(g2) > raw_sample_n:
                idx = rng.choice(len(g2), size=raw_sample_n, replace=False)
                g2 = g2.iloc[idx]
            raw_samples_by_sex.setdefault(str(bid), {})[str(sex_norm)] = [
                {"age_years": float(r.age_years), "value": float(r.value)} for r in g2.itertuples(index=False)
            ]

        sex_binned = compute_binned_long(
            use[use["sex_norm"].isin(["male", "female"])][["biomarker_id", "age_years", "value", "sex_norm"]],
            group_cols=["biomarker_id", "sex_norm"],
        )
        for (bid, sex_norm), g in sex_binned.groupby(["biomarker_id", "sex_norm"], observed=True):
            pts = [
                {
                    "age_bin": str(r.age_bin),
                    "age_mid": float(r.age_mid),
                    "n": int(r.n),
                    "mean": float(r.mean),
                    "std": float(r.std) if pd.notna(r.std) else None,
                    "cv": float(r.cv),
                    "range95_low": float(r.p025) if pd.notna(r.p025) else None,
                    "range95_high": float(r.p975) if pd.notna(r.p975) else None,
                    "ci95_low": float(r.p025) if pd.notna(r.p025) else None,
                    "ci95_high": float(r.p975) if pd.notna(r.p975) else None,
                    "passes_n_threshold": bool(r.passes_n_threshold),
                }
                for r in g.sort_values("age_mid").itertuples(index=False)
            ]
            bid_s = str(bid)
            sex_s = str(sex_norm)
            sex_points_by_id.setdefault(bid_s, {})[sex_s] = pts
            sex_metrics_by_id.setdefault(bid_s, {})[sex_s] = trend_from_points(pts)
    else:
        cv_df["p025"] = np.nan
        cv_df["p975"] = np.nan

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
    metadata["raw_total_n"] = metadata["biomarker_id"].map(raw_counts).fillna(0).astype(int)
    metadata["raw_sample_cap"] = int(raw_sample_n)
    metadata["display_name"] = [make_display_name(n, u) for n, u in zip(metadata["biomarker_name"], metadata["unit"])]
    cat_rows = [
        classify_biomarker(n, v, sf)
        for n, v, sf in zip(metadata["biomarker_name"], metadata["variable_name"], metadata["source_files"])
    ]
    metadata["category"] = [r[0] for r in cat_rows]
    metadata["is_environmental"] = [bool(r[1]) for r in cat_rows]
    metadata["is_core_clinical"] = [bool(r[2]) for r in cat_rows]
    metadata["category_rank"] = metadata["category"].map(CATEGORY_ORDER).fillna(999).astype(int)
    metadata = metadata.sort_values(["category_rank", "display_name", "biomarker_id"]).reset_index(drop=True)

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
    for m in metrics:
        bid = str(m.get("biomarker_id"))
        m["sex_metrics"] = sex_metrics_by_id.get(bid, {})

    series_index: dict[str, str] = {}
    series_payloads: dict[str, dict] = {}
    meta_by_id = metadata.set_index("biomarker_id").to_dict(orient="index")

    for bid, g in cv_df.groupby("biomarker_id", observed=True):
        g = g.sort_values("age_mid")
        rel_path = safe_series_filename(bid)
        md = meta_by_id.get(bid, {})
        series_index[bid] = rel_path
        series_payloads[rel_path] = {
            "biomarker_id": bid,
            "biomarker_name": g["biomarker_name"].iloc[0],
            "display_name": str(md.get("display_name") or make_display_name(
                str(g["biomarker_name"].iloc[0]),
                str(g["unit"].iloc[0] if "unit" in g.columns else ""),
            )),
            "variable_name": g["variable_name"].iloc[0],
            "unit": g["unit"].iloc[0] if "unit" in g.columns else "",
            "category": md.get("category", "Other Clinical"),
            "is_environmental": bool(md.get("is_environmental", False)),
            "is_core_clinical": bool(md.get("is_core_clinical", False)),
            "raw_total_n": int(md.get("raw_total_n", 0)),
            "raw_total_n_by_sex": raw_counts_by_sex.get(str(bid), {}),
            "raw_sample_cap": int(md.get("raw_sample_cap", raw_sample_n)),
            "points": [
                {
                    "age_bin": str(r.age_bin),
                    "age_mid": float(r.age_mid),
                    "n": int(r.n),
                    "mean": float(r.mean),
                    "std": float(r.std) if pd.notna(r.std) else None,
                    "cv": float(r.cv),
                    "range95_low": float(r.p025) if pd.notna(r.p025) else None,
                    "range95_high": float(r.p975) if pd.notna(r.p975) else None,
                    "ci95_low": float(r.p025) if pd.notna(r.p025) else None,
                    "ci95_high": float(r.p975) if pd.notna(r.p975) else None,
                    "passes_n_threshold": bool(r.passes_n_threshold),
                }
                for r in g.itertuples(index=False)
            ],
            "raw_sample": raw_samples.get(str(bid), []),
            "raw_sample_by_sex": raw_samples_by_sex.get(str(bid), {}),
            "sex_points": sex_points_by_id.get(str(bid), {}),
            "sex_metrics": sex_metrics_by_id.get(str(bid), {}),
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
        long_df = pd.read_parquet(long_path, columns=["biomarker_id", "age_years", "value", "sex"])

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
