#!/usr/bin/env python3
"""Build a static interactive HTML dashboard with lazy-loaded biomarker series."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import time
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import skew as scipy_skew
from scipy.stats import spearmanr

from build_aging_biomarkers_dashboard import (
    build_disease_explorer_bundle,
    build_public_manifest,
    load_public_disease_long,
    write_public_dashboard_bundle,
)
from nhanes_common import ensure_dir



ROOT = Path(__file__).resolve().parents[1]
TEMPLATE_PATH = Path(__file__).resolve().parent / "templates" / "dashboard_template.html"


HTML_TEMPLATE = """<!DOCTYPE html>
<html lang=\"en\">
<head>
  <meta charset=\"utf-8\" />
  <meta name=\"viewport\" content=\"width=device-width, initial-scale=1\" />
  <title>nhanes-biomarker-dashboard - __SPECIMEN_TITLE__</title>
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
    .nav-stack {
      display: flex;
      flex-direction: column;
      gap: 10px;
      margin-bottom: 14px;
    }
    .nav-row {
      display: flex;
      align-items: center;
      gap: 10px;
      flex-wrap: wrap;
    }
    .nav-label {
      font-size: 11px;
      color: var(--muted);
      text-transform: uppercase;
      letter-spacing: 0.7px;
      font-weight: 700;
      min-width: 96px;
    }
    .specimen-row {
      background: #fffdf7;
      border: 1px solid var(--line);
      border-radius: 12px;
      padding: 10px 12px;
    }
    .specimen-row .nav-label {
      color: #0b5f58;
    }
    .specimen-tabs,
    .view-tabs {
      display: flex;
      gap: 8px;
      flex-wrap: wrap;
      flex: 1 1 auto;
    }
    .tab-btn {
      border: 1px solid var(--line);
      background: var(--tab);
      border-radius: 10px;
      padding: 8px 12px;
      cursor: pointer;
      font-size: 14px;
      font-weight: 600;
      text-decoration: none;
      color: var(--ink);
      display: inline-flex;
      align-items: center;
      justify-content: center;
    }
    .tab-btn.active {
      background: var(--tab-active);
      border-color: var(--tab-active);
      color: #fff;
    }
    .specimen-tabs .tab-btn {
      padding: 10px 16px;
      border-radius: 999px;
      font-size: 15px;
      font-weight: 700;
      background: #e6f2f0;
      border-color: #cde2de;
    }
    .specimen-tabs .tab-btn.active {
      box-shadow: 0 1px 2px rgba(15, 118, 110, 0.28);
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
    input[type="range"],
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
    input[type="range"] {
      padding: 0;
      border: 0;
      margin: 8px 0 2px 0;
      accent-color: var(--accent);
      background: transparent;
    }
    .trim-caption {
      font-size: 12px;
      color: var(--muted);
      margin: 2px 0 10px 0;
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
    #compare-clalit-plot { width: 100%; height: 480px; }
    .scatter-controls { display: flex; gap: 10px; flex-wrap: wrap; align-items: end; margin-bottom: 10px; }
    .scatter-controls label { text-transform: none; letter-spacing: 0; font-size: 13px; }
    .scatter-controls select, .scatter-controls input { margin: 4px 0 0 0; width: 220px; }
    #scatter-category { height: 160px; }
    .scatter-actions { display: flex; gap: 6px; align-items: center; margin: 2px 0 8px 0; flex-wrap: wrap; }
    .scatter-actions button {
      border: 1px solid var(--line);
      background: #f8f5ef;
      border-radius: 8px;
      padding: 6px 10px;
      cursor: pointer;
      font-size: 12px;
    }
    .scatter-hint { font-size: 12px; color: var(--muted); }
    #scatter-plot { width: 100%; height: 640px; }
    .toggle-btn-active {
      background: var(--accent) !important;
      color: #fff !important;
      border-color: var(--accent) !important;
    }
    .hist-controls { display: flex; gap: 10px; flex-wrap: wrap; align-items: end; margin-bottom: 10px; }
    .hist-controls label { text-transform: none; letter-spacing: 0; font-size: 13px; }
    .hist-controls select, .hist-controls input { margin: 4px 0 0 0; width: 220px; }
    #hist-category { height: 160px; }
    .hist-actions { display: flex; gap: 6px; align-items: center; margin: 2px 0 8px 0; flex-wrap: wrap; }
    .hist-actions button {
      border: 1px solid var(--line);
      background: #f8f5ef;
      border-radius: 8px;
      padding: 6px 10px;
      cursor: pointer;
      font-size: 12px;
    }
    .hist-hint { font-size: 12px; color: var(--muted); }
    #hist-plot { width: 100%; height: 600px; }
    .waterfall-controls { display: flex; gap: 10px; flex-wrap: wrap; align-items: end; margin-bottom: 10px; }
    .waterfall-controls label { text-transform: none; letter-spacing: 0; font-size: 13px; }
    .waterfall-controls select, .waterfall-controls input { margin: 4px 0 0 0; width: 260px; }
    .waterfall-caption { font-size: 12px; color: var(--muted); margin-top: 2px; }
    #waterfall-plot { width: 100%; height: 760px; }

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
      #compare-clalit-plot { height: 480px; }
      #scatter-plot { height: 560px; }
      #hist-plot { height: 520px; }
      #waterfall-plot { height: 620px; }
    }
    @media (max-width: 760px) {
      .nav-row { align-items: flex-start; }
      .nav-label { min-width: 100%; margin-bottom: 2px; }
      .specimen-tabs .tab-btn,
      .view-tabs .tab-btn { flex: 1 1 calc(50% - 8px); text-align: center; }
      .table-wrap {
        max-height: none;
        overflow-x: auto;
      }
      table { font-size: 12px; min-width: 560px; }
      #plot { height: 380px; }
      #compare-plot { height: 500px; }
      #compare-clalit-plot { height: 440px; }
      .compare-controls label { width: 100%; }
      .compare-controls select,
      .compare-controls input { width: 100%; }
      .scatter-controls label { width: 100%; }
      .scatter-controls select,
      .scatter-controls input { width: 100%; }
      #scatter-category { height: 180px; }
      .hist-controls label { width: 100%; }
      .hist-controls select,
      .hist-controls input { width: 100%; }
      #hist-category { height: 180px; }
      .waterfall-controls label { width: 100%; }
      .waterfall-controls select,
      .waterfall-controls input { width: 100%; }
    }
    @media (max-width: 520px) {
      .specimen-tabs .tab-btn,
      .view-tabs .tab-btn { flex: 1 1 100%; }
      .mode-buttons { flex-wrap: wrap; }
      .mode-btn { flex: 1 1 calc(33.333% - 8px); }
      #plot { height: 340px; }
      #compare-plot { height: 440px; }
      #compare-clalit-plot { height: 440px; }
      #scatter-plot { height: 440px; }
      #hist-plot { height: 420px; }
      #waterfall-plot { height: 500px; }
    }
  </style>
</head>
<body>
  <div class=\"wrap\">
    <div class=\"hero\">
      <div>
        <h1 id=\"hero-title\">nhanes-biomarker-dashboard</h1>
        <div id=\"hero-sub\" class=\"sub\">Explore cross-sectional aging trajectories across __SPECIMEN_LOWER__ biomarkers.</div>
      </div>
      <div id=\"status-chip\" class=\"status-chip\">Loading metadata…</div>
    </div>

    <div class=\"nav-stack\" aria-label=\"Dashboard navigation\">
      <div class=\"nav-row specimen-row\" role=\"navigation\" aria-label=\"Specimen\">
        <div class=\"nav-label\">Specimen</div>
        <div class=\"specimen-tabs\">
          __SPECIMEN_SWITCH_LINK__
        </div>
      </div>
      <div class=\"nav-row view-row\" role=\"tablist\" aria-label=\"Analysis View\">
        <div class=\"nav-label\">Analysis View</div>
        <div class=\"view-tabs\">
          <button id=\"tab-dashboard\" class=\"tab-btn active\" type=\"button\">Dashboard</button>
          <button id=\"tab-compare\" class=\"tab-btn\" type=\"button\">Compare Rankings</button>
          <button id=\"tab-scatter\" class=\"tab-btn\" type=\"button\">Scatter Plot</button>
          <button id=\"tab-hist\" class=\"tab-btn\" type=\"button\">Histograms</button>
          <button id=\"tab-waterfall\" class=\"tab-btn\" type=\"button\">Waterfall</button>
          <button id=\"tab-info\" class=\"tab-btn\" type=\"button\">Info & Methods</button>
        </div>
      </div>
    </div>

    <div id=\"panel-dashboard\" class=\"panel active\">
      <div class=\"grid\">
        <div class=\"card sticky\">
          <div class=\"mode-buttons\">
            <button id=\"mode-cv\" class=\"mode-btn active\" type=\"button\">Plot CV</button>
            <button id=\"mode-mean\" class=\"mode-btn\" type=\"button\">Plot Median</button>
            <button id=\"mode-skew\" class=\"mode-btn\" type=\"button\">Plot Skewness</button>
          </div>

          <label for=\"search\">Search Biomarker</label>
              <input id=\"search\" list=\"biomarker-options\" placeholder=\"Type name, code, file...\" autocomplete=\"off\" spellcheck=\"false\" />
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

          <label for=\"trim-slider\">Symmetric Trim Per Tail (%)</label>
          <input id=\"trim-slider\" type=\"range\" min=\"0\" max=\"25\" step=\"5\" value=\"10\" />
          <div id=\"trim-label\" class=\"trim-caption\">10% each tail kept out -> using 10-90 percentile band</div>

          <label class="check-label"><input id="show-low-n" type="checkbox" checked /> Show low-n bins (&lt;30)</label>
          <label id="hide-clalit-wrap" class="check-label"><input id="hide-clalit" type="checkbox" /> Hide Clalit data</label>

          <div id=\"metrics\" class=\"card\" style=\"margin-top:10px;\"></div>
        </div>

        <div class=\"card\">
          <div id=\"plot\"></div>
          <h3 id=\"rank-title\">Biomarkers Ranked by Most Negative Spearman Rho (CV vs age)</h3>
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
          <label>Statistic
            <select id=\"compare-stat\">
              <option value=\"cv\" selected>CV vs age</option>
              <option value=\"mean\">Mean vs age</option>
              <option value=\"skewness\">Skewness vs age</option>
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
          <label>Symmetric trim (% per tail)
            <input id=\"compare-trim-slider\" type=\"range\" min=\"0\" max=\"25\" step=\"5\" value=\"10\" />
            <div id=\"compare-trim-label\" class=\"trim-caption\">10% each tail kept out -> using 10-90 percentile band</div>
          </label>
          <label>Top N
            <input id=\"compare-topn\" type=\"number\" min=\"10\" max=\"200\" step=\"5\" value=\"40\" />
          </label>
        </div>
        <div id="compare-plot"></div>
        <h3 id="compare-clalit-title" style="margin-top: 24px;">Clalit vs NHANES Agreement</h3>
        <div id="compare-clalit-plot"></div>
      </div>
    </div>

    <div id=\"panel-scatter\" class=\"panel\">
      <div class=\"card\">
        <div class=\"scatter-controls\">
          <label>X axis statistic
            <select id=\"scatter-x-stat\">
              <option value=\"cv\" selected>CV vs age (Spearman rho)</option>
              <option value=\"mean\">Mean vs age (Spearman rho)</option>
              <option value=\"skewness\">Skewness vs age (Spearman rho)</option>
            </select>
          </label>
          <label>Y axis statistic
            <select id=\"scatter-y-stat\">
              <option value=\"cv\">CV vs age (Spearman rho)</option>
              <option value=\"mean\" selected>Mean vs age (Spearman rho)</option>
              <option value=\"skewness\">Skewness vs age (Spearman rho)</option>
            </select>
          </label>
          <label>Cohort
            <select id=\"scatter-cohort\">
              <option value=\"pooled\" selected>Pooled</option>
              <option value=\"female\">Female</option>
              <option value=\"male\">Male</option>
              <option value=\"both\">Both (Female + Male)</option>
            </select>
          </label>
          <label>Symmetric trim (% per tail)
            <input id=\"scatter-trim-slider\" type=\"range\" min=\"0\" max=\"25\" step=\"5\" value=\"10\" />
            <div id=\"scatter-trim-label\" class=\"trim-caption\">10% each tail kept out -> using 10-90 percentile band</div>
          </label>
          <label class=\"check-label\"><input id=\"scatter-include-env\" type=\"checkbox\" /> Include environmental/toxicant</label>
        </div>
        <label>Categories (multi-select)</label>
        <div class=\"scatter-actions\">
          <button id=\"scatter-cat-all\" type=\"button\">Select all visible</button>
          <button id=\"scatter-cat-core\" type=\"button\">Clinical/core only</button>
          <button id=\"scatter-cat-clear\" type=\"button\">Clear</button>
          <button id=\"scatter-label-toggle\" type=\"button\">Show labels</button>
          <span id=\"scatter-selection-count\" class=\"scatter-hint\"></span>
        </div>
        <select id=\"scatter-category\" multiple></select>
        <div id=\"scatter-plot\"></div>
      </div>
    </div>

    <div id=\"panel-hist\" class=\"panel\">
      <div class=\"card\">
        <div class=\"hist-controls\">
          <label>Statistic
            <select id=\"hist-stat\">
              <option value=\"cv\" selected>CV vs age (Spearman rho)</option>
              <option value=\"mean\">Mean vs age (Spearman rho)</option>
              <option value=\"skewness\">Skewness vs age (Spearman rho)</option>
            </select>
          </label>
          <label>Cohort
            <select id=\"hist-cohort\">
              <option value=\"pooled\" selected>Pooled</option>
              <option value=\"female\">Female</option>
              <option value=\"male\">Male</option>
              <option value=\"both\">Both (Female + Male)</option>
            </select>
          </label>
          <label>Symmetric trim (% per tail)
            <input id=\"hist-trim-slider\" type=\"range\" min=\"0\" max=\"25\" step=\"5\" value=\"10\" />
            <div id=\"hist-trim-label\" class=\"trim-caption\">10% each tail kept out -> using 10-90 percentile band</div>
          </label>
          <label class=\"check-label\"><input id=\"hist-include-env\" type=\"checkbox\" /> Include environmental/toxicant</label>
        </div>
        <label>Categories (multi-select)</label>
        <div class=\"hist-actions\">
          <button id=\"hist-cat-all\" type=\"button\">Select all visible</button>
          <button id=\"hist-cat-core\" type=\"button\">Clinical/core only</button>
          <button id=\"hist-cat-clear\" type=\"button\">Clear</button>
          <span id=\"hist-selection-count\" class=\"hist-hint\"></span>
        </div>
        <select id=\"hist-category\" multiple></select>
        <div id=\"hist-plot\"></div>
      </div>
    </div>

    <div id=\"panel-waterfall\" class=\"panel\">
      <div class=\"card\">
        <div class=\"waterfall-controls\">
          <label>Search biomarker
            <input id=\"waterfall-search\" list=\"waterfall-biomarker-options\" placeholder=\"Type biomarker name...\" autocomplete=\"off\" spellcheck=\"false\" />
            <datalist id=\"waterfall-biomarker-options\"></datalist>
          </label>
          <label>Biomarker
            <select id=\"waterfall-biomarker\"></select>
          </label>
          <label>Cohort
            <select id=\"waterfall-cohort\">
              <option value=\"pooled\" selected>Pooled</option>
              <option value=\"female\">Female</option>
              <option value=\"male\">Male</option>
            </select>
          </label>
          <label>Symmetric trim (% per tail)
            <input id=\"waterfall-trim-slider\" type=\"range\" min=\"0\" max=\"25\" step=\"5\" value=\"10\" />
            <div id=\"waterfall-trim-label\" class=\"trim-caption\">10% each tail kept out -> using 10-90 percentile band</div>
          </label>
          <label>Min n per age bin
            <input id=\"waterfall-min-n\" type=\"number\" min=\"5\" max=\"100\" step=\"5\" value=\"20\" />
            <div class=\"waterfall-caption\">Bins below this n are hidden from waterfall.</div>
          </label>
        </div>
        <div id=\"waterfall-plot\"></div>
      </div>
    </div>

    <div id=\"panel-info\" class=\"panel\">
      <div class=\"info-grid\">
        <div class=\"card info-card\">
          <h3>What This Analysis Does</h3>
          <p>For each __SPECIMEN_LOWER__ biomarker test, this dashboard pools all NHANES cycles/files into one trajectory.</p>
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
            <li><b>Plot Median</b>: age-binned median with interquartile band (25th-75th percentile) and raw scatter sample.</li>
            <li><b>Plot Skewness</b>: age-binned skewness trend (shape/asymmetry of per-bin values).</li>
            <li>Sex view: pooled, female, male, or both on the same chart (female red, male blue).</li>
            <li>Optional robust mode uses configurable symmetric trimming within each age bin (for example 10-90, 20-80, 25-75) before computing summaries.</li>
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
    const DATA_BASE = '__DATA_BASE__';
    const DATA_VERSION = '__DATA_VERSION__';
    const HAS_CLALIT = __HAS_CLALIT__;
    const SPECIMEN_TITLE = '__SPECIMEN_TITLE__';
    const SPECIMEN_LABEL = '__SPECIMEN_LOWER__';

    const selectEl = document.getElementById('biomarker-select');
    const searchEl = document.getElementById('search');
    const optionsEl = document.getElementById('biomarker-options');
    const showLowNEl = document.getElementById('show-low-n');
    const hideClalitEl = document.getElementById('hide-clalit');
    const hideClalitWrapEl = document.getElementById('hide-clalit-wrap');
    const modeCvBtn = document.getElementById('mode-cv');
    const modeMeanBtn = document.getElementById('mode-mean');
    const modeSkewBtn = document.getElementById('mode-skew');
    const statusChip = document.getElementById('status-chip');
    const heroTitleEl = document.getElementById('hero-title');
    const heroSubEl = document.getElementById('hero-sub');

    const tabDashboardBtn = document.getElementById('tab-dashboard');
    const tabCompareBtn = document.getElementById('tab-compare');
    const tabScatterBtn = document.getElementById('tab-scatter');
    const tabHistBtn = document.getElementById('tab-hist');
    const tabWaterfallBtn = document.getElementById('tab-waterfall');
    const tabInfoBtn = document.getElementById('tab-info');
    const specimenLinks = Array.from(document.querySelectorAll('.specimen-link'));
    const panelDashboard = document.getElementById('panel-dashboard');
    const panelCompare = document.getElementById('panel-compare');
    const panelScatter = document.getElementById('panel-scatter');
    const panelHist = document.getElementById('panel-hist');
    const panelWaterfall = document.getElementById('panel-waterfall');
    const panelInfo = document.getElementById('panel-info');
    const compareClalitTitleEl = document.getElementById('compare-clalit-title');
    const compareSortEl = document.getElementById('compare-sort');
    const compareStatEl = document.getElementById('compare-stat');
    const compareTopNEl = document.getElementById('compare-topn');
    const categoryFilterEl = document.getElementById('category-filter');
    const includeEnvEl = document.getElementById('include-env');
    const compareCategoryEl = document.getElementById('compare-category');
    const compareIncludeEnvEl = document.getElementById('compare-include-env');
    const cohortFilterEl = document.getElementById('cohort-filter');
    const compareCohortEl = document.getElementById('compare-cohort');
    const trimSliderEl = document.getElementById('trim-slider');
    const trimLabelEl = document.getElementById('trim-label');
    const compareTrimSliderEl = document.getElementById('compare-trim-slider');
    const compareTrimLabelEl = document.getElementById('compare-trim-label');
    const rankTitleEl = document.getElementById('rank-title');
    const scatterXStatEl = document.getElementById('scatter-x-stat');
    const scatterYStatEl = document.getElementById('scatter-y-stat');
    const scatterCohortEl = document.getElementById('scatter-cohort');
    const scatterTrimSliderEl = document.getElementById('scatter-trim-slider');
    const scatterTrimLabelEl = document.getElementById('scatter-trim-label');
    const scatterIncludeEnvEl = document.getElementById('scatter-include-env');
    const scatterCategoryEl = document.getElementById('scatter-category');
    const scatterCatAllBtn = document.getElementById('scatter-cat-all');
    const scatterCatCoreBtn = document.getElementById('scatter-cat-core');
    const scatterCatClearBtn = document.getElementById('scatter-cat-clear');
    const scatterLabelToggleBtn = document.getElementById('scatter-label-toggle');
    const scatterSelectionCountEl = document.getElementById('scatter-selection-count');
    const histStatEl = document.getElementById('hist-stat');
    const histCohortEl = document.getElementById('hist-cohort');
    const histTrimSliderEl = document.getElementById('hist-trim-slider');
    const histTrimLabelEl = document.getElementById('hist-trim-label');
    const histIncludeEnvEl = document.getElementById('hist-include-env');
    const histCategoryEl = document.getElementById('hist-category');
    const histCatAllBtn = document.getElementById('hist-cat-all');
    const histCatCoreBtn = document.getElementById('hist-cat-core');
    const histCatClearBtn = document.getElementById('hist-cat-clear');
    const histSelectionCountEl = document.getElementById('hist-selection-count');
    const waterfallSearchEl = document.getElementById('waterfall-search');
    const waterfallOptionsEl = document.getElementById('waterfall-biomarker-options');
    const waterfallBiomarkerEl = document.getElementById('waterfall-biomarker');
    const waterfallCohortEl = document.getElementById('waterfall-cohort');
    const waterfallTrimSliderEl = document.getElementById('waterfall-trim-slider');
    const waterfallTrimLabelEl = document.getElementById('waterfall-trim-label');
    const waterfallMinNEl = document.getElementById('waterfall-min-n');

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
      scatterLabels: false,
      waterfallId: null,
    };

    const WATERFALL_AGE_BINS = [
      { label: '20-29', lo: 20, hi: 30 },
      { label: '30-39', lo: 30, hi: 40 },
      { label: '40-49', lo: 40, hi: 50 },
      { label: '50-59', lo: 50, hi: 60 },
      { label: '60-69', lo: 60, hi: 70 },
      { label: '70-79', lo: 70, hi: 80 },
      { label: '80-89', lo: 80, hi: 90 },
      { label: '90+', lo: 90, hi: 200 },
    ];

    const WATERFALL_QUARTILE_COLORS = ['#4B0055', '#2E6F95', '#3AB47D', '#F2E419'];
    const TOP_TABS = ['dashboard', 'compare', 'scatter', 'hist', 'waterfall', 'info'];
    const TOP_TAB_SET = new Set(TOP_TABS);

    function formatNum(v, d=4) {
      if (v === null || v === undefined || Number.isNaN(v)) return 'NA';
      return Number(v).toFixed(d);
    }

    function normalizeTrimPct(v) {
      const n = Number(v ?? 0);
      if (!Number.isFinite(n)) return 0;
      return Math.max(0, Math.min(25, Math.round(n / 5) * 5));
    }

    function trimPctToMode(pct) {
      const p = normalizeTrimPct(pct);
      if (p <= 0) return 'all';
      return `trim_${p}_${100 - p}`;
    }

    function trimModeToPct(mode) {
      if (!mode || mode === 'all') return 0;
      const m = String(mode).match(/^trim_(\d{1,2})_(\d{1,2})$/);
      if (!m) return 0;
      return normalizeTrimPct(Number(m[1]));
    }

    function trimLabelFromPct(pct) {
      const p = normalizeTrimPct(pct);
      if (p <= 0) return 'Using all values (0-100)';
      return `${p}% each tail kept out -> using ${p}-${100 - p} percentile band`;
    }

    function modeToStat(mode) {
      if (mode === 'mean') return 'mean';
      if (mode === 'skewness') return 'skewness';
      return 'cv';
    }

    function statLabel(statKey) {
      if (statKey === 'mean') return 'Mean';
      if (statKey === 'skewness') return 'Skewness';
      return 'CV';
    }

    function canonicalTopTab(tabName) {
      const clean = String(tabName || '').trim().toLowerCase();
      return TOP_TAB_SET.has(clean) ? clean : 'dashboard';
    }

    function topTabFromHash() {
      const clean = String(window.location.hash || '').replace(/^#/, '').trim().toLowerCase();
      if (!clean) return null;
      return TOP_TAB_SET.has(clean) ? clean : null;
    }

    function setTopTabHash(tabName, replace = false) {
      const next = `#${canonicalTopTab(tabName)}`;
      if (window.location.hash === next) return;
      if (replace) window.history.replaceState(null, '', next);
      else window.history.pushState(null, '', next);
    }

    function syncSpecimenSwitchLinks(tabName) {
      const tabHash = `#${canonicalTopTab(tabName)}`;
      for (const link of specimenLinks) {
        const baseHref = link.dataset.baseHref || String(link.getAttribute('href') || '').split('#')[0];
        if (!baseHref) continue;
        link.dataset.baseHref = baseHref;
        link.setAttribute('href', `${baseHref}${tabHash}`);
      }
    }

    function heroCopy(tabName) {
      if (tabName === 'compare') {
        return {
          title: `NHANES ${SPECIMEN_TITLE} Compare Rankings`,
          sub: `Rank ${SPECIMEN_LABEL} biomarkers by age-trend direction and magnitude across cohorts.`,
        };
      }
      if (tabName === 'scatter') {
        return {
          title: `NHANES ${SPECIMEN_TITLE} Scatter View`,
          sub: `Compare two aging-trend statistics per ${SPECIMEN_LABEL} biomarker in 2D.`,
        };
      }
      if (tabName === 'hist') {
        return {
          title: `NHANES ${SPECIMEN_TITLE} Histogram View`,
          sub: `Inspect the distribution of Spearman trend coefficients across ${SPECIMEN_LABEL} biomarkers.`,
        };
      }
      if (tabName === 'waterfall') {
        return {
          title: `NHANES ${SPECIMEN_TITLE} Waterfall View`,
          sub: `Explore full age-stratified value distributions for a selected ${SPECIMEN_LABEL} biomarker.`,
        };
      }
      if (tabName === 'info') {
        return {
          title: `NHANES ${SPECIMEN_TITLE} Info & Methods`,
          sub: `Review cohort filters, metric definitions, and interpretation notes for the ${SPECIMEN_LABEL} dashboard.`,
        };
      }
      return {
        title: `NHANES ${SPECIMEN_TITLE} Biomarker Variability`,
        sub: `Explore cross-sectional aging trajectories across ${SPECIMEN_LABEL} biomarkers.`,
      };
    }

    function setTopTab(tabName) {
      const resolved = canonicalTopTab(tabName);
      const isDash = resolved === 'dashboard';
      const isCompare = resolved === 'compare';
      const isScatter = resolved === 'scatter';
      const isHist = resolved === 'hist';
      const isWaterfall = resolved === 'waterfall';
      const isInfo = resolved === 'info';
      tabDashboardBtn.classList.toggle('active', isDash);
      tabCompareBtn.classList.toggle('active', isCompare);
      tabScatterBtn.classList.toggle('active', isScatter);
      tabHistBtn.classList.toggle('active', isHist);
      tabWaterfallBtn.classList.toggle('active', isWaterfall);
      tabInfoBtn.classList.toggle('active', isInfo);
      panelDashboard.classList.toggle('active', isDash);
      panelCompare.classList.toggle('active', isCompare);
      panelScatter.classList.toggle('active', isScatter);
      panelHist.classList.toggle('active', isHist);
      panelWaterfall.classList.toggle('active', isWaterfall);
      panelInfo.classList.toggle('active', isInfo);
      const copy = heroCopy(resolved);
      if (heroTitleEl) heroTitleEl.textContent = copy.title;
      if (heroSubEl) heroSubEl.textContent = copy.sub;
      syncSpecimenSwitchLinks(resolved);
    }

    async function activateTopTab(tabName, opts = {}) {
      const resolved = canonicalTopTab(tabName);
      const syncHash = Boolean(opts.syncHash);
      const replaceHash = Boolean(opts.replaceHash);
      setTopTab(resolved);
      if (syncHash) setTopTabHash(resolved, replaceHash);
      if (resolved === 'compare') {
        renderComparePlot();
      } else if (resolved === 'scatter') {
        renderScatterPlot();
      } else if (resolved === 'hist') {
        renderHistogramPlot();
      } else if (resolved === 'waterfall') {
        await renderWaterfallPlot(state.waterfallId);
      }
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
        { value: 'all_non_env', label: `All non-environmental ${SPECIMEN_LABEL} tests` },
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

    function getAllMetricsEnriched() {
      const byId = state.metadataById;
      return state.metrics
        .map(m => {
          const md = byId.get(m.biomarker_id) || {};
          return {
            ...m,
            display_name: md.display_name || m.biomarker_name || m.biomarker_id,
            category: md.category || 'Other Clinical',
            is_environmental: Boolean(md.is_environmental),
            is_core_clinical: Boolean(md.is_core_clinical),
            trends: m.trends || {},
            sex_metrics_by_mode: m.sex_metrics || {},
            trends_by_stat: m.trends_by_stat || {
              cv: m.trends || {},
              mean: m.mean_trends || {},
              skewness: m.skewness_trends || {},
            },
            sex_metrics_by_stat: m.sex_metrics_by_stat || {
              cv: m.sex_metrics || {},
              mean: m.sex_mean_metrics || {},
              skewness: m.sex_skewness_metrics || {},
            },
          };
        });
    }

    function getCompareMetrics() {
      return getAllMetricsEnriched().filter(m => metadataPasses(m, compareCategoryEl.value, compareIncludeEnvEl.checked));
    }

    function setAllTrimSliders(pctRaw) {
      const pct = normalizeTrimPct(pctRaw);
      trimSliderEl.value = String(pct);
      compareTrimSliderEl.value = String(pct);
      scatterTrimSliderEl.value = String(pct);
      histTrimSliderEl.value = String(pct);
      waterfallTrimSliderEl.value = String(pct);
      const txt = trimLabelFromPct(pct);
      trimLabelEl.textContent = txt;
      compareTrimLabelEl.textContent = txt;
      scatterTrimLabelEl.textContent = txt;
      histTrimLabelEl.textContent = txt;
      waterfallTrimLabelEl.textContent = txt;
      return pct;
    }

    function setScatterLabelsEnabled(on) {
      const enabled = Boolean(on);
      state.scatterLabels = enabled;
      scatterLabelToggleBtn.classList.toggle('toggle-btn-active', enabled);
      scatterLabelToggleBtn.textContent = enabled ? 'Hide labels' : 'Show labels';
    }

    function renderScatterCategoryOptions(selectCore=false) {
      const cats = sortedCategories(state.metadata, scatterIncludeEnvEl.checked);
      const prev = new Set(Array.from(scatterCategoryEl.selectedOptions).map(o => o.value));
      const coreCats = new Set(
        state.metadata
          .filter(m => Boolean(m.is_core_clinical) && (scatterIncludeEnvEl.checked || !Boolean(m.is_environmental)))
          .map(m => m.category || 'Other Clinical')
      );
      scatterCategoryEl.innerHTML = '';
      for (const c of cats) {
        const opt = document.createElement('option');
        opt.value = c;
        opt.textContent = c;
        const keep = selectCore ? coreCats.has(c) : (prev.size ? prev.has(c) : true);
        opt.selected = keep;
        scatterCategoryEl.appendChild(opt);
      }
      if (Array.from(scatterCategoryEl.options).every(o => !o.selected)) {
        for (const o of Array.from(scatterCategoryEl.options)) o.selected = true;
      }
      const selected = Array.from(scatterCategoryEl.selectedOptions).length;
      scatterSelectionCountEl.textContent = `${selected}/${cats.length} categories selected`;
    }

    function getScatterSelectedCategories() {
      return new Set(Array.from(scatterCategoryEl.selectedOptions).map(o => o.value));
    }

    function renderHistogramCategoryOptions(selectCore=false) {
      const cats = sortedCategories(state.metadata, histIncludeEnvEl.checked);
      const prev = new Set(Array.from(histCategoryEl.selectedOptions).map(o => o.value));
      const coreCats = new Set(
        state.metadata
          .filter(m => Boolean(m.is_core_clinical) && (histIncludeEnvEl.checked || !Boolean(m.is_environmental)))
          .map(m => m.category || 'Other Clinical')
      );
      histCategoryEl.innerHTML = '';
      for (const c of cats) {
        const opt = document.createElement('option');
        opt.value = c;
        opt.textContent = c;
        const keep = selectCore ? coreCats.has(c) : (prev.size ? prev.has(c) : true);
        opt.selected = keep;
        histCategoryEl.appendChild(opt);
      }
      if (Array.from(histCategoryEl.options).every(o => !o.selected)) {
        for (const o of Array.from(histCategoryEl.options)) o.selected = true;
      }
      const selected = Array.from(histCategoryEl.selectedOptions).length;
      histSelectionCountEl.textContent = `${selected}/${cats.length} categories selected`;
    }

    function getHistogramSelectedCategories() {
      return new Set(Array.from(histCategoryEl.selectedOptions).map(o => o.value));
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

    function renderWaterfallOptions() {
      const opts = state.metadata.slice().sort(
        (a, b) => String(a.display_name || a.biomarker_name || '').localeCompare(String(b.display_name || b.biomarker_name || ''))
      );
      const prev = state.waterfallId || waterfallBiomarkerEl.value || state.currentId;
      waterfallBiomarkerEl.innerHTML = '';
      waterfallOptionsEl.innerHTML = '';
      for (const o of opts) {
        const label = `${o.display_name || o.biomarker_name}`;
        const opt = document.createElement('option');
        opt.value = o.biomarker_id;
        opt.textContent = label;
        waterfallBiomarkerEl.appendChild(opt);

        const dopt = document.createElement('option');
        dopt.value = label;
        waterfallOptionsEl.appendChild(dopt);
      }
      if (!opts.length) {
        state.waterfallId = null;
        return null;
      }
      const next = opts.some(o => o.biomarker_id === prev) ? prev : opts[0].biomarker_id;
      waterfallBiomarkerEl.value = next;
      state.waterfallId = next;
      return next;
    }

    function quantile(values, q) {
      if (!values || values.length === 0) return NaN;
      const sorted = values.slice().sort((a, b) => a - b);
      const pos = (sorted.length - 1) * q;
      const base = Math.floor(pos);
      const rest = pos - base;
      if ((base + 1) < sorted.length) return sorted[base] + rest * (sorted[base + 1] - sorted[base]);
      return sorted[base];
    }

    function assignAgeBin(age) {
      const a = Number(age);
      if (!Number.isFinite(a)) return null;
      for (let i = 0; i < WATERFALL_AGE_BINS.length; i += 1) {
        const b = WATERFALL_AGE_BINS[i];
        if (a >= b.lo && a < b.hi) return b.label;
      }
      return null;
    }

    function gaussianKde(values, xGrid) {
      const n = values.length;
      if (n < 2) return xGrid.map(() => 0);
      const mean = values.reduce((acc, v) => acc + v, 0) / n;
      const variance = values.reduce((acc, v) => acc + ((v - mean) ** 2), 0) / Math.max(1, n - 1);
      const sd = Math.sqrt(Math.max(variance, 1e-12));
      const q25 = quantile(values, 0.25);
      const q75 = quantile(values, 0.75);
      const iqr = Number.isFinite(q25) && Number.isFinite(q75) ? (q75 - q25) : sd;
      let sigma = Math.min(sd, iqr / 1.34);
      if (!Number.isFinite(sigma) || sigma <= 0) sigma = sd;
      if (!Number.isFinite(sigma) || sigma <= 0) sigma = Math.max(1e-3, Math.abs(mean) * 0.05);
      let h = 0.9 * sigma * (n ** -0.2);
      if (!Number.isFinite(h) || h <= 0) h = Math.max(1e-3, sigma * 0.3);
      const norm = 1 / (Math.sqrt(2 * Math.PI) * h * n);
      return xGrid.map((x) => {
        let sum = 0;
        for (const v of values) {
          const z = (x - v) / h;
          sum += Math.exp(-0.5 * z * z);
        }
        return norm * sum;
      });
    }

    function rangeMask(xs, lo, hi) {
      const idx = [];
      for (let i = 0; i < xs.length; i += 1) {
        const x = xs[i];
        if (x >= lo && x <= hi) idx.push(i);
      }
      return idx;
    }

    function renderMetricRows(title, m, rawTotal, rawCap, statKey) {
      if (!m) return `<div class="metric"><b>${title}:</b> no metrics</div>`;
      const isNegative = Boolean(m.negative_flag ?? m.decline_flag);
      const flagCls = isNegative ? 'flag-true' : 'flag-false';
      const stat = statLabel(statKey);
      return `
        <div class="metric"><b>${title} bins:</b> ${m.n_bins ?? 'NA'}</div>
        <div class="metric"><b>${title} Spearman rho (${stat} vs age):</b> ${formatNum(m.spearman_rho, 4)}</div>
        <div class="metric"><b>${title} Spearman p:</b> ${formatNum(m.spearman_p, 5)}</div>
        <div class="metric"><b>${title} Slope ${stat}/year:</b> ${formatNum(m.linear_slope_per_year ?? m.linear_slope_cv_per_year, 6)}</div>
        <div class="metric"><b>${title} Slope log(${stat})/year:</b> ${formatNum(m.linear_slope_log_per_year ?? m.linear_slope_logcv_per_year, 6)}</div>
        <div class="metric"><b>${title} Raw points:</b> up to ${rawCap ?? 'NA'} sampled of ${rawTotal ?? 'NA'} total</div>
        <div class="metric"><b>${title} Negative-trend flag:</b> <span class="${flagCls}">${isNegative}</span></div>
      `;
    }

    function renderMetrics(id, series=null) {
      const pooled = state.metricsById.get(id) || {};
      const md = state.metadataById.get(id) || {};
      const box = document.getElementById('metrics');
      if (!pooled || !Object.keys(pooled).length) {
        box.innerHTML = '<div class="metric">No metrics available.</div>';
        return;
      }

      const cohort = cohortFilterEl.value || 'pooled';
      const trimMode = trimPctToMode(trimSliderEl.value);
      const statKey = modeToStat(state.mode);
      const trendsByStat = pooled.trends_by_stat || {
        cv: pooled.trends || {},
        mean: pooled.mean_trends || {},
        skewness: pooled.skewness_trends || {},
      };
      const sexByStat = pooled.sex_metrics_by_stat || {
        cv: pooled.sex_metrics || {},
        mean: pooled.sex_mean_metrics || {},
        skewness: pooled.sex_skewness_metrics || {},
      };
      const trendByMode = trendsByStat[statKey] || {};
      const pooledMetric = trendByMode[trimMode] || trendByMode.all || null;
      const sexMetricsByMode = sexByStat[statKey] || {};
      const sexMetrics = sexMetricsByMode[trimMode] || sexMetricsByMode.all || {};
      const rawBySex = (series && series.raw_total_n_by_sex) ? series.raw_total_n_by_sex : {};
      const rawCap = md.raw_sample_cap ?? 'NA';
      const trimLabel = trimLabelFromPct(trimSliderEl.value);
      const stat = statLabel(statKey);

      let html = `<div class="metric"><b>Category:</b> ${md.category || 'Other Clinical'}</div>`;
      html += `<div class="metric"><b>Outlier mode:</b> ${trimLabel}</div>`;
      html += `<div class="metric"><b>Ranking/stat view:</b> ${stat} vs age</div>`;
      if (cohort === 'both') {
        html += renderMetricRows('Female', sexMetrics.female || null, rawBySex.female ?? 'NA', rawCap, statKey);
        html += renderMetricRows('Male', sexMetrics.male || null, rawBySex.male ?? 'NA', rawCap, statKey);
      } else if (cohort === 'female' || cohort === 'male') {
        const m = sexMetrics[cohort] || null;
        html += renderMetricRows(cohort === 'female' ? 'Female' : 'Male', m, rawBySex[cohort] ?? 'NA', rawCap, statKey);
      } else {
        html += renderMetricRows('Pooled', pooledMetric, md.raw_total_n ?? 'NA', rawCap, statKey);
      }
      box.innerHTML = html;
    }

    function setMode(mode) {
      state.mode = mode;
      modeCvBtn.classList.toggle('active', mode === 'cv');
      modeMeanBtn.classList.toggle('active', mode === 'mean');
      modeSkewBtn.classList.toggle('active', mode === 'skewness');
    }

    function pickPointsByCohort(s, cohort, trimMode) {
      const mode = trimMode || 'all';
      const byMode = s.points_by_filter || {};
      const sexByMode = s.sex_points_by_filter || {};
      if (cohort === 'female') return (sexByMode[mode] && sexByMode[mode].female) ? sexByMode[mode].female : [];
      if (cohort === 'male') return (sexByMode[mode] && sexByMode[mode].male) ? sexByMode[mode].male : [];
      if (byMode[mode]) return byMode[mode];
      return s.points || [];
    }

    function pickRawByCohort(s, cohort) {
      if (cohort === 'female') return (s.raw_sample_by_sex && s.raw_sample_by_sex.female) ? s.raw_sample_by_sex.female : [];
      if (cohort === 'male') return (s.raw_sample_by_sex && s.raw_sample_by_sex.male) ? s.raw_sample_by_sex.male : [];
      return s.raw_sample || [];
    }

    function lineTrace(points, color, label, valueField, hoverTextFn=null) {
      return {
        x: points.map(p => p.age_mid),
        y: points.map(p => p[valueField]),
        text: points.map(p => hoverTextFn ? hoverTextFn(p) : `age_bin=${p.age_bin}<br>n=${p.n}<br>mean=${formatNum(p.mean, 4)}<br>std=${formatNum(p.std, 4)}<br>cv=${formatNum(p.cv, 4)}<br>skewness=${formatNum(p.skewness, 4)}`),
        mode: 'lines+markers',
        type: 'scatter',
        marker: { size: points.map(p => p.passes_n_threshold ? 8 : 5), color },
        line: { color, width: 2 },
        hovertemplate: '%{text}<extra></extra>',
        name: label
      };
    }

    function ciBandTrace(points, color, label) {
      const ciPoints = points.filter(p => p.q25 !== null && p.q75 !== null);
      if (ciPoints.length < 2) return null;
      return {
        x: ciPoints.map(p => p.age_mid).concat(ciPoints.map(p => p.age_mid).reverse()),
        y: ciPoints.map(p => p.q75).concat(ciPoints.map(p => p.q25).reverse()),
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
      const hideClalit = hideClalitEl.checked;
      const cohort = cohortFilterEl.value || 'pooled';
      const trimMode = trimPctToMode(trimSliderEl.value);

      const traces = [];
      const title = `${s.display_name || s.biomarker_name}`;
      const selectedCohorts = cohort === 'both' ? ['female', 'male'] : [cohort];
      const cohortLabel = { pooled: 'Pooled', female: 'Female', male: 'Male' };
      const band95 = {
        pooled: 'rgba(15,118,110,0.16)',
        female: 'rgba(209,73,91,0.18)',
        male: 'rgba(37,99,235,0.18)',
      };
      for (const c of selectedCohorts) {
        const pointsRaw = pickPointsByCohort(s, c, trimMode);
        let points = showLow ? pointsRaw : pointsRaw.filter(p => p.passes_n_threshold);
        if (!points || points.length === 0) continue;

        if (state.mode === 'cv') {
          points = points.filter(p => p.cv !== null && p.cv !== undefined && Number.isFinite(Number(p.cv)));
          if (points.length === 0) continue;
          traces.push(lineTrace(points, COHORT_COLORS[c], `${cohortLabel[c]} CV`, 'cv'));
          continue;
        }

        if (state.mode === 'skewness') {
          points = points.filter(p => p.skewness !== null && p.skewness !== undefined && Number.isFinite(Number(p.skewness)));
          if (points.length === 0) continue;
          traces.push(lineTrace(
            points,
            COHORT_COLORS[c],
            `${cohortLabel[c]} Skewness`,
            'skewness',
            (p) => `age_bin=${p.age_bin}<br>n=${p.n}<br>skewness=${formatNum(p.skewness, 4)}<br>median=${formatNum(p.median, 4)}<br>mean=${formatNum(p.mean, 4)}<br>cv=${formatNum(p.cv, 4)}`
          ));
          continue;
        }

        const ci = ciBandTrace(points, band95[c], `${cohortLabel[c]} IQR (25th-75th)`);
        if (ci) traces.push(ci);
        traces.push({
          ...lineTrace(
            points,
            COHORT_COLORS[c],
            `${cohortLabel[c]} Median (binned)`,
            'median',
            (p) => `age_bin=${p.age_bin}<br>n=${p.n}<br>median=${formatNum(p.median, 4)}<br>q25=${formatNum(p.q25, 4)}<br>q75=${formatNum(p.q75, 4)}<br>mean=${formatNum(p.mean, 4)}<br>std=${formatNum(p.std, 4)}<br>cv=${formatNum(p.cv, 4)}<br>skewness=${formatNum(p.skewness, 4)}`
          ),
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

      if (HAS_CLALIT && s.clalit_data && !hideClalit) {
        for (const c of selectedCohorts) {
          if (!s.clalit_data[c]) continue;
          let points = showLow ? s.clalit_data[c] : s.clalit_data[c].filter(p => p.passes_n_threshold);
          if (!points || points.length === 0) continue;

          if (state.mode === 'cv') {
            points = points.filter(p => Number.isFinite(p.cv));
            if (points.length === 0) continue;
            traces.push({
              x: points.map(p => p.age_mid),
              y: points.map(p => p.cv),
              text: points.map(p => `Clalit ${cohortLabel[c]}<br>age_bin=${p.age_bin}<br>n=${p.n}<br>cv=${formatNum(p.cv, 4)}<br>mean=${formatNum(p.mean, 4)}`),
              mode: 'lines+markers',
              type: 'scatter',
              marker: { size: 6, color: COHORT_COLORS[c], symbol: 'diamond' },
              line: { color: COHORT_COLORS[c], width: 2, dash: 'dot' },
              hovertemplate: '%{text}<extra></extra>',
              name: `Clalit ${cohortLabel[c]} CV`
            });
            continue;
          }

          if (state.mode === 'skewness') {
            // Clalit data provides log_skewness, which causes sign inversion issues when compared to raw skewness. We skip it.
            continue;
          }

          const clalitBandColors = {
            pooled: 'rgba(15,118,110,0.16)',
            female: 'rgba(209,73,91,0.18)',
            male: 'rgba(37,99,235,0.18)',
          };

          const clalitCi = ciBandTrace(points, clalitBandColors[c], `Clalit ${cohortLabel[c]} IQR (25th-75th)`);
          if (clalitCi) traces.push(clalitCi);

          traces.push({
            x: points.map(p => p.age_mid),
            y: points.map(p => p.median),
            text: points.map(p => `Clalit ${cohortLabel[c]}<br>age_bin=${p.age_bin}<br>n=${p.n}<br>median=${formatNum(p.median, 4)}<br>q25=${formatNum(p.q25, 4)}<br>q75=${formatNum(p.q75, 4)}<br>mean=${formatNum(p.mean, 4)}`),
            mode: 'lines+markers',
            type: 'scatter',
            marker: { size: 6, color: COHORT_COLORS[c], symbol: 'diamond' },
            line: { color: COHORT_COLORS[c], width: 2, dash: 'dot' },
            hovertemplate: '%{text}<extra></extra>',
            name: `Clalit ${cohortLabel[c]} Median`
          });
        }
      }

      renderMetrics(id, s);
      const mobile = window.matchMedia('(max-width: 760px)').matches;
      Plotly.newPlot('plot', traces, {
        title,
        xaxis: { title: 'Age (years)', tickfont: { size: mobile ? 10 : 12 } },
        yaxis: {
          title: state.mode === 'cv'
            ? 'Coefficient of Variation (CV)'
            : state.mode === 'skewness'
              ? 'Skewness (binned)'
              : 'Median Biomarker Value',
          tickfont: { size: mobile ? 10 : 12 }
        },
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

    function metricsForView(rows, cohort, trimMode, statKey='cv') {
      const out = [];
      for (const rec of rows) {
        const trendsByStat = rec.trends_by_stat || { cv: rec.trends || {}, mean: {}, skewness: {} };
        const sexByStat = rec.sex_metrics_by_stat || { cv: rec.sex_metrics_by_mode || {}, mean: {}, skewness: {} };
        const trends = trendsByStat[statKey] || {};
        const sexByMode = sexByStat[statKey] || {};
        const tr = trends[trimMode] || trends.all || null;
        const sexTr = sexByMode[trimMode] || sexByMode.all || {};

        if (cohort === 'both') {
          const fm = sexTr.female || null;
          const ml = sexTr.male || null;
          if (!fm || !ml) continue;
          const rhoF = Number(fm.spearman_rho);
          const rhoM = Number(ml.spearman_rho);
          if (!Number.isFinite(rhoF) || !Number.isFinite(rhoM)) continue;
          out.push({
            ...rec,
            rho: (rhoF + rhoM) / 2,
            p: null,
            n_bins: Math.min(Number(fm.n_bins || 0), Number(ml.n_bins || 0)),
            decline_flag: Boolean((fm.negative_flag ?? fm.decline_flag) && (ml.negative_flag ?? ml.decline_flag)),
            female_metric: fm,
            male_metric: ml,
            rho_female: rhoF,
            rho_male: rhoM,
          });
          continue;
        }

        const m = cohort === 'female' || cohort === 'male' ? (sexTr[cohort] || null) : tr;
        if (!m) continue;
        const rho = Number(m.spearman_rho);
        if (!Number.isFinite(rho)) continue;
          out.push({
            ...rec,
            rho,
            p: m.spearman_p,
            n_bins: m.n_bins,
            decline_flag: Boolean(m.negative_flag ?? m.decline_flag),
            metric: m,
          });
      }
      return out;
    }

    function metricForRecord(rec, cohort, trimMode, statKey='cv') {
      const trendsByStat = rec.trends_by_stat || { cv: rec.trends || {}, mean: {}, skewness: {} };
      const sexByStat = rec.sex_metrics_by_stat || { cv: rec.sex_metrics_by_mode || {}, mean: {}, skewness: {} };
      const trends = trendsByStat[statKey] || {};
      const sexByMode = sexByStat[statKey] || {};
      const tr = trends[trimMode] || trends.all || null;
      const sexTr = sexByMode[trimMode] || sexByMode.all || {};
      if (cohort === 'both') {
        return { female: sexTr.female || null, male: sexTr.male || null };
      }
      if (cohort === 'female' || cohort === 'male') return sexTr[cohort] || null;
      return tr;
    }

    function renderScatterPlot() {
      const xStat = scatterXStatEl.value || 'cv';
      const yStat = scatterYStatEl.value || 'mean';
      const xLabel = statLabel(xStat);
      const yLabel = statLabel(yStat);
      const cohort = scatterCohortEl.value || 'pooled';
      const trimMode = trimPctToMode(scatterTrimSliderEl.value);
      const trimLabel = trimLabelFromPct(scatterTrimSliderEl.value);
      const includeEnv = scatterIncludeEnvEl.checked;
      const showLabels = Boolean(state.scatterLabels);
      const selectedCats = getScatterSelectedCategories();
      const rows = getAllMetricsEnriched().filter(r => {
        if (!includeEnv && r.is_environmental) return false;
        return selectedCats.has(r.category || 'Other Clinical');
      });

      const points = [];
      if (cohort === 'both') {
        for (const r of rows) {
          const mx = metricForRecord(r, 'both', trimMode, xStat);
          const my = metricForRecord(r, 'both', trimMode, yStat);
          for (const sx of ['female', 'male']) {
            const vx = mx ? mx[sx] : null;
            const vy = my ? my[sx] : null;
            const xv = Number(vx?.spearman_rho);
            const yv = Number(vy?.spearman_rho);
            if (!Number.isFinite(xv) || !Number.isFinite(yv)) continue;
            points.push({
              biomarker_id: r.biomarker_id,
              display_name: r.display_name,
              category: r.category || 'Other Clinical',
              sex: sx,
              x: xv,
              y: yv,
              n_bins_x: Number(vx?.n_bins || 0),
              n_bins_y: Number(vy?.n_bins || 0),
              p_x: vx?.spearman_p,
              p_y: vy?.spearman_p,
            });
          }
        }
      } else {
        for (const r of rows) {
          const vx = metricForRecord(r, cohort, trimMode, xStat);
          const vy = metricForRecord(r, cohort, trimMode, yStat);
          const xv = Number(vx?.spearman_rho);
          const yv = Number(vy?.spearman_rho);
          if (!Number.isFinite(xv) || !Number.isFinite(yv)) continue;
          points.push({
            biomarker_id: r.biomarker_id,
            display_name: r.display_name,
            category: r.category || 'Other Clinical',
            sex: cohort,
            x: xv,
            y: yv,
            n_bins_x: Number(vx?.n_bins || 0),
            n_bins_y: Number(vy?.n_bins || 0),
            p_x: vx?.spearman_p,
            p_y: vy?.spearman_p,
          });
        }
      }

      const mobile = window.matchMedia('(max-width: 760px)').matches;
      const scatterDiv = document.getElementById('scatter-plot');
      if (!points.length) {
        Plotly.newPlot('scatter-plot', [], {
          title: 'No biomarkers match current scatter filters',
          xaxis: { title: `Spearman rho (Age vs ${xLabel})` },
          yaxis: { title: `Spearman rho (Age vs ${yLabel})` },
          paper_bgcolor: '#ffffff',
          plot_bgcolor: '#ffffff',
        }, { responsive: true, displaylogo: false });
        return;
      }

      let traces = [];
      const baseType = showLabels ? 'scatter' : 'scattergl';
      const baseMode = showLabels ? 'markers+text' : 'markers';
      if (cohort === 'both') {
        const femalePts = points.filter(p => p.sex === 'female');
        const malePts = points.filter(p => p.sex === 'male');
        traces = [
          {
            type: baseType,
            mode: baseMode,
            name: 'Female',
            marker: { color: COHORT_COLORS.female, size: 8, opacity: 0.74, symbol: 'circle' },
            x: femalePts.map(p => p.x),
            y: femalePts.map(p => p.y),
            text: showLabels ? femalePts.map(p => p.display_name) : undefined,
            textposition: showLabels ? 'top center' : undefined,
            textfont: showLabels ? { size: mobile ? 8 : 9, color: '#9f1239' } : undefined,
            customdata: femalePts.map(p => [p.biomarker_id, p.display_name, p.category, p.n_bins_x, p.n_bins_y, p.p_x, p.p_y]),
            hovertemplate: '%{customdata[1]}<br>sex=female<br>category=%{customdata[2]}<br>x rho=%{x:.4f} (n_bins=%{customdata[3]}, p=%{customdata[5]:.5f})<br>y rho=%{y:.4f} (n_bins=%{customdata[4]}, p=%{customdata[6]:.5f})<br>id=%{customdata[0]}<extra></extra>',
          },
          {
            type: baseType,
            mode: baseMode,
            name: 'Male',
            marker: { color: COHORT_COLORS.male, size: 8, opacity: 0.74, symbol: 'square' },
            x: malePts.map(p => p.x),
            y: malePts.map(p => p.y),
            text: showLabels ? malePts.map(p => p.display_name) : undefined,
            textposition: showLabels ? 'top center' : undefined,
            textfont: showLabels ? { size: mobile ? 8 : 9, color: '#1d4ed8' } : undefined,
            customdata: malePts.map(p => [p.biomarker_id, p.display_name, p.category, p.n_bins_x, p.n_bins_y, p.p_x, p.p_y]),
            hovertemplate: '%{customdata[1]}<br>sex=male<br>category=%{customdata[2]}<br>x rho=%{x:.4f} (n_bins=%{customdata[3]}, p=%{customdata[5]:.5f})<br>y rho=%{y:.4f} (n_bins=%{customdata[4]}, p=%{customdata[6]:.5f})<br>id=%{customdata[0]}<extra></extra>',
          }
        ];
      } else {
        traces = [{
          type: baseType,
          mode: baseMode,
          name: cohort === 'pooled' ? 'Pooled' : (cohort === 'female' ? 'Female' : 'Male'),
          marker: { color: COHORT_COLORS[cohort] || '#0f766e', size: 8, opacity: 0.72 },
          x: points.map(p => p.x),
          y: points.map(p => p.y),
          text: showLabels ? points.map(p => p.display_name) : undefined,
          textposition: showLabels ? 'top center' : undefined,
          textfont: showLabels ? { size: mobile ? 8 : 9, color: '#134e4a' } : undefined,
          customdata: points.map(p => [p.biomarker_id, p.display_name, p.category, p.n_bins_x, p.n_bins_y, p.p_x, p.p_y]),
          hovertemplate: '%{customdata[1]}<br>category=%{customdata[2]}<br>x rho=%{x:.4f} (n_bins=%{customdata[3]}, p=%{customdata[5]:.5f})<br>y rho=%{y:.4f} (n_bins=%{customdata[4]}, p=%{customdata[6]:.5f})<br>id=%{customdata[0]}<extra></extra>',
        }];
      }

      Plotly.newPlot('scatter-plot', traces, {
        title: `Biomarker Scatter: ${xLabel} vs ${yLabel} Spearman`,
        annotations: [{
          xref: 'paper',
          yref: 'paper',
          x: 1,
          y: 1.12,
          showarrow: false,
          text: `Cohort: ${cohort}, outliers: ${trimLabel}, categories: ${selectedCats.size}, n_points: ${points.length}, labels: ${showLabels ? 'on' : 'off'}`,
          font: { size: mobile ? 10 : 12, color: '#5f6b7a' },
        }],
        xaxis: {
          title: `Spearman rho (Age vs ${xLabel})`,
          zeroline: true,
          zerolinecolor: '#c2c8d0',
          range: [-1, 1],
          tickfont: { size: mobile ? 10 : 12 },
        },
        yaxis: {
          title: `Spearman rho (Age vs ${yLabel})`,
          zeroline: true,
          zerolinecolor: '#c2c8d0',
          range: [-1, 1],
          tickfont: { size: mobile ? 10 : 12 },
        },
        margin: mobile ? { t: 62, l: 52, r: 10, b: 52 } : { t: 64, l: 70, r: 14, b: 60 },
        paper_bgcolor: '#ffffff',
        plot_bgcolor: '#ffffff',
      }, { responsive: true, displaylogo: false });

      if (scatterDiv && scatterDiv.removeAllListeners) {
        scatterDiv.removeAllListeners('plotly_click');
      }
      if (scatterDiv && scatterDiv.on) {
        scatterDiv.on('plotly_click', async (evt) => {
          const pt = evt?.points?.[0];
          const id = pt?.customdata?.[0];
          if (!id) return;
          selectEl.value = id;
          state.currentId = id;
          await activateTopTab('dashboard', { syncHash: true });
          renderMetrics(id);
          await renderPlot(id);
        });
      }
    }

    function histogramSummary(values) {
      let negative = 0;
      let positive = 0;
      let zeroish = 0;
      for (const v of values) {
        if (!Number.isFinite(v)) continue;
        if (v < 0) negative += 1;
        else if (v > 0) positive += 1;
        else zeroish += 1;
      }
      return { negative, positive, zeroish, total: negative + positive + zeroish };
    }

    function renderHistogramPlot() {
      const statKey = histStatEl.value || 'cv';
      const stat = statLabel(statKey);
      const cohort = histCohortEl.value || 'pooled';
      const trimMode = trimPctToMode(histTrimSliderEl.value);
      const trimLabel = trimLabelFromPct(histTrimSliderEl.value);
      const includeEnv = histIncludeEnvEl.checked;
      const selectedCats = getHistogramSelectedCategories();
      const rows = getAllMetricsEnriched().filter(r => {
        if (!includeEnv && r.is_environmental) return false;
        return selectedCats.has(r.category || 'Other Clinical');
      });

      const mobile = window.matchMedia('(max-width: 760px)').matches;
      const xbins = { start: -1, end: 1, size: 0.05 };
      let traces = [];
      let annoText = '';

      if (cohort === 'both') {
        const femaleVals = [];
        const maleVals = [];
        for (const r of rows) {
          const fx = metricForRecord(r, 'female', trimMode, statKey);
          const mx = metricForRecord(r, 'male', trimMode, statKey);
          const fr = Number(fx?.spearman_rho);
          const mr = Number(mx?.spearman_rho);
          if (Number.isFinite(fr)) femaleVals.push(fr);
          if (Number.isFinite(mr)) maleVals.push(mr);
        }
        const sf = histogramSummary(femaleVals);
        const sm = histogramSummary(maleVals);
        annoText = `Cohort: both, outliers: ${trimLabel}, categories: ${selectedCats.size}, female n=${sf.total} (neg=${sf.negative}, pos=${sf.positive}), male n=${sm.total} (neg=${sm.negative}, pos=${sm.positive})`;
        traces = [
          {
            type: 'histogram',
            name: 'Female',
            x: femaleVals,
            xbins,
            marker: { color: COHORT_COLORS.female },
            opacity: 0.62,
          },
          {
            type: 'histogram',
            name: 'Male',
            x: maleVals,
            xbins,
            marker: { color: COHORT_COLORS.male },
            opacity: 0.62,
          },
        ];
        if (HAS_CLALIT) {
          const clalitF = [];
          const clalitM = [];
          for (const r of rows) {
            const fc = Number(r.clalit_trends?.[statKey]?.female?.spearman_rho);
            const mc = Number(r.clalit_trends?.[statKey]?.male?.spearman_rho);
            if (Number.isFinite(fc)) clalitF.push(fc);
            if (Number.isFinite(mc)) clalitM.push(mc);
          }
          if (clalitF.length || clalitM.length) {
            traces.push({
              type: 'histogram',
              name: 'Clalit Female',
              x: clalitF,
              xbins,
              marker: { color: COHORT_COLORS.female, line: { width: 1, color: '#ffffff' } },
              opacity: 0.8,
              histnorm: '',
            }, {
              type: 'histogram',
              name: 'Clalit Male',
              x: clalitM,
              xbins,
              marker: { color: COHORT_COLORS.male, line: { width: 1, color: '#ffffff' } },
              opacity: 0.8,
            });
          }
        }
      } else {
        const values = [];
        for (const r of rows) {
          const m = metricForRecord(r, cohort, trimMode, statKey);
          const rho = Number(m?.spearman_rho);
          if (Number.isFinite(rho)) values.push(rho);
        }
        if (HAS_CLALIT) {
          const clalitVals = [];
          for (const r of rows) {
            const cr = Number(r.clalit_trends?.[statKey]?.[cohort]?.spearman_rho);
            if (Number.isFinite(cr)) clalitVals.push(cr);
          }
          if (clalitVals.length) {
            traces.push({
              type: 'histogram',
              name: `Clalit ${cohort === 'pooled' ? 'Pooled' : (cohort === 'female' ? 'Female' : 'Male')}`,
              x: clalitVals,
              xbins,
              marker: { color: '#fb923c', line: { width: 1, color: '#ffffff' } },
              opacity: 0.9,
            });
          }
        }
        const s = histogramSummary(values);
        annoText = `Cohort: ${cohort}, outliers: ${trimLabel}, categories: ${selectedCats.size}, n=${s.total} (neg=${s.negative}, pos=${s.positive}, zero=${s.zeroish})`;
        traces = [
          {
            type: 'histogram',
            name: cohort === 'pooled' ? 'Pooled' : (cohort === 'female' ? 'Female' : 'Male'),
            x: values,
            xbins,
            marker: { color: COHORT_COLORS[cohort] || '#0f766e' },
            opacity: 0.78,
          },
        ];
      }

      const noData = traces.every(t => !t.x || t.x.length === 0);
      if (noData) {
        Plotly.newPlot('hist-plot', [], {
          title: 'No biomarkers match current histogram filters',
          xaxis: { title: `Spearman rho (Age vs ${stat})`, range: [-1, 1] },
          yaxis: { title: 'Count of biomarkers' },
          paper_bgcolor: '#ffffff',
          plot_bgcolor: '#ffffff',
        }, { responsive: true, displaylogo: false });
        return;
      }

      Plotly.newPlot('hist-plot', traces, {
        title: `Histogram of Spearman rho: ${stat} vs age`,
        barmode: 'overlay',
        annotations: [{
          xref: 'paper',
          yref: 'paper',
          x: 1,
          y: 1.12,
          showarrow: false,
          text: annoText,
          font: { size: mobile ? 10 : 12, color: '#5f6b7a' },
        }],
        xaxis: {
          title: `Spearman rho (Age vs ${stat})`,
          range: [-1, 1],
          tickfont: { size: mobile ? 10 : 12 },
          zeroline: true,
          zerolinecolor: '#c2c8d0',
        },
        yaxis: {
          title: 'Count of biomarkers',
          tickfont: { size: mobile ? 10 : 12 },
        },
        legend: { orientation: 'h' },
        margin: mobile ? { t: 66, l: 52, r: 10, b: 52 } : { t: 68, l: 70, r: 14, b: 60 },
        paper_bgcolor: '#ffffff',
        plot_bgcolor: '#ffffff',
      }, { responsive: true, displaylogo: false });
    }

    async function applyWaterfallSearch() {
      const q = waterfallSearchEl.value.toLowerCase().trim();
      if (!q) return;
      const hit = state.metadata.find(
        m => `${m.display_name || ''} ${m.biomarker_name || ''} ${m.variable_name || ''}`
          .toLowerCase()
          .includes(q)
      );
      if (!hit) return;
      waterfallBiomarkerEl.value = hit.biomarker_id;
      state.waterfallId = hit.biomarker_id;
      await renderWaterfallPlot(hit.biomarker_id);
    }

    async function renderWaterfallPlot(biomarkerId = null) {
      const id = biomarkerId || waterfallBiomarkerEl.value || state.currentId;
      if (!id) return;
      const s = await loadSeries(id);
      if (!s) return;
      state.waterfallId = id;

      const cohort = waterfallCohortEl.value || 'pooled';
      const trimPct = normalizeTrimPct(waterfallTrimSliderEl.value);
      const trimLo = trimPct / 100;
      const trimHi = 1 - trimLo;
      const minN = Math.max(5, Math.min(1000, Number(waterfallMinNEl.value || 20)));
      waterfallMinNEl.value = String(minN);

      let raw = [];
      if (cohort === 'female' || cohort === 'male') {
        raw = (s.raw_sample_by_sex && s.raw_sample_by_sex[cohort]) ? s.raw_sample_by_sex[cohort] : [];
      } else {
        raw = s.raw_sample || [];
      }

      const bins = {};
      for (const b of WATERFALL_AGE_BINS) bins[b.label] = [];
      for (const p of raw) {
        const age = Number(p.age_years);
        const value = Number(p.value);
        if (!Number.isFinite(age) || !Number.isFinite(value)) continue;
        const label = assignAgeBin(age);
        if (!label) continue;
        bins[label].push(value);
      }

      const rows = [];
      for (const b of WATERFALL_AGE_BINS) {
        const vals = (bins[b.label] || []).slice().sort((a, z) => a - z);
        if (vals.length < minN) continue;
        let trimmed = vals;
        if (trimPct > 0) {
          const loCut = quantile(vals, trimLo);
          const hiCut = quantile(vals, trimHi);
          trimmed = vals.filter(v => v >= loCut && v <= hiCut);
        }
        if (trimmed.length < minN) continue;
        const q1 = quantile(trimmed, 0.25);
        const q2 = quantile(trimmed, 0.50);
        const q3 = quantile(trimmed, 0.75);
        rows.push({ label: b.label, values: trimmed, q1, q2, q3, n: trimmed.length });
      }

      if (!rows.length) {
        Plotly.newPlot('waterfall-plot', [], {
          title: 'No age bins pass minimum n for this biomarker/cohort',
          xaxis: { title: s.display_name || s.biomarker_name || id },
          yaxis: { title: 'Age bin' },
          paper_bgcolor: '#ffffff',
          plot_bgcolor: '#ffffff',
        }, { responsive: true, displaylogo: false });
        return;
      }

      let globalMin = Infinity;
      let globalMax = -Infinity;
      let allNonNegative = true;
      for (const r of rows) {
        const lo = Math.min(...r.values);
        const hi = Math.max(...r.values);
        if (lo < 0) allNonNegative = false;
        if (lo < globalMin) globalMin = lo;
        if (hi > globalMax) globalMax = hi;
      }
      const span = Math.max(1e-6, globalMax - globalMin);
      const pad = span * 0.08;
      const xMin = allNonNegative ? 0 : (globalMin - pad);
      const xMax = globalMax + pad;
      const gridN = 180;
      const xGrid = Array.from({ length: gridN }, (_, i) => xMin + (i * (xMax - xMin) / (gridN - 1)));

      const withDensity = rows.map(r => ({ ...r, density: gaussianKde(r.values, xGrid) }));
      let maxD = 0;
      for (const r of withDensity) {
        for (const d of r.density) if (d > maxD) maxD = d;
      }
      const amp = 0.82;

      const traces = [];
      const quartNames = ['Q1', 'Q2', 'Q3', 'Q4'];
      const mobile = window.matchMedia('(max-width: 760px)').matches;

      const yPos = withDensity.map((_, idx) => withDensity.length - 1 - idx);
      for (let rowIdx = 0; rowIdx < withDensity.length; rowIdx += 1) {
        const r = withDensity[rowIdx];
        const yBase = yPos[rowIdx];
        const yCurve = r.density.map(d => yBase + (maxD > 0 ? (d / maxD) * amp : 0));
        const bounds = [Number.NEGATIVE_INFINITY, r.q1, r.q2, r.q3, Number.POSITIVE_INFINITY];

        for (let q = 0; q < 4; q += 1) {
          const lo = bounds[q];
          const hi = bounds[q + 1];
          const idx = rangeMask(xGrid, lo, hi);
          if (idx.length < 2) continue;
          const xs = idx.map(i => xGrid[i]);
          const ys = idx.map(i => yCurve[i]);
          traces.push({
            type: 'scatter',
            mode: 'lines',
            x: xs.concat(xs.slice().reverse()),
            y: ys.concat(xs.map(() => yBase).reverse()),
            fill: 'toself',
            fillcolor: WATERFALL_QUARTILE_COLORS[q],
            line: { color: 'rgba(0,0,0,0)' },
            name: quartNames[q],
            legendgroup: quartNames[q],
            showlegend: rowIdx === 0,
            opacity: 0.96,
            hovertemplate: `${r.label}<br>${quartNames[q]}<br>n=${r.n}<br>Q1=${formatNum(r.q1, 4)}<br>Median=${formatNum(r.q2, 4)}<br>Q3=${formatNum(r.q3, 4)}<extra></extra>`,
          });
        }

        traces.push({
          type: 'scatter',
          mode: 'lines',
          x: xGrid,
          y: yCurve,
          line: { color: '#111827', width: 1.6 },
          hovertemplate: `${r.label}<br>n=${r.n}<br>Q1=${formatNum(r.q1, 4)}<br>Median=${formatNum(r.q2, 4)}<br>Q3=${formatNum(r.q3, 4)}<extra>density</extra>`,
          name: `${r.label} density`,
          showlegend: false,
        });
      }

      Plotly.newPlot('waterfall-plot', traces, {
        title: `${s.display_name || s.biomarker_name} — age-stratified waterfall (${cohort})`,
        annotations: [{
          xref: 'paper',
          yref: 'paper',
          x: 1,
          y: 1.1,
          showarrow: false,
          text: `Outliers: ${trimLabelFromPct(trimPct)}, min n/bin: ${minN}, bins shown: ${withDensity.length}`,
          font: { size: mobile ? 10 : 12, color: '#5f6b7a' },
        }],
        xaxis: { title: 'Biomarker value', tickfont: { size: mobile ? 10 : 12 } },
        yaxis: {
          title: 'Age bin',
          tickmode: 'array',
          tickvals: yPos,
          ticktext: withDensity.map(r => r.label),
          tickfont: { size: mobile ? 10 : 12 },
        },
        legend: { orientation: 'v', title: { text: 'Quartiles' } },
        margin: mobile ? { t: 72, l: 72, r: 12, b: 52 } : { t: 72, l: 88, r: 16, b: 60 },
        paper_bgcolor: '#ffffff',
        plot_bgcolor: '#ffffff',
      }, { responsive: true, displaylogo: false });
    }

    function renderRankTable() {
      const tbl = document.getElementById('rank-table');
      const visible = new Set(getDashboardMetadata().map(m => m.biomarker_id));
      const cohort = cohortFilterEl.value || 'pooled';
      const trimMode = trimPctToMode(trimSliderEl.value);
      const statKey = modeToStat(state.mode);
      const stat = statLabel(statKey);
      if (rankTitleEl) rankTitleEl.textContent = `Biomarkers Ranked by Most Negative Spearman Rho (${stat} vs age)`;
      const ranked = metricsForView(
        getCompareMetrics().filter(r => visible.has(r.biomarker_id)),
        cohort,
        trimMode,
        statKey
      ).sort((a, b) => (a.rho ?? 999) - (b.rho ?? 999));
      const top = ranked.slice(0, 200);
      let html = `<thead><tr><th>Biomarker</th><th>Spearman rho (${stat})</th><th>p</th><th>Negative trend</th></tr></thead><tbody>`;
      for (const r of top) {
        html += `<tr data-id="${r.biomarker_id}"><td>${r.display_name}</td><td>${formatNum(r.rho, 4)}</td><td>${formatNum(r.p, 5)}</td><td>${r.decline_flag}</td></tr>`;
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
      const statKey = compareStatEl.value || 'cv';
      const stat = statLabel(statKey);
      const cohort = compareCohortEl.value || 'pooled';
      const trimMode = trimPctToMode(compareTrimSliderEl.value);
      const topN = Math.max(10, Math.min(200, Number(compareTopNEl.value || 40)));
      const trimLabel = trimLabelFromPct(compareTrimSliderEl.value);
      compareTopNEl.value = String(topN);

      let ranked = metricsForView(getCompareMetrics(), cohort, trimMode, statKey).slice();
      const rankVal = (m) => (mode === 'absolute' ? Math.abs(m.rho) : m.rho);
      if (mode === 'negative') ranked.sort((a, b) => rankVal(a) - rankVal(b));
      if (mode === 'positive') ranked.sort((a, b) => rankVal(b) - rankVal(a));
      if (mode === 'absolute') ranked.sort((a, b) => rankVal(b) - rankVal(a));
      ranked = ranked.slice(0, topN);

      const y = ranked.map(r => r.display_name).reverse();
      const categoryLabel = compareCategoryEl.options[compareCategoryEl.selectedIndex]?.textContent || 'All';
      let traces = [];
      let xTitle = `Spearman rho (Age vs ${stat})`;

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
        xTitle = `Spearman rho (female vs male, Age vs ${stat})`;
      } else {
        const x = ranked.map(r => Number(r.rho)).reverse();
        const custom = ranked.map(r => {
          return [r.p, r.n_bins, r.decline_flag, r.biomarker_id, r.category];
        }).reverse();
        const colors = x.map(v => (v < 0 ? '#0f766e' : '#b45309'));
        traces = [{
          type: 'bar',
          orientation: 'h',
          y,
          x,
          marker: { color: colors },
          customdata: custom,
          hovertemplate: 'rho=%{x:.4f}<br>p=%{customdata[0]:.5f}<br>n_bins=%{customdata[1]}<br>negative_trend=%{customdata[2]}<br>id=%{customdata[3]}<br>category=%{customdata[4]}<extra></extra>',
          name: cohort === 'female' ? 'Female' : cohort === 'male' ? 'Male' : 'Pooled',
        }];
      }

      const mobile = window.matchMedia('(max-width: 760px)').matches;
      Plotly.newPlot('compare-plot', traces, {
        title: mode === 'negative' ? `Top ${topN} Most Negative Spearman Biomarkers (${stat} vs age)` :
               mode === 'positive' ? `Top ${topN} Most Positive Spearman Biomarkers (${stat} vs age)` :
               `Top ${topN} Largest |Spearman| Biomarkers (${stat} vs age)`,
        annotations: [{
          xref: 'paper',
          yref: 'paper',
          x: 1,
          y: 1.12,
          showarrow: false,
          text: `Filter: ${categoryLabel}${compareIncludeEnvEl.checked ? ' (env included)' : ''}, cohort: ${cohort}, statistic: ${stat}, outliers: ${trimLabel}`,
          font: { size: mobile ? 10 : 12, color: '#5f6b7a' },
        }],
        barmode: cohort === 'both' ? 'group' : 'relative',
        xaxis: { title: xTitle, tickfont: { size: mobile ? 10 : 12 } },
        yaxis: { automargin: true, tickfont: { size: mobile ? 10 : 12 } },
        margin: mobile ? { t: 64, l: 150, r: 10, b: 44 } : { t: 56, l: 260, r: 16, b: 54 },
        paper_bgcolor: '#ffffff',
        plot_bgcolor: '#ffffff',
      }, { responsive: true, displaylogo: false });

      if (!HAS_CLALIT) {
        Plotly.newPlot('compare-clalit-plot', [], {
          title: 'No external cohort overlay configured for this specimen',
          xaxis: { visible: false },
          yaxis: { visible: false },
          margin: mobile ? { t: 44, l: 20, r: 10, b: 20 } : { t: 44, l: 30, r: 16, b: 28 },
          paper_bgcolor: '#ffffff',
          plot_bgcolor: '#ffffff',
        }, { responsive: true, displaylogo: false });
        return;
      }

      let cTraces = [];
      if (cohort === 'both') {
        const bothRanked = ranked.filter(r => r.clalit_trends && r.clalit_trends[statKey] && r.clalit_trends[statKey].female && r.clalit_trends[statKey].male);
        cTraces = [
          {
            type: 'scatter',
            mode: 'markers+text',
            name: 'Female',
            marker: { color: COHORT_COLORS.female, size: 8, symbol: 'circle' },
            x: bothRanked.map(r => Number(r.rho_female)),
            y: bothRanked.map(r => Number(r.clalit_trends[statKey].female.spearman_rho)),
            text: bothRanked.map(r => r.display_name),
            textposition: 'top center',
            textfont: { size: mobile ? 8 : 9, color: '#9f1239' },
            customdata: bothRanked.map(r => [r.biomarker_id, r.category]),
            hovertemplate: '%{text}<br>sex=female<br>category=%{customdata[1]}<br>NHANES rho=%{x:.4f}<br>Clalit rho=%{y:.4f}<extra></extra>',
          },
          {
            type: 'scatter',
            mode: 'markers+text',
            name: 'Male',
            marker: { color: COHORT_COLORS.male, size: 8, symbol: 'square' },
            x: bothRanked.map(r => Number(r.rho_male)),
            y: bothRanked.map(r => Number(r.clalit_trends[statKey].male.spearman_rho)),
            text: bothRanked.map(r => r.display_name),
            textposition: 'top center',
            textfont: { size: mobile ? 8 : 9, color: '#1d4ed8' },
            customdata: bothRanked.map(r => [r.biomarker_id, r.category]),
            hovertemplate: '%{text}<br>sex=male<br>category=%{customdata[1]}<br>NHANES rho=%{x:.4f}<br>Clalit rho=%{y:.4f}<extra></extra>',
          }
        ];
      } else {
        const rankedC = ranked.filter(r => r.clalit_trends && r.clalit_trends[statKey] && r.clalit_trends[statKey][cohort]);
        cTraces = [{
          type: 'scatter',
          mode: 'markers+text',
          name: cohort === 'pooled' ? 'Pooled' : (cohort === 'female' ? 'Female' : 'Male'),
          marker: { color: COHORT_COLORS[cohort] || '#0f766e', size: 8 },
          x: rankedC.map(r => Number(r.rho)),
          y: rankedC.map(r => Number(r.clalit_trends[statKey][cohort].spearman_rho)),
          text: rankedC.map(r => r.display_name),
          textposition: 'top center',
          textfont: { size: mobile ? 8 : 9, color: '#134e4a' },
          customdata: rankedC.map(r => [r.biomarker_id, r.category]),
          hovertemplate: '%{text}<br>category=%{customdata[1]}<br>NHANES rho=%{x:.4f}<br>Clalit rho=%{y:.4f}<extra></extra>',
        }];
      }

      cTraces.push({
        x: [-1, 1],
        y: [-1, 1],
        mode: 'lines',
        name: 'Agreement Diagonal',
        line: { color: 'rgba(0,0,0,0.2)', dash: 'dash' },
        hoverinfo: 'none'
      });

      Plotly.newPlot('compare-clalit-plot', cTraces, {
        title: `Clalit vs NHANES Spearman rho (${stat} vs age)`,
        xaxis: { title: `NHANES Spearman rho (${stat})`, range: [-1, 1], zeroline: true },
        yaxis: { title: `Clalit Spearman rho (${stat})`, range: [-1, 1], zeroline: true },
        margin: mobile ? { t: 40, l: 52, r: 10, b: 44 } : { t: 40, l: 60, r: 16, b: 54 },
        paper_bgcolor: '#ffffff',
        plot_bgcolor: '#ffffff',
        showlegend: false,
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
        waterfallBiomarkerEl.value = hit.biomarker_id;
        state.waterfallId = hit.biomarker_id;
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
      hideClalitEl.checked = !HAS_CLALIT;
      includeEnvEl.checked = false;
      compareIncludeEnvEl.checked = false;
      scatterIncludeEnvEl.checked = false;
      histIncludeEnvEl.checked = false;
      if (!HAS_CLALIT) {
        if (hideClalitWrapEl) hideClalitWrapEl.style.display = 'none';
        if (compareClalitTitleEl) compareClalitTitleEl.textContent = 'External Cohort Agreement';
      }
      setScatterLabelsEnabled(false);
      setAllTrimSliders(10);
      renderCategorySelect(categoryFilterEl, includeEnvEl.checked, 'all_core');
      renderCategorySelect(compareCategoryEl, compareIncludeEnvEl.checked, 'all_core');
      renderScatterCategoryOptions(false);
      renderHistogramCategoryOptions(false);
      renderWaterfallOptions();

      await refreshDashboardFromFilters();
      renderComparePlot();
      renderScatterPlot();
      renderHistogramPlot();
      await renderWaterfallPlot(state.waterfallId);

      statusChip.textContent = `Ready: ${state.metadata.length} ${SPECIMEN_LABEL} biomarkers indexed`;

      tabDashboardBtn.addEventListener('click', () => {
        activateTopTab('dashboard', { syncHash: true }).catch(console.error);
      });
      tabCompareBtn.addEventListener('click', () => {
        activateTopTab('compare', { syncHash: true }).catch(console.error);
      });
      tabScatterBtn.addEventListener('click', () => {
        activateTopTab('scatter', { syncHash: true }).catch(console.error);
      });
      tabHistBtn.addEventListener('click', () => {
        activateTopTab('hist', { syncHash: true }).catch(console.error);
      });
      tabWaterfallBtn.addEventListener('click', () => {
        activateTopTab('waterfall', { syncHash: true }).catch(console.error);
      });
      tabInfoBtn.addEventListener('click', () => {
        activateTopTab('info', { syncHash: true }).catch(console.error);
      });
      for (const link of specimenLinks) {
        link.addEventListener('click', () => {
          const current = topTabFromHash() || 'dashboard';
          syncSpecimenSwitchLinks(current);
        });
      }
      compareSortEl.addEventListener('change', renderComparePlot);
      compareStatEl.addEventListener('change', renderComparePlot);
      compareTopNEl.addEventListener('change', renderComparePlot);
      compareCategoryEl.addEventListener('change', renderComparePlot);
      compareCohortEl.addEventListener('change', async () => {
        cohortFilterEl.value = compareCohortEl.value;
        scatterCohortEl.value = compareCohortEl.value;
        histCohortEl.value = compareCohortEl.value;
        if (compareCohortEl.value !== 'both') waterfallCohortEl.value = compareCohortEl.value;
        renderRankTable();
        renderComparePlot();
        renderScatterPlot();
        renderHistogramPlot();
        if (compareCohortEl.value !== 'both') await renderWaterfallPlot(state.waterfallId);
        if (state.currentId) await renderPlot(state.currentId);
      });
      compareTrimSliderEl.addEventListener('input', () => {
        setAllTrimSliders(compareTrimSliderEl.value);
        renderRankTable();
        if (state.currentId) renderPlot(state.currentId);
        renderComparePlot();
        renderScatterPlot();
        renderHistogramPlot();
        renderWaterfallPlot(state.waterfallId);
      });
      compareIncludeEnvEl.addEventListener('change', () => {
        renderCategorySelect(compareCategoryEl, compareIncludeEnvEl.checked, compareCategoryEl.value);
        renderComparePlot();
      });
      selectEl.addEventListener('change', async () => {
        const id = selectEl.value;
        state.currentId = id;
        if (Array.from(waterfallBiomarkerEl.options).some(o => o.value === id)) {
          waterfallBiomarkerEl.value = id;
          state.waterfallId = id;
        }
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
        compareCohortEl.value = cohortFilterEl.value;
        scatterCohortEl.value = cohortFilterEl.value;
        histCohortEl.value = cohortFilterEl.value;
        if (cohortFilterEl.value !== 'both') waterfallCohortEl.value = cohortFilterEl.value;
        renderRankTable();
        renderComparePlot();
        renderScatterPlot();
        renderHistogramPlot();
        if (cohortFilterEl.value !== 'both') await renderWaterfallPlot(state.waterfallId);
        if (state.currentId) await renderPlot(state.currentId);
      });
      trimSliderEl.addEventListener('input', async () => {
        setAllTrimSliders(trimSliderEl.value);
        renderRankTable();
        renderComparePlot();
        renderScatterPlot();
        renderHistogramPlot();
        renderWaterfallPlot(state.waterfallId);
        if (state.currentId) await renderPlot(state.currentId);
      });
      showLowNEl.addEventListener('change', async () => {
        if (state.currentId) await renderPlot(state.currentId);
      });
      hideClalitEl.addEventListener('change', async () => {
        if (state.currentId) await renderPlot(state.currentId);
      });
      modeCvBtn.addEventListener('click', async () => {
        setMode('cv');
        renderRankTable();
        if (state.currentId) await renderPlot(state.currentId);
      });
      modeMeanBtn.addEventListener('click', async () => {
        setMode('mean');
        renderRankTable();
        if (state.currentId) await renderPlot(state.currentId);
      });
      modeSkewBtn.addEventListener('click', async () => {
        setMode('skewness');
        renderRankTable();
        if (state.currentId) await renderPlot(state.currentId);
      });
      scatterXStatEl.addEventListener('change', renderScatterPlot);
      scatterYStatEl.addEventListener('change', renderScatterPlot);
      scatterCohortEl.addEventListener('change', () => {
        cohortFilterEl.value = scatterCohortEl.value;
        compareCohortEl.value = scatterCohortEl.value;
        histCohortEl.value = scatterCohortEl.value;
        if (scatterCohortEl.value !== 'both') waterfallCohortEl.value = scatterCohortEl.value;
        renderRankTable();
        renderComparePlot();
        renderHistogramPlot();
        if (scatterCohortEl.value !== 'both') renderWaterfallPlot(state.waterfallId);
        if (state.currentId) renderPlot(state.currentId);
        renderScatterPlot();
      });
      scatterTrimSliderEl.addEventListener('input', () => {
        setAllTrimSliders(scatterTrimSliderEl.value);
        renderRankTable();
        renderComparePlot();
        renderHistogramPlot();
        renderWaterfallPlot(state.waterfallId);
        if (state.currentId) renderPlot(state.currentId);
        renderScatterPlot();
      });
      scatterIncludeEnvEl.addEventListener('change', () => {
        renderScatterCategoryOptions(false);
        renderScatterPlot();
      });
      scatterCategoryEl.addEventListener('change', () => {
        const cats = sortedCategories(state.metadata, scatterIncludeEnvEl.checked);
        const selected = Array.from(scatterCategoryEl.selectedOptions).length;
        scatterSelectionCountEl.textContent = `${selected}/${cats.length} categories selected`;
        renderScatterPlot();
      });
      scatterCatAllBtn.addEventListener('click', () => {
        for (const o of Array.from(scatterCategoryEl.options)) o.selected = true;
        renderScatterCategoryOptions(false);
        renderScatterPlot();
      });
      scatterCatCoreBtn.addEventListener('click', () => {
        renderScatterCategoryOptions(true);
        renderScatterPlot();
      });
      scatterCatClearBtn.addEventListener('click', () => {
        for (const o of Array.from(scatterCategoryEl.options)) o.selected = false;
        const cats = sortedCategories(state.metadata, scatterIncludeEnvEl.checked);
        scatterSelectionCountEl.textContent = `0/${cats.length} categories selected`;
        renderScatterPlot();
      });
      scatterLabelToggleBtn.addEventListener('click', () => {
        setScatterLabelsEnabled(!state.scatterLabels);
        renderScatterPlot();
      });
      histStatEl.addEventListener('change', renderHistogramPlot);
      histCohortEl.addEventListener('change', async () => {
        cohortFilterEl.value = histCohortEl.value;
        compareCohortEl.value = histCohortEl.value;
        scatterCohortEl.value = histCohortEl.value;
        if (histCohortEl.value !== 'both') waterfallCohortEl.value = histCohortEl.value;
        renderRankTable();
        renderComparePlot();
        renderScatterPlot();
        renderHistogramPlot();
        if (histCohortEl.value !== 'both') await renderWaterfallPlot(state.waterfallId);
        if (state.currentId) await renderPlot(state.currentId);
      });
      histTrimSliderEl.addEventListener('input', () => {
        setAllTrimSliders(histTrimSliderEl.value);
        renderRankTable();
        renderComparePlot();
        renderScatterPlot();
        renderHistogramPlot();
        renderWaterfallPlot(state.waterfallId);
        if (state.currentId) renderPlot(state.currentId);
      });
      histIncludeEnvEl.addEventListener('change', () => {
        renderHistogramCategoryOptions(false);
        renderHistogramPlot();
      });
      histCategoryEl.addEventListener('change', () => {
        const cats = sortedCategories(state.metadata, histIncludeEnvEl.checked);
        const selected = Array.from(histCategoryEl.selectedOptions).length;
        histSelectionCountEl.textContent = `${selected}/${cats.length} categories selected`;
        renderHistogramPlot();
      });
      histCatAllBtn.addEventListener('click', () => {
        for (const o of Array.from(histCategoryEl.options)) o.selected = true;
        renderHistogramCategoryOptions(false);
        renderHistogramPlot();
      });
      histCatCoreBtn.addEventListener('click', () => {
        renderHistogramCategoryOptions(true);
        renderHistogramPlot();
      });
      histCatClearBtn.addEventListener('click', () => {
        for (const o of Array.from(histCategoryEl.options)) o.selected = false;
        const cats = sortedCategories(state.metadata, histIncludeEnvEl.checked);
        histSelectionCountEl.textContent = `0/${cats.length} categories selected`;
        renderHistogramPlot();
      });
      waterfallBiomarkerEl.addEventListener('change', async () => {
        const id = waterfallBiomarkerEl.value;
        state.waterfallId = id;
        if (Array.from(selectEl.options).some(o => o.value === id)) {
          selectEl.value = id;
          state.currentId = id;
        }
        await renderWaterfallPlot(id);
      });
      waterfallSearchEl.addEventListener('change', applyWaterfallSearch);
      waterfallSearchEl.addEventListener('keyup', (e) => { if (e.key === 'Enter') applyWaterfallSearch(); });
      waterfallCohortEl.addEventListener('change', async () => {
        cohortFilterEl.value = waterfallCohortEl.value;
        compareCohortEl.value = waterfallCohortEl.value;
        scatterCohortEl.value = waterfallCohortEl.value;
        histCohortEl.value = waterfallCohortEl.value;
        renderRankTable();
        renderComparePlot();
        renderScatterPlot();
        renderHistogramPlot();
        if (state.currentId) await renderPlot(state.currentId);
        await renderWaterfallPlot(state.waterfallId);
      });
      waterfallTrimSliderEl.addEventListener('input', () => {
        setAllTrimSliders(waterfallTrimSliderEl.value);
        renderRankTable();
        renderComparePlot();
        renderScatterPlot();
        renderHistogramPlot();
        if (state.currentId) renderPlot(state.currentId);
        renderWaterfallPlot(state.waterfallId);
      });
      waterfallMinNEl.addEventListener('change', () => {
        renderWaterfallPlot(state.waterfallId);
      });
      window.addEventListener('hashchange', () => {
        const hashTab = topTabFromHash();
        const fallbackToDashboard = hashTab === null;
        activateTopTab(hashTab || 'dashboard', {
          syncHash: fallbackToDashboard,
          replaceHash: fallbackToDashboard,
        }).catch(console.error);
      });
      window.addEventListener('resize', () => {
        const plotEl = document.getElementById('plot');
        const compareEl = document.getElementById('compare-plot');
        const scatterEl = document.getElementById('scatter-plot');
        const histEl = document.getElementById('hist-plot');
        const waterfallEl = document.getElementById('waterfall-plot');
        if (plotEl) Plotly.Plots.resize(plotEl);
        if (compareEl) Plotly.Plots.resize(compareEl);
        if (scatterEl) Plotly.Plots.resize(scatterEl);
        if (histEl) Plotly.Plots.resize(histEl);
        if (waterfallEl) Plotly.Plots.resize(waterfallEl);
      });
      const initialHashTab = topTabFromHash();
      await activateTopTab(initialHashTab || 'dashboard', {
        syncHash: true,
        replaceHash: true,
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


def clean_urinary_ui_base(name: str) -> str:
    s = str(name or "").strip()
    orig = s
    # Remove leading chemistry locants for UI readability in urinary list labels.
    s = re.sub(r"^\s*(?:\d+[a-z]?(?:\s*,\s*\d+[a-z]?){0,6})\s*-\s*", "", s)
    s = re.sub(r"^\s*#\d+\s*", "", s)
    s = re.sub(r"\s+", " ", s).strip()
    s = s or orig
    if s and s[0].islower():
        s = s[0].upper() + s[1:]
    return s


def parse_terminal_unit(label: str) -> tuple[str, str]:
    s = str(label or "").strip()
    m = re.search(r"\(([^()]*)\)\s*$", s)
    if not m:
        return s, ""
    unit = m.group(1).strip()
    base = s[: m.start()].strip().rstrip(",")
    return base, unit


def make_display_name(name: str, unit: str, specimen_kind: str = "blood") -> str:
    base = clean_display_base(name)
    if specimen_kind == "urine":
        base = clean_urinary_ui_base(base)
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


AGE_BINS = list(np.arange(20, 90, 5))
AGE_LABELS = [f"{a}-{a+4}" for a in range(20, 85, 5)]
AGE_MIDS = {lab: mid for lab, mid in zip(AGE_LABELS, [a + 2.5 for a in range(20, 85, 5)])}
TRIM_PCTS = [0, 5, 10, 15, 20, 25]
DERIVED_NLR_ID = "neutrophil to lymphocyte ratio"
DERIVED_NLR_NAME = "Neutrophil-to-lymphocyte ratio (NLR)"


def trim_mode_key(pct: int) -> str:
    p = int(pct)
    if p <= 0:
        return "all"
    return f"trim_{p}_{100-p}"


def trim_mode_quantiles(mode: str) -> tuple[float, float] | None:
    if mode == "all":
        return None
    m = re.match(r"trim_(\d{1,2})_(\d{1,2})$", str(mode))
    if not m:
        return None
    lo = int(m.group(1))
    hi = int(m.group(2))
    if lo < 0 or hi > 100 or lo >= hi:
        return None
    return lo / 100.0, hi / 100.0


def slope(x: np.ndarray, y: np.ndarray) -> float:
    if len(x) < 2:
        return np.nan
    return float(np.polyfit(x, y, 1)[0])


def compute_binned_long(
    df: pd.DataFrame,
    group_cols: list[str],
    trim_quantiles: tuple[float, float] | None = None,
) -> pd.DataFrame:
    tmp = df.copy()
    tmp["age_bin"] = pd.cut(tmp["age_years"], bins=AGE_BINS, labels=AGE_LABELS, right=False, include_lowest=True)
    tmp["age_mid"] = tmp["age_bin"].map(AGE_MIDS).astype(float)
    tmp = tmp.dropna(subset=["age_bin", "value"])

    keys = group_cols + ["age_bin", "age_mid"]
    if trim_quantiles is not None:
        q_lo, q_hi = trim_quantiles
        q_tbl = (
            tmp.groupby(keys, observed=True)["value"]
            .quantile([q_lo, q_hi])
            .unstack(level=-1)
            .rename(columns={q_lo: "trim_lo", q_hi: "trim_hi"})
            .reset_index()
        )
        tmp = tmp.merge(q_tbl, on=keys, how="left")
        tmp = tmp[(tmp["value"] >= tmp["trim_lo"]) & (tmp["value"] <= tmp["trim_hi"])].copy()

    grouped = (
        tmp.groupby(keys, observed=True)["value"]
        .agg(
            n="count",
            mean="mean",
            std="std",
            median="median",
            q25=lambda s: float(np.nanpercentile(s.to_numpy(dtype=float), 25)),
            q75=lambda s: float(np.nanpercentile(s.to_numpy(dtype=float), 75)),
            p10=lambda s: float(np.nanpercentile(s.to_numpy(dtype=float), 10)),
            p90=lambda s: float(np.nanpercentile(s.to_numpy(dtype=float), 90)),
            skewness=lambda s: float(scipy_skew(s.to_numpy(dtype=float), bias=False, nan_policy="omit")),
        )
        .reset_index()
    )
    grouped["cv"] = grouped["std"] / grouped["mean"].abs()
    grouped.loc[grouped["mean"].abs() < 1e-8, "cv"] = np.nan
    grouped["quantile_skewness"] = quantile_skewness_from_stats(grouped["q25"], grouped["median"], grouped["q75"])
    grouped["passes_n_threshold"] = grouped["n"] >= 30
    return grouped.reset_index(drop=True)


def quantile_skewness_from_stats(q25, median, q75):
    q25_s = pd.to_numeric(q25, errors="coerce")
    median_s = pd.to_numeric(median, errors="coerce")
    q75_s = pd.to_numeric(q75, errors="coerce")
    denom = q75_s - q25_s
    out = (q75_s + q25_s - 2.0 * median_s) / denom
    return out.where(denom.abs() > 1e-12, np.nan)


def trend_from_points(points: list[dict], value_key: str) -> dict:
    eligible = [
        p
        for p in points
        if bool(p.get("passes_n_threshold"))
        and p.get(value_key) is not None
        and pd.notna(p.get(value_key))
    ]
    x = np.asarray([float(p["age_mid"]) for p in eligible], dtype=float)
    y = np.asarray([float(p[value_key]) for p in eligible], dtype=float)
    rho = np.nan
    pval = np.nan
    if len(y) >= 2:
        rho, pval = spearmanr(x, y)
    pos = y > 0
    out = {
        "n_bins": int(len(eligible)),
        "spearman_rho": float(rho) if pd.notna(rho) else None,
        "spearman_p": float(pval) if pd.notna(pval) else None,
        "linear_slope_per_year": float(slope(x, y)) if len(y) >= 2 else None,
        "linear_slope_log_per_year": float(slope(x[pos], np.log(y[pos]))) if int(pos.sum()) >= 2 else None,
    }
    out["negative_flag"] = bool(
        out["n_bins"] >= 5
        and out["spearman_rho"] is not None
        and out["spearman_p"] is not None
        and out["linear_slope_per_year"] is not None
        and out["spearman_rho"] < 0
        and out["spearman_p"] < 0.05
        and out["linear_slope_per_year"] < 0
    )
    if value_key == "cv":
        out["linear_slope_cv_per_year"] = out["linear_slope_per_year"]
        out["linear_slope_logcv_per_year"] = out["linear_slope_log_per_year"]
        out["decline_flag"] = out["negative_flag"]
    return out


CLALIT_SCALE_COLUMNS = [
    "min",
    "q1",
    "q25",
    "median",
    "mad",
    "se",
    "q75",
    "q3",
    "max",
    "mean",
    "sd",
    "p10",
    "p25",
    "p75",
    "p90",
    "geom_mean",
]


def expand_clalit_mapping_target(target: object) -> list[dict[str, object]]:
    if isinstance(target, list):
        out: list[dict[str, object]] = []
        for item in target:
            out.extend(expand_clalit_mapping_target(item))
        return out

    if isinstance(target, dict):
        biomarker_id = str(target.get("biomarker_id") or target.get("id") or "").strip()
        if not biomarker_id:
            return []
        scale_factor = float(target.get("scale_factor", 1.0) or 1.0)
        scale_reason = str(target.get("scale_reason") or "").strip()
        return [
            {
                "biomarker_id": biomarker_id,
                "scale_factor": scale_factor,
                "scale_reason": scale_reason,
            }
        ]

    biomarker_id = str(target or "").strip()
    if not biomarker_id:
        return []
    return [{"biomarker_id": biomarker_id, "scale_factor": 1.0, "scale_reason": ""}]


def process_clalit_data(clalit_f: pd.DataFrame, clalit_m: pd.DataFrame, mapping: dict) -> dict:
    if clalit_f is None or clalit_m is None or not mapping:
        return {}
    df_f = clalit_f.copy()
    df_m = clalit_m.copy()
    df_f['sex_norm'] = 'female'
    df_m['sex_norm'] = 'male'
    df = pd.concat([df_f, df_m], ignore_index=True)
    df['mapped_targets'] = df['test'].map(mapping)
    df = df.dropna(subset=['mapped_targets']).copy()
    # Allow one Clalit test to map to multiple NHANES biomarker IDs and optional scale factors.
    df['mapped_targets'] = df['mapped_targets'].apply(expand_clalit_mapping_target)
    df = df.explode('mapped_targets').copy()
    df = df[df['mapped_targets'].notna()].copy()
    df['biomarker_id'] = df['mapped_targets'].apply(lambda v: str(v.get('biomarker_id') or '').strip())
    df['scale_factor'] = df['mapped_targets'].apply(lambda v: float(v.get('scale_factor', 1.0) or 1.0))
    df['scale_reason'] = df['mapped_targets'].apply(lambda v: str(v.get('scale_reason') or '').strip())
    df = df.drop(columns=['mapped_targets'])
    df = df[df['biomarker_id'] != ""].copy()
    for col in CLALIT_SCALE_COLUMNS:
        if col not in df.columns:
            continue
        df[col] = pd.to_numeric(df[col], errors='coerce') * df['scale_factor']
    
    df['age_bin'] = pd.cut(df['age'], bins=AGE_BINS, labels=AGE_LABELS, right=False, include_lowest=True)
    df = df.dropna(subset=['age_bin']).copy()
    df['age_mid'] = df['age_bin'].map(AGE_MIDS).astype(float)
    
    clalit_payload = {}
    
    for (bid, sex), g in df.groupby(['biomarker_id', 'sex_norm'], observed=True):
        pooled = []
        for age_bin, g_age in g.groupby('age_bin', observed=True):
            if g_age.empty:
                continue
            n_tot = int(g_age['n'].sum())
            if n_tot < 30:
                continue
            mean_v = float(np.average(g_age['mean'], weights=g_age['n']))
            var_v = np.average(g_age['sd']**2 + (g_age['mean'] - mean_v)**2, weights=g_age['n'])
            std_v = float(np.sqrt(var_v)) if var_v > 0 else 0.0
            cv_v = std_v / abs(mean_v) if abs(mean_v) > 1e-8 else np.nan
            median_v = float(np.average(g_age['median'], weights=g_age['n']))
            q25_v = float(np.average(g_age['q25'], weights=g_age['n'])) if 'q25' in g_age.columns else None
            q75_v = float(np.average(g_age['q75'], weights=g_age['n'])) if 'q75' in g_age.columns else None
            qskew_v = quantile_skewness_from_stats(pd.Series([q25_v]), pd.Series([median_v]), pd.Series([q75_v])).iloc[0]
            
            p = {
                "age_bin": str(age_bin),
                "age_mid": float(g_age['age_mid'].iloc[0]),
                "n": n_tot,
                "mean": mean_v,
                "std": std_v,
                "median": median_v,
                "q25": q25_v,
                "q75": q75_v,
                "quantile_skewness": float(qskew_v) if pd.notna(qskew_v) else None,
                "cv": float(cv_v),
                "passes_n_threshold": True
            }
            pooled.append(p)
        clalit_payload.setdefault(bid, {})[sex] = pooled
        
    for bid, g in df.groupby('biomarker_id', observed=True):
        pooled = []
        for age_bin, g_age in g.groupby('age_bin', observed=True):
            if g_age.empty:
                continue
            n_tot = int(g_age['n'].sum())
            if n_tot < 30:
                continue
            mean_v = float(np.average(g_age['mean'], weights=g_age['n']))
            var_v = np.average(g_age['sd']**2 + (g_age['mean'] - mean_v)**2, weights=g_age['n'])
            std_v = float(np.sqrt(var_v)) if var_v > 0 else 0.0
            cv_v = std_v / abs(mean_v) if abs(mean_v) > 1e-8 else np.nan
            median_v = float(np.average(g_age['median'], weights=g_age['n']))
            q25_v = float(np.average(g_age['q25'], weights=g_age['n'])) if 'q25' in g_age.columns else None
            q75_v = float(np.average(g_age['q75'], weights=g_age['n'])) if 'q75' in g_age.columns else None
            qskew_v = quantile_skewness_from_stats(pd.Series([q25_v]), pd.Series([median_v]), pd.Series([q75_v])).iloc[0]
            
            p = {
                "age_bin": str(age_bin),
                "age_mid": float(g_age['age_mid'].iloc[0]),
                "n": n_tot,
                "mean": mean_v,
                "std": std_v,
                "median": median_v,
                "q25": q25_v,
                "q75": q75_v,
                "quantile_skewness": float(qskew_v) if pd.notna(qskew_v) else None,
                "cv": float(cv_v),
                "passes_n_threshold": True
            }
            pooled.append(p)
        clalit_payload.setdefault(bid, {})['pooled'] = pooled
        
    return clalit_payload


def append_neutrophil_lymphocyte_ratio(
    long_df: pd.DataFrame | None,
    catalog_df: pd.DataFrame | None,
    specimen_kind: str,
) -> tuple[pd.DataFrame | None, pd.DataFrame | None]:
    """Append a derived neutrophil-to-lymphocyte ratio biomarker to blood inputs."""
    if specimen_kind != "blood" or long_df is None or long_df.empty:
        return long_df, catalog_df

    required = {"seqn", "age_years", "sex", "biomarker_id", "value"}
    if not required.issubset(long_df.columns):
        return long_df, catalog_df

    key_cols = [c for c in ["seqn", "cycle_start_year", "age_years", "sex"] if c in long_df.columns]
    use = long_df[key_cols + ["biomarker_id", "value"]].dropna(subset=["biomarker_id", "value"]).copy()
    use["bid_norm"] = use["biomarker_id"].astype(str).str.strip().str.lower()

    neut = (
        use[use["bid_norm"] == "segmented neutrophils num"]
        .groupby(key_cols, observed=True)["value"]
        .mean()
        .reset_index(name="neutrophils")
    )
    lymph = (
        use[use["bid_norm"] == "lymphocyte number"]
        .groupby(key_cols, observed=True)["value"]
        .mean()
        .reset_index(name="lymphocytes")
    )
    merged = neut.merge(lymph, on=key_cols, how="inner")
    merged = merged.replace([np.inf, -np.inf], np.nan).dropna(subset=["neutrophils", "lymphocytes"])
    merged = merged[merged["lymphocytes"] > 0].copy()
    if merged.empty:
        return long_df, catalog_df

    merged["value"] = merged["neutrophils"] / merged["lymphocytes"]
    merged = merged.replace([np.inf, -np.inf], np.nan).dropna(subset=["value"]).copy()
    if merged.empty:
        return long_df, catalog_df

    derived = pd.DataFrame(index=np.arange(len(merged)))
    for c in long_df.columns:
        derived[c] = np.nan
    for c in key_cols:
        derived[c] = merged[c].to_numpy()
    derived["biomarker_id"] = DERIVED_NLR_ID
    derived["value"] = merged["value"].to_numpy(dtype=float)
    if "variable_name" in derived.columns:
        derived["variable_name"] = DERIVED_NLR_ID
    if "biomarker_name" in derived.columns:
        derived["biomarker_name"] = DERIVED_NLR_NAME
    if "source_data_file" in derived.columns:
        derived["source_data_file"] = "DERIVED"
    if "unit" in derived.columns:
        derived["unit"] = ""
    if "healthy_flag" in derived.columns:
        derived["healthy_flag"] = np.nan
    if "exclusion_reason" in derived.columns:
        derived["exclusion_reason"] = ""

    long_out = pd.concat([long_df, derived[long_df.columns]], ignore_index=True)

    catalog_out = catalog_df
    if catalog_out is not None and not catalog_out.empty:
        has_row = catalog_out["biomarker_id"].astype(str).str.lower().eq(DERIVED_NLR_ID).any()
        if not has_row:
            new_row = {c: np.nan for c in catalog_out.columns}
            new_row["biomarker_id"] = DERIVED_NLR_ID
            if "variable_name" in new_row:
                new_row["variable_name"] = DERIVED_NLR_ID
            if "biomarker_name" in new_row:
                new_row["biomarker_name"] = DERIVED_NLR_NAME
            if "unit" in new_row:
                new_row["unit"] = ""
            if "source_file_count" in new_row:
                new_row["source_file_count"] = 2
            if "source_files" in new_row:
                new_row["source_files"] = "DERIVED"
            if "source_variable_count" in new_row:
                new_row["source_variable_count"] = 2
            if "source_variables" in new_row:
                new_row["source_variables"] = "segmented neutrophils num|lymphocyte number"
            catalog_out = pd.concat([catalog_out, pd.DataFrame([new_row])], ignore_index=True)

    return long_out, catalog_out


def build_outputs(
    cv_df: pd.DataFrame,
    metrics_df: pd.DataFrame,
    catalog_df: pd.DataFrame | None,
    long_df: pd.DataFrame | None,
    raw_sample_n: int,
    random_seed: int,
    specimen_kind: str = "blood",
    clalit_f_df: pd.DataFrame | None = None,
    clalit_m_df: pd.DataFrame | None = None,
    clalit_map: dict | None = None,
    sr_comparison_bundle: dict | None = None,
) -> tuple[pd.DataFrame, list[dict], dict[str, str], dict[str, dict]]:
    cv_df = cv_df.copy()
    long_df, catalog_df = append_neutrophil_lymphocyte_ratio(long_df, catalog_df, specimen_kind)
    if "variable_name" not in cv_df.columns:
        cv_df["variable_name"] = cv_df["biomarker_id"]
    if "unit" not in cv_df.columns:
        cv_df["unit"] = ""

    def grouped_to_points_map(df: pd.DataFrame) -> dict[str, list[dict]]:
        out: dict[str, list[dict]] = {}
        if df is None or df.empty:
            return out
        for bid, g in df.groupby("biomarker_id", observed=True):
            pts = []
            for r in g.sort_values("age_mid").itertuples(index=False):
                mean_v = float(getattr(r, "mean"))
                std_v = getattr(r, "std", np.nan)
                med_v = getattr(r, "median", np.nan)
                q25_v = getattr(r, "q25", np.nan)
                q75_v = getattr(r, "q75", np.nan)
                p10_v = getattr(r, "p10", np.nan)
                p90_v = getattr(r, "p90", np.nan)
                skew_v = getattr(r, "skewness", np.nan)
                qskew_v = getattr(r, "quantile_skewness", np.nan)
                cv_v = getattr(r, "cv", np.nan)
                pts.append(
                    {
                        "age_bin": str(getattr(r, "age_bin")),
                        "age_mid": float(getattr(r, "age_mid")),
                        "n": int(getattr(r, "n")),
                        "mean": mean_v,
                        "std": float(std_v) if pd.notna(std_v) else None,
                        "median": float(med_v) if pd.notna(med_v) else mean_v,
                        "q25": float(q25_v) if pd.notna(q25_v) else None,
                        "q75": float(q75_v) if pd.notna(q75_v) else None,
                        "p10": float(p10_v) if pd.notna(p10_v) else None,
                        "p90": float(p90_v) if pd.notna(p90_v) else None,
                        "skewness": float(skew_v) if pd.notna(skew_v) else None,
                        "quantile_skewness": float(qskew_v) if pd.notna(qskew_v) else None,
                        "cv": float(cv_v) if pd.notna(cv_v) else None,
                        "passes_n_threshold": bool(getattr(r, "passes_n_threshold")),
                    }
                )
            out[str(bid)] = pts
        return out

    def grouped_to_sex_points_map(df: pd.DataFrame) -> dict[str, dict[str, list[dict]]]:
        out: dict[str, dict[str, list[dict]]] = {}
        if df is None or df.empty:
            return out
        for (bid, sex_norm), g in df.groupby(["biomarker_id", "sex_norm"], observed=True):
            out.setdefault(str(bid), {})[str(sex_norm)] = grouped_to_points_map(g)[str(bid)]
        return out

    raw_samples: dict[str, list[dict]] = {}
    raw_samples_by_sex: dict[str, dict[str, list[dict]]] = {}
    raw_counts: dict[str, int] = {}
    raw_counts_by_sex: dict[str, dict[str, int]] = {}
    pooled_points_by_mode: dict[str, dict[str, list[dict]]] = {}
    sex_points_by_mode: dict[str, dict[str, dict[str, list[dict]]]] = {}
    pooled_trends_by_mode_cv: dict[str, dict[str, dict]] = {}
    pooled_trends_by_mode_std: dict[str, dict[str, dict]] = {}
    pooled_trends_by_mode_mean: dict[str, dict[str, dict]] = {}
    pooled_trends_by_mode_skew: dict[str, dict[str, dict]] = {}
    pooled_trends_by_mode_qskew: dict[str, dict[str, dict]] = {}
    sex_trends_by_mode_cv: dict[str, dict[str, dict[str, dict]]] = {}
    sex_trends_by_mode_std: dict[str, dict[str, dict[str, dict]]] = {}
    sex_trends_by_mode_mean: dict[str, dict[str, dict[str, dict]]] = {}
    sex_trends_by_mode_skew: dict[str, dict[str, dict[str, dict]]] = {}
    sex_trends_by_mode_qskew: dict[str, dict[str, dict[str, dict]]] = {}

    if long_df is not None and not long_df.empty:
        use = long_df[["biomarker_id", "age_years", "value", "sex"]].dropna(subset=["biomarker_id", "age_years", "value"])
        use["sex_norm"] = use["sex"].astype(str).str.strip().str.lower()
        use.loc[~use["sex_norm"].isin(["male", "female"]), "sex_norm"] = "unknown"

        sex_use = use[use["sex_norm"].isin(["male", "female"])][["biomarker_id", "age_years", "value", "sex_norm"]]
        for pct in TRIM_PCTS:
            mode = trim_mode_key(pct)
            q = trim_mode_quantiles(mode)
            pooled_binned = compute_binned_long(
                use[["biomarker_id", "age_years", "value"]],
                group_cols=["biomarker_id"],
                trim_quantiles=q,
            )
            sex_binned = compute_binned_long(
                sex_use,
                group_cols=["biomarker_id", "sex_norm"],
                trim_quantiles=q,
            )
            pooled_pts = grouped_to_points_map(pooled_binned)
            sex_pts = grouped_to_sex_points_map(sex_binned)
            pooled_points_by_mode[mode] = pooled_pts
            sex_points_by_mode[mode] = sex_pts
            pooled_trends_by_mode_cv[mode] = {bid: trend_from_points(pts, "cv") for bid, pts in pooled_pts.items()}
            pooled_trends_by_mode_std[mode] = {bid: trend_from_points(pts, "std") for bid, pts in pooled_pts.items()}
            pooled_trends_by_mode_mean[mode] = {bid: trend_from_points(pts, "mean") for bid, pts in pooled_pts.items()}
            pooled_trends_by_mode_skew[mode] = {
                bid: trend_from_points(pts, "skewness") for bid, pts in pooled_pts.items()
            }
            pooled_trends_by_mode_qskew[mode] = {
                bid: trend_from_points(pts, "quantile_skewness") for bid, pts in pooled_pts.items()
            }
            sex_trends_by_mode_cv[mode] = {
                bid: {sx: trend_from_points(pts, "cv") for sx, pts in by_sex.items()} for bid, by_sex in sex_pts.items()
            }
            sex_trends_by_mode_std[mode] = {
                bid: {sx: trend_from_points(pts, "std") for sx, pts in by_sex.items()} for bid, by_sex in sex_pts.items()
            }
            sex_trends_by_mode_mean[mode] = {
                bid: {sx: trend_from_points(pts, "mean") for sx, pts in by_sex.items()} for bid, by_sex in sex_pts.items()
            }
            sex_trends_by_mode_skew[mode] = {
                bid: {sx: trend_from_points(pts, "skewness") for sx, pts in by_sex.items()}
                for bid, by_sex in sex_pts.items()
            }
            sex_trends_by_mode_qskew[mode] = {
                bid: {sx: trend_from_points(pts, "quantile_skewness") for sx, pts in by_sex.items()}
                for bid, by_sex in sex_pts.items()
            }

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
        by_biomarker = [(str(bid), g) for bid, g in use.groupby("biomarker_id", observed=True)]
        non_derived_groups = [(bid, g) for bid, g in by_biomarker if bid != DERIVED_NLR_ID]
        derived_groups = [(bid, g) for bid, g in by_biomarker if bid == DERIVED_NLR_ID]
        for bid, g in non_derived_groups + derived_groups:
            g_pool = g[["age_years", "value"]].dropna()
            if len(g_pool) > raw_sample_n:
                idx = rng.choice(len(g_pool), size=raw_sample_n, replace=False)
                g_pool = g_pool.iloc[idx]
            raw_samples[str(bid)] = [{"age_years": float(r.age_years), "value": float(r.value)} for r in g_pool.itertuples(index=False)]

        by_biomarker_sex = [
            ((str(bid), str(sex_norm)), g)
            for (bid, sex_norm), g in use[use["sex_norm"].isin(["male", "female"])].groupby(
                ["biomarker_id", "sex_norm"], observed=True
            )
        ]
        non_derived_sex_groups = [item for item in by_biomarker_sex if item[0][0] != DERIVED_NLR_ID]
        derived_sex_groups = [item for item in by_biomarker_sex if item[0][0] == DERIVED_NLR_ID]
        for (bid, sex_norm), g in non_derived_sex_groups + derived_sex_groups:
            g2 = g[["age_years", "value"]].dropna()
            if len(g2) > raw_sample_n:
                idx = rng.choice(len(g2), size=raw_sample_n, replace=False)
                g2 = g2.iloc[idx]
            raw_samples_by_sex.setdefault(str(bid), {})[str(sex_norm)] = [
                {"age_years": float(r.age_years), "value": float(r.value)} for r in g2.itertuples(index=False)
            ]
    else:
        # Fallback without participant-level long table.
        base = cv_df.copy()
        base["age_bin"] = base["age_bin"].astype(str)
        if "median" not in base.columns:
            base["median"] = base["mean"]
        for col in ["q25", "q75", "p10", "p90"]:
            if col not in base.columns:
                base[col] = np.nan
        if "skewness" not in base.columns:
            base["skewness"] = np.nan
        if "quantile_skewness" not in base.columns:
            base["quantile_skewness"] = quantile_skewness_from_stats(base["q25"], base["median"], base["q75"])
        for pct in TRIM_PCTS:
            mode = trim_mode_key(pct)
            pooled_points_by_mode[mode] = grouped_to_points_map(base)
            pooled_trends_by_mode_cv[mode] = {
                bid: trend_from_points(pts, "cv") for bid, pts in pooled_points_by_mode[mode].items()
            }
            pooled_trends_by_mode_std[mode] = {
                bid: trend_from_points(pts, "std") for bid, pts in pooled_points_by_mode[mode].items()
            }
            pooled_trends_by_mode_mean[mode] = {
                bid: trend_from_points(pts, "mean") for bid, pts in pooled_points_by_mode[mode].items()
            }
            pooled_trends_by_mode_skew[mode] = {
                bid: trend_from_points(pts, "skewness") for bid, pts in pooled_points_by_mode[mode].items()
            }
            pooled_trends_by_mode_qskew[mode] = {
                bid: trend_from_points(pts, "quantile_skewness") for bid, pts in pooled_points_by_mode[mode].items()
            }
            sex_points_by_mode[mode] = {}
            sex_trends_by_mode_cv[mode] = {}
            sex_trends_by_mode_std[mode] = {}
            sex_trends_by_mode_mean[mode] = {}
            sex_trends_by_mode_skew[mode] = {}
            sex_trends_by_mode_qskew[mode] = {}

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
    metadata["display_name"] = [
        make_display_name(n, u, specimen_kind=specimen_kind)
        for n, u in zip(metadata["biomarker_name"], metadata["unit"])
    ]
    cat_rows = [
        classify_biomarker(n, v, sf)
        for n, v, sf in zip(metadata["biomarker_name"], metadata["variable_name"], metadata["source_files"])
    ]
    metadata["category"] = [r[0] for r in cat_rows]
    metadata["is_environmental"] = [bool(r[1]) for r in cat_rows]
    metadata["is_core_clinical"] = [bool(r[2]) for r in cat_rows]
    metadata["category_rank"] = metadata["category"].map(CATEGORY_ORDER).fillna(999).astype(int)
    metadata = metadata.sort_values(["category_rank", "display_name", "biomarker_id"]).reset_index(drop=True)

    clalit_data_map = process_clalit_data(clalit_f_df, clalit_m_df, clalit_map)
    sr_summary_by_id = (sr_comparison_bundle or {}).get("summary_by_biomarker") or {}
    sr_rank_summary_by_id = (sr_comparison_bundle or {}).get("rank_summary_by_biomarker") or {}

    metrics: list[dict] = []
    for r in metadata.itertuples(index=False):
        bid = str(r.biomarker_id)
        fallback_cv = {
            "n_bins": 0,
            "spearman_rho": None,
            "spearman_p": None,
            "linear_slope_per_year": None,
            "linear_slope_log_per_year": None,
            "linear_slope_cv_per_year": None,
            "linear_slope_logcv_per_year": None,
            "negative_flag": False,
            "decline_flag": False,
        }
        fallback_other = {
            "n_bins": 0,
            "spearman_rho": None,
            "spearman_p": None,
            "linear_slope_per_year": None,
            "linear_slope_log_per_year": None,
            "negative_flag": False,
        }
        modes = [trim_mode_key(p) for p in TRIM_PCTS]
        trends_cv = {mode: pooled_trends_by_mode_cv.get(mode, {}).get(bid, fallback_cv) for mode in modes}
        trends_std = {mode: pooled_trends_by_mode_std.get(mode, {}).get(bid, fallback_other) for mode in modes}
        trends_mean = {mode: pooled_trends_by_mode_mean.get(mode, {}).get(bid, fallback_other) for mode in modes}
        trends_skew = {mode: pooled_trends_by_mode_skew.get(mode, {}).get(bid, fallback_other) for mode in modes}
        trends_qskew = {mode: pooled_trends_by_mode_qskew.get(mode, {}).get(bid, fallback_other) for mode in modes}
        sex_metrics_cv = {mode: sex_trends_by_mode_cv.get(mode, {}).get(bid, {}) for mode in modes}
        sex_metrics_std = {mode: sex_trends_by_mode_std.get(mode, {}).get(bid, {}) for mode in modes}
        sex_metrics_mean = {mode: sex_trends_by_mode_mean.get(mode, {}).get(bid, {}) for mode in modes}
        sex_metrics_skew = {mode: sex_trends_by_mode_skew.get(mode, {}).get(bid, {}) for mode in modes}
        sex_metrics_qskew = {mode: sex_trends_by_mode_qskew.get(mode, {}).get(bid, {}) for mode in modes}
        trend_all = trends_cv.get("all", fallback_cv)

        c_trends = {"cv": {}, "std": {}, "mean": {}, "skewness": {}, "quantile_skewness": {}}
        for c_sex, pts in clalit_data_map.get(bid, {}).items():
            if not pts:
                continue
            c_trends["cv"][c_sex] = trend_from_points(pts, "cv")
            c_trends["std"][c_sex] = trend_from_points(pts, "std")
            c_trends["mean"][c_sex] = trend_from_points(pts, "mean")
            c_trends["skewness"][c_sex] = trend_from_points(pts, "skewness")
            c_trends["quantile_skewness"][c_sex] = trend_from_points(pts, "quantile_skewness")

        metrics.append(
            {
                "biomarker_id": bid,
                "biomarker_name": str(r.biomarker_name),
                "n_bins": trend_all.get("n_bins"),
                "spearman_rho": trend_all.get("spearman_rho"),
                "spearman_p": trend_all.get("spearman_p"),
                "linear_slope_cv_per_year": trend_all.get("linear_slope_cv_per_year"),
                "linear_slope_logcv_per_year": trend_all.get("linear_slope_logcv_per_year"),
                "decline_flag": trend_all.get("decline_flag"),
                "trends": trends_cv,
                "std_trends": trends_std,
                "mean_trends": trends_mean,
                "skewness_trends": trends_skew,
                "quantile_skewness_trends": trends_qskew,
                "trends_by_stat": {
                    "cv": trends_cv,
                    "std": trends_std,
                    "mean": trends_mean,
                    "skewness": trends_skew,
                    "quantile_skewness": trends_qskew,
                },
                "sex_metrics": sex_metrics_cv,
                "sex_std_metrics": sex_metrics_std,
                "sex_mean_metrics": sex_metrics_mean,
                "sex_skewness_metrics": sex_metrics_skew,
                "sex_quantile_skewness_metrics": sex_metrics_qskew,
                "sex_metrics_by_stat": {
                    "cv": sex_metrics_cv,
                    "std": sex_metrics_std,
                    "mean": sex_metrics_mean,
                    "skewness": sex_metrics_skew,
                    "quantile_skewness": sex_metrics_qskew,
                },
                "clalit_trends": c_trends,
                "sr_comparison_summary": sr_summary_by_id.get(bid),
                "sr_rank_comparison_summary": sr_rank_summary_by_id.get(bid),
            }
        )

    series_index: dict[str, str] = {}
    series_payloads: dict[str, dict] = {}

    meta_by_id = metadata.set_index("biomarker_id").to_dict(orient="index")

    for bid in metadata["biomarker_id"].astype(str).tolist():
        rel_path = safe_series_filename(bid)
        md = meta_by_id.get(bid, {})
        sr_detail_payload = sr_detail_payload_for_id(sr_comparison_bundle, bid) or {}
        series_index[bid] = rel_path
        points_by_filter = {mode: pooled_points_by_mode.get(mode, {}).get(bid, []) for mode in [trim_mode_key(p) for p in TRIM_PCTS]}
        sex_points_by_filter = {mode: sex_points_by_mode.get(mode, {}).get(bid, {}) for mode in [trim_mode_key(p) for p in TRIM_PCTS]}
        trends_by_filter_cv = {mode: pooled_trends_by_mode_cv.get(mode, {}).get(bid, {}) for mode in [trim_mode_key(p) for p in TRIM_PCTS]}
        trends_by_filter_std = {mode: pooled_trends_by_mode_std.get(mode, {}).get(bid, {}) for mode in [trim_mode_key(p) for p in TRIM_PCTS]}
        trends_by_filter_mean = {mode: pooled_trends_by_mode_mean.get(mode, {}).get(bid, {}) for mode in [trim_mode_key(p) for p in TRIM_PCTS]}
        trends_by_filter_skew = {mode: pooled_trends_by_mode_skew.get(mode, {}).get(bid, {}) for mode in [trim_mode_key(p) for p in TRIM_PCTS]}
        trends_by_filter_qskew = {mode: pooled_trends_by_mode_qskew.get(mode, {}).get(bid, {}) for mode in [trim_mode_key(p) for p in TRIM_PCTS]}
        sex_trends_filter_cv = {mode: sex_trends_by_mode_cv.get(mode, {}).get(bid, {}) for mode in [trim_mode_key(p) for p in TRIM_PCTS]}
        sex_trends_filter_std = {mode: sex_trends_by_mode_std.get(mode, {}).get(bid, {}) for mode in [trim_mode_key(p) for p in TRIM_PCTS]}
        sex_trends_filter_mean = {mode: sex_trends_by_mode_mean.get(mode, {}).get(bid, {}) for mode in [trim_mode_key(p) for p in TRIM_PCTS]}
        sex_trends_filter_skew = {mode: sex_trends_by_mode_skew.get(mode, {}).get(bid, {}) for mode in [trim_mode_key(p) for p in TRIM_PCTS]}
        sex_trends_filter_qskew = {mode: sex_trends_by_mode_qskew.get(mode, {}).get(bid, {}) for mode in [trim_mode_key(p) for p in TRIM_PCTS]}
        all_points = points_by_filter.get("all", [])
        series_payloads[rel_path] = {
            "biomarker_id": bid,
            "biomarker_name": str(md.get("biomarker_name") or bid),
            "display_name": str(md.get("display_name") or make_display_name(
                str(md.get("biomarker_name") or bid),
                str(md.get("unit") or ""),
                specimen_kind=specimen_kind,
            )),
            "variable_name": str(md.get("variable_name") or bid),
            "unit": str(md.get("unit") or ""),
            "category": md.get("category", "Other Clinical"),
            "is_environmental": bool(md.get("is_environmental", False)),
            "is_core_clinical": bool(md.get("is_core_clinical", False)),
            "raw_total_n": int(md.get("raw_total_n", 0)),
            "raw_total_n_by_sex": raw_counts_by_sex.get(str(bid), {}),
            "raw_sample_cap": int(md.get("raw_sample_cap", raw_sample_n)),
            "points": all_points,
            "points_by_filter": points_by_filter,
            "raw_sample": raw_samples.get(str(bid), []),
            "raw_sample_by_sex": raw_samples_by_sex.get(str(bid), {}),
            "sex_points": sex_points_by_filter.get("all", {}),
            "sex_points_by_filter": sex_points_by_filter,
            "trends": trends_by_filter_cv,
            "std_trends": trends_by_filter_std,
            "mean_trends": trends_by_filter_mean,
            "skewness_trends": trends_by_filter_skew,
            "quantile_skewness_trends": trends_by_filter_qskew,
            "trends_by_stat": {
                "cv": trends_by_filter_cv,
                "std": trends_by_filter_std,
                "mean": trends_by_filter_mean,
                "skewness": trends_by_filter_skew,
                "quantile_skewness": trends_by_filter_qskew,
            },
            "sex_metrics": sex_trends_filter_cv,
            "sex_std_metrics": sex_trends_filter_std,
            "sex_mean_metrics": sex_trends_filter_mean,
            "sex_skewness_metrics": sex_trends_filter_skew,
            "sex_quantile_skewness_metrics": sex_trends_filter_qskew,
            "sex_metrics_by_stat": {
                "cv": sex_trends_filter_cv,
                "std": sex_trends_filter_std,
                "mean": sex_trends_filter_mean,
                "skewness": sex_trends_filter_skew,
                "quantile_skewness": sex_trends_filter_qskew,
            },
            "clalit_data": clalit_data_map.get(str(bid)),
            "sr_comparison": sr_detail_payload.get("sr_comparison"),
            "sr_rank_comparison": sr_detail_payload.get("sr_rank_comparison"),
        }

    return metadata, metrics, series_index, series_payloads


def render_dashboard_html(
    data_base: str,
    specimen_title: str,
    specimen_lower: str,
    has_clalit: bool,
    has_sr_comparison: bool = False,
    specimen_switch_link: str = "",
) -> str:
    data_version = str(int(time.time()))
    template = TEMPLATE_PATH.read_text(encoding="utf-8") if TEMPLATE_PATH.exists() else HTML_TEMPLATE
    sr_tab_html = '<button id="tab-sr-comparison" class="tab-btn" type="button">SR Comparison</button>' if has_sr_comparison else ""
    sr_panel_html = (
        """
    <div id="panel-sr-comparison" class="panel">
      <div class="card">
        <div class="panel-header">
          <div>
            <h2 class="panel-title">SR Comparison</h2>
            <p class="panel-copy">Compare pooled blood biomarkers against the alive-only SR-model `X` distribution in matching 5-year age bins.</p>
          </div>
        </div>
        <div class="sr-controls">
          <label>Method
            <select id="sr-method-mode">
              <option value="qq" selected>QQ / Shape</option>
              <option value="rank">Rank-Wasserstein</option>
            </select>
          </label>
          <label>Search biomarker
            <input id="sr-search" list="sr-biomarker-options" placeholder="Type biomarker name..." autocomplete="off" spellcheck="false" />
            <datalist id="sr-biomarker-options"></datalist>
          </label>
          <label>Biomarker
            <select id="sr-biomarker"></select>
          </label>
          <label class="check-label"><input id="sr-include-env" type="checkbox" /> Include environmental/toxicant</label>
          <label>Current selection
            <div id="sr-selected-biomarker" class="sr-selected">Choose a biomarker to inspect its SR Q-Q fit.</div>
          </label>
          <label>Age bin
            <input id="sr-age-bin-slider" type="range" min="0" max="12" step="1" value="6" />
            <div id="sr-age-bin-label" class="trim-caption">50-54</div>
          </label>
          <label>Comparison trim
            <select id="sr-trim-mode">
              <option value="all">0% each tail</option>
              <option value="trim_3_97">3% each tail</option>
              <option value="trim_5_95">5% each tail</option>
              <option value="trim_10_90">10% each tail</option>
            </select>
          </label>
          <label>Sort main table by
            <select id="sr-sort-field">
              <option value="mean_r2" selected>Mean R²</option>
              <option value="mean_wasserstein_z">Mean z-Wasserstein</option>
              <option value="current_bin_r2">Selected-bin R²</option>
              <option value="current_bin_wasserstein_z">Selected-bin z-Wasserstein</option>
              <option value="min_r2">Minimum R²</option>
              <option value="median_r2">Median R²</option>
              <option value="min_wasserstein_z">Minimum z-Wasserstein</option>
              <option value="median_wasserstein_z">Median z-Wasserstein</option>
              <option value="valid_bin_count">Valid bins</option>
              <option value="mean_slope_m">Mean m</option>
              <option value="slope_m_sd">SD m</option>
              <option value="mean_intercept_c">Mean c</option>
              <option value="intercept_c_sd">SD c</option>
            </select>
          </label>
          <label>Order
            <select id="sr-sort-direction">
              <option value="desc" selected>Descending</option>
              <option value="asc">Ascending</option>
            </select>
          </label>
        </div>
        <div class="filter-test-picker sr-category-picker">
          <label for="sr-category-search">Choose categories to include</label>
          <input id="sr-category-search" type="search" placeholder="Search categories..." autocomplete="off" spellcheck="false" />
          <div class="filter-test-actions">
            <button id="sr-category-all" type="button">Select all visible</button>
            <button id="sr-category-core" type="button">Core clinical only</button>
            <button id="sr-category-clear" type="button">Clear selection</button>
            <span id="sr-category-selection-count" class="filter-test-hint"></span>
          </div>
          <select id="sr-category-multi" multiple></select>
        </div>
        <div id="sr-summary-strip" class="sr-stat-strip"></div>
        <div class="sr-qq-shell">
          <div id="sr-qq-plot"></div>
        </div>
        <div class="sr-mini-grid">
          <div class="sr-mini-shell">
            <div id="sr-r2-plot"></div>
          </div>
          <div class="sr-mini-shell">
            <div id="sr-coef-plot"></div>
          </div>
        </div>
        <div class="sr-section-head">
          <h3>Biomarker Ranking</h3>
          <p>Click a row to switch the shared biomarker selection and refresh the SR plots.</p>
        </div>
        <div class="table-wrap"><table id="sr-rank-table"></table></div>
        <div class="sr-section-head">
          <h3>Per-Bin Fit Details</h3>
          <p>Inspect how `R²`, z-scored Wasserstein distance, slope `m`, and intercept `c` change across age bins for the selected biomarker.</p>
        </div>
        <div class="table-wrap"><table id="sr-bin-table"></table></div>
      </div>
    </div>
        """.strip("\n")
        if has_sr_comparison
        else ""
    )
    return (
        template.replace("__DATA_VERSION__", data_version)
        .replace("__DATA_BASE__", data_base)
        .replace("__SPECIMEN_TITLE__", specimen_title)
        .replace("__SPECIMEN_LOWER__", specimen_lower)
        .replace("__HAS_CLALIT__", "true" if has_clalit else "false")
        .replace("__HAS_SR_COMPARISON__", "true" if has_sr_comparison else "false")
        .replace("__SR_COMPARISON_TAB__", sr_tab_html)
        .replace("__SR_COMPARISON_PANEL__", sr_panel_html)
        .replace("__SPECIMEN_SWITCH_LINK__", specimen_switch_link)
    )


def load_sr_comparison_bundle(root: str | Path | None) -> dict | None:
    if root is None:
        return None

    payload_path = Path(root) / "dashboard_payload.json"
    if not payload_path.exists():
        return None

    payload = json.loads(payload_path.read_text(encoding="utf-8"))
    payload["_root"] = str(Path(root))
    payload["_detail_cache"] = {}
    return payload


def sr_detail_payload_for_id(sr_comparison_bundle: dict | None, biomarker_id: str) -> dict | None:
    if not sr_comparison_bundle:
        return None

    inline_qq = (sr_comparison_bundle.get("detail_by_biomarker") or {}).get(biomarker_id)
    inline_rank = (sr_comparison_bundle.get("rank_detail_by_biomarker") or {}).get(biomarker_id)
    if inline_qq is not None or inline_rank is not None:
        return {
            "sr_comparison": inline_qq,
            "sr_rank_comparison": inline_rank,
        }

    detail_index = sr_comparison_bundle.get("detail_index_by_biomarker") or {}
    relative_path = detail_index.get(biomarker_id)
    if not relative_path:
        return None

    cache = sr_comparison_bundle.setdefault("_detail_cache", {})
    if biomarker_id in cache:
        return cache[biomarker_id]

    root = sr_comparison_bundle.get("_root")
    if not root:
        return None

    detail_path = Path(root) / relative_path
    if not detail_path.exists():
        return None

    detail_payload = json.loads(detail_path.read_text(encoding="utf-8"))
    cache[biomarker_id] = detail_payload
    return detail_payload


def dashboard_shared_payloads(sr_comparison_bundle: dict | None) -> dict[str, dict]:
    if not sr_comparison_bundle:
        return {}

    sr_waterfall_reference = sr_comparison_bundle.get("sr_waterfall_reference")
    sr_rank_reference = sr_comparison_bundle.get("sr_rank_reference")
    payloads: dict[str, dict] = {}
    if sr_waterfall_reference:
        payloads["sr_waterfall_reference.json"] = sr_waterfall_reference
    if sr_rank_reference:
        payloads["sr_rank_reference.json"] = sr_rank_reference
    return payloads


def write_dashboard_bundle(
    out_html: Path,
    out_json: Path,
    data_dir_name: str,
    metadata: pd.DataFrame,
    metrics: list[dict],
    series_index: dict[str, str],
    series_payloads: dict[str, dict],
    raw_sample_n: int,
    shared_payloads: dict[str, dict] | None = None,
) -> None:
    data_dir = out_html.parent / data_dir_name
    series_dir = data_dir / "series"
    shared_payloads = shared_payloads or {}

    ensure_dir(out_html.parent)
    ensure_dir(data_dir)
    ensure_dir(series_dir)
    ensure_dir(out_json.parent)

    for old in series_dir.glob("*.json"):
        old.unlink()

    (data_dir / "metadata.json").write_text(
        json.dumps(metadata.to_dict(orient="records"), ensure_ascii=True, allow_nan=False), encoding="utf-8"
    )
    (data_dir / "metrics.json").write_text(json.dumps(metrics, ensure_ascii=True, allow_nan=False), encoding="utf-8")
    (data_dir / "series_index.json").write_text(
        json.dumps(series_index, ensure_ascii=True, allow_nan=False), encoding="utf-8"
    )

    shared_file_names = ["sr_waterfall_reference.json", "sr_rank_reference.json"]
    for file_name in shared_file_names:
        shared_path = data_dir / file_name
        if file_name in shared_payloads:
            shared_path.write_text(
                json.dumps(shared_payloads[file_name], ensure_ascii=True, allow_nan=False),
                encoding="utf-8",
            )
            continue
        if shared_path.exists():
            shared_path.unlink()

    for rel, payload in series_payloads.items():
        p = data_dir / rel
        ensure_dir(p.parent)
        p.write_text(json.dumps(payload, ensure_ascii=True, allow_nan=False), encoding="utf-8")

    summary_payload = {
        "metadata_count": len(metadata),
        "metrics_count": len(metrics),
        "series_count": len(series_payloads),
        "raw_sample_n": raw_sample_n,
        "data_dir": str(data_dir),
    }
    out_json.write_text(json.dumps(summary_payload, ensure_ascii=True, indent=2, allow_nan=False), encoding="utf-8")

    print(f"Wrote metadata: {data_dir / 'metadata.json'}")
    print(f"Wrote metrics: {data_dir / 'metrics.json'}")
    print(f"Wrote series index: {data_dir / 'series_index.json'}")
    print(f"Wrote {len(series_payloads)} series files under: {series_dir}")
    print(f"Wrote dashboard summary JSON: {out_json}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--cv", default="data/processed/cv_by_age.parquet")
    ap.add_argument("--cv-all", default="data/processed/cv_by_age_all.parquet")
    ap.add_argument("--metrics", default="data/processed/cv_trend_metrics.parquet")
    ap.add_argument("--catalog", default="data/processed/biomarker_catalog.parquet")
    ap.add_argument("--long", default="data/processed/biomarker_long.parquet")
    ap.add_argument("--urine-cv", default="data/processed/urine/cv_by_age.parquet")
    ap.add_argument("--urine-cv-all", default="data/processed/urine/cv_by_age_all.parquet")
    ap.add_argument("--urine-metrics", default="data/processed/urine/cv_trend_metrics.parquet")
    ap.add_argument("--urine-catalog", default="data/processed/urine/biomarker_catalog.parquet")
    ap.add_argument("--urine-long", default="data/processed/urine/biomarker_long.parquet")
    ap.add_argument("--raw-sample-n", type=int, default=1200)
    ap.add_argument("--random-seed", type=int, default=42)
    ap.add_argument("--out", default="dashboard/index.html")
    ap.add_argument("--json-out", default="dashboard/dashboard_data.json")
    ap.add_argument("--urine-out", default="dashboard/urinary.html")
    ap.add_argument("--urine-json-out", default="dashboard/dashboard_data_urine.json")
    ap.add_argument("--aging-public-out", default="dashboard/aging_biomarkers_dashboard.html")
    ap.add_argument("--aging-public-json-out", default="dashboard/dashboard_data_aging_biomarkers.json")
    ap.add_argument(
        "--aging-biomarkers-catalog-csv",
        default=str(ROOT / "projects" / "aging_biomarkers" / "catalog" / "aging_biomarkers.csv"),
    )
    ap.add_argument("--participant-flags", default="data/processed/participant_health_flags.parquet")
    ap.add_argument("--raw-dir", default="data/raw")
    ap.add_argument("--variable-screening-summary", default="data/processed/variable_screening_summary.csv")
    ap.add_argument("--duplicate-merge-map", default="data/processed/duplicate_merge_map.csv")
    ap.add_argument("--clalit-f", default="data/clalit/females_all_statistics.csv")
    ap.add_argument("--clalit-m", default="data/clalit/males_all_statistics.csv")
    ap.add_argument("--clalit-map", default="data/clalit_mapping.json")
    ap.add_argument("--sr-comparison-root", default=str(ROOT / "projects" / "sr_comparison" / "blood"))
    args = ap.parse_args()

    def load_inputs(cv_all_path: str, cv_path: str, metrics_path: str, catalog_path: str, long_path: str) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame | None, pd.DataFrame | None]:
        preferred_cv = Path(cv_all_path)
        if not preferred_cv.exists():
            preferred_cv = Path(cv_path)
        cv_df = pd.read_parquet(preferred_cv)
        metrics_df = pd.read_parquet(metrics_path) if Path(metrics_path).exists() else pd.DataFrame()
        cat_path = Path(catalog_path)
        catalog_df = pd.read_parquet(cat_path) if cat_path.exists() else None
        lg_path = Path(long_path)
        long_df = None
        if lg_path.exists():
            long_df = pd.read_parquet(
                lg_path,
                columns=["seqn", "cycle_start_year", "biomarker_id", "age_years", "value", "sex"],
            )
        return cv_df, metrics_df, catalog_df, long_df

    blood_cv_df, blood_metrics_df, blood_catalog_df, blood_long_df = load_inputs(
        cv_all_path=args.cv_all,
        cv_path=args.cv,
        metrics_path=args.metrics,
        catalog_path=args.catalog,
        long_path=args.long,
    )
    urine_cv_df, urine_metrics_df, urine_catalog_df, urine_long_df = load_inputs(
        cv_all_path=args.urine_cv_all,
        cv_path=args.urine_cv,
        metrics_path=args.urine_metrics,
        catalog_path=args.urine_catalog,
        long_path=args.urine_long,
    )

    clalit_f = pd.read_csv(args.clalit_f) if Path(args.clalit_f).exists() else None
    clalit_m = pd.read_csv(args.clalit_m) if Path(args.clalit_m).exists() else None
    clalit_map = None
    if Path(args.clalit_map).exists():
        with open(args.clalit_map) as f:
            clalit_map = json.load(f)
    participant_flags = pd.read_parquet(args.participant_flags) if Path(args.participant_flags).exists() else None
    has_blood_clalit = bool(clalit_f is not None and clalit_m is not None and clalit_map)
    sr_comparison_bundle = load_sr_comparison_bundle(args.sr_comparison_root)
    has_blood_sr_comparison = bool(sr_comparison_bundle)

    blood_metadata, blood_metrics, blood_series_index, blood_series_payloads = build_outputs(
        cv_df=blood_cv_df,
        metrics_df=blood_metrics_df,
        catalog_df=blood_catalog_df,
        long_df=blood_long_df,
        raw_sample_n=args.raw_sample_n,
        random_seed=args.random_seed,
        specimen_kind="blood",
        clalit_f_df=clalit_f if has_blood_clalit else None,
        clalit_m_df=clalit_m if has_blood_clalit else None,
        clalit_map=clalit_map if has_blood_clalit else None,
        sr_comparison_bundle=sr_comparison_bundle,
    )
    urine_metadata, urine_metrics, urine_series_index, urine_series_payloads = build_outputs(
        cv_df=urine_cv_df,
        metrics_df=urine_metrics_df,
        catalog_df=urine_catalog_df,
        long_df=urine_long_df,
        raw_sample_n=args.raw_sample_n,
        random_seed=args.random_seed,
        specimen_kind="urine",
        clalit_f_df=None,
        clalit_m_df=None,
        clalit_map=None,
    )

    blood_out_html = Path(args.out)
    blood_out_json = Path(args.json_out)
    urine_out_html = Path(args.urine_out)
    urine_out_json = Path(args.urine_json_out)
    aging_public_out_html = Path(args.aging_public_out)
    aging_public_out_json = Path(args.aging_public_json_out)

    write_dashboard_bundle(
        out_html=blood_out_html,
        out_json=blood_out_json,
        data_dir_name="data",
        metadata=blood_metadata,
        metrics=blood_metrics,
        series_index=blood_series_index,
        series_payloads=blood_series_payloads,
        raw_sample_n=args.raw_sample_n,
        shared_payloads=dashboard_shared_payloads(sr_comparison_bundle),
    )
    public_manifest = build_public_manifest(
        metadata=blood_metadata,
        series_index=blood_series_index,
        series_payloads=blood_series_payloads,
        aging_catalog_csv=args.aging_biomarkers_catalog_csv,
    )
    disease_long_df = load_public_disease_long(
        public_manifest=public_manifest,
        participant_flags=participant_flags,
        raw_dir=args.raw_dir,
        screening_summary_path=args.variable_screening_summary,
        merge_map_path=args.duplicate_merge_map,
    )
    disease_bundle = build_disease_explorer_bundle(
        public_manifest=public_manifest,
        long_df=disease_long_df,
        participant_flags=participant_flags,
    )
    write_public_dashboard_bundle(
        out_html=aging_public_out_html,
        out_json=aging_public_out_json,
        data_dir_name="aging_biomarkers_public",
        manifest=public_manifest,
        disease_bundle=disease_bundle,
    )
    write_dashboard_bundle(
        out_html=urine_out_html,
        out_json=urine_out_json,
        data_dir_name="data_urine",
        metadata=urine_metadata,
        metrics=urine_metrics,
        series_index=urine_series_index,
        series_payloads=urine_series_payloads,
        raw_sample_n=args.raw_sample_n,
        shared_payloads=None,
    )

    blood_self_href = os.path.relpath(blood_out_html, start=blood_out_html.parent).replace(os.sep, "/")
    urine_from_blood_href = os.path.relpath(urine_out_html, start=blood_out_html.parent).replace(os.sep, "/")
    blood_from_urine_href = os.path.relpath(blood_out_html, start=urine_out_html.parent).replace(os.sep, "/")
    urine_self_href = os.path.relpath(urine_out_html, start=urine_out_html.parent).replace(os.sep, "/")

    blood_switch_links = (
        f'<a class="tab-btn specimen-link active" data-base-href="{blood_self_href}" href="{blood_self_href}#dashboard">Blood Tests</a>'
        f'<a class="tab-btn specimen-link" data-base-href="{urine_from_blood_href}" href="{urine_from_blood_href}#dashboard">Urinary Tests</a>'
    )
    urine_switch_links = (
        f'<a class="tab-btn specimen-link" data-base-href="{blood_from_urine_href}" href="{blood_from_urine_href}#dashboard">Blood Tests</a>'
        f'<a class="tab-btn specimen-link active" data-base-href="{urine_self_href}" href="{urine_self_href}#dashboard">Urinary Tests</a>'
    )

    blood_out_html.write_text(
        render_dashboard_html(
            data_base="./data",
            specimen_title="Blood",
            specimen_lower="blood",
            has_clalit=has_blood_clalit,
            has_sr_comparison=has_blood_sr_comparison,
            specimen_switch_link=blood_switch_links,
        ),
        encoding="utf-8",
    )
    urine_out_html.write_text(
        render_dashboard_html(
            data_base="./data_urine",
            specimen_title="Urinary",
            specimen_lower="urinary",
            has_clalit=False,
            has_sr_comparison=False,
            specimen_switch_link=urine_switch_links,
        ),
        encoding="utf-8",
    )
    print(f"Wrote dashboard HTML: {blood_out_html}")
    print(f"Wrote dashboard HTML: {urine_out_html}")


if __name__ == "__main__":
    main()
