# NHANES Blood Biomarker Explorer

Interactive explorer for age-related blood biomarker trajectories in NHANES.

This project builds a static web dashboard where users can search biomarkers, compare trends across age and sex, and inspect ranking metrics across hundreds of blood tests.

## Scripts
- `src/discover_nhanes.py` discovers laboratory variable metadata and blood candidates.
- `src/download_nhanes.py` downloads required NHANES XPT files.
- `src/build_analysis_dataset.py` creates harmonized healthy-adult biomarker long data.
- `src/compute_cv_metrics.py` computes CV-by-age bins and decline metrics.
- `src/build_dashboard.py` builds static interactive HTML dashboard.

## Run Order
```bash
python3 src/discover_nhanes.py --component Laboratory --verify-urls
python3 src/download_nhanes.py --manifest data/processed/lab_variable_manifest.parquet
python3 src/build_analysis_dataset.py --raw data/raw --manifest data/processed/lab_variable_manifest.parquet --out data/processed
python3 src/compute_cv_metrics.py --in data/processed/biomarker_long.parquet --out data/processed
python3 src/build_dashboard.py --cv data/processed/cv_by_age.parquet --cv-all data/processed/cv_by_age_all.parquet --metrics data/processed/cv_trend_metrics.parquet --out dashboard/index.html --json-out dashboard/dashboard_data.json
```

## Open the dashboard
- Local:
  - Double-click `Open_NHANES_Dashboard.command`
  - It starts a local server and opens `http://127.0.0.1:8765/dashboard/index.html`
- Online:
  - Open the GitHub Pages site (if enabled in your repo settings):
  - `https://<github-username>.github.io/<repo-name>/`

## Performance model (on-demand data loading)
- `dashboard/index.html` now loads only metadata + metrics initially.
- Per-biomarker point series are stored in:
  - `dashboard/data/series/*.json`
- Series are fetched ad hoc only when a biomarker is selected/searched.

## Plot modes
- Use the top buttons in the dashboard:
  - `Plot CV`: CV vs age.
  - `Plot Median`: median vs age with:
    - interquartile range (IQR) band (25th-75th percentile)
    - raw scatter sample (age vs value) for the selected biomarker
  - `Symmetric Trim Per Tail (%)`: optional robust trimming within each age bin before summary stats are computed (for example 10-90, 20-80, 25-75).

## Info tab
- Use `Info & Methods` (top tab) for:
  - analysis scope and filtering
  - healthy cohort definition
  - decline flag criteria
  - interpretation notes for CV and median views

## Compare tab
- Use `Compare Rankings` (top tab) to compare biomarkers by Spearman trend quickly.
- Controls:
  - sort mode: most negative, most positive, or largest absolute Spearman
  - symmetric trim (% per tail), shared with dashboard outlier mode
  - top N count
- Visual:
  - horizontal bar chart with hover details (`rho`, `p`, `n_bins`, `decline`, biomarker id)

## Pooling and variable screening
- Biomarkers are pooled across NHANES cycles/files by normalized test name (not only by code name).
- Example: different code names for the same test (e.g., `LBX*` and `SST*`) are merged when they refer to the same analyte/test.
- Compatible unit variants are converted and pooled (e.g., g/dL and g/L for albumin); incompatible unit systems remain separate entries.
- Non-analytic fields are removed before analysis:
  - comment/result code fields
  - questionnaire-style text fields
  - duplicate/technical assay fields
  - low-information categorical numeric fields
- CRP/hs-CRP are included as pooled blood biomarkers.
- Screening audit is written to:
  - `data/processed/variable_screening_summary.csv`
- Pooled catalog is written to:
  - `data/processed/biomarker_catalog.parquet`

## Median mode interpretation
- `Plot Median` displays the age-binned median and IQR band (25th-75th percentile).
- Trimming is symmetric by tail and is applied within each age bin before computing plotted summaries and trend metrics.
- Raw sampled points are shown in median view to visualize spread and outliers.

## Tests
```bash
python3 -m unittest discover -s tests -p 'test_*.py'
```
