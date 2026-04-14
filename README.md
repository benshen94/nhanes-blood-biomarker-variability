# nhanes-biomarker-dashboard

Interactive explorer for age-related blood and urinary biomarker trajectories in the NHANES dataset.

This project builds static web dashboards where users can search biomarkers, compare trends across age and sex, inspect ranking metrics across hundreds of tests, and browse a curated public-facing aging biomarker explorer.

Documentation rule: when dashboard features/metrics change, update this README in the same commit.

## Scripts
- `src/discover_nhanes.py` discovers laboratory variable metadata and tags both blood and urinary candidates in the manifest (`is_blood_candidate`, `is_urine_candidate`).
- `src/download_nhanes.py` downloads required NHANES XPT files (lab + demographics + questionnaire modules including `DIQ/MCQ/KIQ/BPQ/OSQ/VIQ/PFQ/HUQ`), with candidate selection controlled by `--candidate-column`.
- `src/build_analysis_dataset.py` creates harmonized healthy-adult biomarker long data, with candidate selection controlled by `--candidate-column`.
- `src/compute_cv_metrics.py` computes CV-by-age bins and decline metrics.
- `src/build_sr_comparison.py` reruns or reuses cached SR fits from `projects/sr_fits/`, bins each SR-model `X` distribution on the NHANES 5-year age bins, and builds both the Q-Q and rank-based SR comparison payloads under `projects/sr_comparison/blood/`.
  - In rank mode, biomarker values are trimmed within each age bin before pooling and ranking, but the alive-only SR `X` values are not trimmed.
  - Rank mode now supports `0%`, `3%`, `5%`, and `10%` biomarker tail trimming.
- `projects/sr_fits/fit_registry.json` is the source of truth for the available SR reference fits shown in the dashboard. Each fit gets its own cached trajectories, manifest, and compact SR reference payloads under `projects/sr_fits/<fit_key>/`.
- `src/run_custom_sr_fit.py` runs a custom SR simulation through the external aging codebase and saves a combined survival/hazard PNG, a waterfall PNG, a yearly alive-only `X` summary CSV/PNG (`mean`, `std`, `cv`, quantile-skewness), and a parameter JSON under `output/sr_fits_results/`.
- `src/build_dashboard.py` builds the blood dashboard (`dashboard/index.html`), urinary dashboard (`dashboard/urinary.html`), and the public-facing aging biomarkers dashboard (`dashboard/aging_biomarkers_dashboard.html`), and writes the shared SR fit manifest plus SR waterfall/rank reference assets into `dashboard/data/` when the blood SR payload is available.
- `src/build_clalit_quartiles.py` aggregates the Clalit single-year ridgeline densities into NHANES-style 5-year bins (`20-24` through `95-99`) and writes a combined female/male quartile export to `data/clalit/clalit_quartiles.csv`.
- `src/aging_biomarker_analysis.py` now prefers `data/clalit/clalit_quartiles.csv` for the Clalit summary-rho PCA branch, using quartile-derived median, \( \mathrm{IQR} / Q_{50} \) CV, and quantile skewness instead of the legacy Clalit standard-deviation features.
- `src/build_aging_biomarkers_dashboard.py` builds the curated manifest and HTML bundle for the public-facing blood-only aging biomarkers explorer.
  - It also writes disease-default metadata for the guided Disease Explorer and a small `surprising.json` payload for the shareable `What’s Surprising?` tab.
  - If `AGING_PUBLIC_GA4_ID` is set at build time, it injects Google Analytics 4 into the public dashboard HTML.
- `src/templates/dashboard_template.html` is the shared dashboard UI template used by `src/build_dashboard.py` for both specimen outputs.
- `src/templates/aging_biomarkers_dashboard_template.html` is the standalone editorial-science template used for the public-facing aging biomarkers explorer.
- `dashboard/longevity-explorer.html` is the short GitHub Pages alias that redirects to the audience-facing aging biomarkers explorer.
- `scripts/export_public_dashboard_site.py` exports only the audience-facing static site files into `../biomarker_dashboard/` for use in a separate public GitHub Pages repo.
- `src/plot_km_kidney_liver.py` generates Kaplan-Meier survival plots for broad disease cohorts vs full cohort using linked mortality files (follow-up and age-timescale outputs).
- `src/cluster_km_shapes.py` clusters disease KM curve shapes with multiple distances and algorithms, and writes visual diagnostics to `output/km_shape_clustering/`.
- `src/fpca_km_shapes.py` runs functional-PCA style decomposition of disease KM curves, clusters in fPCA score space, and writes outputs to `output/fPCA/`.

## Reproductive health questionnaire data
- Public NHANES reproductive-health questionnaire files are now stored locally at:
  - `data/raw/2001/RHQ_B.xpt`
  - `data/raw/2003/RHQ_C.xpt`
  - `data/raw/2005/RHQ_D.xpt`
  - `data/raw/2007/RHQ_E.xpt`
  - `data/raw/2009/RHQ_F.xpt`
  - `data/raw/2011/RHQ_G.xpt`
  - `data/raw/2013/RHQ_H.xpt`
  - `data/raw/2015/RHQ_I.xpt`
  - `data/raw/2017/RHQ_J.xpt`
- The main menarche variable is `RHQ010`, labeled `Age when first menstrual period occurred`.
- In practice, `RHQ010` is the NHANES age-at-menarche field. There is not a separate public variable for `menarche` versus `first period`; they are the same concept here.
- `RHQ020` is the age-range fallback when exact age at first menstrual period is not reported.
- Download and presence summaries are tracked in `data/processed/download_manifest_reproductive_health.csv` and `data/processed/reproductive_health_menarche_inventory.csv`.
- NHANES also documents fuller restricted-use `RHQ_*_R` files for some cycles. Those are RDC-only and are not publicly downloadable.

## Run Order
```bash
python3 src/discover_nhanes.py --component Laboratory --verify-urls
python3 src/download_nhanes.py --manifest data/processed/lab_variable_manifest.parquet
python3 src/build_analysis_dataset.py --raw data/raw --manifest data/processed/lab_variable_manifest.parquet --out data/processed
python3 src/compute_cv_metrics.py --in data/processed/biomarker_long.parquet --out data/processed
python3 src/download_nhanes.py --manifest data/processed/lab_variable_manifest.parquet --candidate-column is_urine_candidate --download-manifest data/processed/download_manifest_urine.csv
python3 src/build_analysis_dataset.py --raw data/raw --manifest data/processed/lab_variable_manifest.parquet --out data/processed/urine --candidate-column is_urine_candidate
python3 src/compute_cv_metrics.py --in data/processed/urine/biomarker_long.parquet --out data/processed/urine
python3 src/build_sr_comparison.py --out-root projects/sr_comparison/blood
python3 src/build_dashboard.py
AGING_PUBLIC_GA4_ID=G-XXXXXXXXXX python3 src/build_dashboard.py
python3 src/plot_km_kidney_liver.py --participants data/processed/participant_health_flags.parquet --mortality-dir data/raw/mortality --png-out output/km_kidney_liver_vs_full.png --csv-out output/km_kidney_liver_counts.csv --png-age-out output/km_kidney_liver_vs_full_by_age.png --csv-age-out output/km_kidney_liver_counts_by_age.csv --png-all-disease-panels-age-out output/km_all_diseases_vs_full_by_age_panels.png --csv-all-disease-age-out output/km_all_diseases_age_summary.csv --age-summary-csv-out output/km_kidney_liver_age_summary.csv --steepness-png-out output/steepness_longevity_disease.png --png-asthma-age-out output/km_asthma_vs_full_by_age.png --csv-asthma-age-out output/km_asthma_counts_by_age.csv --min-disease-n 100
python3 src/cluster_km_shapes.py --participants data/processed/participant_health_flags.parquet --mortality-dir data/raw/mortality --out-dir output/km_shape_clustering --min-disease-n 100 --k-min 2 --k-max 8 --seed 42
python3 src/fpca_km_shapes.py --participants data/processed/participant_health_flags.parquet --mortality-dir data/raw/mortality --out-dir output/fPCA --min-disease-n 100 --k-min 2 --k-max 8 --seed 42
```

## Kaplan-Meier outputs
- Follow-up timeline:
  - `output/km_kidney_liver_vs_full.png`
  - `output/km_kidney_liver_counts.csv`
  - x-axis: months since interview (`permth_int`/`permth_exm`)
- Age timeline (delayed entry / left truncation):
  - `output/km_kidney_liver_vs_full_by_age.png`
  - `output/km_kidney_liver_counts_by_age.csv`
  - x-axis: age in years
  - entry age: interview age
  - exit age: interview age + mortality/censor follow-up
- Age-shape summary:
  - `output/km_kidney_liver_age_summary.csv`
  - per cohort: median lifespan, Q1/Q3 lifespan, IQR lifespan, steepness (`median / IQR`)
- Relative disease scatter:
  - `output/steepness_longevity_disease.png`
  - x-axis: disease-cohort median lifespan divided by full-cohort median lifespan
  - y-axis: disease-cohort steepness divided by full-cohort steepness
  - includes dashed reference lines at `(1,1)`
- All-disease age panel plot:
  - `output/km_all_diseases_vs_full_by_age_panels.png`
  - one panel per available disease flag (each panel overlays full cohort + disease cohort KM curve)
  - lifelines confidence intervals are shown for both curves in each panel
- All-disease age summary table:
  - `output/km_all_diseases_age_summary.csv`
  - includes `n`, `deaths`, median lifespan, Q1/Q3, IQR, and steepness for each disease cohort
- Asthma (separate age-timescale curve):
  - `output/km_asthma_vs_full_by_age.png`
  - `output/km_asthma_counts_by_age.csv`
  - compares asthma (`MCQ010==1`) vs full cohort on age timeline

## KM Shape Clustering
- Output folder:
  - `output/km_shape_clustering/`
- Curves used:
  - disease age-timescale KM curves (left-truncated with interview-age entry), sampled on a shared age grid
- Distances computed:
  - cosine (raw KM)
  - euclidean (raw KM)
  - correlation (raw KM)
  - cosine (shape-normalized KM)
  - euclidean (derivative profile)
  - DTW (shape-normalized KM)
- Clustering methods compared:
  - hierarchical average linkage (cosine)
  - hierarchical average linkage (DTW)
  - k-medoids (cosine)
  - k-means (euclidean features)
  - spectral clustering (cosine affinity)
  - agglomerative ward
- Key outputs:
  - `cluster_method_summary.csv` (best K + silhouette per method)
  - `cluster_assignments.csv` (disease cluster membership for each method)
  - `nearest_neighbors_cosine_raw.csv` (top nearest curve neighbors for each disease)
  - `dendrogram_cosine_raw.png` (who merges with who)
  - `heatmap_*.png` + `pairwise_distance_*.csv` (similarity maps)
  - `consensus_similarity_heatmap.png` (how consistently pairs co-cluster across methods)
  - `cluster_overlays_by_method.png` (cluster medoid/median shape overlays)
  - `mds_cosine_raw.png` (2D map of disease-curve similarity)

## fPCA of KM Curves
- Output folder:
  - `output/fPCA/`
- Functional representation:
  - disease age-timescale KM curves are sampled on a common age grid and smoothed into continuous functions
  - fPCA is run on centered smoothed functions
- Core outputs:
  - `summary.txt` (cohort count, selected k, and FPC1/FPC2 variance capture)
  - `fpca_explained_variance.csv` + `scree_explained_variance.png`
  - `fpca_eigenfunctions.csv` + `eigenfunctions_top_components.png`
  - `modes_of_variation_pc1_pc2.png` (mean ± 2 SD score along FPC1/FPC2)
  - `scores_scatter_pc1_pc2_clusters.png` (disease points on first two fPCA axes)
  - `silhouette_vs_k_fpca_scores.png` + `fpca_k_selection_silhouette.csv`
  - `fpca_scores_clusters.csv` (cluster assignment per disease)
  - `cluster_overlays_fpca2d_kmeans.png` (curve overlays by fPCA-based cluster)
  - `reconstruction_examples_pc1_pc2.png` (2-component reconstruction diagnostics)
  - `nearest_neighbors_fpca2d.csv` (closest diseases in fPCA score space)

## Open the dashboard
- Local:
  - Double-click `Open_NHANES_Dashboard.command`
  - It starts a local server and opens `http://127.0.0.1:8765/dashboard/index.html` (blood dashboard)
  - Urinary dashboard is at `http://127.0.0.1:8765/dashboard/urinary.html` (or use the `Urinary Tests` tab button inside the blood dashboard)
  - Public aging biomarkers dashboard is at `http://127.0.0.1:8765/dashboard/aging_biomarkers_dashboard.html`
- Online:
- Open the GitHub Pages site (if enabled in your repo settings):
- Blood: `https://<github-username>.github.io/<repo-name>/dashboard/index.html`
- Urinary: `https://<github-username>.github.io/<repo-name>/dashboard/urinary.html`
- Public aging biomarkers: `https://<github-username>.github.io/<repo-name>/dashboard/aging_biomarkers_dashboard.html`
- Public aging biomarkers short link: `https://<github-username>.github.io/<repo-name>/dashboard/longevity-explorer.html`

## Dashboard UI architecture
- The blood and urinary dashboards share one HTML/CSS/JS shell from `src/templates/dashboard_template.html`.
- `src/build_dashboard.py` injects specimen-specific metadata, dataset paths, counts, and specimen-switch links into that template, then writes:
  - `dashboard/index.html`
  - `dashboard/urinary.html`
- The public-facing aging biomarkers dashboard uses a separate shell from `src/templates/aging_biomarkers_dashboard_template.html`.
- `src/build_dashboard.py` also calls `src/build_aging_biomarkers_dashboard.py` to write:
  - `dashboard/aging_biomarkers_dashboard.html`
  - `dashboard/dashboard_data_aging_biomarkers.json`
  - `dashboard/aging_biomarkers_public/manifest.json`
  - `dashboard/aging_biomarkers_public/disease_index.json`
  - `dashboard/aging_biomarkers_public/surprising.json`
- The public dashboard now includes:
  - a guided Disease Explorer with disease-specific starter biomarker chips
  - disease comparisons that only draw age bins with \( n \ge 30 \) for the selected assay/cohort
  - disease median plots shaded by \( \mathrm{SEM} \) or \( \mathrm{SD} \) around the age-bin median, using the same spread toggle as the main explorer
  - a `What’s Surprising?` tab for shareable counterintuitive aging patterns
  - `5-95 trimmed` as the default public trim mode
- The redesign keeps the existing static single-page model and DOM IDs used by the inline dashboard logic, while modernizing:
  - hero/header hierarchy
  - specimen and analysis navigation
  - control rails and panel cards
  - metric summary presentation
  - focus states, contrast, and responsive behavior
- Plotly traces and metric semantics are unchanged; only presentation defaults such as fonts, spacing, grid contrast, legends, and empty-state styling were updated.

## Navigation model (specimen-first)
- The dashboard now uses two navigation levels:
  - Row 1 (`Specimen`): `Blood Tests` and `Urinary Tests`
  - Row 2 (`Analysis View`): `Dashboard`, `Compare Rankings`, `Filter Tests`, `Scatter Plot`, `Histograms`, `Waterfall`, `Info & Methods`
- `Blood Tests` / `Urinary Tests` are parent context controls, not peer tabs with analysis views.
- Active analysis view is URL-hash state:
  - `#dashboard`, `#compare`, `#filter-tests`, `#scatter`, `#hist`, `#waterfall`, `#info`
- Switching specimen preserves the current analysis view:
  - `index.html#scatter` -> `urinary.html#scatter`
  - `urinary.html#compare` -> `index.html#compare`
- Browser back/forward keeps panel state and hero title/subtitle synchronized via `hashchange` handling.

## Performance model (on-demand data loading)
- `dashboard/index.html` (blood) and `dashboard/urinary.html` (urinary) each load only metadata + metrics initially.
- The dashboard now renders only the active top tab on startup instead of pre-rendering every tab.
- Per-biomarker point series are stored in:
  - `dashboard/data/series/*.json`
  - `dashboard/data_urine/series/*.json`
- Series are fetched ad hoc only when a biomarker is selected/searched.
- The blood dashboard also lazy-loads shared SR fit payloads:
  - `dashboard/data/sr_fit_manifest.json`
  - `dashboard/data/sr_waterfall_references.json`
  - the fit manifest lists the available SR fit keys/labels, and the waterfall payload contains compact quantile-sampled SR-model `X` distributions for every registered fit
- The blood dashboard also uses one shared SR rank reference payload for `Rank-Wasserstein` mode:
  - `dashboard/data/sr_rank_references.json`
  - it contains the SR normalized-rank distributions (`1` to `100`) for each registered SR fit, SR trim mode, and 5-year age bin, so those SR ranks are not duplicated inside every biomarker series file

## Public aging biomarkers dashboard
- Artifact:
  - `dashboard/aging_biomarkers_dashboard.html`
- Summary payload:
  - `dashboard/dashboard_data_aging_biomarkers.json`
- Curated manifest:
  - `dashboard/aging_biomarkers_public/manifest.json`
- Disease explorer index:
  - `dashboard/aging_biomarkers_public/disease_index.json`
- Disease explorer detail payloads:
  - `dashboard/aging_biomarkers_public/diseases/*.json`
- UI notes:
  - `docs/public_dashboard_minimalism/README.md`
- Data contract:
  - built from the matched blood rows in `projects/aging_biomarkers/catalog/aging_biomarkers.csv`
  - each manifest row includes the public display name, collection assignment, aging metadata, source-series path, and precomputed public metrics
  - public metrics are precomputed for `pooled`, `female`, and `male`, each in `raw` and `10-90 trimmed` contexts
  - the public dashboard reuses the existing blood detail series files in `dashboard/data/series/*.json` for lazy-loaded chart detail
  - the main public explorer still relies on the healthy-only blood long table in `data/processed/biomarker_long.parquet`
  - the disease explorer rebuilds only the curated public biomarker subset from raw NHANES lab files, then joins `data/processed/participant_health_flags.parquet` so excluded disease cohorts can be compared against the same healthy baseline in matched age bins
- Navigation:
  - `Start Here`
  - `Explore a Biomarker`
  - `Disease Explorer`
  - `What Changes Most?`
  - `Compare Biomarkers`
  - `Blood Age`
  - `About the Data`
- Explore views:
  - `Typical level`
  - `Spread`
  - `Tail shape`
  - `Sex split`
- Disease Explorer:
  - healthy-only main tabs stay unchanged
  - this tab intentionally reintroduces selected disease cohorts (`diabetes`, `hypertension`, `cvd`, `kidney`, `liver`, `cancer`, `asthma`, `thyroid_problem`, `stroke`)
  - per condition, one biomarker can be compared at a time between the healthy baseline and the selected disease cohort using the same pooled/female/male and raw vs `10-90 trimmed` logic
- Blood Age:
  - browser-only calculator tab for the open PhenoAge blood model
  - uses birth date, blood draw date, and 9 routine biomarkers with in-browser unit conversion
  - cites the Levine et al. and Liu et al. NHANES papers directly in the UI
- State model:
  - URL hash restores tab, biomarker, disease condition, disease biomarker, cohort, trim mode, view, spread metric, compare mode, and compare-set selection
- Scope:
  - focused on the curated aging biomarker subset and familiar clinical markers rather than the full blood dashboard inventory
  - main exploration uses the healthy cohort exclusions documented below; disease comparisons live in their own tab so they do not contaminate the healthy-aging views
  - the public-facing page now includes a visible disclaimer that it is for education and research exploration, not diagnosis or medical advice
  - the public-facing page intentionally keeps secondary explanation, methods notes, and caveats behind click-to-open bubbles so the default view stays readable

## Plot modes
- In `Dashboard` analysis view, use:
  - `Plot CV`: CV vs age.
  - `Plot SD`: standard deviation vs age.
  - `Plot Median`: median vs age with:
    - interquartile range (IQR) band (25th-75th percentile)
    - raw scatter sample (age vs value) for the selected biomarker
  - `Plot Skewness`: classic moment skewness vs age (distribution asymmetry per age bin).
  - `Plot Quantile Skewness`: Bowley/Galton quantile skewness vs age, defined as `(Q3 + Q1 - 2*median) / (Q3 - Q1)`.
  - `Full View`: a 2x2 dashboard view that shows `Median`, `Standard deviation`, `CV`, and one selectable skewness panel for the selected biomarker.
    - use the `Full view skew metric` selector to choose either classic `Skewness` or `Quantile skewness`
    - each subplot shows the NHANES Spearman rho with age above the panel
    - in `Both` cohort mode, female and male rho values are shown separately in each subplot
    - if Clalit overlay data exist for that biomarker, Clalit trajectories are included in the median, standard deviation, selected skewness panel, and CV panel without extra rho text
  - All ranking/filtering/scatter/histogram views now compute and use the same age-trend metrics for `CV`, `SD`, `Median`, `Skewness`, and `Quantile skewness`:
    - `n_bins`
    - `Spearman rho`
    - `Spearman p`
    - linear slope
    - linear log-slope when defined
  - `Symmetric Trim Per Tail (%)`: optional robust trimming within each age bin before summary stats are computed (for example 10-90, 20-80, 25-75).
  - Age bins for these summary trajectories run from `20-24` through `80-84`; ages `85+` are excluded from the binned trend calculations.
  - Sex view: `Pooled`, `Female`, `Male`, `Both (Female + Male)`.
    - In sex-specific views, trimming is done within each sex separately (not on pooled male+female values).

## Info tab
- Use `Info & Methods` (top tab) for:
  - analysis scope and filtering
  - healthy cohort definition
  - decline flag criteria
  - interpretation notes for CV/SD/median/skewness/quantile-skewness views

## Healthy Exclusion Rules
- Adults only (`age >= 20`).
- Excluded if any of:
  - pregnancy (`RIDEXPRG == 1`)
  - diagnosed diabetes (`DIQ010 == 1`)
  - diagnosed CVD (`MCQ160B/C/D/E/F == 1`)
  - cancer history (`MCQ220 == 1`)
  - weak/failing kidneys (`KIQ022 == 1`)
  - liver disease history (`MCQ160L == 1`, or newer liver variables `MCQ500/MCQ510A-F == 1`)
- Asthma (`MCQ010`) is tracked in `participant_health_flags.parquet` for survival analyses, but is **not** used as a healthy-cohort exclusion in biomarker dashboard analyses.
- Additional disease flags (for example heart attack, stroke, thyroid, bronchitis, hypertension, overweight, osteoporosis) are tracked for survival analyses and are also **not** used as extra healthy-cohort exclusions in biomarker dashboard analyses.

## Compare tab
- Use `Compare Rankings` (top tab) to compare biomarkers by Spearman trend quickly.
- Controls:
  - statistic: `CV vs age`, `Standard deviation vs age`, `Mean vs age`, `Skewness vs age`, or `Quantile skewness vs age`
  - sort mode: most negative, most positive, or largest absolute Spearman
  - symmetric trim (% per tail), shared with dashboard outlier mode
  - cohort: pooled, female, male, or both
  - top N count
- Visual:
  - horizontal bar chart with hover details (`rho`, `p`, `n_bins`, negative-trend flag, biomarker id)
  - in `Both` cohort mode, female and male bars are shown side-by-side on the same biomarker list
  - blood dashboard includes a `Clalit vs NHANES Agreement` scatter panel; urinary dashboard keeps the panel but shows a placeholder (no urinary Clalit overlay configured)

## Waterfall tab
- Use `Waterfall` (top tab) to inspect age-stratified full-value distributions for one biomarker.
- Controls:
  - biomarker search + selector
  - cohort selector (`Pooled`, `Female`, `Male`)
  - symmetric trim slider shared with the rest of the dashboard
  - minimum `n` per age bin
  - blood only: `Show SR model X side by side`
- Default mode keeps the legacy 10-year waterfall bins (`20-29` through `80-89`, plus `90+` when present).
- When `Show SR model X side by side` is enabled:
  - the waterfall switches to the SR comparison 5-year bins (`20-24` through `80-84`)
  - the left panel shows the selected biomarker
  - the right panel shows the selected cached SR-fit `X` reference
  - the biomarker panel still respects the trim slider, but the SR `X` panel always stays alive-only with no tail clipping
  - the SR panel is pooled-only and is meant for visual diagnosis of the SR Q-Q score, not as a separate sex-specific reference
  - the waterfall SR-fit selector lets you switch between `SR original fit`, `SR alternative fit`, and future registered fits

## SR comparison outputs
- `src/build_sr_comparison.py` writes:
  - `projects/sr_comparison/blood/fits/original/biomarker_qq_summary.csv`
  - `projects/sr_comparison/blood/fits/original/biomarker_qq_detail.csv`
  - `projects/sr_comparison/blood/fits/alternative/biomarker_qq_summary.csv`
  - `projects/sr_comparison/blood/fits/alternative/biomarker_qq_detail.csv`
  - `projects/sr_comparison/blood/dashboard_payload.json`
  - `projects/sr_comparison/blood/run_manifest.json`
- `dashboard_payload.json` now includes:
  - per-biomarker SR Q-Q summaries and detail rows for multiple biomarker trim modes (`0%`, `3%`, `5%`, `10%` per tail)
  - fit-keyed summary/detail maps so the dashboard can switch between multiple SR reference fits without duplicating biomarker series payloads
  - `sr_fit_manifest`, which lists the available SR fits and the default fit key
  - `sr_reference_bins_by_fit`, `sr_waterfall_references`, and `sr_rank_references` for the shared fit-level SR caches

## Filter Tests tab
- Use `Filter Tests` (top tab) to build logical clause filters over trend metrics and return matching tests.
- Controls:
  - sex group: `Female`, `Male`, or `Both (Female + Male)`
  - symmetric trim slider (shared globally)
  - `Clause Constructor` for creating and editing reusable clauses
  - `Logical Stage` for nested expressions such as `C1 AND (C2 OR C3)`
  - stage tokens can be inserted by button and reordered/removed directly on the stage
  - `Saved Statement Bank` to save a full logical stage plus its clause library in browser storage, then load/update/delete it later
  - searchable multi-select test picker with quick actions (`Select all visible`, `Core clinical only`, `Clear selection`)
  - optional include/exclude environmental-toxicant assays
- Clause fields:
  - statistic: `CV`, `Standard deviation`, `Mean`, `Skewness`
  - metric: `Spearman rho`, `Spearman p-value`, `n bins`, `Slope/year`, `Slope log/year`
  - comparator: `<`, `<=`, `>`, `>=`, `==`, `!=`
  - numeric threshold
- Output:
  - matching biomarker table for the active specimen page (blood or urinary), with per-clause pass/value columns
  - per-row plot toggle checkbox so filtered tests can be included or removed from the overlay plot without changing the filter
  - `Export CSV` downloads the current filtered table, including the clause definitions, clause match flags, and clause metric values used for each returned biomarker
  - `Filtered Test Overlay` plot can switch between:
    - median normalized to the nearest age-30 bin
    - CV normalized to the nearest age-30 bin
    - standard deviation normalized to the nearest age-30 bin
    - skewness normalized to the nearest age-30 bin
    - quantile skewness normalized to the nearest age-30 bin
    - `Full view (all metrics)` to show median, standard deviation, CV, and one selectable skewness panel together for the current filtered set
    - use the `Full view skew metric` selector to choose classic skewness or quantile skewness in that 2x2 overlay view
  - in `Both` cohort mode, overlay traces are split into female and male trajectories for each selected biomarker
  - clicking a result opens that biomarker in `Dashboard`

## Scatter tab
- Use `Scatter Plot` (top tab) to compare biomarkers in 2D across trend metrics.
- Axes:
  - choose X and Y independently from `CV vs age`, `Standard deviation vs age`, `Mean vs age`, `Skewness vs age`, or `Quantile skewness vs age` (each axis uses Spearman rho of age vs selected statistic)
- Controls:
  - cohort: `Pooled`, `Female`, `Male`, `Both (Female + Male)`
  - symmetric trim slider (shared with Dashboard/Compare)
  - include/exclude environmental-toxicant assays
  - category multi-select with combinations (for example `Routine - CBC` + `Specialized - Inflammatory`)
  - quick category actions: select all, core-only preset, clear
  - `Show labels` toggle button to annotate each dot with biomarker name directly on the chart
- Visual behavior:
  - each point is one biomarker
  - in `Both` cohort mode, female and male are separate point layers (red/blue)
  - clicking a point opens that biomarker in the main Dashboard tab

## Histograms tab
- Use `Histograms` (top tab) to see the distribution of Spearman rho values across biomarkers.
- Metric:
  - choose one metric at a time: `CV vs age`, `Standard deviation vs age`, `Mean vs age`, `Skewness vs age`, or `Quantile skewness vs age`
- Controls:
  - cohort: `Pooled`, `Female`, `Male`, `Both (Female + Male)`
  - symmetric trim slider (shared with Dashboard/Compare/Scatter)
  - include/exclude environmental-toxicant assays
  - category multi-select with combinations and quick actions (all/core/clear)
- Visual behavior:
  - histogram x-axis is fixed to Spearman range `[-1, 1]`
  - in `Both` cohort mode, female and male histograms are overlaid
  - annotation shows counts of negative/positive rho values so you can quickly see how many biomarkers decrease or increase with age

## Waterfall tab
- Use `Waterfall` (top tab) to inspect one biomarker’s full value distribution across age strata.
- Controls:
  - biomarker search + biomarker selector
  - cohort: `Pooled`, `Female`, `Male`
  - symmetric trim slider (shared with all tabs)
  - minimum `n` per age bin (bins below threshold are hidden)
- Visual behavior:
  - age is stratified into 10-year bins (`20-29`, `30-39`, ..., `90+`)
  - each age-bin density is drawn as a stacked ridgeline/waterfall profile
  - profile is segmented into quartiles (`Q1`..`Q4`) using quartile color bands
  - for biomarkers with nonnegative observed values, the waterfall x-axis is floored at `0` (prevents KDE tails from visually extending below zero)
  - hover shows per-bin `n`, `Q1`, `Median`, and `Q3`

## Trend metrics in rankings
- Spearman is computed between age-bin midpoint and the selected statistic (`CV`, `Standard deviation`, `Mean`, or `Skewness`) after the selected trim mode.
- `Negative trend` flag is true when:
  - `n_bins >= 5`
  - `spearman_rho < 0`
  - `spearman_p < 0.05`
  - linear slope of the selected statistic vs age is negative
- The legacy CV-specific `decline` metric is preserved for CV and aligns with negative-trend behavior in CV mode.

## Pooling and variable screening
- Biomarkers are pooled across NHANES cycles/files by normalized test name (not only by code name).
- Example: different code names for the same test (e.g., `LBX*` and `SST*`) are merged when they refer to the same analyte/test.
- Compatible unit variants are converted and pooled, including:
  - simple scale changes (for example `g/dL` vs `g/L`, `ng/mL` vs `ug/L`)
  - paired NHANES conventional/SI fields inferred from the raw XPT data (for example `mg/dL` vs `mmol/L`, `pg/mL` vs `pmol/L`)
  - missing-unit aliases when NHANES reuses the same analyte label without printing the unit in some files
- Known blood-test alias merges now include cases such as:
  - `Albumin` + `Albumin (g/dL)` + `Albumin (g/L)`
  - `Creatinine (mg/dL)` + `Creatinine (umol/L)`
  - `Triglyceride` + `Triglycerides`
  - `Vitamin B12` + `Vitamin B12, serum`
  - `Folate, serum` + `Serum folate`
  - `Cholesterol` + `Total Cholesterol`
- Intentionally incompatible representations remain separate (for example `%` fatty-acid composition vs concentration units, and serology `IgG` vs `IgM` assays).
- Non-analytic fields are removed before analysis:
  - comment/result code fields
  - questionnaire-style text fields
  - duplicate/technical assay fields
  - alternate equation-specific LDL calculation outputs that duplicate the plain LDL biomarker (`Friedewald`, `Martin-Hopkins`, `NIH equation 2`, and explicit `LBDLDLN = ...` formula labels)
  - low-information categorical numeric fields
- CRP/hs-CRP are included as pooled blood biomarkers.
- A derived blood biomarker is added during dashboard build:
  - `Neutrophil-to-lymphocyte ratio (NLR)` = `Segmented neutrophils num (1000 cell/uL)` / `Lymphocyte number (1000 cells/uL)`
  - computed per participant per cycle from the long table, then passed through all dashboard analyses (pooled, sex-specific, and all trim modes).
- Screening audit is written to:
  - `data/processed/variable_screening_summary.csv`
  - `data/processed/urine/variable_screening_summary.csv`
- Duplicate-merge documentation is written to:
  - `data/processed/duplicate_merge_map.csv`
  - `data/processed/duplicate_merge_summary.csv`
  - `data/processed/duplicate_merge_report.md`
- Blood duplicate pooling now has an explicit downstream compatibility check for Clalit overlays:
  - `data/clalit_mapping.json` should target the post-merge pooled NHANES biomarker IDs, not the legacy `name__unit` IDs
  - Clalit mappings can optionally include a `scale_factor` when Clalit is stored in a unit that was merged into a different pooled NHANES display unit (current example: free T4 `pmol/L` -> `ng/dL`)
- Pooled catalog is written to:
  - `data/processed/biomarker_catalog.parquet`
  - `data/processed/urine/biomarker_catalog.parquet`

## SR comparison tab
- `SR comparison` is available on the blood dashboard only.
- Purpose:
  - compare pooled NHANES blood biomarkers against the SR-model `X` distribution by age-bin shape, not by shared units
  - let the user switch between multiple registered SR reference fits
  - support two comparison methods inside one tab:
    - `QQ / Shape`
    - `Rank-Wasserstein`
- Build inputs:
  - NHANES blood long table: `data/processed/biomarker_long.parquet`
  - SR fit registry: `projects/sr_fits/fit_registry.json`
  - external SR code rooted at the configured SR script / SR package paths stored in that registry
  - cached local output root: `projects/sr_comparison/blood/`
- Analysis contract:
  - age bins are `20-24` through `80-84`
  - SR `X` is sampled from the alive-only simulation cohort at each age-bin midpoint and is never tail-trimmed
  - pooled blood biomarkers only in v1
  - `QQ / Shape` mode:
    - NHANES biomarker values are evaluated under four symmetric tail-trim modes: `0%`, `3%`, `5%`, and `10%` per tail
    - per-bin fit is \( \text{biomarker quantile} = m \cdot \text{SR quantile} + c \)
    - z-scored Wasserstein distance is computed per age bin after z-scoring SR and biomarker values separately within that bin
  - `Rank-Wasserstein` mode:
    - trimming options are `0%`, `3%`, `5%`, and `10%` per tail
    - trimming is applied within each age bin first, then the trimmed values from all age bins are pooled and converted to normalized ranks from \(1\) to \(100\)
    - each age-bin comparison uses the subset of pooled normalized ranks that belong to that age bin
    - ties are broken with a deterministic seeded random ordering inside each exact-value tie group
    - Wasserstein distance is then computed on those normalized-rank distributions for each age bin
- UI behavior:
  - one method switch between `QQ / Shape` and `Rank-Wasserstein`
  - one SR-fit selector to switch between `SR original fit`, `SR alternative fit`, and future registered fits
  - in `QQ / Shape` mode:
    - one selected-bin Q-Q plot
    - one `R²(age)` plot
    - one coefficient plot for `m(age)` and `c(age)`
  - in `Rank-Wasserstein` mode:
    - one selected-bin percentile-rank CDF overlay
    - one `Rank-Wasserstein(age)` plot
    - one placeholder panel noting that rank mode does not use `m` or `c`
  - searchable multi-select SR category picker with `Select all visible`, `Core clinical only`, and `Clear selection`
  - `Specialized - Nutritional/Vitamin` is excluded from the SR comparison UI entirely
  - main sortable biomarker table changes columns and default sort by method:
    - `QQ / Shape`: selected-bin and aggregate `R²` and z-Wasserstein values, plus `mean/SD m`, `mean/SD c`, and valid-bin count
    - `Rank-Wasserstein`: mean/current/min/median rank-Wasserstein plus valid-bin count
  - secondary per-bin detail table changes by method:
    - `QQ / Shape`: `R²`, z-Wasserstein, `m`, `c`, and quartiles
    - `Rank-Wasserstein`: rank-Wasserstein, NHANES \(n\), and SR \(n\)
  - the SR trim selector changes behavior by method:
    - `QQ / Shape`: trims biomarker tails within the selected age bin comparison
    - `Rank-Wasserstein`: trims within each age bin before pooled percentile-ranking
- Interpretation:
  - `QQ / Shape`:
    - high `R²` means the biomarker and SR model share a similar distribution shape in that age bin
    - lower z-Wasserstein means the z-scored biomarker and SR distributions are closer overall, especially in tail placement
    - `m` and `c` capture age-dependent scaling and offset differences even when shape agreement remains strong
  - `Rank-Wasserstein`:
    - lower rank-Wasserstein means the biomarker age bin occupies a similar relative percentile-rank region as the SR age bin inside each variable's own pooled lifespan distribution

## Median mode interpretation
- `Plot Median` displays the age-binned median and IQR band (25th-75th percentile).
- Trimming is symmetric by tail and is applied within each age bin before computing plotted summaries and trend metrics.
- Raw sampled points are shown in median view to visualize spread and outliers.

## Tests
```bash
python3 -m unittest discover -s tests -p 'test_*.py'
```

## Dashboard validation workflow
- Regenerate both dashboards after UI or metric-surface changes:
```bash
python3 src/build_sr_comparison.py --out-root projects/sr_comparison/blood
python3 src/build_dashboard.py
```
- To enable basic public traffic analytics on the audience-facing dashboard, rebuild with:
```bash
AGING_PUBLIC_GA4_ID=G-XXXXXXXXXX python3 src/build_dashboard.py
```
- The public dashboard sends GA4 pageviews plus custom events for tab switches, biomarker opens, disease-condition changes, compare-set additions, chart saves, and Blood Age calculations.
- Serve locally for browser validation:
```bash
python3 -m http.server 8765 --directory .
```
- Playwright/manual validation checklist used for the current redesign:
  - blood tabs: `#dashboard`, `#compare`, `#filter-tests`, `#scatter`, `#hist`, `#waterfall`, `#sr-comparison`, `#info`
  - urinary spot-checks with preserved hash navigation (for example `urinary.html#compare` and switch back to `index.html#compare`)
  - public dashboard tabs: `aging_biomarkers_dashboard.html#tab=start`, `explore`, `disease`, `rankings`, `surprising`, `compare`, `calculator`, and `about` states via hash restore
  - representative interactions:
    - dashboard mode, cohort, and trim changes
    - compare statistic, sort, cohort, and top-N changes
    - scatter X/Y metrics, category filtering, and label toggle
    - histogram metric, cohort, and category filtering
    - waterfall biomarker search/selection, cohort, and minimum-n changes
    - SR comparison method switch, trim changes, age-bin slider, ranking sort changes, biomarker row switching, and urine fallback from `#sr-comparison` to `#dashboard`
    - filter-tests clause editing and execution
    - public dashboard biomarker search, disease-condition switching, disease starter chips, pooled/female/male switching, raw vs `5-95 trimmed` vs `10-90 trimmed`, surprise-card jumps, compare-set add/remove, Blood Age calculation, and chart export
  - keyboard focus visibility across specimen links, top tabs, and form controls
  - mobile-width spot checks for stacked navigation and control layouts
- Example screenshots from the current pass were written to `output/playwright/`.

## Clalit Data Integration
- We map Israeli Clalit clinical data to NHANES biomarkers to allow direct visual overlay of age-trajectory statistics on the dashboard.
- Clalit overlays are currently configured for the blood dashboard (`dashboard/index.html`) and not for urinary dashboard (`dashboard/urinary.html`).
- Overall availability and mapped linkage between tests are tracked in `data/data_availability.csv`.
- Mapping scripts (using Jaccard string similarity filtering) live in `scripts/match_clalit_nhanes.py` and manual overrides loop is in `scripts/match_clalit_nhanes_round2.py`.
- The final JSON index connecting Clalit test keys to NHANES IDs is read from `data/clalit_mapping.json`.
- `data/clalit_mapping.json` supports mapping one Clalit test code to multiple NHANES biomarker IDs (JSON array) when the same analyte appears under multiple NHANES pooled IDs (for example, CRP aliases).
- `data/clalit_mapping.json` also supports object targets with `biomarker_id` and `scale_factor`, which are applied before plotting when the Clalit source unit differs from the pooled NHANES unit.
- A post-build audit of mapping coverage/validity is written to `output/clalit_mapping_audit.csv` (`mapped_valid` vs `unmapped`).
- `python3 src/build_clalit_quartiles.py` writes `data/clalit/clalit_quartiles.csv`.
  - The export keeps one row per `sex x test x 5-year age_bin x scale_type`.
  - Age bins follow the NHANES 5-year convention through `80-84`, then continue as `85-89`, `90-94`, and `95-99`.
  - Quartiles are derived from weighted mixtures of the single-year ridgeline densities, using the per-age `n` from `data/clalit/females_all_statistics.csv` and `data/clalit/males_all_statistics.csv` when available, and equal-by-age fallback only when the statistics tables have no weights for that test.
  - `scale_type=regular` stores quartiles on the raw biomarker scale; `scale_type=log` stores quartiles on the log scale and also includes `raw_q*` back-transformed to the original biomarker scale.
  - The file also carries `unit`, `unit_source`, and `unit_confidence`; a small number of tests still have intentionally blank units where the current repo data do not support a defensible Clalit-specific label.
- The search dropdown uses a browser `datalist`; if native browser history is enabled it can appear as a second block beneath the live biomarker matches, so dashboard search inputs now explicitly disable native autocomplete to avoid stale pre-merge biomarker suggestions.
