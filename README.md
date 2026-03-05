# nhanes-biomarker-dashboard

Interactive explorer for age-related blood and urinary biomarker trajectories in the NHANES dataset.

This project builds static web dashboards where users can search biomarkers, compare trends across age and sex, and inspect ranking metrics across hundreds of tests.

Documentation rule: when dashboard features/metrics change, update this README in the same commit.

## Scripts
- `src/discover_nhanes.py` discovers laboratory variable metadata and tags both blood and urinary candidates in the manifest (`is_blood_candidate`, `is_urine_candidate`).
- `src/download_nhanes.py` downloads required NHANES XPT files (lab + demographics + questionnaire modules including `DIQ/MCQ/KIQ/BPQ/OSQ/VIQ/PFQ/HUQ`), with candidate selection controlled by `--candidate-column`.
- `src/build_analysis_dataset.py` creates harmonized healthy-adult biomarker long data, with candidate selection controlled by `--candidate-column`.
- `src/compute_cv_metrics.py` computes CV-by-age bins and decline metrics.
- `src/build_dashboard.py` builds both static interactive HTML dashboards: blood (`dashboard/index.html`) and urinary (`dashboard/urinary.html`).
- `src/plot_km_kidney_liver.py` generates Kaplan-Meier survival plots for broad disease cohorts vs full cohort using linked mortality files (follow-up and age-timescale outputs).
- `src/cluster_km_shapes.py` clusters disease KM curve shapes with multiple distances and algorithms, and writes visual diagnostics to `output/km_shape_clustering/`.
- `src/fpca_km_shapes.py` runs functional-PCA style decomposition of disease KM curves, clusters in fPCA score space, and writes outputs to `output/fPCA/`.

## Run Order
```bash
python3 src/discover_nhanes.py --component Laboratory --verify-urls
python3 src/download_nhanes.py --manifest data/processed/lab_variable_manifest.parquet
python3 src/build_analysis_dataset.py --raw data/raw --manifest data/processed/lab_variable_manifest.parquet --out data/processed
python3 src/compute_cv_metrics.py --in data/processed/biomarker_long.parquet --out data/processed
python3 src/download_nhanes.py --manifest data/processed/lab_variable_manifest.parquet --candidate-column is_urine_candidate --download-manifest data/processed/download_manifest_urine.csv
python3 src/build_analysis_dataset.py --raw data/raw --manifest data/processed/lab_variable_manifest.parquet --out data/processed/urine --candidate-column is_urine_candidate
python3 src/compute_cv_metrics.py --in data/processed/urine/biomarker_long.parquet --out data/processed/urine
python3 src/build_dashboard.py
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
- Online:
  - Open the GitHub Pages site (if enabled in your repo settings):
  - Blood: `https://<github-username>.github.io/<repo-name>/dashboard/index.html`
  - Urinary: `https://<github-username>.github.io/<repo-name>/dashboard/urinary.html`

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
- Per-biomarker point series are stored in:
  - `dashboard/data/series/*.json`
  - `dashboard/data_urine/series/*.json`
- Series are fetched ad hoc only when a biomarker is selected/searched.

## Plot modes
- In `Dashboard` analysis view, use:
  - `Plot CV`: CV vs age.
  - `Plot SD`: standard deviation vs age.
  - `Plot Median`: median vs age with:
    - interquartile range (IQR) band (25th-75th percentile)
    - raw scatter sample (age vs value) for the selected biomarker
  - `Plot Skewness`: skewness vs age (distribution asymmetry per age bin).
  - `Symmetric Trim Per Tail (%)`: optional robust trimming within each age bin before summary stats are computed (for example 10-90, 20-80, 25-75).
  - Sex view: `Pooled`, `Female`, `Male`, `Both (Female + Male)`.
    - In sex-specific views, trimming is done within each sex separately (not on pooled male+female values).

## Info tab
- Use `Info & Methods` (top tab) for:
  - analysis scope and filtering
  - healthy cohort definition
  - decline flag criteria
  - interpretation notes for CV/SD/median/skewness views

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
  - statistic: `CV vs age`, `Standard deviation vs age`, `Mean vs age`, or `Skewness vs age`
  - sort mode: most negative, most positive, or largest absolute Spearman
  - symmetric trim (% per tail), shared with dashboard outlier mode
  - cohort: pooled, female, male, or both
  - top N count
- Visual:
  - horizontal bar chart with hover details (`rho`, `p`, `n_bins`, negative-trend flag, biomarker id)
  - in `Both` cohort mode, female and male bars are shown side-by-side on the same biomarker list
  - blood dashboard includes a `Clalit vs NHANES Agreement` scatter panel; urinary dashboard keeps the panel but shows a placeholder (no urinary Clalit overlay configured)

## Filter Tests tab
- Use `Filter Tests` (top tab) to build logical clause filters over trend metrics and return matching tests.
- Controls:
  - sex group: `Female`, `Male`, or `Both (Female + Male)`
  - symmetric trim slider (shared globally)
  - logical combiner: `AND` or `OR`
  - add/remove any number of clauses
  - optional include/exclude environmental-toxicant assays
- Clause fields:
  - statistic: `CV`, `Standard deviation`, `Mean`, `Skewness`
  - metric: `Spearman rho`, `Spearman p-value`, `n bins`, `Slope/year`, `Slope log/year`
  - comparator: `<`, `<=`, `>`, `>=`, `==`, `!=`
  - numeric threshold
- Output:
  - matching biomarker table for the active specimen page (blood or urinary), with clause values per biomarker
  - clicking a result opens that biomarker in `Dashboard`

## Scatter tab
- Use `Scatter Plot` (top tab) to compare biomarkers in 2D across trend metrics.
- Axes:
  - choose X and Y independently from `CV vs age`, `Standard deviation vs age`, `Mean vs age`, `Skewness vs age` (each axis uses Spearman rho of age vs selected statistic)
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
  - choose one metric at a time: `CV vs age`, `Standard deviation vs age`, `Mean vs age`, or `Skewness vs age`
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
- Compatible unit variants are converted and pooled (e.g., g/dL and g/L for albumin); incompatible unit systems remain separate entries.
- Non-analytic fields are removed before analysis:
  - comment/result code fields
  - questionnaire-style text fields
  - duplicate/technical assay fields
  - low-information categorical numeric fields
- CRP/hs-CRP are included as pooled blood biomarkers.
- A derived blood biomarker is added during dashboard build:
  - `Neutrophil-to-lymphocyte ratio (NLR)` = `Segmented neutrophils num (1000 cell/uL)` / `Lymphocyte number (1000 cells/uL)`
  - computed per participant per cycle from the long table, then passed through all dashboard analyses (pooled, sex-specific, and all trim modes).
- Screening audit is written to:
  - `data/processed/variable_screening_summary.csv`
  - `data/processed/urine/variable_screening_summary.csv`
- Pooled catalog is written to:
  - `data/processed/biomarker_catalog.parquet`
  - `data/processed/urine/biomarker_catalog.parquet`

## Median mode interpretation
- `Plot Median` displays the age-binned median and IQR band (25th-75th percentile).
- Trimming is symmetric by tail and is applied within each age bin before computing plotted summaries and trend metrics.
- Raw sampled points are shown in median view to visualize spread and outliers.

## Tests
```bash
python3 -m unittest discover -s tests -p 'test_*.py'
```

## Clalit Data Integration
- We map Israeli Clalit clinical data to NHANES biomarkers to allow direct visual overlay of age-trajectory statistics on the dashboard.
- Clalit overlays are currently configured for the blood dashboard (`dashboard/index.html`) and not for urinary dashboard (`dashboard/urinary.html`).
- Overall availability and mapped linkage between tests are tracked in `data/data_availability.csv`.
- Mapping scripts (using Jaccard string similarity filtering) live in `scripts/match_clalit_nhanes.py` and manual overrides loop is in `scripts/match_clalit_nhanes_round2.py`.
- The final JSON index connecting Clalit test keys to NHANES IDs is read from `data/clalit_mapping.json`.
- `data/clalit_mapping.json` supports mapping one Clalit test code to multiple NHANES biomarker IDs (JSON array) when the same analyte appears under multiple NHANES pooled IDs (for example, CRP aliases).
- A post-build audit of mapping coverage/validity is written to `output/clalit_mapping_audit.csv` (`mapped_valid` vs `unmapped`).
