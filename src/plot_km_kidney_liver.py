#!/usr/bin/env python3
"""Kaplan-Meier survival curves for chronic-condition cohorts vs full NHANES cohort.

Uses NHANES linked mortality public-use files (2019 linkage release) and the
processed participant health flags table.
"""

from __future__ import annotations

import argparse
import math
import pickle
import re
from pathlib import Path
from typing import Iterable

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import requests
from lifelines import KaplanMeierFitter
from lifelines.exceptions import StatError
from matplotlib.lines import Line2D

from nhanes_common import ensure_dir


MORTALITY_FILES = [
    "NHANES_1999_2000_MORT_2019_PUBLIC.dat",
    "NHANES_2001_2002_MORT_2019_PUBLIC.dat",
    "NHANES_2003_2004_MORT_2019_PUBLIC.dat",
    "NHANES_2005_2006_MORT_2019_PUBLIC.dat",
    "NHANES_2007_2008_MORT_2019_PUBLIC.dat",
    "NHANES_2009_2010_MORT_2019_PUBLIC.dat",
    "NHANES_2011_2012_MORT_2019_PUBLIC.dat",
    "NHANES_2013_2014_MORT_2019_PUBLIC.dat",
    "NHANES_2015_2016_MORT_2019_PUBLIC.dat",
    "NHANES_2017_2018_MORT_2019_PUBLIC.dat",
]
MORT_BASE_URL = "https://ftp.cdc.gov/pub/Health_Statistics/NCHS/datalinkage/linked_mortality/"
COHORT_COLORS = {
    "full": "#111827",
    "diabetes": "#ea580c",
    "kidney": "#2563eb",
    "liver": "#dc2626",
    "asthma": "#059669",
}
DISEASE_META = {
    "diabetes": ("Diabetes", "DIQ010"),
    "asthma": ("Asthma", "MCQ010"),
    "kidney": ("Kidney disease", "KIQ022"),
    "liver": ("Liver disease", "MCQ160L/MCQ500/MCQ510A-F"),
    "cancer": ("Cancer/malignancy", "MCQ220"),
    "cvd": ("Any major CVD", "MCQ160B-F composite"),
    "hypertension": ("Hypertension", "BPQ020"),
    "osteoporosis": ("Osteoporosis", "OSQ060"),
    "cataract_operation": ("Cataract operation", "VIQ070"),
    "adl_disability": ("ADL disability", "ADDLDIS/ADLDIS"),
    "iadl_disability": ("IADL disability", "IADLDIS"),
    "arthritis": ("Arthritis", "MCQ160A"),
    "heart_failure": ("Congestive heart failure", "MCQ160B"),
    "coronary_heart_disease": ("Coronary heart disease", "MCQ160C"),
    "angina": ("Angina", "MCQ160D"),
    "heart_attack": ("Heart attack", "MCQ160E"),
    "stroke": ("Stroke", "MCQ160F"),
    "emphysema": ("Emphysema", "MCQ160G"),
    "overweight": ("Overweight", "MCQ160J"),
    "chronic_bronchitis": ("Chronic bronchitis", "MCQ160K"),
    "liver_condition": ("Liver condition", "MCQ160L"),
    "thyroid_problem": ("Thyroid problem", "MCQ160M"),
    "still_chronic_bronchitis": ("Still have chronic bronchitis", "MCQ170K"),
    "still_liver_condition": ("Still have liver condition", "MCQ170L"),
    "still_thyroid_problem": ("Still have thyroid problem", "MCQ170M"),
}
NON_DISEASE_COLUMNS = {
    "seqn",
    "cycle_start_year",
    "age_years",
    "sex",
    "pregnant",
    "healthy_flag",
    "exclusion_reason",
}
AGING_ROOT = Path(__file__).resolve().parents[2]
SR_UNDERLAY_CACHE_PKL = AGING_ROOT / "python/notebooks/thresholds, noise/saved_results/sr_usa_2019_steepness_longevity_curves.pkl"
SR_UNDERLAY_SOURCE_PKL = AGING_ROOT / "python/notebooks/thresholds, noise/saved_results/param_variation_results_usa_2019.pkl"
SR_PARAM_ORDER = ("eta", "beta", "kappa", "epsilon", "Xc")
SR_PARAM_COLORS = {
    "eta": "blue",
    "beta": "green",
    "kappa": "grey",
    "epsilon": "orange",
    "Xc": "purple",
}


def _km_time_at_survival_prob(kmf: KaplanMeierFitter, target_survival: float) -> float:
    """Return first age where S(age) <= target_survival via linear interpolation."""
    surv = kmf.survival_function_.iloc[:, 0]
    times = surv.index.to_numpy(dtype=float)
    probs = surv.to_numpy(dtype=float)
    hit = np.where(probs <= target_survival)[0]
    if len(hit) == 0:
        return float("nan")
    i = int(hit[0])
    if i == 0:
        return float(times[0])
    t0, t1 = float(times[i - 1]), float(times[i])
    s0, s1 = float(probs[i - 1]), float(probs[i])
    if s1 == s0:
        return t1
    w = (s0 - target_survival) / (s0 - s1)
    return t0 + w * (t1 - t0)


def _is_true_mask(s: pd.Series) -> pd.Series:
    if str(s.dtype) in {"bool", "boolean"}:
        return s.fillna(False).astype(bool)
    return pd.to_numeric(s, errors="coerce").eq(1)


def _disease_label(col: str) -> tuple[str, str]:
    if col in DISEASE_META:
        return DISEASE_META[col]
    m = re.fullmatch(r"(mcq\d+[a-z]?)_condition", str(col).lower())
    if m:
        code = m.group(1).upper()
        return (f"{code} condition", code)
    return (str(col).replace("_", " ").strip().title(), "")


def _km_summary_row(kmf: KaplanMeierFitter, cohort_key: str, cohort: str, code: str, n: int, deaths: int) -> dict:
    median_age = float(_km_time_at_survival_prob(kmf, 0.5))
    q1_age = float(_km_time_at_survival_prob(kmf, 0.75))
    q3_age = float(_km_time_at_survival_prob(kmf, 0.25))
    iqr_age = q3_age - q1_age if np.isfinite(q1_age) and np.isfinite(q3_age) else float("nan")
    steepness = (
        median_age / iqr_age
        if np.isfinite(median_age) and np.isfinite(iqr_age) and iqr_age > 0
        else float("nan")
    )
    return {
        "cohort_key": cohort_key,
        "cohort": cohort,
        "code": code,
        "median_age_years": median_age,
        "q1_age_years": q1_age,
        "q3_age_years": q3_age,
        "iqr_age_years": iqr_age,
        "steepness_median_over_iqr": steepness,
        "n": int(n),
        "deaths": int(deaths),
    }


def to_num(s: pd.Series) -> pd.Series:
    return pd.to_numeric(s, errors="coerce")


def download_mortality_files(out_dir: Path, files: Iterable[str]) -> None:
    ensure_dir(out_dir)
    for fn in files:
        p = out_dir / fn
        if p.exists() and p.stat().st_size > 0:
            continue
        r = requests.get(MORT_BASE_URL + fn, timeout=120)
        r.raise_for_status()
        p.write_bytes(r.content)


def load_mortality(mort_dir: Path, files: Iterable[str]) -> pd.DataFrame:
    rows = []
    # Based on NCHS Stata read-in program for NHANES 2019 linkage:
    # seqn 1-6, eligstat 15, mortstat 16, ucod 17-19, diabetes 20,
    # hyperten 21, permth_int 43-45, permth_exm 46-48.
    colspecs = [(0, 6), (14, 15), (15, 16), (16, 19), (19, 20), (20, 21), (42, 45), (45, 48)]
    names = ["seqn", "eligstat", "mortstat", "ucod_leading", "diabetes_mcod", "hyperten_mcod", "permth_int", "permth_exm"]
    for fn in files:
        p = mort_dir / fn
        m = pd.read_fwf(p, colspecs=colspecs, names=names, dtype=str)
        m["seqn"] = to_num(m["seqn"]).astype("Int64")
        for c in names[1:]:
            m[c] = to_num(m[c])
        cycle_start = int(fn.split("_")[1])
        m["cycle_start_year"] = cycle_start
        rows.append(m)
    return pd.concat(rows, ignore_index=True)


def _fit_age_km(sub: pd.DataFrame, label: str) -> KaplanMeierFitter | None:
    kmf = KaplanMeierFitter()
    try:
        kmf.fit(
            durations=sub["end_age"],
            event_observed=sub["event"],
            entry=sub["entry_age"],
            label=label,
        )
    except StatError:
        return None
    return kmf


def _collect_disease_columns(part: pd.DataFrame) -> list[str]:
    cols = []
    for c in part.columns:
        if c in NON_DISEASE_COLUMNS:
            continue
        s = part[c]
        mask = _is_true_mask(s)
        if int(mask.sum()) <= 0:
            continue
        cols.append(c)
    return sorted(set(cols), key=lambda c: (_disease_label(c)[0].lower(), c))


def _build_sr_underlay_curves(
    results: dict,
    from_t: int,
    steepness_metric: str,
    longevity_metric: str,
    ignore_kappa: bool = True,
    include_h_ext: bool = True,
) -> dict:
    baseline = results.get("baseline", {}).get(from_t)
    if baseline is None:
        raise KeyError(f"Missing SR baseline for from_t={from_t}")

    baseline_steep = baseline.get(steepness_metric)
    baseline_longevity = baseline.get(longevity_metric)
    if baseline_steep is None or baseline_longevity is None:
        raise KeyError(
            f"Missing SR baseline metrics for {steepness_metric}/{longevity_metric} at from_t={from_t}"
        )
    if baseline_steep == 0 or baseline_longevity == 0:
        raise ValueError("SR baseline metric is zero; cannot normalize underlay curves.")

    curves: dict[str, list[tuple[float, float, float]]] = {}
    params = [p for p in SR_PARAM_ORDER if not (ignore_kappa and p == "kappa")]
    for param in params:
        param_results = results.get(param, {})
        points: list[tuple[float, float, float]] = []
        for factor in sorted(param_results):
            by_age = param_results[factor]
            if from_t not in by_age:
                continue
            vals = by_age[from_t]
            steep = vals.get(steepness_metric)
            longevity = vals.get(longevity_metric)
            if steep is None or longevity is None:
                continue
            if not (np.isfinite(steep) and np.isfinite(longevity)):
                continue
            points.append((float(factor), float(longevity / baseline_longevity), float(steep / baseline_steep)))
        if points:
            curves[param] = points

    if include_h_ext and "h_ext" in results:
        h_points: list[tuple[float, float, float]] = []
        for h_val in sorted(results["h_ext"]):
            by_age = results["h_ext"][h_val]
            if from_t not in by_age:
                continue
            vals = by_age[from_t]
            steep = vals.get(steepness_metric)
            longevity = vals.get(longevity_metric)
            if steep is None or longevity is None:
                continue
            if not (np.isfinite(steep) and np.isfinite(longevity)):
                continue
            h_points.append((float(h_val), float(longevity / baseline_longevity), float(steep / baseline_steep)))
        if h_points:
            curves["h_ext"] = h_points

    return curves


def _load_sr_underlay_curves(
    from_t: int = 20,
    steepness_metric: str = "steepness_iqr_absolute",
    longevity_metric: str = "t_median_absolute",
    ignore_kappa: bool = True,
    include_h_ext: bool = True,
) -> dict:
    expected_meta = {
        "source_pkl": str(SR_UNDERLAY_SOURCE_PKL),
        "from_t": int(from_t),
        "steepness_metric": steepness_metric,
        "longevity_metric": longevity_metric,
        "ignore_kappa": bool(ignore_kappa),
        "include_h_ext": bool(include_h_ext),
    }

    if SR_UNDERLAY_CACHE_PKL.exists():
        try:
            with open(SR_UNDERLAY_CACHE_PKL, "rb") as f:
                cached = pickle.load(f)
            if cached.get("metadata") == expected_meta and cached.get("curves"):
                return cached["curves"]
        except Exception:
            pass

    with open(SR_UNDERLAY_SOURCE_PKL, "rb") as f:
        results = pickle.load(f)
    curves = _build_sr_underlay_curves(
        results=results,
        from_t=from_t,
        steepness_metric=steepness_metric,
        longevity_metric=longevity_metric,
        ignore_kappa=ignore_kappa,
        include_h_ext=include_h_ext,
    )
    payload = {"metadata": expected_meta, "curves": curves}
    SR_UNDERLAY_CACHE_PKL.parent.mkdir(parents=True, exist_ok=True)
    with open(SR_UNDERLAY_CACHE_PKL, "wb") as f:
        pickle.dump(payload, f)
    return curves


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--participants", default="data/processed/participant_health_flags.parquet")
    ap.add_argument("--mortality-dir", default="data/raw/mortality")
    ap.add_argument("--png-out", default="output/km_kidney_liver_vs_full.png")
    ap.add_argument("--csv-out", default="output/km_kidney_liver_counts.csv")
    ap.add_argument("--png-age-out", default="output/km_kidney_liver_vs_full_by_age.png")
    ap.add_argument("--csv-age-out", default="output/km_kidney_liver_counts_by_age.csv")
    ap.add_argument("--png-asthma-age-out", default="output/km_asthma_vs_full_by_age.png")
    ap.add_argument("--csv-asthma-age-out", default="output/km_asthma_counts_by_age.csv")
    ap.add_argument("--png-all-disease-panels-age-out", default="output/km_all_diseases_vs_full_by_age_panels.png")
    ap.add_argument("--csv-all-disease-age-out", default="output/km_all_diseases_age_summary.csv")
    ap.add_argument("--age-summary-csv-out", default="output/km_kidney_liver_age_summary.csv")
    ap.add_argument("--steepness-png-out", default="output/steepness_longevity_disease.png")
    ap.add_argument("--sr-underlay-from-t", type=int, default=20)
    ap.add_argument("--no-sr-underlay", action="store_true")
    ap.add_argument("--min-disease-n", type=int, default=100)
    args = ap.parse_args()

    part = pd.read_parquet(args.participants)
    required = {"seqn", "cycle_start_year", "age_years"}
    missing = required.difference(set(part.columns))
    if missing:
        raise RuntimeError(f"participant file missing required columns: {sorted(missing)}")

    part = part.copy()
    part = part[part["age_years"] >= 20].copy()

    mort_dir = Path(args.mortality_dir)
    download_mortality_files(mort_dir, MORTALITY_FILES)
    mort = load_mortality(mort_dir, MORTALITY_FILES)

    df = part.merge(mort, on=["seqn", "cycle_start_year"], how="left")
    df = df[df["eligstat"] == 1].copy()
    df["time_months"] = df["permth_int"].where(df["permth_int"].notna(), df["permth_exm"])
    df["event"] = (df["mortstat"] == 1).astype(int)
    df = df[df["time_months"].notna()].copy()

    # Keep legacy follow-up plot (diabetes/kidney/liver vs full) for continuity.
    legacy_followup = [
        ("full", "Full cohort (age>=20, eligstat=1)", pd.Series(True, index=df.index), COHORT_COLORS["full"]),
        (
            "diabetes",
            "Diabetes (DIQ010=1)",
            _is_true_mask(df.get("diabetes", pd.Series(False, index=df.index))),
            COHORT_COLORS["diabetes"],
        ),
        (
            "kidney",
            "Kidney disease (KIQ022=1)",
            _is_true_mask(df.get("kidney", pd.Series(False, index=df.index))),
            COHORT_COLORS["kidney"],
        ),
        (
            "liver",
            "Liver disease (MCQ160L/MCQ500/MCQ510*=1)",
            _is_true_mask(df.get("liver", pd.Series(False, index=df.index))),
            COHORT_COLORS["liver"],
        ),
    ]

    ensure_dir(Path(args.png_out).parent)
    fig, ax = plt.subplots(figsize=(10, 7), dpi=150)
    kmf = KaplanMeierFitter()
    count_rows = []

    for _, label, mask, color in legacy_followup:
        sub = df.loc[mask].copy()
        if sub.empty:
            continue
        kmf.fit(sub["time_months"], event_observed=sub["event"], label=label)
        kmf.plot_survival_function(ax=ax, ci_show=True, color=color, linewidth=2)
        count_rows.append(
            {
                "cohort": label,
                "n": int(len(sub)),
                "deaths": int(sub["event"].sum()),
                "censored": int((sub["event"] == 0).sum()),
                "max_followup_months": float(sub["time_months"].max()),
            }
        )

    ax.set_title("NHANES Kaplan-Meier Survival: Diabetes/Kidney/Liver vs Full Cohort")
    ax.set_xlabel("Follow-up time (months, from interview)")
    ax.set_ylabel("Survival probability")
    ax.set_ylim(0, 1.0)
    ax.grid(alpha=0.25)
    ax.legend(loc="best", frameon=False)
    fig.tight_layout()
    fig.savefig(args.png_out)
    plt.close(fig)

    ensure_dir(Path(args.csv_out).parent)
    pd.DataFrame(count_rows).to_csv(args.csv_out, index=False)

    # Age-timescale KM (delayed entry / left truncation): each participant enters
    # the risk set at interview age and exits at age-at-death or age-at-censoring.
    df_age = df[df["age_years"].notna()].copy()
    df_age["entry_age"] = pd.to_numeric(df_age["age_years"], errors="coerce")
    df_age["end_age"] = df_age["entry_age"] + (pd.to_numeric(df_age["time_months"], errors="coerce") / 12.0)
    df_age = df_age[df_age["entry_age"].notna() & df_age["end_age"].notna()].copy()
    df_age = df_age[df_age["end_age"] > df_age["entry_age"]].copy()

    # Legacy age plot (diabetes/kidney/liver vs full).
    legacy_age = [
        ("full", "Full cohort (age>=20, eligstat=1)", pd.Series(True, index=df_age.index), COHORT_COLORS["full"]),
        (
            "diabetes",
            "Diabetes (DIQ010=1)",
            _is_true_mask(df_age.get("diabetes", pd.Series(False, index=df_age.index))),
            COHORT_COLORS["diabetes"],
        ),
        (
            "kidney",
            "Kidney disease (KIQ022=1)",
            _is_true_mask(df_age.get("kidney", pd.Series(False, index=df_age.index))),
            COHORT_COLORS["kidney"],
        ),
        (
            "liver",
            "Liver disease (MCQ160L/MCQ500/MCQ510*=1)",
            _is_true_mask(df_age.get("liver", pd.Series(False, index=df_age.index))),
            COHORT_COLORS["liver"],
        ),
    ]

    ensure_dir(Path(args.png_age_out).parent)
    fig_age, ax_age = plt.subplots(figsize=(10, 7), dpi=150)
    count_rows_age = []

    for key, label, mask, color in legacy_age:
        sub = df_age.loc[mask].copy()
        if sub.empty:
            continue
        kmf_age = _fit_age_km(sub, label)
        if kmf_age is None:
            continue
        kmf_age.plot_survival_function(ax=ax_age, ci_show=True, color=color, linewidth=2)
        count_rows_age.append(
            {
                "cohort": label,
                "cohort_key": key,
                "n": int(len(sub)),
                "deaths": int(sub["event"].sum()),
                "censored": int((sub["event"] == 0).sum()),
                "min_entry_age_years": float(sub["entry_age"].min()),
                "max_entry_age_years": float(sub["entry_age"].max()),
                "max_end_age_years": float(sub["end_age"].max()),
            }
        )

    ax_age.set_title("NHANES Kaplan-Meier Survival by Age: Diabetes/Kidney/Liver vs Full Cohort")
    ax_age.set_xlabel("Age (years)")
    ax_age.set_ylabel("Survival probability")
    ax_age.set_ylim(0, 1.0)
    ax_age.grid(alpha=0.25)
    ax_age.legend(loc="best", frameon=False)
    fig_age.tight_layout()
    fig_age.savefig(args.png_age_out)
    plt.close(fig_age)

    ensure_dir(Path(args.csv_age_out).parent)
    pd.DataFrame(count_rows_age).to_csv(args.csv_age_out, index=False)

    # All disease cohorts present in participant table.
    disease_cols = _collect_disease_columns(part)
    full_kmf_age = _fit_age_km(df_age, "Full cohort")
    if full_kmf_age is None:
        raise RuntimeError("Failed to fit age-timescale KM for full cohort")
    full_summary = _km_summary_row(
        full_kmf_age,
        cohort_key="full",
        cohort="Full cohort (age>=20, eligstat=1)",
        code="FULL",
        n=len(df_age),
        deaths=int(df_age["event"].sum()),
    )

    disease_rows = []
    cmap = plt.get_cmap("tab20")
    disease_color = {}
    for i, col in enumerate(disease_cols):
        if col in COHORT_COLORS:
            disease_color[col] = COHORT_COLORS[col]
        else:
            disease_color[col] = cmap(i % cmap.N)

    for col in disease_cols:
        mask = _is_true_mask(df_age[col])
        sub = df_age.loc[mask].copy()
        if sub.empty or len(sub) < int(args.min_disease_n):
            continue
        label_text, code_text = _disease_label(col)
        display = f"{label_text} ({code_text})" if code_text else label_text
        kmf_cond = _fit_age_km(sub, display)
        if kmf_cond is None:
            continue
        row = _km_summary_row(
            kmf_cond,
            cohort_key=col,
            cohort=display,
            code=code_text,
            n=len(sub),
            deaths=int(sub["event"].sum()),
        )
        row["color"] = disease_color[col]
        disease_rows.append(row)

    summary_df = pd.DataFrame([full_summary] + disease_rows)
    ensure_dir(Path(args.age_summary_csv_out).parent)
    summary_df.to_csv(args.age_summary_csv_out, index=False)
    ensure_dir(Path(args.csv_all_disease_age_out).parent)
    summary_df.to_csv(args.csv_all_disease_age_out, index=False)

    # Steepness-longevity scatter: all disease cohorts relative to full.
    full_row = summary_df.loc[summary_df["cohort_key"] == "full"]
    if not full_row.empty:
        median_full = float(full_row["median_age_years"].iloc[0])
        steep_full = float(full_row["steepness_median_over_iqr"].iloc[0])
        rel_df = summary_df[summary_df["cohort_key"] != "full"].copy()
        rel_df["relative_median_longevity"] = rel_df["median_age_years"] / median_full
        rel_df["relative_steepness"] = rel_df["steepness_median_over_iqr"] / steep_full
        rel_df = rel_df[
            np.isfinite(rel_df["relative_median_longevity"])
            & np.isfinite(rel_df["relative_steepness"])
        ].copy()

        ensure_dir(Path(args.steepness_png_out).parent)
        fig_rel, ax_rel = plt.subplots(figsize=(12, 9), dpi=170)
        if rel_df.empty:
            x_window = (0.5, 1.5)
            y_window = (0.5, 1.5)
        else:
            x_min = float(rel_df["relative_median_longevity"].min())
            x_max = float(rel_df["relative_median_longevity"].max())
            y_min = float(rel_df["relative_steepness"].min())
            y_max = float(rel_df["relative_steepness"].max())
            x_pad = max(0.03, 0.08 * max(x_max - x_min, 1e-6))
            y_pad = max(0.03, 0.08 * max(y_max - y_min, 1e-6))
            x_window = (x_min - x_pad, x_max + x_pad)
            y_window = (y_min - y_pad, y_max + y_pad)

        if not args.no_sr_underlay:
            try:
                sr_curves = _load_sr_underlay_curves(
                    from_t=int(args.sr_underlay_from_t),
                    steepness_metric="steepness_iqr_absolute",
                    longevity_metric="t_median_absolute",
                    ignore_kappa=True,
                    include_h_ext=True,
                )
                for param in [p for p in SR_PARAM_ORDER if p != "kappa"]:
                    points = sr_curves.get(param, [])
                    if not points:
                        continue
                    filtered = [
                        (pt[1], pt[2])
                        for pt in points
                        if (x_window[0] - 0.05) <= pt[1] <= (x_window[1] + 0.05)
                        and (y_window[0] - 0.05) <= pt[2] <= (y_window[1] + 0.05)
                    ]
                    if len(filtered) < 2:
                        continue
                    xs = [pt[0] for pt in filtered]
                    ys = [pt[1] for pt in filtered]
                    ax_rel.plot(
                        xs,
                        ys,
                        color=SR_PARAM_COLORS.get(param, "#475569"),
                        linewidth=3.0,
                        alpha=0.65,
                        zorder=1,
                    )
                h_points = sr_curves.get("h_ext", [])
                h_filtered = [
                    (pt[1], pt[2])
                    for pt in h_points
                    if (x_window[0] - 0.05) <= pt[1] <= (x_window[1] + 0.05)
                    and (y_window[0] - 0.05) <= pt[2] <= (y_window[1] + 0.05)
                ]
                if len(h_filtered) >= 2:
                    ax_rel.plot(
                        [pt[0] for pt in h_filtered],
                        [pt[1] for pt in h_filtered],
                        color="red",
                        linewidth=2.8,
                        alpha=0.55,
                        zorder=1,
                    )
            except Exception as exc:
                print(f"Warning: failed to draw SR underlay curves: {exc}")

        max_n = max(float(rel_df["n"].max()), 1.0) if not rel_df.empty else 1.0
        for _, row in rel_df.iterrows():
            key = str(row["cohort_key"])
            x = float(row["relative_median_longevity"])
            y = float(row["relative_steepness"])
            dot_size = 120 + 380 * (float(row["n"]) / max_n)
            c = disease_color.get(key, "#0f766e")
            ax_rel.scatter(
                x,
                y,
                s=dot_size,
                color=c,
                alpha=0.88,
                edgecolor="white",
                linewidth=1.0,
                zorder=3,
            )
            short = str(row["cohort"]).split(" (")[0]
            ax_rel.text(x + 0.004, y + 0.004, short, fontsize=8.5, alpha=0.92)

        ax_rel.axvline(1.0, color="#64748b", linestyle="--", linewidth=1.4)
        ax_rel.axhline(1.0, color="#64748b", linestyle="--", linewidth=1.4)
        ax_rel.set_xlim(*x_window)
        ax_rel.set_ylim(*y_window)
        ax_rel.set_title("Disease Cohorts: Relative Longevity vs Relative Steepness")
        ax_rel.set_xlabel("Median lifespan / Full-cohort median lifespan")
        ax_rel.set_ylabel("Steepness (median/IQR) / Full-cohort steepness")
        ax_rel.grid(alpha=0.25, zorder=0)
        fig_rel.tight_layout()
        fig_rel.savefig(args.steepness_png_out)
        plt.close(fig_rel)

    # Multi-panel age-timescale KM: each panel is one condition vs full cohort.
    if disease_rows:
        n_panels = len(disease_rows)
        ncols = 4
        nrows = int(math.ceil(n_panels / ncols))
        ensure_dir(Path(args.png_all_disease_panels_age_out).parent)
        fig_p, axes = plt.subplots(nrows, ncols, figsize=(6.0 * ncols, 3.2 * nrows), dpi=160, sharex=True, sharey=True)
        axes_list = np.array(axes).reshape(-1)

        for i, row in enumerate(disease_rows):
            axp = axes_list[i]
            key = str(row["cohort_key"])
            mask = _is_true_mask(df_age[key])
            sub = df_age.loc[mask].copy()
            if sub.empty:
                axp.axis("off")
                continue

            full_kmf_age.plot_survival_function(
                ax=axp,
                ci_show=True,
                ci_alpha=0.15,
                ci_force_lines=False,
                color=COHORT_COLORS["full"],
                linewidth=1.8,
            )
            kmf_cond = _fit_age_km(sub, row["cohort"])
            if kmf_cond is None:
                axp.axis("off")
                continue
            kmf_cond.plot_survival_function(
                ax=axp,
                ci_show=True,
                ci_alpha=0.15,
                ci_force_lines=False,
                color=disease_color.get(key, "#0f766e"),
                linewidth=2.0,
            )

            title_short = str(row["cohort"]).split(" (")[0]
            axp.set_title(f"{title_short}\nn={int(row['n'])}, deaths={int(row['deaths'])}", fontsize=9)
            axp.set_xlabel("Age")
            axp.set_ylabel("Survival")
            axp.grid(alpha=0.2)
            axp.set_ylim(0, 1)

        for j in range(n_panels, len(axes_list)):
            axes_list[j].axis("off")

        legend_handles = [
            Line2D([0], [0], color=COHORT_COLORS["full"], lw=2, label="Full cohort"),
            Line2D([0], [0], color="#0f766e", lw=2, label="Condition cohort"),
        ]
        fig_p.legend(handles=legend_handles, loc="upper center", ncol=2, frameon=False)
        fig_p.suptitle("NHANES Age-Timescale Kaplan-Meier: Full Cohort vs Condition Cohorts", y=0.995, fontsize=14)
        fig_p.tight_layout(rect=[0, 0, 1, 0.975])
        fig_p.savefig(args.png_all_disease_panels_age_out)
        plt.close(fig_p)

    # Preserve separate asthma-vs-full age plot as explicit convenience output.
    if "asthma" in df_age.columns:
        asthma_mask = _is_true_mask(df_age["asthma"])
        cohorts_asthma_age = [
            ("Full cohort (age>=20, eligstat=1)", pd.Series(True, index=df_age.index), COHORT_COLORS["full"]),
            ("Asthma (MCQ010=1)", asthma_mask, COHORT_COLORS["asthma"]),
        ]
        ensure_dir(Path(args.png_asthma_age_out).parent)
        fig_a, ax_a = plt.subplots(figsize=(10, 7), dpi=150)
        asthma_rows = []
        for label, mask, color in cohorts_asthma_age:
            sub = df_age.loc[mask].copy()
            if sub.empty:
                continue
            kmf_a = _fit_age_km(sub, label)
            if kmf_a is None:
                continue
            kmf_a.plot_survival_function(ax=ax_a, ci_show=True, color=color, linewidth=2.2)
            asthma_rows.append(
                {
                    "cohort": label,
                    "n": int(len(sub)),
                    "deaths": int(sub["event"].sum()),
                    "censored": int((sub["event"] == 0).sum()),
                    "min_entry_age_years": float(sub["entry_age"].min()),
                    "max_entry_age_years": float(sub["entry_age"].max()),
                    "max_end_age_years": float(sub["end_age"].max()),
                }
            )
        ax_a.set_title("NHANES Kaplan-Meier Survival by Age: Asthma vs Full Cohort")
        ax_a.set_xlabel("Age (years)")
        ax_a.set_ylabel("Survival probability")
        ax_a.set_ylim(0, 1.0)
        ax_a.grid(alpha=0.25)
        ax_a.legend(loc="best", frameon=False)
        fig_a.tight_layout()
        fig_a.savefig(args.png_asthma_age_out)
        plt.close(fig_a)

        ensure_dir(Path(args.csv_asthma_age_out).parent)
        pd.DataFrame(asthma_rows).to_csv(args.csv_asthma_age_out, index=False)

    print(f"Wrote KM plot: {args.png_out}")
    print(f"Wrote cohort counts: {args.csv_out}")
    print(f"Wrote age-timescale KM plot: {args.png_age_out}")
    print(f"Wrote age-timescale cohort counts: {args.csv_age_out}")
    print(f"Wrote age-timescale summary: {args.age_summary_csv_out}")
    print(f"Wrote all-disease age summary: {args.csv_all_disease_age_out}")
    print(f"Wrote steepness/longevity scatter: {args.steepness_png_out}")
    print(f"Wrote all-disease panel KM plot: {args.png_all_disease_panels_age_out}")
    print(f"Wrote asthma age-timescale KM plot: {args.png_asthma_age_out}")
    print(f"Wrote asthma age-timescale cohort counts: {args.csv_asthma_age_out}")


if __name__ == "__main__":
    main()
