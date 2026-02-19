#!/usr/bin/env python3
"""Kaplan-Meier survival curves for kidney/liver/diabetes vs full NHANES cohort.

Uses NHANES linked mortality public-use files (2019 linkage release) and the
processed participant health flags table.
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Iterable

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import requests
from lifelines import KaplanMeierFitter

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


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--participants", default="data/processed/participant_health_flags.parquet")
    ap.add_argument("--mortality-dir", default="data/raw/mortality")
    ap.add_argument("--png-out", default="output/km_kidney_liver_vs_full.png")
    ap.add_argument("--csv-out", default="output/km_kidney_liver_counts.csv")
    ap.add_argument("--png-age-out", default="output/km_kidney_liver_vs_full_by_age.png")
    ap.add_argument("--csv-age-out", default="output/km_kidney_liver_counts_by_age.csv")
    ap.add_argument("--age-summary-csv-out", default="output/km_kidney_liver_age_summary.csv")
    ap.add_argument("--steepness-png-out", default="output/steepness_longevity_disease.png")
    args = ap.parse_args()

    part = pd.read_parquet(args.participants)
    required = {"seqn", "cycle_start_year", "age_years", "kidney", "liver", "diabetes"}
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

    cohorts = [
        ("full", "Full cohort (age>=20, eligstat=1)", pd.Series(True, index=df.index), COHORT_COLORS["full"]),
        ("diabetes", "Diabetes (DIQ010=1)", df["diabetes"] == True, COHORT_COLORS["diabetes"]),  # noqa: E712
        ("kidney", "Kidney disease (KIQ022=1)", df["kidney"] == True, COHORT_COLORS["kidney"]),  # noqa: E712
        ("liver", "Liver disease (MCQ160L/MCQ500/MCQ510*=1)", df["liver"] == True, COHORT_COLORS["liver"]),  # noqa: E712
    ]

    ensure_dir(Path(args.png_out).parent)
    fig, ax = plt.subplots(figsize=(10, 7), dpi=150)
    kmf = KaplanMeierFitter()
    count_rows = []

    for key, label, mask, color in cohorts:
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

    cohorts_age = [
        ("full", "Full cohort (age>=20, eligstat=1)", pd.Series(True, index=df_age.index), COHORT_COLORS["full"]),
        ("diabetes", "Diabetes (DIQ010=1)", df_age["diabetes"] == True, COHORT_COLORS["diabetes"]),  # noqa: E712
        ("kidney", "Kidney disease (KIQ022=1)", df_age["kidney"] == True, COHORT_COLORS["kidney"]),  # noqa: E712
        ("liver", "Liver disease (MCQ160L/MCQ500/MCQ510*=1)", df_age["liver"] == True, COHORT_COLORS["liver"]),  # noqa: E712
    ]

    ensure_dir(Path(args.png_age_out).parent)
    fig_age, ax_age = plt.subplots(figsize=(10, 7), dpi=150)
    kmf_age = KaplanMeierFitter()
    count_rows_age = []
    age_summary_rows = []

    for key, label, mask, color in cohorts_age:
        sub = df_age.loc[mask].copy()
        if sub.empty:
            continue
        kmf_age.fit(
            durations=sub["end_age"],
            event_observed=sub["event"],
            entry=sub["entry_age"],
            label=label,
        )
        kmf_age.plot_survival_function(ax=ax_age, ci_show=True, color=color, linewidth=2)
        median_age = float(_km_time_at_survival_prob(kmf_age, 0.5))
        q1_age = float(_km_time_at_survival_prob(kmf_age, 0.75))
        q3_age = float(_km_time_at_survival_prob(kmf_age, 0.25))
        iqr_age = q3_age - q1_age if np.isfinite(q1_age) and np.isfinite(q3_age) else float("nan")
        steepness = (
            median_age / iqr_age
            if np.isfinite(median_age) and np.isfinite(iqr_age) and iqr_age > 0
            else float("nan")
        )
        count_rows_age.append(
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
        age_summary_rows.append(
            {
                "cohort_key": key,
                "cohort": label,
                "median_age_years": median_age,
                "q1_age_years": q1_age,
                "q3_age_years": q3_age,
                "iqr_age_years": iqr_age,
                "steepness_median_over_iqr": steepness,
                "n": int(len(sub)),
                "deaths": int(sub["event"].sum()),
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

    summary_df = pd.DataFrame(age_summary_rows)
    ensure_dir(Path(args.age_summary_csv_out).parent)
    summary_df.to_csv(args.age_summary_csv_out, index=False)

    full_row = summary_df.loc[summary_df["cohort_key"] == "full"]
    if not full_row.empty:
        median_full = float(full_row["median_age_years"].iloc[0])
        steep_full = float(full_row["steepness_median_over_iqr"].iloc[0])
        rel_df = summary_df[summary_df["cohort_key"].isin(["diabetes", "kidney", "liver"])].copy()
        rel_df["relative_median_longevity"] = rel_df["median_age_years"] / median_full
        rel_df["relative_steepness"] = rel_df["steepness_median_over_iqr"] / steep_full
        rel_df = rel_df[np.isfinite(rel_df["relative_median_longevity"]) & np.isfinite(rel_df["relative_steepness"])].copy()

        ensure_dir(Path(args.steepness_png_out).parent)
        fig_rel, ax_rel = plt.subplots(figsize=(8.5, 7), dpi=160)
        for _, row in rel_df.iterrows():
            key = str(row["cohort_key"])
            x = float(row["relative_median_longevity"])
            y = float(row["relative_steepness"])
            ax_rel.scatter(
                x,
                y,
                s=320,
                color=COHORT_COLORS[key],
                alpha=0.9,
                edgecolor="white",
                linewidth=1.2,
                zorder=3,
            )
            ax_rel.text(x, y + 0.02, row["cohort"].split(" (")[0], ha="center", va="bottom", fontsize=11)

        ax_rel.axvline(1.0, color="#64748b", linestyle="--", linewidth=1.4)
        ax_rel.axhline(1.0, color="#64748b", linestyle="--", linewidth=1.4)
        ax_rel.set_title("Disease Cohorts: Relative Longevity vs Relative Steepness")
        ax_rel.set_xlabel("Median lifespan / Full-cohort median lifespan")
        ax_rel.set_ylabel("Steepness (median/IQR) / Full-cohort steepness")
        ax_rel.grid(alpha=0.25, zorder=0)
        fig_rel.tight_layout()
        fig_rel.savefig(args.steepness_png_out)
        plt.close(fig_rel)

    print(f"Wrote KM plot: {args.png_out}")
    print(f"Wrote cohort counts: {args.csv_out}")
    print(f"Wrote age-timescale KM plot: {args.png_age_out}")
    print(f"Wrote age-timescale cohort counts: {args.csv_age_out}")
    print(f"Wrote age-timescale summary: {args.age_summary_csv_out}")
    print(f"Wrote steepness/longevity scatter: {args.steepness_png_out}")


if __name__ == "__main__":
    main()
