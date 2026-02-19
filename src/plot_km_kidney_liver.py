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
        ("Full cohort (age>=20, eligstat=1)", pd.Series(True, index=df.index), "#334155"),
        ("Diabetes (DIQ010=1)", df["diabetes"] == True, "#7c3aed"),  # noqa: E712
        ("Kidney disease (KIQ022=1)", df["kidney"] == True, "#1d4ed8"),  # noqa: E712
        ("Liver disease (MCQ160L/MCQ500/MCQ510*=1)", df["liver"] == True, "#b91c1c"),  # noqa: E712
    ]

    ensure_dir(Path(args.png_out).parent)
    fig, ax = plt.subplots(figsize=(10, 7), dpi=150)
    kmf = KaplanMeierFitter()
    count_rows = []

    for label, mask, color in cohorts:
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

    print(f"Wrote KM plot: {args.png_out}")
    print(f"Wrote cohort counts: {args.csv_out}")


if __name__ == "__main__":
    main()
