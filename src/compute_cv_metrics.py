#!/usr/bin/env python3
"""Compute age-binned CV and biomarker trend metrics."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

from nhanes_common import ensure_dir


def assign_age_bins(age: pd.Series) -> tuple[pd.Series, pd.Series]:
    edges = list(np.arange(20, 90, 5)) + [200]
    labels = [f"{a}-{a+4}" for a in range(20, 85, 5)] + ["85+"]
    mids = [a + 2.5 for a in range(20, 85, 5)] + [87.5]

    b = pd.cut(age, bins=edges, labels=labels, right=False, include_lowest=True)
    m = b.map(dict(zip(labels, mids))).astype(float)
    return b, m


def compute_binned(df: pd.DataFrame) -> pd.DataFrame:
    tmp = df.copy()
    tmp["age_bin"], tmp["age_mid"] = assign_age_bins(tmp["age_years"])
    tmp = tmp.dropna(subset=["age_bin", "value"])

    group_cols = ["biomarker_id", "biomarker_name"]
    if "unit" in tmp.columns:
        group_cols.append("unit")

    grouped = (
        tmp.groupby(group_cols + ["age_bin", "age_mid"], observed=True)["value"]
        .agg(["count", "mean", "std"])
        .reset_index()
        .rename(columns={"count": "n"})
    )

    grouped["cv"] = grouped["std"] / grouped["mean"].abs()
    grouped.loc[grouped["mean"].abs() < 1e-8, "cv"] = np.nan
    grouped["passes_n_threshold"] = grouped["n"] >= 30
    grouped = grouped.dropna(subset=["cv"]).reset_index(drop=True)
    return grouped


def slope(x: np.ndarray, y: np.ndarray) -> float:
    if len(x) < 2:
        return np.nan
    coef = np.polyfit(x, y, 1)
    return float(coef[0])


def compute_trends(cv_df: pd.DataFrame) -> pd.DataFrame:
    eligible = cv_df[cv_df["passes_n_threshold"]].copy()

    rows = []
    for (bid, bname), g in eligible.groupby(["biomarker_id", "biomarker_name"], observed=True):
        g = g.sort_values("age_mid")
        x = g["age_mid"].to_numpy(dtype=float)
        y = g["cv"].to_numpy(dtype=float)
        pos = y > 0

        rho, p = (np.nan, np.nan)
        if len(y) >= 2:
            rho, p = spearmanr(x, y)

        row = {
            "biomarker_id": bid,
            "biomarker_name": bname,
            "n_bins": int(len(g)),
            "spearman_rho": float(rho) if pd.notna(rho) else np.nan,
            "spearman_p": float(p) if pd.notna(p) else np.nan,
            "linear_slope_cv_per_year": slope(x, y),
            "linear_slope_logcv_per_year": slope(x[pos], np.log(y[pos])) if pos.sum() >= 2 else np.nan,
        }
        row["decline_flag"] = bool(
            row["n_bins"] >= 5
            and pd.notna(row["spearman_rho"])
            and pd.notna(row["spearman_p"])
            and row["spearman_rho"] < 0
            and row["spearman_p"] < 0.05
            and pd.notna(row["linear_slope_cv_per_year"])
            and row["linear_slope_cv_per_year"] < 0
        )
        rows.append(row)

    return pd.DataFrame(rows)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--in", dest="input_path", default="data/processed/biomarker_long.parquet")
    ap.add_argument("--out", default="data/processed")
    args = ap.parse_args()

    out_dir = Path(args.out)
    ensure_dir(out_dir)

    df = pd.read_parquet(args.input_path)
    cv_all = compute_binned(df)

    cv_all.to_parquet(out_dir / "cv_by_age_all.parquet", index=False)
    cv_main = cv_all[cv_all["passes_n_threshold"]].copy()
    cv_main.to_parquet(out_dir / "cv_by_age.parquet", index=False)

    trends = compute_trends(cv_all)
    trends.to_parquet(out_dir / "cv_trend_metrics.parquet", index=False)

    print(f"cv_by_age_all rows: {len(cv_all):,}")
    print(f"cv_by_age rows (n>=30): {len(cv_main):,}")
    print(f"trend rows: {len(trends):,}")


if __name__ == "__main__":
    main()
