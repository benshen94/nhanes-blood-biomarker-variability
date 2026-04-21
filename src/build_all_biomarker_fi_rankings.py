#!/usr/bin/env python3
"""Rank NHANES biomarkers by pooled percentile correlation with the HRS-overlap FI."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import pearsonr


ROOT = Path(__file__).resolve().parents[1]
FRAILTY_PANEL_PATH = ROOT / "output" / "frailty" / "frailty_panel.csv.gz"
BIOMARKER_LONG_PATH = ROOT / "data" / "processed" / "biomarker_long.parquet"
OUTPUT_DIR = ROOT / "output" / "frailty_all_biomarker_scan"

MIN_POINTS_PER_AGE_BIN = 200
MIN_AGE_BINS_FOR_RANKING = 3
FI_COLUMN = "fi_hrs_overlap_22"
REPORT_TOP_N = 200
SHOW_TOP_N = 25


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    fi_panel = load_fi_panel()
    biomarker_long = load_biomarker_long()
    merged = merge_fi_and_biomarkers(fi_panel, biomarker_long)
    merged = add_biomarker_percentiles(merged)

    age_bin_stats = build_age_bin_stats(merged)
    eligible_merged = keep_eligible_age_bins(merged, age_bin_stats)
    biomarker_summary = build_biomarker_summary(age_bin_stats, eligible_merged)

    age_bin_stats.to_csv(OUTPUT_DIR / "biomarker_age_bin_correlations.csv", index=False)
    biomarker_summary.to_csv(OUTPUT_DIR / "biomarker_rankings.csv", index=False)
    biomarker_summary.loc[biomarker_summary["age_bins_used"] >= MIN_AGE_BINS_FOR_RANKING].to_csv(
        OUTPUT_DIR / "biomarker_rankings_multibin.csv",
        index=False,
    )
    write_markdown_report(biomarker_summary)


def load_fi_panel() -> pd.DataFrame:
    panel = pd.read_csv(
        FRAILTY_PANEL_PATH,
        usecols=["seqn", "cycle_start_year", "age_bin", FI_COLUMN],
    )
    panel = panel.loc[panel[FI_COLUMN].notna()].copy()
    panel[f"{FI_COLUMN}_pct"] = age_bin_percentile(panel, FI_COLUMN)
    return panel


def load_biomarker_long() -> pd.DataFrame:
    columns = [
        "seqn",
        "cycle_start_year",
        "biomarker_id",
        "biomarker_name",
        "value",
        "unit",
    ]
    return pd.read_parquet(BIOMARKER_LONG_PATH, columns=columns)


def merge_fi_and_biomarkers(fi_panel: pd.DataFrame, biomarker_long: pd.DataFrame) -> pd.DataFrame:
    merged = biomarker_long.merge(
        fi_panel,
        on=["seqn", "cycle_start_year"],
        how="inner",
    )
    merged["value"] = pd.to_numeric(merged["value"], errors="coerce")
    merged = merged.dropna(subset=["age_bin", "value", f"{FI_COLUMN}_pct"]).copy()
    return merged


def add_biomarker_percentiles(merged: pd.DataFrame) -> pd.DataFrame:
    merged = merged.copy()
    merged["biomarker_pct"] = (
        merged.groupby(["biomarker_id", "age_bin"], observed=True)["value"]
        .rank(method="average", pct=True)
        .mul(100.0)
    )
    return merged


def build_age_bin_stats(merged: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, object]] = []

    grouped = merged.groupby(["biomarker_id", "biomarker_name", "unit", "age_bin"], observed=True)
    for (biomarker_id, biomarker_name, unit, age_bin), frame in grouped:
        n_points = int(frame.shape[0])
        if n_points <= MIN_POINTS_PER_AGE_BIN:
            continue

        x = frame[f"{FI_COLUMN}_pct"].to_numpy()
        y = frame["biomarker_pct"].to_numpy()
        if np.unique(y).size < 2:
            continue

        pearson_r, p_value = pearsonr(x, y)
        rows.append(
            {
                "biomarker_id": biomarker_id,
                "biomarker_name": biomarker_name,
                "unit": unit,
                "age_bin": age_bin,
                "n_points": n_points,
                "pearson_r": pearson_r,
                "p_value": p_value,
            }
        )

    stats = pd.DataFrame(rows)
    if stats.empty:
        return stats

    stats["age_bin_lower"] = stats["age_bin"].map(age_bin_lower_bound)
    stats = stats.sort_values(["biomarker_name", "age_bin_lower"]).reset_index(drop=True)
    return stats


def keep_eligible_age_bins(merged: pd.DataFrame, age_bin_stats: pd.DataFrame) -> pd.DataFrame:
    if age_bin_stats.empty:
        return merged.iloc[0:0].copy()

    eligible_bins = age_bin_stats.loc[:, ["biomarker_id", "age_bin"]].drop_duplicates()
    eligible_merged = merged.merge(
        eligible_bins,
        on=["biomarker_id", "age_bin"],
        how="inner",
    )
    return eligible_merged


def build_biomarker_summary(
    age_bin_stats: pd.DataFrame,
    eligible_merged: pd.DataFrame,
) -> pd.DataFrame:
    if age_bin_stats.empty or eligible_merged.empty:
        return pd.DataFrame()

    age_bin_summary = (
        age_bin_stats.groupby(["biomarker_id", "biomarker_name", "unit"], as_index=False)
        .agg(
            age_bins_used=("age_bin", "nunique"),
            eligible_total_n=("n_points", "sum"),
            mean_age_bin_r=("pearson_r", "mean"),
            std_age_bin_r=("pearson_r", "std"),
            mean_age_bin_p=("p_value", "mean"),
        )
    )
    age_bin_summary["sem_age_bin_r"] = age_bin_summary["std_age_bin_r"] / np.sqrt(
        age_bin_summary["age_bins_used"]
    )
    age_bin_summary["std_age_bin_r"] = age_bin_summary["std_age_bin_r"].fillna(0.0)
    age_bin_summary["sem_age_bin_r"] = age_bin_summary["sem_age_bin_r"].fillna(0.0)

    pooled_rows: list[dict[str, object]] = []
    grouped = eligible_merged.groupby(["biomarker_id", "biomarker_name", "unit"], observed=True)
    for (biomarker_id, biomarker_name, unit), frame in grouped:
        pooled_n = int(frame.shape[0])
        if pooled_n <= 1:
            continue

        x = frame[f"{FI_COLUMN}_pct"].to_numpy()
        y = frame["biomarker_pct"].to_numpy()
        if np.unique(y).size < 2:
            continue

        pooled_r, pooled_p = pearsonr(x, y)
        pooled_rows.append(
            {
                "biomarker_id": biomarker_id,
                "biomarker_name": biomarker_name,
                "unit": unit,
                "pooled_n": pooled_n,
                "pooled_r": pooled_r,
                "pooled_p": pooled_p,
            }
        )

    pooled_summary = pd.DataFrame(pooled_rows)
    if pooled_summary.empty:
        return pd.DataFrame()

    summary = age_bin_summary.merge(
        pooled_summary,
        on=["biomarker_id", "biomarker_name", "unit"],
        how="inner",
    )
    summary = summary.sort_values(
        ["pooled_r", "age_bins_used", "pooled_n"],
        ascending=[False, False, False],
    ).reset_index(drop=True)
    return summary


def write_markdown_report(biomarker_summary: pd.DataFrame) -> None:
    report_summary = biomarker_summary.loc[
        biomarker_summary["age_bins_used"] >= MIN_AGE_BINS_FOR_RANKING
    ].copy()

    lines = [
        "# FI HRS-overlap vs NHANES Biomarkers",
        "",
        "This report ranks harmonized NHANES blood biomarkers by pooled Pearson correlation with the HRS-overlap FI percentile.",
        "",
        f"Step 1: keep only biomarker age bins with \\(n > {MIN_POINTS_PER_AGE_BIN}\\).",
        "Step 2: within each remaining age bin, convert FI and biomarker values to percentiles.",
        "Step 3: stack all of those percentile dots across age bins for each biomarker and compute one pooled Pearson correlation.",
        f"Ranked markdown list includes only biomarkers with at least \\({MIN_AGE_BINS_FOR_RANKING}\\) eligible age bins.",
        "",
        "Columns:",
        "- `pooled_r`: Pearson correlation after pooling all eligible percentile dots across age bins",
        "- `pooled_p`: p-value for that pooled Pearson correlation",
        "- `pooled_n`: total pooled participant-dots used in that pooled correlation",
        "- `age_bins_used`: number of eligible age bins contributing to the summary",
        "- `mean_age_bin_r`: mean Pearson correlation across eligible age bins",
        "- `sem_age_bin_r`: standard error of the mean age-bin correlation",
        "",
        "Summary tables:",
        "",
    ]

    if report_summary.empty:
        lines.append("No eligible biomarker summaries were produced.")
    else:
        positive = report_summary.sort_values(
            ["pooled_r", "age_bins_used", "pooled_n"],
            ascending=[False, False, False],
        ).head(SHOW_TOP_N)
        negative = report_summary.sort_values(
            ["pooled_r", "age_bins_used", "pooled_n"],
            ascending=[True, False, False],
        ).head(SHOW_TOP_N)
        strongest_absolute = report_summary.assign(
            abs_pooled_r=report_summary["pooled_r"].abs()
        ).sort_values(
            ["abs_pooled_r", "age_bins_used", "pooled_n"],
            ascending=[False, False, False],
        ).head(SHOW_TOP_N)
        full_ranked = report_summary.head(REPORT_TOP_N)

        lines.extend(
            [
                f"Top {SHOW_TOP_N} positive pooled correlations:",
                "",
                dataframe_to_markdown(
                    rename_for_report(positive)[
                        [
                            "biomarker",
                            "unit",
                            "pooled_r",
                            "pooled_p",
                            "pooled_n",
                            "age_bins_used",
                            "mean_age_bin_r",
                            "sem_age_bin_r",
                        ]
                    ]
                ),
                "",
                f"Top {SHOW_TOP_N} inverse pooled correlations:",
                "",
                dataframe_to_markdown(
                    rename_for_report(negative)[
                        [
                            "biomarker",
                            "unit",
                            "pooled_r",
                            "pooled_p",
                            "pooled_n",
                            "age_bins_used",
                            "mean_age_bin_r",
                            "sem_age_bin_r",
                        ]
                    ]
                ),
                "",
                f"Top {SHOW_TOP_N} strongest absolute pooled correlations:",
                "",
                dataframe_to_markdown(
                    rename_for_report(strongest_absolute)[
                        [
                            "biomarker",
                            "unit",
                            "pooled_r",
                            "pooled_p",
                            "pooled_n",
                            "age_bins_used",
                            "mean_age_bin_r",
                            "sem_age_bin_r",
                        ]
                    ]
                ),
                "",
                f"Full ranked list, top {REPORT_TOP_N} by pooled correlation:",
                "",
                dataframe_to_markdown(
                    rename_for_report(full_ranked)[
                        [
                            "biomarker",
                            "unit",
                            "pooled_r",
                            "pooled_p",
                            "pooled_n",
                            "age_bins_used",
                            "mean_age_bin_r",
                            "sem_age_bin_r",
                        ]
                    ]
                ),
            ]
        )
        lines.append("Full ranking CSV:")
        lines.append(f"- `{OUTPUT_DIR / 'biomarker_rankings.csv'}`")
        lines.append("Multi-bin ranking CSV:")
        lines.append(f"- `{OUTPUT_DIR / 'biomarker_rankings_multibin.csv'}`")
        lines.append("Full age-bin correlation CSV:")
        lines.append(f"- `{OUTPUT_DIR / 'biomarker_age_bin_correlations.csv'}`")

    (OUTPUT_DIR / "FI_HRS_overlap_all_biomarkers_ranked.md").write_text("\n".join(lines))


def rename_for_report(frame: pd.DataFrame) -> pd.DataFrame:
    return frame.rename(
        columns={
            "biomarker_name": "biomarker",
            "unit": "unit",
            "pooled_r": "pooled_r",
            "pooled_p": "pooled_p",
            "pooled_n": "pooled_n",
            "age_bins_used": "age_bins_used",
            "mean_age_bin_r": "mean_age_bin_r",
            "sem_age_bin_r": "sem_age_bin_r",
        }
    )


def age_bin_percentile(panel: pd.DataFrame, value_column: str) -> pd.Series:
    out = pd.Series(np.nan, index=panel.index, dtype=float)

    for age_bin in ordered_age_bins(panel["age_bin"]):
        frame = panel.loc[panel["age_bin"] == age_bin].copy()
        values = pd.to_numeric(frame[value_column], errors="coerce")
        valid = values.notna()
        if valid.sum() == 0:
            continue

        ranks = values.loc[valid].rank(method="average", pct=True) * 100.0
        out.loc[ranks.index] = ranks

    return out


def ordered_age_bins(values: pd.Series | list[str]) -> list[str]:
    unique_bins = [value for value in pd.Series(values).dropna().unique().tolist()]
    return sorted(unique_bins, key=age_bin_lower_bound)


def age_bin_lower_bound(age_bin: str) -> int:
    text = str(age_bin)
    if "+" in text:
        return int(text.replace("+", ""))
    return int(text.split("-")[0])


def dataframe_to_markdown(frame: pd.DataFrame) -> str:
    formatted = frame.copy()

    for column in formatted.columns:
        if pd.api.types.is_float_dtype(formatted[column]):
            if "p" in column.lower():
                formatted[column] = formatted[column].map(format_p_value)
            else:
                formatted[column] = formatted[column].map(lambda value: f"{value:.3f}")

    return formatted.to_markdown(index=False)


def format_p_value(value: float) -> str:
    if pd.isna(value):
        return "nan"
    if value < 1e-4:
        return f"{value:.1e}"
    return f"{value:.4f}"


if __name__ == "__main__":
    main()
