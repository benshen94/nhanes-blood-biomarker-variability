#!/usr/bin/env python3
"""Rank NHANES biomarkers by pooled percentile correlation with the HRS-overlap FI.

This version uses the largest relevant cohort for the FI analysis:
- start from the FI participant panel
- read raw NHANES blood lab files for those FI participants only
- harmonize biomarkers with the existing pooling rules
- do not apply the healthy-only biomarker exclusion
- compute within-age-bin percentiles, then pool dots across eligible age bins
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
from scipy.stats import pearsonr

from build_analysis_dataset import (
    build_pooling_map,
    is_comment_or_code_variable,
    is_continuous_numeric,
    normalize_seqn,
    read_xpt_columns,
)


ROOT = Path(__file__).resolve().parents[1]
RAW_DIR = ROOT / "data" / "raw"
LAB_MANIFEST_PATH = ROOT / "data" / "processed" / "lab_variable_manifest.parquet"
FRAILTY_PANEL_PATH = ROOT / "output" / "frailty" / "frailty_panel.csv.gz"
OUTPUT_DIR = ROOT / "output" / "frailty_all_biomarker_scan"
CACHE_PATH = OUTPUT_DIR / "fi_biomarker_long_all_participants.parquet"

MIN_POINTS_PER_AGE_BIN = 200
MIN_AGE_BINS_FOR_RANKING = 3
FI_COLUMN = "fi_hrs_overlap_22"
REPORT_TOP_N = 200
SHOW_TOP_N = 25


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    fi_panel = load_fi_panel()
    biomarker_long = load_or_build_biomarker_long(fi_panel)
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
    panel["seqn"] = pd.to_numeric(panel["seqn"], errors="coerce").astype("Int64")
    panel["cycle_start_year"] = pd.to_numeric(panel["cycle_start_year"], errors="coerce").astype("Int64")
    return panel


def load_or_build_biomarker_long(fi_panel: pd.DataFrame) -> pd.DataFrame:
    if CACHE_PATH.exists():
        return pd.read_parquet(CACHE_PATH)

    lab_manifest = pd.read_parquet(LAB_MANIFEST_PATH)
    biomarker_long = build_biomarker_long_for_fi(fi_panel, lab_manifest)
    return biomarker_long


def build_biomarker_long_for_fi(fi_panel: pd.DataFrame, lab_manifest: pd.DataFrame) -> pd.DataFrame:
    fi_cycles = sorted(fi_panel["cycle_start_year"].dropna().astype(int).unique().tolist())
    fi_people_by_cycle = build_fi_people_lookup(fi_panel)

    selected = lab_manifest.loc[lab_manifest["is_blood_candidate"].fillna(False)].copy()
    selected["cycle_start_year"] = pd.to_numeric(selected["cycle_start_year"], errors="coerce")
    selected = selected.loc[selected["cycle_start_year"].isin(fi_cycles)].copy()
    selected = selected.drop_duplicates(subset=["xpt_url", "variable_name"]).reset_index(drop=True)

    file_meta = (
        selected[
            ["data_file_name", "cycle_start_year", "cycle_end_year", "xpt_url"]
        ]
        .drop_duplicates(subset=["xpt_url"])
        .set_index("xpt_url")
    )
    vars_by_url = {
        url: frame[["variable_name", "variable_desc"]].drop_duplicates().reset_index(drop=True)
        for url, frame in selected.groupby("xpt_url")
    }

    pooling_map_df = build_pooling_map(lab_manifest, raw_dir=RAW_DIR, candidate_column="is_blood_candidate")
    pooling_map = pooling_map_df.set_index(["variable_name", "variable_desc"]).to_dict(orient="index")

    writer: Optional[pq.ParquetWriter] = None

    for url, vars_df in vars_by_url.items():
        meta = file_meta.loc[url]
        cycle_start_year = int(meta["cycle_start_year"])
        fi_people = fi_people_by_cycle.get(cycle_start_year)
        if fi_people is None or fi_people.empty:
            continue

        xpt_path = raw_path_from_url(url)
        if not xpt_path.exists():
            continue

        try:
            data = read_xpt_columns(xpt_path)
        except Exception:
            continue

        if "SEQN" not in data.columns:
            continue

        data["seqn"] = normalize_seqn(data)
        data = data[data["seqn"].isin(fi_people["seqn"])].copy()
        if data.empty:
            continue

        for _, variable_row in vars_df.iterrows():
            long_frame = build_variable_long_frame(
                data=data,
                variable_name=str(variable_row["variable_name"]),
                variable_desc=str(variable_row["variable_desc"]),
                cycle_start_year=cycle_start_year,
                cycle_end_year=meta["cycle_end_year"],
                data_file_name=str(meta["data_file_name"]),
                pooling_map=pooling_map,
            )
            if long_frame is None or long_frame.empty:
                continue

            table = pa.Table.from_pandas(long_frame, preserve_index=False)
            if writer is None:
                writer = pq.ParquetWriter(CACHE_PATH, table.schema)
            writer.write_table(table)

    if writer is not None:
        writer.close()

    if not CACHE_PATH.exists():
        return pd.DataFrame(
            columns=["seqn", "cycle_start_year", "biomarker_id", "biomarker_name", "unit", "value"]
        )

    return pd.read_parquet(CACHE_PATH)


def build_fi_people_lookup(fi_panel: pd.DataFrame) -> dict[int, pd.DataFrame]:
    lookup: dict[int, pd.DataFrame] = {}

    for cycle_start_year, frame in fi_panel.groupby("cycle_start_year", observed=True):
        year = int(cycle_start_year)
        lookup[year] = frame.loc[:, ["seqn"]].drop_duplicates().copy()

    return lookup


def raw_path_from_url(url: str) -> Path:
    year = int(url.split("/Public/")[1].split("/")[0])
    filename = Path(url).name
    return RAW_DIR / str(year) / filename


def build_variable_long_frame(
    data: pd.DataFrame,
    variable_name: str,
    variable_desc: str,
    cycle_start_year: int,
    cycle_end_year: object,
    data_file_name: str,
    pooling_map: dict[tuple[str, str], dict[str, object]],
) -> Optional[pd.DataFrame]:
    if variable_name not in data.columns:
        return None
    if variable_name == "SEQN" or variable_name.startswith("WT"):
        return None
    if is_comment_or_code_variable(variable_name, variable_desc):
        return None

    pooling_key = (variable_name, variable_desc)
    pool = pooling_map.get(pooling_key)
    if pool is None:
        return None
    if not is_continuous_numeric(data[variable_name]):
        return None

    frame = pd.DataFrame(
        {
            "seqn": data["seqn"],
            "value": pd.to_numeric(data[variable_name], errors="coerce"),
        }
    )
    frame = frame.dropna(subset=["seqn", "value"]).copy()
    if frame.empty:
        return None

    factor = float(pool.get("conversion_factor_to_pooled_unit", 1.0))
    if factor != 1.0:
        frame["value"] = frame["value"] * factor

    frame["cycle_start_year"] = cycle_start_year
    frame["cycle_end_year"] = cycle_end_year
    frame["biomarker_id"] = str(pool["pooled_id"])
    frame["biomarker_name"] = str(pool["pooled_name"])
    frame["unit"] = str(pool["pooled_unit"] or "")
    frame["variable_name"] = variable_name
    frame["source_data_file"] = data_file_name

    return frame[
        [
            "seqn",
            "cycle_start_year",
            "cycle_end_year",
            "biomarker_id",
            "variable_name",
            "biomarker_name",
            "source_data_file",
            "unit",
            "value",
        ]
    ]


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
    return merged.merge(eligible_bins, on=["biomarker_id", "age_bin"], how="inner")


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
        f"Step 1: read raw NHANES blood lab files for all FI participants, without the healthy-only filter.",
        f"Step 2: keep only biomarker age bins with \\(n > {MIN_POINTS_PER_AGE_BIN}\\).",
        "Step 3: within each remaining age bin, convert FI and biomarker values to percentiles.",
        "Step 4: stack all of those percentile dots across age bins for each biomarker and compute one pooled Pearson correlation.",
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
        lines.append("Cached all-participant FI biomarker long table:")
        lines.append(f"- `{CACHE_PATH}`")

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
