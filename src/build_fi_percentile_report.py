#!/usr/bin/env python3
"""Build percentile-mapping report between FI percentiles and biomarker percentiles."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import math

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
FRAILTY_DIR = ROOT / "output" / "frailty"
PANEL_PATH = FRAILTY_DIR / "frailty_panel.csv.gz"
OUTPUT_DIR = ROOT / "output" / "frailty_percentile_report"
FIGURE_DIR = OUTPUT_DIR / "figures"

MIN_POINTS_PER_PANEL = 200
FI_COLUMNS = [
    "fi_hrs_overlap_22",
    "fi_hrs_overlap_memory_23",
    "fi_screened_19",
]


@dataclass(frozen=True)
class TestSpec:
    key: str
    label: str
    column: str
    unit: str


TEST_SPECS = [
    TestSpec("crp", "CRP", "crp_mg_dl", "mg/dL"),
    TestSpec("hscrp", "hs-CRP", "hscrp_mg_l", "mg/L"),
    TestSpec("insulin", "Insulin", "insulin_uU_ml", "uU/mL"),
    TestSpec("glucose", "Glucose", "fasting_glucose_mg_dl", "mg/dL"),
    TestSpec("triglycerides", "Triglycerides", "triglycerides_mg_dl", "mg/dL"),
    TestSpec("eosinophils", "Eosinophils", "eosinophils_abs_1000_uL", "1000 cells/uL"),
]

FI_LABELS = {
    "fi_hrs_overlap_22": "FI HRS-overlap 22",
    "fi_hrs_overlap_memory_23": "FI HRS-overlap+memory 23",
    "fi_screened_19": "FI screened 19",
}

FI_COLORS = {
    "fi_hrs_overlap_22": "#1f77b4",
    "fi_hrs_overlap_memory_23": "#d62728",
    "fi_screened_19": "#2ca02c",
}


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    FIGURE_DIR.mkdir(parents=True, exist_ok=True)

    panel = pd.read_csv(PANEL_PATH)
    panel = add_percentile_columns(panel)

    all_summary_frames: list[pd.DataFrame] = []
    markdown_lines = build_markdown_header()

    for test_spec in TEST_SPECS:
        summary = build_test_summary(panel, test_spec)
        if summary.empty:
            continue

        figure_path = FIGURE_DIR / f"{test_spec.key}_fi_percentile_scatter.png"
        make_test_figure(panel, summary, test_spec, figure_path)

        summary.to_csv(OUTPUT_DIR / f"{test_spec.key}_panel_summary.csv", index=False)
        all_summary_frames.append(summary)
        markdown_lines.extend(build_markdown_section(test_spec, summary, figure_path))

    if all_summary_frames:
        combined_summary = pd.concat(all_summary_frames, ignore_index=True)
        combined_summary.to_csv(OUTPUT_DIR / "all_tests_panel_summary.csv", index=False)
        markdown_lines.extend(build_markdown_overall_section(combined_summary))

    report_path = OUTPUT_DIR / "FI_percentile_mapping_report.md"
    report_path.write_text("\n".join(markdown_lines))


def add_percentile_columns(panel: pd.DataFrame) -> pd.DataFrame:
    panel = panel.copy()

    for fi_column in FI_COLUMNS:
        percentile_column = f"{fi_column}_pct"
        panel[percentile_column] = age_bin_percentile(panel, fi_column)

    for test_spec in TEST_SPECS:
        percentile_column = f"{test_spec.column}_pct"
        panel[percentile_column] = age_bin_percentile(panel, test_spec.column)

    return panel


def age_bin_percentile(panel: pd.DataFrame, value_column: str) -> pd.Series:
    out = pd.Series(np.nan, index=panel.index, dtype=float)

    for age_bin, frame in panel.groupby("age_bin", observed=False):
        if pd.isna(age_bin):
            continue

        values = pd.to_numeric(frame[value_column], errors="coerce")
        valid = values.notna()
        if valid.sum() == 0:
            continue

        ranks = values.loc[valid].rank(method="average", pct=True) * 100.0
        out.loc[ranks.index] = ranks

    return out


def build_test_summary(panel: pd.DataFrame, test_spec: TestSpec) -> pd.DataFrame:
    rows = []
    biomarker_pct_column = f"{test_spec.column}_pct"

    for age_bin in sorted(panel["age_bin"].dropna().unique()):
        age_frame = panel.loc[panel["age_bin"] == age_bin].copy()

        for fi_column in FI_COLUMNS:
            fi_pct_column = f"{fi_column}_pct"
            pair = age_frame[[fi_column, fi_pct_column, test_spec.column, biomarker_pct_column]].dropna()
            n_points = int(pair.shape[0])

            if n_points <= 1:
                continue

            pearson_r = float(pair[fi_pct_column].corr(pair[biomarker_pct_column], method="pearson"))
            r_squared = pearson_r**2 if not math.isnan(pearson_r) else np.nan

            rows.append(
                {
                    "test_key": test_spec.key,
                    "test_label": test_spec.label,
                    "age_bin": age_bin,
                    "fi_column": fi_column,
                    "fi_label": FI_LABELS[fi_column],
                    "n_points": n_points,
                    "pearson_r": pearson_r,
                    "r_squared": r_squared,
                    "mean_abs_distance_from_diagonal": float(
                        np.abs(pair[fi_pct_column] - pair[biomarker_pct_column]).mean()
                    ),
                }
            )

    summary = pd.DataFrame(rows)
    if summary.empty:
        return summary

    relevant_age_bins = (
        summary.groupby("age_bin")["n_points"]
        .max()
        .reset_index()
        .loc[lambda df: df["n_points"] > MIN_POINTS_PER_PANEL, "age_bin"]
        .tolist()
    )

    summary = summary.loc[summary["age_bin"].isin(relevant_age_bins)].copy()
    summary["panel_included"] = summary["n_points"] > MIN_POINTS_PER_PANEL
    return summary


def make_test_figure(
    panel: pd.DataFrame,
    summary: pd.DataFrame,
    test_spec: TestSpec,
    figure_path: Path,
) -> None:
    age_bins = summary["age_bin"].drop_duplicates().tolist()
    n_panels = len(age_bins)
    n_cols = min(3, n_panels)
    n_rows = math.ceil(n_panels / n_cols)

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(6.0 * n_cols, 5.2 * n_rows), squeeze=False)
    axes_flat = axes.flatten()

    mean_r2_map = (
        summary.loc[summary["panel_included"]]
        .groupby("fi_label")["r_squared"]
        .mean()
        .to_dict()
    )
    subtitle = " | ".join(
        f"{fi_label} mean $R^2$={mean_r2_map.get(fi_label, float('nan')):.3f}"
        for fi_label in [FI_LABELS[col] for col in FI_COLUMNS]
    )
    fig.suptitle(
        f"{test_spec.label} percentile vs FI percentile within age bins\n{subtitle}",
        fontsize=16,
        y=0.98,
    )

    biomarker_pct_column = f"{test_spec.column}_pct"

    for axis, age_bin in zip(axes_flat, age_bins):
        age_frame = panel.loc[panel["age_bin"] == age_bin].copy()

        axis.plot([0, 100], [0, 100], linestyle="--", color="black", linewidth=1, alpha=0.7)

        text_lines = []
        for fi_column in FI_COLUMNS:
            fi_pct_column = f"{fi_column}_pct"
            pair = age_frame[[fi_pct_column, biomarker_pct_column]].dropna()
            n_points = int(pair.shape[0])
            if n_points <= MIN_POINTS_PER_PANEL:
                text_lines.append(f"{FI_LABELS[fi_column]}: n={n_points}")
                continue

            axis.scatter(
                pair[fi_pct_column],
                pair[biomarker_pct_column],
                s=7,
                alpha=0.18,
                color=FI_COLORS[fi_column],
                edgecolors="none",
                rasterized=True,
            )

            row = summary.loc[
                (summary["age_bin"] == age_bin) & (summary["fi_column"] == fi_column)
            ].iloc[0]
            text_lines.append(
                f"{FI_LABELS[fi_column]}: n={n_points}, $R^2$={row['r_squared']:.3f}"
            )

        axis.set_title(f"Age {age_bin}")
        axis.set_xlim(0, 100)
        axis.set_ylim(0, 100)
        axis.set_xlabel("FI percentile within age bin")
        axis.set_ylabel(f"{test_spec.label} percentile within age bin")
        axis.text(
            0.02,
            0.98,
            "\n".join(text_lines),
            transform=axis.transAxes,
            ha="left",
            va="top",
            fontsize=9,
            bbox={"facecolor": "white", "alpha": 0.75, "edgecolor": "#cccccc"},
        )

    for axis in axes_flat[n_panels:]:
        axis.axis("off")

    handles = [
        plt.Line2D([], [], linestyle="", marker="o", color=FI_COLORS[fi_column], markersize=6, label=FI_LABELS[fi_column])
        for fi_column in FI_COLUMNS
    ]
    fig.legend(handles=handles, loc="lower center", ncol=3, frameon=False, bbox_to_anchor=(0.5, 0.01))
    fig.tight_layout(rect=[0.02, 0.05, 1.0, 0.94])
    fig.savefig(figure_path, dpi=220)
    plt.close(fig)


def build_markdown_header() -> list[str]:
    return [
        "# FI Percentile Mapping Report",
        "",
        "This report compares within-age-bin frailty-index percentiles against within-age-bin biomarker percentiles.",
        "",
        "Interpretation:",
        "- Each dot is one person.",
        "- The x-axis is that person's FI percentile within their age bin.",
        "- The y-axis is that person's biomarker percentile within the same age bin.",
        "- Perfect percentile mapping would place points on the diagonal \\(x=y\\).",
        "- \\(R^2\\) here is Pearson correlation squared between FI percentile and biomarker percentile within each age bin.",
        "",
        f"Panels are shown only for age bins where at least \\(n > {MIN_POINTS_PER_PANEL}\\) people have both FI and biomarker data.",
        "",
    ]


def build_markdown_section(test_spec: TestSpec, summary: pd.DataFrame, figure_path: Path) -> list[str]:
    lines = [
        f"## {test_spec.label}",
        "",
        f"![{test_spec.label} percentile mapping]({figure_path})",
        "",
    ]

    mean_table = (
        summary.loc[summary["panel_included"]]
        .groupby("fi_label", as_index=False)
        .agg(
            mean_r_squared=("r_squared", "mean"),
            mean_abs_distance_from_diagonal=("mean_abs_distance_from_diagonal", "mean"),
            age_bins_used=("age_bin", "nunique"),
        )
        .sort_values("fi_label")
    )

    lines.extend(
        [
            "Mean across included age bins:",
            "",
            dataframe_to_markdown(mean_table.rename(columns={
                "fi_label": "FI",
                "mean_r_squared": "mean_R2",
                "mean_abs_distance_from_diagonal": "mean_abs_distance",
                "age_bins_used": "age_bins_used",
            })),
            "",
            "Panel-level summary:",
            "",
            dataframe_to_markdown(
                summary.loc[:, ["age_bin", "fi_label", "n_points", "r_squared", "mean_abs_distance_from_diagonal"]]
                .rename(columns={
                    "age_bin": "age_bin",
                    "fi_label": "FI",
                    "n_points": "n",
                    "r_squared": "R2",
                    "mean_abs_distance_from_diagonal": "mean_abs_distance",
                })
                .sort_values(["age_bin", "FI"])
            ),
            "",
        ]
    )
    return lines


def build_markdown_overall_section(summary: pd.DataFrame) -> list[str]:
    overall = (
        summary.loc[summary["panel_included"]]
        .groupby(["test_label", "fi_label"], as_index=False)
        .agg(
            mean_r_squared=("r_squared", "mean"),
            mean_abs_distance_from_diagonal=("mean_abs_distance_from_diagonal", "mean"),
            age_bins_used=("age_bin", "nunique"),
        )
        .sort_values(["test_label", "fi_label"])
    )

    return [
        "## Overall Summary",
        "",
        dataframe_to_markdown(overall.rename(columns={
            "test_label": "test",
            "fi_label": "FI",
            "mean_r_squared": "mean_R2",
            "mean_abs_distance_from_diagonal": "mean_abs_distance",
            "age_bins_used": "age_bins_used",
        })),
        "",
    ]


def dataframe_to_markdown(frame: pd.DataFrame) -> str:
    formatted = frame.copy()

    for column in formatted.columns:
        if pd.api.types.is_float_dtype(formatted[column]):
            formatted[column] = formatted[column].map(lambda value: f"{value:.3f}")

    return formatted.to_markdown(index=False)


if __name__ == "__main__":
    main()
