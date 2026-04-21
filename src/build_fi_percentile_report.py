#!/usr/bin/env python3
"""Build percentile-mapping report between FI percentiles and biomarker percentiles."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import math

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import pearsonr


ROOT = Path(__file__).resolve().parents[1]
FRAILTY_DIR = ROOT / "output" / "frailty"
PANEL_PATH = FRAILTY_DIR / "frailty_panel.csv.gz"
OUTPUT_DIR = ROOT / "output" / "frailty_percentile_report"
FIGURE_DIR = OUTPUT_DIR / "figures"

MIN_POINTS_PER_PANEL = 200
FI_COLUMNS = ["fi_hrs_overlap_22"]


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

TEST_COLORS = {
    "CRP": "#d62728",
    "hs-CRP": "#1f77b4",
    "Insulin": "#2ca02c",
    "Glucose": "#ff7f0e",
    "Triglycerides": "#9467bd",
    "Eosinophils": "#8c564b",
}

FI_LABELS = {"fi_hrs_overlap_22": "FI HRS-overlap 22"}

FI_COLORS = {"fi_hrs_overlap_22": "#1f77b4"}


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
        combined_summary["age_bin_lower"] = combined_summary["age_bin"].map(age_bin_lower_bound)
        combined_summary = combined_summary.sort_values(
            ["age_bin_lower", "test_label", "fi_label"]
        ).reset_index(drop=True)
        combined_summary.to_csv(OUTPUT_DIR / "all_tests_panel_summary.csv", index=False)
        make_age_bin_correlation_plot(combined_summary, FIGURE_DIR / "age_bin_correlation_by_test.png")
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


def ordered_age_bins(values: pd.Series | list[str]) -> list[str]:
    unique_bins = [value for value in pd.Series(values).dropna().unique().tolist()]
    return sorted(unique_bins, key=age_bin_lower_bound)


def age_bin_lower_bound(age_bin: str) -> int:
    text = str(age_bin)
    if "+" in text:
        return int(text.replace("+", ""))
    return int(text.split("-")[0])


def build_test_summary(panel: pd.DataFrame, test_spec: TestSpec) -> pd.DataFrame:
    rows = []
    biomarker_pct_column = f"{test_spec.column}_pct"

    for age_bin in ordered_age_bins(panel["age_bin"]):
        age_frame = panel.loc[panel["age_bin"] == age_bin].copy()

        for fi_column in FI_COLUMNS:
            fi_pct_column = f"{fi_column}_pct"
            pair = age_frame[[fi_column, fi_pct_column, test_spec.column, biomarker_pct_column]].dropna()
            n_points = int(pair.shape[0])

            if n_points <= 1:
                continue

            pearson_r, p_value = pearsonr(pair[fi_pct_column], pair[biomarker_pct_column])

            rows.append(
                {
                    "test_key": test_spec.key,
                    "test_label": test_spec.label,
                    "age_bin": age_bin,
                    "fi_column": fi_column,
                    "fi_label": FI_LABELS[fi_column],
                    "n_points": n_points,
                    "pearson_r": pearson_r,
                    "p_value": p_value,
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
    summary["age_bin_lower"] = summary["age_bin"].map(age_bin_lower_bound)
    summary = summary.sort_values(["age_bin_lower", "fi_label"]).reset_index(drop=True)
    return summary


def make_test_figure(
    panel: pd.DataFrame,
    summary: pd.DataFrame,
    test_spec: TestSpec,
    figure_path: Path,
) -> None:
    age_bins = ordered_age_bins(summary["age_bin"])
    n_panels = len(age_bins)
    n_cols = min(3, n_panels)
    n_rows = math.ceil(n_panels / n_cols)

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(6.0 * n_cols, 5.2 * n_rows), squeeze=False)
    axes_flat = axes.flatten()

    mean_r = float(summary.loc[summary["panel_included"], "pearson_r"].mean())
    median_p = float(summary.loc[summary["panel_included"], "p_value"].median())
    fig.suptitle(
        f"{test_spec.label} percentile vs FI HRS-overlap 22 percentile within age bins\n"
        f"mean $r$={mean_r:.3f} | median p={format_p_value(median_p)}",
        fontsize=16,
        y=0.98,
    )

    biomarker_pct_column = f"{test_spec.column}_pct"

    for axis, age_bin in zip(axes_flat, age_bins):
        age_frame = panel.loc[panel["age_bin"] == age_bin].copy()

        axis.plot([0, 100], [0, 100], linestyle="--", color="black", linewidth=1, alpha=0.7)

        fi_column = FI_COLUMNS[0]
        fi_pct_column = f"{fi_column}_pct"
        pair = age_frame[[fi_pct_column, biomarker_pct_column]].dropna()
        n_points = int(pair.shape[0])

        text_lines = []
        if n_points <= MIN_POINTS_PER_PANEL:
            text_lines.append(f"{FI_LABELS[fi_column]}: n={n_points}")
        else:
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
                f"n={n_points}, $r$={row['pearson_r']:.3f}, p={format_p_value(row['p_value'])}"
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

    fig.tight_layout(rect=[0.02, 0.02, 1.0, 0.94])
    fig.savefig(figure_path, dpi=220)
    plt.close(fig)


def make_age_bin_correlation_plot(summary: pd.DataFrame, figure_path: Path) -> None:
    plot_data = summary.loc[summary["panel_included"]].copy()
    if plot_data.empty:
        return

    age_bin_order = ordered_age_bins(plot_data["age_bin"])
    x_positions = np.arange(len(age_bin_order))
    age_to_x = {age_bin: index for index, age_bin in enumerate(age_bin_order)}

    fig, axis = plt.subplots(figsize=(11, 6))
    fig.patch.set_facecolor("#fbfbfd")
    axis.set_facecolor("#ffffff")

    for test_label, frame in plot_data.groupby("test_label", sort=False):
        frame = frame.copy()
        frame["x"] = frame["age_bin"].map(age_to_x)
        frame = frame.sort_values("x")

        color = TEST_COLORS.get(test_label, "#333333")
        axis.plot(
            frame["x"],
            frame["pearson_r"],
            color=color,
            linewidth=2.5,
            marker="o",
            markersize=6,
            label=test_label,
        )

    axis.axhline(0.0, color="#999999", linewidth=1, linestyle="--", alpha=0.8)
    axis.set_xticks(x_positions)
    axis.set_xticklabels(age_bin_order, rotation=35, ha="right")
    axis.set_ylabel("Pearson correlation (r)")
    axis.set_xlabel("Age bin")
    axis.set_title("FI HRS-overlap percentile correlation with biomarkers by age bin")
    axis.grid(axis="y", color="#d9dbe7", linewidth=0.8, alpha=0.8)
    axis.spines["top"].set_visible(False)
    axis.spines["right"].set_visible(False)
    axis.legend(frameon=False, ncol=3, loc="upper center", bbox_to_anchor=(0.5, -0.12))

    fig.tight_layout()
    fig.savefig(figure_path, dpi=220, bbox_inches="tight")
    plt.close(fig)


def build_markdown_header() -> list[str]:
    return [
        "# FI Percentile Mapping Report",
        "",
        "This report compares within-age-bin HRS-overlap frailty-index percentiles against within-age-bin biomarker percentiles.",
        "",
        "Interpretation:",
        "- Each dot is one person.",
        "- The x-axis is that person's `fi_hrs_overlap_22` percentile within their age bin.",
        "- The y-axis is that person's biomarker percentile within the same age bin.",
        "- Perfect percentile mapping would place points on the diagonal \\(x=y\\).",
        "- `r` here is the Pearson correlation between FI percentile and biomarker percentile within each age bin.",
        "- `p` is the corresponding two-sided significance test for non-zero Pearson correlation.",
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
            mean_pearson_r=("pearson_r", "mean"),
            median_p_value=("p_value", "median"),
            mean_abs_distance_from_diagonal=("mean_abs_distance_from_diagonal", "mean"),
            age_bins_used=("age_bin", "nunique"),
        )
    )

    lines.extend(
        [
            "Mean across included age bins:",
            "",
            dataframe_to_markdown(mean_table.rename(columns={
                "fi_label": "FI",
                "mean_pearson_r": "mean_r",
                "median_p_value": "median_p",
                "mean_abs_distance_from_diagonal": "mean_abs_distance",
                "age_bins_used": "age_bins_used",
            })),
            "",
            "Panel-level summary:",
            "",
            dataframe_to_markdown(
                summary.loc[
                    :,
                    [
                        "age_bin",
                        "fi_label",
                        "n_points",
                        "pearson_r",
                        "p_value",
                        "mean_abs_distance_from_diagonal",
                    ],
                ]
                .rename(columns={
                    "age_bin": "age_bin",
                    "fi_label": "FI",
                    "n_points": "n",
                    "pearson_r": "r",
                    "p_value": "p",
                    "mean_abs_distance_from_diagonal": "mean_abs_distance",
                })
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
            mean_pearson_r=("pearson_r", "mean"),
            mean_p_value=("p_value", "mean"),
            mean_abs_distance_from_diagonal=("mean_abs_distance_from_diagonal", "mean"),
            age_bins_used=("age_bin", "nunique"),
        )
        .sort_values(["mean_pearson_r", "test_label"], ascending=[False, True])
    )

    return [
        "## Overall Summary",
        "",
        "Tests ranked in descending order by mean Pearson correlation across included age bins.",
        "",
        dataframe_to_markdown(overall.rename(columns={
            "test_label": "test",
            "fi_label": "FI",
            "mean_pearson_r": "mean_r",
            "mean_p_value": "mean_p",
            "mean_abs_distance_from_diagonal": "mean_abs_distance",
            "age_bins_used": "age_bins_used",
        })),
        "",
    ]


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
