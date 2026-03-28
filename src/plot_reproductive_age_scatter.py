#!/usr/bin/env python3
"""Build reproductive age scatter plots from local NHANES questionnaire data."""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import pearsonr

from nhanes_common import ensure_dir
from nhanes_mortality import (
    PUBLIC_MORTALITY_FILES,
    download_public_mortality_files,
    load_mortality,
    merge_participants_with_mortality,
)


@dataclass(frozen=True)
class CycleConfig:
    cycle_start_year: int
    demo_file: str
    rhq_file: str


CYCLES = [
    CycleConfig(2001, "DEMO_B.xpt", "RHQ_B.xpt"),
    CycleConfig(2003, "DEMO_C.xpt", "RHQ_C.xpt"),
    CycleConfig(2005, "DEMO_D.xpt", "RHQ_D.xpt"),
    CycleConfig(2007, "DEMO_E.xpt", "RHQ_E.xpt"),
    CycleConfig(2009, "DEMO_F.xpt", "RHQ_F.xpt"),
    CycleConfig(2011, "DEMO_G.xpt", "RHQ_G.xpt"),
    CycleConfig(2013, "DEMO_H.xpt", "RHQ_H.xpt"),
    CycleConfig(2015, "DEMO_I.xpt", "RHQ_I.xpt"),
    CycleConfig(2017, "DEMO_J.xpt", "RHQ_J.xpt"),
]

MENARCHE_INVALID_CODES = {0, 7, 9, 77, 99, 777, 999}
MENOPAUSE_INVALID_CODES = {7, 9, 77, 99, 777, 999}
INTERVIEW_AGE_INVALID_CODES = {7, 9, 77, 99}
ROBUST_NATURAL_MENOPAUSE_MIN_CYCLE = 2007
ROBUST_NATURAL_MENOPAUSE_MIN_AGE = 40
ROBUST_NATURAL_MENOPAUSE_MAX_AGE = 65


def clean_age(series: pd.Series, *, invalid_codes: set[int], min_age: float, max_age: float) -> pd.Series:
    values = pd.to_numeric(series, errors="coerce")
    values = values.where(~values.isin(invalid_codes))
    values = values.where(values >= min_age)
    values = values.where(values <= max_age)
    return values


def clean_flag(series: pd.Series) -> pd.Series:
    return pd.to_numeric(series, errors="coerce")


def get_series(df: pd.DataFrame, column: str) -> pd.Series:
    if column in df.columns:
        return df[column]
    return pd.Series(pd.NA, index=df.index)


def load_xpt(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"Missing NHANES file: {path}")
    return pd.read_sas(path, format="xport", encoding="utf-8")


def build_reproductive_frame(
    raw_dir: Path,
    *,
    min_menopause_cycle_year: int = ROBUST_NATURAL_MENOPAUSE_MIN_CYCLE,
    min_natural_menopause_age: float = ROBUST_NATURAL_MENOPAUSE_MIN_AGE,
    max_natural_menopause_age: float = ROBUST_NATURAL_MENOPAUSE_MAX_AGE,
) -> pd.DataFrame:
    frames: list[pd.DataFrame] = []

    for config in CYCLES:
        frame = build_cycle_frame(
            raw_dir,
            config,
            min_menopause_cycle_year=min_menopause_cycle_year,
            min_natural_menopause_age=min_natural_menopause_age,
            max_natural_menopause_age=max_natural_menopause_age,
        )
        frames.append(frame)

    if not frames:
        return pd.DataFrame()

    combined = pd.concat(frames, ignore_index=True)
    combined = combined.loc[combined["sex_code"].eq(2)].copy()
    return combined


def build_cycle_frame(
    raw_dir: Path,
    config: CycleConfig,
    *,
    min_menopause_cycle_year: int = ROBUST_NATURAL_MENOPAUSE_MIN_CYCLE,
    min_natural_menopause_age: float = ROBUST_NATURAL_MENOPAUSE_MIN_AGE,
    max_natural_menopause_age: float = ROBUST_NATURAL_MENOPAUSE_MAX_AGE,
) -> pd.DataFrame:
    cycle_dir = raw_dir / str(config.cycle_start_year)
    demo = load_xpt(cycle_dir / config.demo_file)
    rhq = load_xpt(cycle_dir / config.rhq_file)

    demo_subset = pd.DataFrame(
        {
            "seqn": pd.to_numeric(demo.get("SEQN"), errors="coerce").astype("Int64"),
            "sex_code": clean_flag(demo.get("RIAGENDR")),
            "age_years": clean_age(
                demo.get("RIDAGEYR"),
                invalid_codes=INTERVIEW_AGE_INVALID_CODES,
                min_age=0,
                max_age=120,
            ),
        }
    )

    rhq_subset = pd.DataFrame(
        {
            "seqn": pd.to_numeric(rhq.get("SEQN"), errors="coerce").astype("Int64"),
            "age_menarche": clean_age(
                get_series(rhq, "RHQ010"),
                invalid_codes=MENARCHE_INVALID_CODES,
                min_age=5,
                max_age=25,
            ),
            "age_last_period": clean_age(
                get_series(rhq, "RHQ060"),
                invalid_codes=MENOPAUSE_INVALID_CODES,
                min_age=20,
                max_age=65,
            ),
        }
    )

    rhq_subset["age_menopause"] = build_natural_menopause_age(
        rhq=rhq,
        cycle_start_year=config.cycle_start_year,
        age_last_period=rhq_subset["age_last_period"],
        min_menopause_cycle_year=min_menopause_cycle_year,
        min_natural_menopause_age=min_natural_menopause_age,
        max_natural_menopause_age=max_natural_menopause_age,
    )
    rhq_subset = rhq_subset.drop(columns=["age_last_period"])

    merged = demo_subset.merge(rhq_subset, on="seqn", how="inner")
    merged["cycle_start_year"] = config.cycle_start_year
    return merged


def build_no_period_flag(rhq: pd.DataFrame) -> pd.Series:
    if "RHQ031" in rhq.columns:
        return clean_flag(rhq["RHQ031"]).eq(2)

    if "RHQ030" in rhq.columns:
        return clean_flag(rhq["RHQ030"]).eq(2)

    return pd.Series(False, index=rhq.index)


def build_menopause_reason_flag(rhq: pd.DataFrame, cycle_start_year: int) -> pd.Series:
    if cycle_start_year < 2007:
        return pd.Series(False, index=rhq.index)

    if cycle_start_year <= 2011:
        reason = clean_flag(get_series(rhq, "RHD042")).eq(7)
        no_hysterectomy = clean_flag(get_series(rhq, "RHD280")).eq(2)
        return reason & no_hysterectomy

    reason = clean_flag(get_series(rhq, "RHD043")).eq(7)
    return reason


def build_natural_menopause_age(
    *,
    rhq: pd.DataFrame,
    cycle_start_year: int,
    age_last_period: pd.Series,
    min_menopause_cycle_year: int,
    min_natural_menopause_age: float,
    max_natural_menopause_age: float,
) -> pd.Series:
    if cycle_start_year < min_menopause_cycle_year:
        return pd.Series(np.nan, index=rhq.index, dtype="float64")

    has_no_period_last_12_months = build_no_period_flag(rhq)
    menopause_reason_ok = build_menopause_reason_flag(rhq, cycle_start_year)

    age_menopause = age_last_period.where(has_no_period_last_12_months & menopause_reason_ok)
    age_menopause = age_menopause.where(age_menopause >= min_natural_menopause_age)
    age_menopause = age_menopause.where(age_menopause <= max_natural_menopause_age)
    return age_menopause


def add_mortality_columns(reproductive: pd.DataFrame, mortality_dir: Path) -> pd.DataFrame:
    download_public_mortality_files(mortality_dir, PUBLIC_MORTALITY_FILES)
    mortality = load_mortality(mortality_dir, PUBLIC_MORTALITY_FILES)
    merged = merge_participants_with_mortality(reproductive, mortality)

    merged["mortstat"] = pd.to_numeric(merged.get("mortstat"), errors="coerce")
    merged["permth_int"] = pd.to_numeric(merged.get("permth_int"), errors="coerce")
    merged["permth_exm"] = pd.to_numeric(merged.get("permth_exm"), errors="coerce")
    merged["agedeath"] = pd.to_numeric(merged.get("agedeath"), errors="coerce")

    merged["time_months"] = merged["permth_int"].where(merged["permth_int"].notna(), merged["permth_exm"])
    merged["age_at_death"] = merged["agedeath"].where(
        merged["agedeath"].notna(),
        merged["age_years"] + merged["time_months"] / 12.0,
    )
    merged["age_at_death"] = merged["age_at_death"].where(merged["mortstat"].eq(1))
    return merged


def build_plot_frame(df: pd.DataFrame, x_col: str, y_col: str) -> pd.DataFrame:
    subset = df[[x_col, y_col, "seqn", "cycle_start_year"]].copy()
    subset = subset.dropna(subset=[x_col, y_col])
    subset = subset.loc[np.isfinite(subset[x_col]) & np.isfinite(subset[y_col])].copy()
    subset = subset.loc[subset[y_col] >= subset[x_col]].copy()
    return subset


def save_scatter_plot(
    df: pd.DataFrame,
    *,
    x_col: str,
    y_col: str,
    x_label: str,
    y_label: str,
    title: str,
    out_path: Path,
) -> dict[str, float | int | str]:
    if len(df) < 3:
        raise ValueError(f"Need at least 3 rows to plot {out_path.name}; found {len(df)}")

    r_value, p_value = pearsonr(df[x_col], df[y_col])

    fig, ax = plt.subplots(figsize=(8, 6))
    ax.scatter(df[x_col], df[y_col], s=24, alpha=0.55, color="#1f77b4", edgecolor="none")

    slope, intercept = np.polyfit(df[x_col], df[y_col], deg=1)
    x_line = np.linspace(df[x_col].min(), df[x_col].max(), 100)
    y_line = intercept + slope * x_line
    ax.plot(x_line, y_line, color="#d62728", linewidth=2)

    ax.set_title(title)
    ax.set_xlabel(x_label)
    ax.set_ylabel(y_label)
    ax.grid(alpha=0.25)

    annotation = f"n = {len(df)}\nPearson r = {r_value:.3f}\np = {p_value:.3g}"
    ax.text(
        0.03,
        0.97,
        annotation,
        transform=ax.transAxes,
        va="top",
        ha="left",
        bbox={"boxstyle": "round,pad=0.4", "facecolor": "white", "alpha": 0.9, "edgecolor": "#cccccc"},
    )

    fig.tight_layout()
    fig.savefig(out_path, dpi=200)
    plt.close(fig)

    return {
        "plot_name": out_path.stem,
        "n": int(len(df)),
        "pearson_r": float(r_value),
        "pearson_p_value": float(p_value),
    }


def save_density_plot(
    series: pd.Series,
    *,
    x_label: str,
    title: str,
    out_path: Path,
) -> dict[str, float | int | str]:
    values = pd.to_numeric(series, errors="coerce").dropna()
    values = values.loc[np.isfinite(values)]

    if len(values) < 3:
        raise ValueError(f"Need at least 3 values to plot {out_path.name}; found {len(values)}")

    fig, ax = plt.subplots(figsize=(8, 6))
    values.plot.kde(ax=ax, linewidth=2, color="#1f77b4")
    ax.set_title(title)
    ax.set_xlabel(x_label)
    ax.set_ylabel("Density")
    ax.grid(alpha=0.25)

    annotation = f"n = {len(values)}\nmean = {values.mean():.2f}\nmedian = {values.median():.2f}"
    ax.text(
        0.03,
        0.97,
        annotation,
        transform=ax.transAxes,
        va="top",
        ha="left",
        bbox={"boxstyle": "round,pad=0.4", "facecolor": "white", "alpha": 0.9, "edgecolor": "#cccccc"},
    )

    fig.tight_layout()
    fig.savefig(out_path, dpi=200)
    plt.close(fig)

    return {
        "plot_name": out_path.stem,
        "n": int(len(values)),
        "mean": float(values.mean()),
        "median": float(values.median()),
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--raw-dir", default="data/raw")
    ap.add_argument("--mortality-dir", default="data/raw/mortality")
    ap.add_argument("--out-dir", default="output/plots")
    ap.add_argument("--min-menopause-cycle-year", type=int, default=ROBUST_NATURAL_MENOPAUSE_MIN_CYCLE)
    ap.add_argument("--min-natural-menopause-age", type=float, default=ROBUST_NATURAL_MENOPAUSE_MIN_AGE)
    ap.add_argument("--max-natural-menopause-age", type=float, default=ROBUST_NATURAL_MENOPAUSE_MAX_AGE)
    args = ap.parse_args()

    raw_dir = Path(args.raw_dir)
    mortality_dir = Path(args.mortality_dir)
    out_dir = Path(args.out_dir)
    ensure_dir(out_dir)

    reproductive = build_reproductive_frame(
        raw_dir,
        min_menopause_cycle_year=args.min_menopause_cycle_year,
        min_natural_menopause_age=args.min_natural_menopause_age,
        max_natural_menopause_age=args.max_natural_menopause_age,
    )
    merged = add_mortality_columns(reproductive, mortality_dir)

    plot_specs = [
        {
            "plot_name": "menarche_vs_menopause",
            "x_col": "age_menarche",
            "y_col": "age_menopause",
            "x_label": "Age at menarche",
            "y_label": "Age at menopause",
            "title": "Age at menarche vs age at menopause",
            "frame": build_plot_frame(merged, "age_menarche", "age_menopause"),
        },
        {
            "plot_name": "menarche_vs_age_at_death",
            "x_col": "age_menarche",
            "y_col": "age_at_death",
            "x_label": "Age at menarche",
            "y_label": "Age at death",
            "title": "Age at menarche vs age at death",
            "frame": build_plot_frame(merged.loc[merged["mortstat"].eq(1)], "age_menarche", "age_at_death"),
        },
        {
            "plot_name": "menopause_vs_age_at_death",
            "x_col": "age_menopause",
            "y_col": "age_at_death",
            "x_label": "Age at menopause",
            "y_label": "Age at death",
            "title": "Age at menopause vs age at death",
            "frame": build_plot_frame(merged.loc[merged["mortstat"].eq(1)], "age_menopause", "age_at_death"),
        },
    ]

    summary_rows = []
    for spec in plot_specs:
        plot_path = out_dir / f"{spec['plot_name']}.png"
        data_path = out_dir / f"{spec['plot_name']}_data.csv"

        frame = spec["frame"]
        frame.to_csv(data_path, index=False)

        summary = save_scatter_plot(
            frame,
            x_col=spec["x_col"],
            y_col=spec["y_col"],
            x_label=spec["x_label"],
            y_label=spec["y_label"],
            title=spec["title"],
            out_path=plot_path,
        )
        summary_rows.append(summary)

    density_specs = [
        {
            "plot_name": "menarche_age_kde",
            "series": merged["age_menarche"],
            "x_label": "Age at menarche",
            "title": "Age at menarche distribution",
        },
        {
            "plot_name": "menopause_age_kde",
            "series": merged["age_menopause"],
            "x_label": "Age at menopause",
            "title": "Age at menopause distribution",
        },
    ]

    density_rows = []
    for spec in density_specs:
        plot_path = out_dir / f"{spec['plot_name']}.png"
        summary = save_density_plot(
            spec["series"],
            x_label=spec["x_label"],
            title=spec["title"],
            out_path=plot_path,
        )
        density_rows.append(summary)

    summary_df = pd.DataFrame(summary_rows)
    summary_path = out_dir / "reproductive_scatter_summary.csv"
    summary_df.to_csv(summary_path, index=False)

    density_df = pd.DataFrame(density_rows)
    density_path = out_dir / "reproductive_density_summary.csv"
    density_df.to_csv(density_path, index=False)

    print(summary_df.to_string(index=False))
    print()
    print(density_df.to_string(index=False))
    print(f"\nSaved plots to {out_dir}")


if __name__ == "__main__":
    main()
