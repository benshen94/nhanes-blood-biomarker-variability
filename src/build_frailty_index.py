#!/usr/bin/env python3
"""Build a participant-level frailty index from local NHANES files.

This script is intentionally conservative and easy to audit.

Main design choices:
- Use NHANES 2005-2017 for the primary FI because local files contain a stable
  set of HRS-overlap chronic conditions, self-rated health, and function items.
- Treat NHANES as repeated cross-sectional. Each row is one participant in one
  exam cycle, not a longitudinal person-wave panel as in HRS.
- Build HRS-overlap FI variants first, then a screened FI using simple quality
  checks inspired by the Rockwood / Theou guidance.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import math

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pyreadstat
from scipy.stats import gaussian_kde


ROOT = Path(__file__).resolve().parents[1]
RAW_DIR = ROOT / "data" / "raw"
OUTPUT_DIR = ROOT / "output" / "frailty"
FIGURE_DIR = OUTPUT_DIR / "figures"
CYCLES = [2005, 2007, 2009, 2011, 2013, 2015, 2017]


@dataclass(frozen=True)
class DeficitSpec:
    name: str
    source: str
    question: str
    domain: str
    coding: str
    variant_group: str


FI_DEFINITIONS = [
    DeficitSpec("hypertension", "BPQ020", "Ever told you had high blood pressure", "disease", "1=yes, 2=no", "clinical"),
    DeficitSpec("diabetes", "DIQ010", "Doctor told you have diabetes", "disease", "1=yes, 2=no", "clinical"),
    DeficitSpec("cancer", "MCQ220", "Ever told you had cancer or malignancy", "disease", "1=yes, 2=no", "clinical"),
    DeficitSpec("emphysema", "MCQ160G", "Ever told you had emphysema", "disease", "1=yes, 2=no", "clinical"),
    DeficitSpec("chronic_bronchitis_ever", "MCQ160K", "Ever told you had chronic bronchitis", "disease", "1=yes, 2=no", "clinical"),
    DeficitSpec("lung_disease_any", "MCQ160G|MCQ160K", "Any chronic lung disease proxy", "disease", "1 if emphysema or chronic bronchitis", "clinical"),
    DeficitSpec("heart_failure", "MCQ160B", "Ever told had congestive heart failure", "disease", "1=yes, 2=no", "clinical"),
    DeficitSpec("coronary_heart_disease", "MCQ160C", "Ever told had coronary heart disease", "disease", "1=yes, 2=no", "clinical"),
    DeficitSpec("angina", "MCQ160D", "Ever told had angina", "disease", "1=yes, 2=no", "clinical"),
    DeficitSpec("heart_attack", "MCQ160E", "Ever told had heart attack", "disease", "1=yes, 2=no", "clinical"),
    DeficitSpec("heart_disease_any", "MCQ160B|MCQ160C|MCQ160D|MCQ160E", "Any major heart disease proxy", "disease", "1 if CHF, CHD, angina, or MI", "clinical"),
    DeficitSpec("stroke", "MCQ160F", "Ever told had stroke", "disease", "1=yes, 2=no", "clinical"),
    DeficitSpec("arthritis", "MCQ160A", "Doctor ever said you had arthritis", "disease", "1=yes, 2=no", "clinical"),
    DeficitSpec("general_health", "HUQ010", "General health condition", "global_health", r"\(1=\) excellent to \(5=\) poor, recoded to \(0\) to \(1\)", "clinical"),
    DeficitSpec("memory_problem", "PFQ057", "Experience confusion or memory problems", "cognition", "1=yes, 2=no", "clinical"),
    DeficitSpec("manage_money", "PFQ061A", "Managing money difficulty", "iadl", r"\(1=\) no, \(2-4=\) deficit, \(5=\) does not do", "clinical"),
    DeficitSpec("house_chore", "PFQ061F", "House chore difficulty", "iadl", r"\(1=\) no, \(2-4=\) deficit, \(5=\) does not do", "clinical"),
    DeficitSpec("prepare_meals", "PFQ061G", "Preparing meals difficulty", "iadl", r"\(1=\) no, \(2-4=\) deficit, \(5=\) does not do", "clinical"),
    DeficitSpec("walk_room", "PFQ061H", "Walking between rooms on same floor", "adl", r"\(1=\) no, \(2-4=\) deficit, \(5=\) does not do", "clinical"),
    DeficitSpec("chair_rise", "PFQ061I", "Standing up from armless chair difficulty", "mobility", r"\(1=\) no, \(2-4=\) deficit, \(5=\) does not do", "clinical"),
    DeficitSpec("bed_transfer", "PFQ061J", "Getting in and out of bed difficulty", "adl", r"\(1=\) no, \(2-4=\) deficit, \(5=\) does not do", "clinical"),
    DeficitSpec("eat_utensils", "PFQ061K", "Using fork, knife, or cup difficulty", "adl", r"\(1=\) no, \(2-4=\) deficit, \(5=\) does not do", "clinical"),
    DeficitSpec("dress_self", "PFQ061L", "Dressing yourself difficulty", "adl", r"\(1=\) no, \(2-4=\) deficit, \(5=\) does not do", "clinical"),
    DeficitSpec("walk_quarter_mile", "PFQ061B", "Walking for a quarter mile difficulty", "mobility", r"\(1=\) no, \(2-4=\) deficit, \(5=\) does not do", "clinical"),
    DeficitSpec("walk_up_steps", "PFQ061C", "Walking up ten steps difficulty", "mobility", r"\(1=\) no, \(2-4=\) deficit, \(5=\) does not do", "clinical"),
    DeficitSpec("stoop", "PFQ061D", "Stooping, crouching, kneeling difficulty", "mobility", r"\(1=\) no, \(2-4=\) deficit, \(5=\) does not do", "clinical"),
    DeficitSpec("lift_carry", "PFQ061E", "Lifting or carrying difficulty", "mobility", r"\(1=\) no, \(2-4=\) deficit, \(5=\) does not do", "clinical"),
    DeficitSpec("reach_overhead", "PFQ061O", "Reaching up over head difficulty", "mobility", r"\(1=\) no, \(2-4=\) deficit, \(5=\) does not do", "clinical"),
    DeficitSpec("push_pull", "PFQ061T", "Push or pull large objects difficulty", "mobility", r"\(1=\) no, \(2-4=\) deficit, \(5=\) does not do", "clinical"),
    DeficitSpec("grasp_small_objects", "PFQ061P", "Grasping or holding small objects difficulty", "mobility", r"\(1=\) no, \(2-4=\) deficit, \(5=\) does not do", "clinical"),
]

HRS_OVERLAP_ITEMS = [
    "def_hypertension",
    "def_diabetes",
    "def_cancer",
    "def_lung_disease_any",
    "def_heart_disease_any",
    "def_stroke",
    "def_arthritis",
    "def_general_health",
    "def_walk_room",
    "def_bed_transfer",
    "def_eat_utensils",
    "def_dress_self",
    "def_manage_money",
    "def_house_chore",
    "def_prepare_meals",
    "def_chair_rise",
    "def_walk_quarter_mile",
    "def_walk_up_steps",
    "def_stoop",
    "def_lift_carry",
    "def_push_pull",
    "def_reach_overhead",
]

HRS_OVERLAP_MEMORY_ITEMS = [*HRS_OVERLAP_ITEMS, "def_memory_problem"]


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    FIGURE_DIR.mkdir(parents=True, exist_ok=True)

    panel = build_panel()
    panel = add_recoded_deficits(panel)

    variant_map, screening_log = build_variant_map(panel)
    fi_panel = add_frailty_scores(panel, variant_map)
    primary_variant = "fi_hrs_overlap_22"

    catalog = build_deficit_catalog(variant_map)
    variant_summary = build_variant_summary(fi_panel, variant_map)
    cycle_summary = build_cycle_summary(fi_panel)
    age_bin_summary = build_age_bin_summary(fi_panel)
    biomarker_overlap = build_biomarker_overlap_counts(fi_panel, primary_variant)

    fi_panel.to_csv(OUTPUT_DIR / "frailty_panel.csv.gz", index=False)
    catalog.to_csv(OUTPUT_DIR / "deficit_catalog.csv", index=False)
    screening_log.to_csv(OUTPUT_DIR / "screening_log.csv", index=False)
    variant_summary.to_csv(OUTPUT_DIR / "variant_summary.csv", index=False)
    cycle_summary.to_csv(OUTPUT_DIR / "trajectory_by_cycle.csv", index=False)
    age_bin_summary.to_csv(OUTPUT_DIR / "distribution_by_age_bin.csv", index=False)
    biomarker_overlap.to_csv(OUTPUT_DIR / "biomarker_overlap_counts.csv", index=False)

    write_readme(variant_summary, primary_variant, biomarker_overlap)
    make_cycle_plot(cycle_summary)
    make_age_plot(age_bin_summary)
    make_age_bin_kde_plot(fi_panel, primary_variant)


def build_panel() -> pd.DataFrame:
    rows = []

    for cycle_year in CYCLES:
        cycle_panel = build_cycle_panel(cycle_year)
        if cycle_panel.empty:
            continue
        rows.append(cycle_panel)

    if not rows:
        raise RuntimeError("No NHANES frailty data could be assembled from local raw files.")

    panel = pd.concat(rows, ignore_index=True)
    panel = panel.loc[panel["age_years"] >= 60].copy()

    panel["is_female"] = np.where(panel["sex"] == "Female", 1.0, 0.0)
    panel["age_bin"] = pd.cut(
        panel["age_years"],
        bins=[60, 65, 70, 75, 80, 85, 90, math.inf],
        right=False,
        labels=["60-64", "65-69", "70-74", "75-79", "80-84", "85-89", "90+"],
    )
    return panel


def build_cycle_panel(cycle_year: int) -> pd.DataFrame:
    demo_path = find_first(cycle_year, ["DEMO*.xpt", "P_DEMO.xpt"])
    if demo_path is None:
        return pd.DataFrame()

    demo_columns = ["SEQN", "RIDAGEYR", "RIAGENDR", "WTINT2YR", "WTMEC2YR"]
    demo = read_xpt(demo_path, demo_columns)
    if demo.empty:
        return pd.DataFrame()

    panel = pd.DataFrame(
        {
            "seqn": to_int_series(demo["SEQN"]),
            "age_years": to_float_series(demo["RIDAGEYR"]),
            "sex": to_float_series(demo["RIAGENDR"]).map({1.0: "Male", 2.0: "Female"}),
            "wtint2yr": to_float_series(demo.get("WTINT2YR")),
            "wtmec2yr": to_float_series(demo.get("WTMEC2YR")),
            "cycle_start_year": float(cycle_year),
        }
    )

    panel = panel.dropna(subset=["seqn", "age_years"]).copy()

    merge_into(panel, cycle_year, "BPQ*.xpt", ["BPQ020"])
    merge_into(panel, cycle_year, "DIQ*.xpt", ["DIQ010"])
    merge_into(panel, cycle_year, "KIQ_U*.xpt", ["KIQ022"])
    merge_into(panel, cycle_year, "MCQ*.xpt", ["MCQ010", "MCQ160A", "MCQ160B", "MCQ160C", "MCQ160D", "MCQ160E", "MCQ160F", "MCQ160G", "MCQ160K", "MCQ170K", "MCQ170L", "MCQ170M", "MCQ220"])
    merge_into(panel, cycle_year, "OSQ*.xpt", ["OSQ060"])
    merge_into(panel, cycle_year, "HUQ*.xpt", ["HUQ010"])
    merge_into(panel, cycle_year, "PFQ*.xpt", ["PFQ057", "PFQ061A", "PFQ061B", "PFQ061C", "PFQ061D", "PFQ061E", "PFQ061F", "PFQ061G", "PFQ061H", "PFQ061I", "PFQ061J", "PFQ061K", "PFQ061L", "PFQ061M", "PFQ061N", "PFQ061O", "PFQ061P", "PFQ061T"])
    merge_into(panel, cycle_year, "CRP*.xpt", ["LBXCRP"], {"LBXCRP": "crp_mg_dl"})
    merge_into(panel, cycle_year, "HSCRP*.xpt", ["LBXHSCRP"], {"LBXHSCRP": "hscrp_mg_l"})
    merge_into(panel, cycle_year, "INS*.xpt", ["LBXIN"], {"LBXIN": "insulin_uU_ml"})
    merge_into(panel, cycle_year, "GLU*.xpt", ["LBXGLU"], {"LBXGLU": "fasting_glucose_mg_dl"})
    merge_into(panel, cycle_year, "TRIGLY*.xpt", ["LBXTR"], {"LBXTR": "triglycerides_mg_dl"})
    merge_into(panel, cycle_year, "CBC*.xpt", ["LBDEONO", "LBXHGB", "LBXRDW", "LBXWBCSI"], {"LBDEONO": "eosinophils_abs_1000_uL"})

    panel = panel.drop_duplicates(subset=["seqn", "cycle_start_year"], keep="last")
    return panel


def find_first(cycle_year: int, patterns: list[str]) -> Path | None:
    cycle_dir = RAW_DIR / str(cycle_year)
    if not cycle_dir.exists():
        return None

    for pattern in patterns:
        hits = sorted(cycle_dir.glob(pattern))
        if hits:
            return hits[0]

    return None


def merge_into(
    panel: pd.DataFrame,
    cycle_year: int,
    pattern: str,
    columns: list[str],
    rename_map: dict[str, str] | None = None,
) -> None:
    path = find_first(cycle_year, [pattern])
    if path is None:
        return

    frame = read_xpt(path, ["SEQN", *columns])
    if frame.empty:
        return

    if rename_map:
        frame = frame.rename(columns=rename_map)

    frame = frame.rename(columns={"SEQN": "seqn"})
    frame["seqn"] = to_int_series(frame["seqn"])
    merged = panel.merge(frame, on="seqn", how="left")

    for column in frame.columns:
        if column == "seqn":
            continue
        panel[column] = merged[column]


def read_xpt(path: Path, usecols: list[str]) -> pd.DataFrame:
    if not usecols:
        return pd.DataFrame()

    try:
        frame, _ = pyreadstat.read_xport(str(path), usecols=usecols)
        return frame
    except Exception:
        return pd.DataFrame()


def to_int_series(series: pd.Series | None) -> pd.Series:
    if series is None:
        return pd.Series(dtype="Int64")
    return pd.to_numeric(series, errors="coerce").astype("Int64")


def to_float_series(series: pd.Series | None) -> pd.Series:
    if series is None:
        return pd.Series(dtype=float)
    return pd.to_numeric(series, errors="coerce")


def add_recoded_deficits(panel: pd.DataFrame) -> pd.DataFrame:
    panel = panel.copy()

    binary_yes_no = [
        "hypertension",
        "diabetes",
        "cancer",
        "arthritis",
        "heart_failure",
        "coronary_heart_disease",
        "angina",
        "heart_attack",
        "stroke",
        "emphysema",
        "chronic_bronchitis_ever",
        "memory_problem",
    ]
    difficulty_items = [
        "manage_money",
        "walk_quarter_mile",
        "walk_up_steps",
        "stoop",
        "lift_carry",
        "house_chore",
        "prepare_meals",
        "walk_room",
        "chair_rise",
        "bed_transfer",
        "eat_utensils",
        "dress_self",
        "reach_overhead",
        "grasp_small_objects",
        "push_pull",
    ]

    source_map = {
        "hypertension": "BPQ020",
        "diabetes": "DIQ010",
        "cancer": "MCQ220",
        "arthritis": "MCQ160A",
        "heart_failure": "MCQ160B",
        "coronary_heart_disease": "MCQ160C",
        "angina": "MCQ160D",
        "heart_attack": "MCQ160E",
        "stroke": "MCQ160F",
        "emphysema": "MCQ160G",
        "chronic_bronchitis_ever": "MCQ160K",
        "memory_problem": "PFQ057",
        "manage_money": "PFQ061A",
        "walk_quarter_mile": "PFQ061B",
        "walk_up_steps": "PFQ061C",
        "stoop": "PFQ061D",
        "lift_carry": "PFQ061E",
        "house_chore": "PFQ061F",
        "prepare_meals": "PFQ061G",
        "walk_room": "PFQ061H",
        "chair_rise": "PFQ061I",
        "bed_transfer": "PFQ061J",
        "eat_utensils": "PFQ061K",
        "dress_self": "PFQ061L",
        "reach_overhead": "PFQ061O",
        "grasp_small_objects": "PFQ061P",
        "push_pull": "PFQ061T",
    }

    for name in binary_yes_no:
        panel[f"def_{name}"] = recode_binary_yes_no(panel.get(source_map[name]))

    for name in difficulty_items:
        panel[f"def_{name}"] = recode_difficulty(panel.get(source_map[name]))

    panel["def_general_health"] = recode_self_rated_health(panel.get("HUQ010"))
    panel["def_lung_disease_any"] = recode_any_yes_no(
        [panel.get("MCQ160G"), panel.get("MCQ160K")]
    )
    panel["def_heart_disease_any"] = recode_any_yes_no(
        [panel.get("MCQ160B"), panel.get("MCQ160C"), panel.get("MCQ160D"), panel.get("MCQ160E")]
    )

    return panel


def recode_binary_yes_no(series: pd.Series | None) -> pd.Series:
    if series is None:
        return pd.Series(dtype=float)

    out = pd.Series(np.nan, index=series.index, dtype=float)
    numeric = pd.to_numeric(series, errors="coerce")
    out.loc[numeric == 1] = 1.0
    out.loc[numeric == 2] = 0.0
    return out


def recode_difficulty(series: pd.Series | None) -> pd.Series:
    if series is None:
        return pd.Series(dtype=float)

    out = pd.Series(np.nan, index=series.index, dtype=float)
    numeric = pd.to_numeric(series, errors="coerce")
    out.loc[numeric == 1] = 0.0
    out.loc[numeric.isin([2, 3, 4])] = 1.0
    return out


def recode_any_yes_no(series_list: list[pd.Series | None]) -> pd.Series:
    valid_series = [pd.to_numeric(series, errors="coerce") for series in series_list if series is not None]
    if not valid_series:
        return pd.Series(dtype=float)

    frame = pd.concat(valid_series, axis=1)
    out = pd.Series(np.nan, index=frame.index, dtype=float)
    any_observed = frame.notna().any(axis=1)
    any_yes = frame.eq(1).any(axis=1)
    all_no = ((frame == 2) | frame.isna()).all(axis=1)

    out.loc[any_observed & all_no] = 0.0
    out.loc[any_observed & any_yes] = 1.0
    return out


def recode_self_rated_health(series: pd.Series | None) -> pd.Series:
    if series is None:
        return pd.Series(dtype=float)

    numeric = pd.to_numeric(series, errors="coerce")
    numeric = numeric.where(numeric.between(1, 5), np.nan)
    return (numeric - 1.0) / 4.0


def build_variant_map(panel: pd.DataFrame) -> tuple[dict[str, list[str]], pd.DataFrame]:
    screened_items, screening_log = run_screening(panel, HRS_OVERLAP_MEMORY_ITEMS)
    variant_map = {
        f"fi_hrs_overlap_{len(HRS_OVERLAP_ITEMS)}": HRS_OVERLAP_ITEMS,
        f"fi_hrs_overlap_memory_{len(HRS_OVERLAP_MEMORY_ITEMS)}": HRS_OVERLAP_MEMORY_ITEMS,
        f"fi_screened_{len(screened_items)}": screened_items,
    }
    return variant_map, screening_log


def run_screening(panel: pd.DataFrame, candidate_columns: list[str]) -> tuple[list[str], pd.DataFrame]:
    provisional = []
    screening_rows = []

    for column in candidate_columns:
        wave_missingness = cycle_missingness_rates(panel, column)
        max_cycle_missingness = max(wave_missingness.values())
        prevalence = panel[column].mean(skipna=True)
        age_correlation = panel[["age_years", column]].corr(method="spearman").iloc[0, 1]

        reason = "retained_before_correlation"

        if max_cycle_missingness > 0.20:
            reason = "excluded_high_cycle_missingness"
        elif pd.isna(prevalence) or prevalence < 0.01 or prevalence > 0.80:
            reason = "excluded_prevalence_out_of_range"
        elif pd.isna(age_correlation) or age_correlation <= 0:
            reason = "excluded_nonpositive_age_correlation"
        else:
            provisional.append(column)

        screening_rows.append(
            {
                "column": column,
                "max_cycle_missingness": max_cycle_missingness,
                "prevalence": prevalence,
                "age_spearman_r": age_correlation,
                "screen_result": reason,
            }
        )

    retained = remove_redundant_items(panel, provisional)
    screening_log = pd.DataFrame(screening_rows)
    screening_log["retained_after_correlation_screen"] = screening_log["column"].isin(provisional)
    screening_log["retained_final"] = screening_log["column"].isin(retained)
    return retained, screening_log


def cycle_missingness_rates(panel: pd.DataFrame, column: str) -> dict[float, float]:
    out = {}

    for cycle_year, frame in panel.groupby("cycle_start_year"):
        out[cycle_year] = float(frame[column].isna().mean())

    return out


def remove_redundant_items(panel: pd.DataFrame, columns: list[str]) -> list[str]:
    retained = []

    for column in columns:
        keep = True

        for prior in retained:
            corr = panel[[prior, column]].corr(method="spearman").iloc[0, 1]
            if pd.notna(corr) and abs(corr) >= 0.70:
                keep = False
                break

        if keep:
            retained.append(column)

    return retained


def add_frailty_scores(panel: pd.DataFrame, variant_map: dict[str, list[str]]) -> pd.DataFrame:
    out = panel.copy()

    for variant_name, columns in variant_map.items():
        observed = out[columns].notna().sum(axis=1)
        deficit_sum = out[columns].sum(axis=1, skipna=True)
        minimum_observed = math.ceil(0.8 * len(columns))

        out[f"{variant_name}_observed"] = observed
        out[variant_name] = np.where(observed >= minimum_observed, deficit_sum / observed, np.nan)

    return out


def build_deficit_catalog(variant_map: dict[str, list[str]]) -> pd.DataFrame:
    by_name = {f"def_{definition.name}": definition for definition in FI_DEFINITIONS}
    rows = []

    for variant_name, columns in variant_map.items():
        for column in columns:
            definition = by_name[column]
            rows.append(
                {
                    "variant": variant_name,
                    "deficit_column": column,
                    "source_variable": definition.source,
                    "question": definition.question,
                    "domain": definition.domain,
                    "coding": definition.coding,
                }
            )

    return pd.DataFrame(rows)


def build_variant_summary(panel: pd.DataFrame, variant_map: dict[str, list[str]]) -> pd.DataFrame:
    rows = []

    for variant_name, columns in variant_map.items():
        series = panel[variant_name].dropna()
        if series.empty:
            continue

        rows.append(
            {
                "variant": variant_name,
                "n_items": len(columns),
                "n_people": int(series.shape[0]),
                "mean": float(series.mean()),
                "sd": float(series.std()),
                "median": float(series.median()),
                "p90": float(series.quantile(0.90)),
                "p99": float(series.quantile(0.99)),
                "age_spearman_r": float(panel[["age_years", variant_name]].corr(method="spearman").iloc[0, 1]),
                "female_minus_male_mean": float(
                    panel.loc[panel["sex"] == "Female", variant_name].mean()
                    - panel.loc[panel["sex"] == "Male", variant_name].mean()
                ),
            }
        )

    return pd.DataFrame(rows)


def build_cycle_summary(panel: pd.DataFrame) -> pd.DataFrame:
    variants = [column for column in panel.columns if column.startswith("fi_") and not column.endswith("_observed")]
    rows = []

    for cycle_year, frame in panel.groupby("cycle_start_year"):
        for variant_name in variants:
            series = frame[variant_name].dropna()
            if series.empty:
                continue
            rows.append(
                {
                    "cycle_start_year": cycle_year,
                    "variant": variant_name,
                    "n_people": int(series.shape[0]),
                    "mean": float(series.mean()),
                    "median": float(series.median()),
                }
            )

    return pd.DataFrame(rows)


def build_age_bin_summary(panel: pd.DataFrame) -> pd.DataFrame:
    variants = [column for column in panel.columns if column.startswith("fi_") and not column.endswith("_observed")]
    rows = []

    for age_bin, frame in panel.groupby("age_bin", observed=False):
        if pd.isna(age_bin):
            continue

        for variant_name in variants:
            series = frame[variant_name].dropna()
            if series.empty:
                continue
            rows.append(
                {
                    "age_bin": str(age_bin),
                    "variant": variant_name,
                    "n_people": int(series.shape[0]),
                    "mean": float(series.mean()),
                    "median": float(series.median()),
                }
            )

    return pd.DataFrame(rows)


def build_biomarker_overlap_counts(panel: pd.DataFrame, primary_variant: str) -> pd.DataFrame:
    subset = panel.loc[panel[primary_variant].notna()].copy()
    rows = [
        {
            "marker": "fi_total",
            "column": primary_variant,
            "n_with_marker": int(subset.shape[0]),
        }
    ]

    marker_map = {
        "crp": "crp_mg_dl",
        "hscrp": "hscrp_mg_l",
        "insulin": "insulin_uU_ml",
        "glucose": "fasting_glucose_mg_dl",
        "triglycerides": "triglycerides_mg_dl",
        "eosinophils": "eosinophils_abs_1000_uL",
    }

    for marker, column in marker_map.items():
        rows.append(
            {
                "marker": marker,
                "column": column,
                "n_with_marker": int(subset[column].notna().sum()),
            }
        )

    rows.append(
        {
            "marker": "all_requested_markers",
            "column": "|".join(marker_map.values()),
            "n_with_marker": int(subset.dropna(subset=list(marker_map.values())).shape[0]),
        }
    )
    return pd.DataFrame(rows)


def write_readme(
    variant_summary: pd.DataFrame,
    primary_variant: str,
    biomarker_overlap: pd.DataFrame,
) -> None:
    lines = [
        "# NHANES Frailty Index Outputs",
        "",
        "This folder contains participant-level frailty indices built from local NHANES raw files.",
        "",
        "## Main point",
        "",
        "NHANES has enough information to build a defensible frailty index per participant, but not an HRS-style longitudinal participant-wave panel.",
        "",
        "Reason:",
        "- HRS follows the same people over repeated waves.",
        "- NHANES is repeated cross-sectional, so each participant appears in one exam cycle only.",
        "",
        "That means the valid NHANES product is:",
        "- a per-person FI within each survey cycle",
        "- cross-cycle comparisons of FI distributions",
        "- age-bin summaries within and across cycles",
        "",
        "## Cycles used",
        "",
        "Primary FI construction uses NHANES 2005 through 2017 in adults aged \\(60+\\).",
        "",
        "The aligned FI variants are built only from domains that NHANES can match reasonably well to HRS over this span:",
        "- chronic disease burden",
        "- self-rated health",
        "- ADL limitations",
        "- IADL limitations",
        "- mobility limitations",
        "- one simple cognition proxy for the broader variant",
        "",
        "## Scoring rule",
        "",
        "$$",
        r"FI_i = \frac{\sum_{j=1}^{m} d_{ij}}{\sum_{j=1}^{m} I(d_{ij}\ \text{observed})}",
        "$$",
        "",
        "I only score a participant when at least \\(80\\%\\) of the deficits for that variant are observed.",
        "",
        "## Outputs",
        "",
        "- `frailty_panel.csv.gz`: participant-level FI panel",
        "- `deficit_catalog.csv`: exact deficit list used in each FI variant",
        "- `screening_log.csv`: screening results for the broad candidate deficits",
        "- `variant_summary.csv`: high-level validation summaries",
        "- `trajectory_by_cycle.csv`: mean FI by NHANES cycle",
        "- `distribution_by_age_bin.csv`: mean FI by age bin",
        "- `biomarker_overlap_counts.csv`: counts of FI-scored participants with requested biomarkers",
        "- `figures/kde_by_age_bin_fi_hrs_overlap_22.png`: KDE curves for the primary aligned FI by age bin",
        "",
        "## Variant summary",
        "",
    ]

    for row in variant_summary.itertuples(index=False):
        lines.append(
            f"- `{row.variant}`: mean \\(= {row.mean:.3f}\\), age Spearman \\(= {row.age_spearman_r:.3f}\\), "
            f"female-minus-male mean \\(= {row.female_minus_male_mean:.3f}\\), \\(99^{{th}}\\) percentile \\(= {row.p99:.3f}\\)"
        )

    lines.extend(
        [
            "",
            "## Primary variant",
            "",
            f"The main HRS-aligned NHANES FI is `{primary_variant}`.",
            "",
            "This is the closest local analog to the HRS morbidity-function FI because it stays inside shared questionnaire domains rather than adding NHANES-specific lab deficits.",
            "",
            "## Biomarker overlap counts",
            "",
            "## Interpretation",
            "",
            "This is useful as an NHANES frailty phenotype in the deficit-accumulation sense, and it should work for downstream cross-sectional and mortality-linked analyses.",
            "",
            "It is not a replacement for the longitudinal HRS FI if the scientific question depends on within-person frailty trajectories.",
            "",
        ]
    )

    insertion_index = lines.index("## Interpretation")
    biomarker_lines = [
        f"- `{row.marker}`: \\(n = {row.n_with_marker}\\)"
        for row in biomarker_overlap.itertuples(index=False)
    ]
    lines[insertion_index:insertion_index] = biomarker_lines + [""]

    (OUTPUT_DIR / "README.md").write_text("\n".join(lines))


def make_cycle_plot(summary: pd.DataFrame) -> None:
    if summary.empty:
        return

    plt.figure(figsize=(8, 4.5))
    for variant_name, frame in summary.groupby("variant"):
        plt.plot(frame["cycle_start_year"], frame["mean"], marker="o", label=variant_name)
    plt.xlabel("NHANES cycle start year")
    plt.ylabel("Mean frailty index")
    plt.title("NHANES frailty index by cycle")
    plt.legend(frameon=False)
    plt.tight_layout()
    plt.savefig(FIGURE_DIR / "trajectory_by_cycle.png", dpi=200)
    plt.close()


def make_age_plot(summary: pd.DataFrame) -> None:
    if summary.empty:
        return

    plt.figure(figsize=(8, 4.5))
    for variant_name, frame in summary.groupby("variant"):
        plt.plot(frame["age_bin"], frame["mean"], marker="o", label=variant_name)
    plt.xlabel("Age bin")
    plt.ylabel("Mean frailty index")
    plt.title("NHANES frailty index by age bin")
    plt.xticks(rotation=30)
    plt.legend(frameon=False)
    plt.tight_layout()
    plt.savefig(FIGURE_DIR / "distribution_by_age_bin.png", dpi=200)
    plt.close()


def make_age_bin_kde_plot(panel: pd.DataFrame, variant_name: str) -> None:
    subset = panel.loc[panel[variant_name].notna()].copy()
    if subset.empty:
        return

    x_max = max(0.8, float(subset[variant_name].quantile(0.995)))
    xs = np.linspace(0.0, x_max, 300)

    plt.figure(figsize=(9, 5.5))
    for age_bin, frame in subset.groupby("age_bin", observed=False):
        if pd.isna(age_bin):
            continue

        values = frame[variant_name].dropna().to_numpy()
        if values.size < 20 or np.unique(values).size < 2:
            continue

        density = gaussian_kde(values)
        plt.plot(xs, density(xs), linewidth=2, label=f"{age_bin} (n={values.size})")

    plt.xlabel("Frailty index")
    plt.ylabel("Density")
    plt.title(f"KDE of {variant_name} by age bin")
    plt.legend(frameon=False, fontsize=8)
    plt.tight_layout()
    plt.savefig(FIGURE_DIR / f"kde_by_age_bin_{variant_name}.png", dpi=200)
    plt.close()


if __name__ == "__main__":
    main()
