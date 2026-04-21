#!/usr/bin/env python3
"""Build a participant-level frailty index from local NHANES files.

This script is intentionally conservative and easy to audit.

Main design choices:
- Use NHANES 2005-2017 for the primary FI because local files contain a stable
  mix of chronic conditions, self-rated health, function, and common labs.
- Treat NHANES as repeated cross-sectional. Each row is one participant in one
  exam cycle, not a longitudinal person-wave panel as in HRS.
- Build a broad candidate FI first, then a screened FI using simple quality
  checks inspired by the Rockwood / Theou guidance.
+"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import math

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pyreadstat


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


CLINICAL_DEFINITIONS = [
    DeficitSpec("hypertension", "BPQ020", "Ever told you had high blood pressure", "disease", "1=yes, 2=no", "clinical"),
    DeficitSpec("diabetes", "DIQ010", "Doctor told you have diabetes", "disease", "1=yes, 2=no", "clinical"),
    DeficitSpec("asthma", "MCQ010", "Ever been told you have asthma", "disease", "1=yes, 2=no", "clinical"),
    DeficitSpec("cancer", "MCQ220", "Ever told you had cancer or malignancy", "disease", "1=yes, 2=no", "clinical"),
    DeficitSpec("kidney", "KIQ022", "Weak or failing kidneys in the past 12 months", "disease", "1=yes, 2=no", "clinical"),
    DeficitSpec("arthritis", "MCQ160A", "Doctor ever said you had arthritis", "disease", "1=yes, 2=no", "clinical"),
    DeficitSpec("heart_failure", "MCQ160B", "Ever told had congestive heart failure", "disease", "1=yes, 2=no", "clinical"),
    DeficitSpec("coronary_heart_disease", "MCQ160C", "Ever told had coronary heart disease", "disease", "1=yes, 2=no", "clinical"),
    DeficitSpec("angina", "MCQ160D", "Ever told had angina", "disease", "1=yes, 2=no", "clinical"),
    DeficitSpec("heart_attack", "MCQ160E", "Ever told had heart attack", "disease", "1=yes, 2=no", "clinical"),
    DeficitSpec("stroke", "MCQ160F", "Ever told had stroke", "disease", "1=yes, 2=no", "clinical"),
    DeficitSpec("emphysema", "MCQ160G", "Ever told you had emphysema", "disease", "1=yes, 2=no", "clinical"),
    DeficitSpec("chronic_bronchitis", "MCQ170K", "Still have chronic bronchitis", "disease", "1=yes, 2=no", "clinical"),
    DeficitSpec("liver_condition", "MCQ170L", "Still have a liver condition", "disease", "1=yes, 2=no", "clinical"),
    DeficitSpec("thyroid_problem", "MCQ170M", "Still have thyroid problem", "disease", "1=yes, 2=no", "clinical"),
    DeficitSpec("osteoporosis", "OSQ060", "Ever told had osteoporosis", "disease", "1=yes, 2=no", "clinical"),
    DeficitSpec("general_health", "HUQ010", "General health condition", "global_health", r"\(1=\) excellent to \(5=\) poor, recoded to \(0\) to \(1\)", "clinical"),
    DeficitSpec("memory_problem", "PFQ057", "Experience confusion or memory problems", "symptom", "1=yes, 2=no", "clinical"),
    DeficitSpec("special_equipment_walk", "PFQ054", "Need special equipment to walk", "function", "1=yes, 2=no", "clinical"),
    DeficitSpec("special_healthcare_equipment", "PFQ090", "Require special healthcare equipment", "function", "1=yes, 2=no", "clinical"),
    DeficitSpec("manage_money", "PFQ061A", "Managing money difficulty", "iadl", r"\(1=\) no, \(2-4=\) deficit, \(5=\) does not do", "clinical"),
    DeficitSpec("walk_quarter_mile", "PFQ061B", "Walking for a quarter mile difficulty", "mobility", r"\(1=\) no, \(2-4=\) deficit, \(5=\) does not do", "clinical"),
    DeficitSpec("walk_up_steps", "PFQ061C", "Walking up ten steps difficulty", "mobility", r"\(1=\) no, \(2-4=\) deficit, \(5=\) does not do", "clinical"),
    DeficitSpec("stoop", "PFQ061D", "Stooping, crouching, kneeling difficulty", "mobility", r"\(1=\) no, \(2-4=\) deficit, \(5=\) does not do", "clinical"),
    DeficitSpec("lift_carry", "PFQ061E", "Lifting or carrying difficulty", "mobility", r"\(1=\) no, \(2-4=\) deficit, \(5=\) does not do", "clinical"),
    DeficitSpec("prepare_meals", "PFQ061G", "Preparing meals difficulty", "iadl", r"\(1=\) no, \(2-4=\) deficit, \(5=\) does not do", "clinical"),
    DeficitSpec("walk_room", "PFQ061H", "Walking between rooms on same floor", "adl", r"\(1=\) no, \(2-4=\) deficit, \(5=\) does not do", "clinical"),
    DeficitSpec("chair_rise", "PFQ061I", "Standing up from armless chair difficulty", "adl", r"\(1=\) no, \(2-4=\) deficit, \(5=\) does not do", "clinical"),
    DeficitSpec("bed_transfer", "PFQ061J", "Getting in and out of bed difficulty", "adl", r"\(1=\) no, \(2-4=\) deficit, \(5=\) does not do", "clinical"),
    DeficitSpec("eat_utensils", "PFQ061K", "Using fork, knife, or cup difficulty", "adl", r"\(1=\) no, \(2-4=\) deficit, \(5=\) does not do", "clinical"),
    DeficitSpec("dress_self", "PFQ061L", "Dressing yourself difficulty", "adl", r"\(1=\) no, \(2-4=\) deficit, \(5=\) does not do", "clinical"),
    DeficitSpec("stand_long", "PFQ061M", "Standing for long periods difficulty", "mobility", r"\(1=\) no, \(2-4=\) deficit, \(5=\) does not do", "clinical"),
    DeficitSpec("sit_long", "PFQ061N", "Sitting for long periods difficulty", "mobility", r"\(1=\) no, \(2-4=\) deficit, \(5=\) does not do", "clinical"),
    DeficitSpec("reach_overhead", "PFQ061O", "Reaching up over head difficulty", "mobility", r"\(1=\) no, \(2-4=\) deficit, \(5=\) does not do", "clinical"),
    DeficitSpec("grasp_small_objects", "PFQ061P", "Grasping or holding small objects difficulty", "mobility", r"\(1=\) no, \(2-4=\) deficit, \(5=\) does not do", "clinical"),
    DeficitSpec("social_attend", "PFQ061R", "Attending social event difficulty", "social_function", r"\(1=\) no, \(2-4=\) deficit, \(5=\) does not do", "clinical"),
    DeficitSpec("push_pull", "PFQ061T", "Push or pull large objects difficulty", "mobility", r"\(1=\) no, \(2-4=\) deficit, \(5=\) does not do", "clinical"),
]

LAB_DEFINITIONS = [
    DeficitSpec("albumin_low", "LBXSAL", "Serum albumin low", "lab", "1 if albumin < 3.5 g/dL", "lab"),
    DeficitSpec("bun_high", "LBXSBU", "Blood urea nitrogen high", "lab", "1 if BUN > 20 mg/dL", "lab"),
    DeficitSpec("creatinine_high", "LBXSCR", "Serum creatinine high", "lab", "1 if creatinine > 1.3 mg/dL", "lab"),
    DeficitSpec("glucose_high", "LBXSGL", "Serum glucose high", "lab", "1 if glucose >= 126 mg/dL", "lab"),
    DeficitSpec("hemoglobin_low", "LBXHGB", "Hemoglobin low", "lab", "1 if < 13 g/dL in men or < 12 g/dL in women", "lab"),
    DeficitSpec("rdw_high", "LBXRDW", "Red cell distribution width high", "lab", "1 if RDW > 14.5%", "lab"),
    DeficitSpec("wbc_abnormal", "LBXWBCSI", "White blood cell count abnormal", "lab", "1 if WBC < 4 or > 11 (1000 cells/uL)", "lab"),
    DeficitSpec("uric_acid_high", "LBXSUA", "Uric acid high", "lab", "1 if > 7.0 mg/dL in men or > 6.0 mg/dL in women", "lab"),
]


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    FIGURE_DIR.mkdir(parents=True, exist_ok=True)

    panel = build_panel()
    panel = add_recoded_deficits(panel)

    variant_map, screening_log = build_variant_map(panel)
    fi_panel = add_frailty_scores(panel, variant_map)

    catalog = build_deficit_catalog(variant_map)
    variant_summary = build_variant_summary(fi_panel, variant_map)
    cycle_summary = build_cycle_summary(fi_panel)
    age_bin_summary = build_age_bin_summary(fi_panel)

    fi_panel.to_csv(OUTPUT_DIR / "frailty_panel.csv.gz", index=False)
    catalog.to_csv(OUTPUT_DIR / "deficit_catalog.csv", index=False)
    screening_log.to_csv(OUTPUT_DIR / "screening_log.csv", index=False)
    variant_summary.to_csv(OUTPUT_DIR / "variant_summary.csv", index=False)
    cycle_summary.to_csv(OUTPUT_DIR / "trajectory_by_cycle.csv", index=False)
    age_bin_summary.to_csv(OUTPUT_DIR / "distribution_by_age_bin.csv", index=False)

    write_readme(variant_summary)
    make_cycle_plot(cycle_summary)
    make_age_plot(age_bin_summary)


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
    merge_into(panel, cycle_year, "MCQ*.xpt", ["MCQ010", "MCQ160A", "MCQ160B", "MCQ160C", "MCQ160D", "MCQ160E", "MCQ160F", "MCQ160G", "MCQ170K", "MCQ170L", "MCQ170M", "MCQ220"])
    merge_into(panel, cycle_year, "OSQ*.xpt", ["OSQ060"])
    merge_into(panel, cycle_year, "HUQ*.xpt", ["HUQ010"])
    merge_into(panel, cycle_year, "PFQ*.xpt", ["PFQ054", "PFQ057", "PFQ090", "PFQ061A", "PFQ061B", "PFQ061C", "PFQ061D", "PFQ061E", "PFQ061G", "PFQ061H", "PFQ061I", "PFQ061J", "PFQ061K", "PFQ061L", "PFQ061M", "PFQ061N", "PFQ061O", "PFQ061P", "PFQ061R", "PFQ061T"])
    merge_into(panel, cycle_year, "BIOPRO*.xpt", ["LBXSAL", "LBXSBU", "LBXSCR", "LBXSGL", "LBXSUA"])
    merge_into(panel, cycle_year, "CBC*.xpt", ["LBXHGB", "LBXRDW", "LBXWBCSI"])

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


def merge_into(panel: pd.DataFrame, cycle_year: int, pattern: str, columns: list[str]) -> None:
    path = find_first(cycle_year, [pattern])
    if path is None:
        return

    frame = read_xpt(path, ["SEQN", *columns])
    if frame.empty:
        return

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
        "asthma",
        "cancer",
        "kidney",
        "arthritis",
        "heart_failure",
        "coronary_heart_disease",
        "angina",
        "heart_attack",
        "stroke",
        "emphysema",
        "chronic_bronchitis",
        "liver_condition",
        "thyroid_problem",
        "osteoporosis",
        "memory_problem",
        "special_equipment_walk",
        "special_healthcare_equipment",
    ]
    difficulty_items = [
        "manage_money",
        "walk_quarter_mile",
        "walk_up_steps",
        "stoop",
        "lift_carry",
        "prepare_meals",
        "walk_room",
        "chair_rise",
        "bed_transfer",
        "eat_utensils",
        "dress_self",
        "stand_long",
        "sit_long",
        "reach_overhead",
        "grasp_small_objects",
        "social_attend",
        "push_pull",
    ]

    source_map = {
        "hypertension": "BPQ020",
        "diabetes": "DIQ010",
        "asthma": "MCQ010",
        "cancer": "MCQ220",
        "kidney": "KIQ022",
        "arthritis": "MCQ160A",
        "heart_failure": "MCQ160B",
        "coronary_heart_disease": "MCQ160C",
        "angina": "MCQ160D",
        "heart_attack": "MCQ160E",
        "stroke": "MCQ160F",
        "emphysema": "MCQ160G",
        "chronic_bronchitis": "MCQ170K",
        "liver_condition": "MCQ170L",
        "thyroid_problem": "MCQ170M",
        "osteoporosis": "OSQ060",
        "memory_problem": "PFQ057",
        "special_equipment_walk": "PFQ054",
        "special_healthcare_equipment": "PFQ090",
        "manage_money": "PFQ061A",
        "walk_quarter_mile": "PFQ061B",
        "walk_up_steps": "PFQ061C",
        "stoop": "PFQ061D",
        "lift_carry": "PFQ061E",
        "prepare_meals": "PFQ061G",
        "walk_room": "PFQ061H",
        "chair_rise": "PFQ061I",
        "bed_transfer": "PFQ061J",
        "eat_utensils": "PFQ061K",
        "dress_self": "PFQ061L",
        "stand_long": "PFQ061M",
        "sit_long": "PFQ061N",
        "reach_overhead": "PFQ061O",
        "grasp_small_objects": "PFQ061P",
        "social_attend": "PFQ061R",
        "push_pull": "PFQ061T",
    }

    for name in binary_yes_no:
        panel[f"def_{name}"] = recode_binary_yes_no(panel.get(source_map[name]))

    for name in difficulty_items:
        panel[f"def_{name}"] = recode_difficulty(panel.get(source_map[name]))

    panel["def_general_health"] = recode_self_rated_health(panel.get("HUQ010"))
    panel["def_albumin_low"] = make_binary_threshold(panel.get("LBXSAL"), low=3.5)
    panel["def_bun_high"] = make_binary_threshold(panel.get("LBXSBU"), high=20.0)
    panel["def_creatinine_high"] = make_binary_threshold(panel.get("LBXSCR"), high=1.3)
    panel["def_glucose_high"] = make_binary_threshold(panel.get("LBXSGL"), high=126.0, high_inclusive=True)
    panel["def_hemoglobin_low"] = recode_low_hemoglobin(panel.get("LBXHGB"), panel.get("sex"))
    panel["def_rdw_high"] = make_binary_threshold(panel.get("LBXRDW"), high=14.5)
    panel["def_wbc_abnormal"] = recode_wbc(panel.get("LBXWBCSI"))
    panel["def_uric_acid_high"] = recode_uric_acid(panel.get("LBXSUA"), panel.get("sex"))

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


def recode_self_rated_health(series: pd.Series | None) -> pd.Series:
    if series is None:
        return pd.Series(dtype=float)

    numeric = pd.to_numeric(series, errors="coerce")
    numeric = numeric.where(numeric.between(1, 5), np.nan)
    return (numeric - 1.0) / 4.0


def make_binary_threshold(
    series: pd.Series | None,
    low: float | None = None,
    high: float | None = None,
    high_inclusive: bool = False,
) -> pd.Series:
    if series is None:
        return pd.Series(dtype=float)

    numeric = pd.to_numeric(series, errors="coerce")
    out = pd.Series(0.0, index=numeric.index, dtype=float)
    out.loc[numeric.isna()] = np.nan

    if low is not None:
        out.loc[numeric < low] = 1.0

    if high is not None and high_inclusive:
        out.loc[numeric >= high] = 1.0
    elif high is not None:
        out.loc[numeric > high] = 1.0

    return out


def recode_low_hemoglobin(series: pd.Series | None, sex: pd.Series | None) -> pd.Series:
    if series is None or sex is None:
        return pd.Series(dtype=float)

    numeric = pd.to_numeric(series, errors="coerce")
    out = pd.Series(np.nan, index=numeric.index, dtype=float)

    male_mask = sex.eq("Male") & numeric.notna()
    female_mask = sex.eq("Female") & numeric.notna()

    out.loc[male_mask] = 0.0
    out.loc[female_mask] = 0.0
    out.loc[male_mask & (numeric < 13.0)] = 1.0
    out.loc[female_mask & (numeric < 12.0)] = 1.0
    return out


def recode_wbc(series: pd.Series | None) -> pd.Series:
    if series is None:
        return pd.Series(dtype=float)

    numeric = pd.to_numeric(series, errors="coerce")
    out = pd.Series(np.nan, index=numeric.index, dtype=float)
    valid = numeric.notna()
    out.loc[valid] = 0.0
    out.loc[valid & ((numeric < 4.0) | (numeric > 11.0))] = 1.0
    return out


def recode_uric_acid(series: pd.Series | None, sex: pd.Series | None) -> pd.Series:
    if series is None or sex is None:
        return pd.Series(dtype=float)

    numeric = pd.to_numeric(series, errors="coerce")
    out = pd.Series(np.nan, index=numeric.index, dtype=float)

    male_mask = sex.eq("Male") & numeric.notna()
    female_mask = sex.eq("Female") & numeric.notna()

    out.loc[male_mask] = 0.0
    out.loc[female_mask] = 0.0
    out.loc[male_mask & (numeric > 7.0)] = 1.0
    out.loc[female_mask & (numeric > 6.0)] = 1.0
    return out


def build_variant_map(panel: pd.DataFrame) -> tuple[dict[str, list[str]], pd.DataFrame]:
    clinical_items = [f"def_{definition.name}" for definition in CLINICAL_DEFINITIONS]
    lab_items = [f"def_{definition.name}" for definition in LAB_DEFINITIONS]
    broad_items = [*clinical_items, *lab_items]

    screened_items, screening_log = run_screening(panel, broad_items)
    variant_map = {
        f"fi_clinical_{len(clinical_items)}": clinical_items,
        f"fi_broad_{len(broad_items)}": broad_items,
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
    all_definitions = [*CLINICAL_DEFINITIONS, *LAB_DEFINITIONS]
    by_name = {f"def_{definition.name}": definition for definition in all_definitions}
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


def write_readme(variant_summary: pd.DataFrame) -> None:
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
        "Primary FI construction uses NHANES 2005 through 2017, because that is the cleanest local span with stable function questions plus common chemistry and CBC measures.",
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
            "## Interpretation",
            "",
            "This is useful as an NHANES frailty phenotype in the deficit-accumulation sense, and it should work for downstream cross-sectional and mortality-linked analyses.",
            "",
            "It is not a replacement for the longitudinal HRS FI if the scientific question depends on within-person frailty trajectories.",
            "",
        ]
    )

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


if __name__ == "__main__":
    main()
