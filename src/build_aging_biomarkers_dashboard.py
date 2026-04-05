#!/usr/bin/env python3
"""Build the public-facing aging biomarkers dashboard bundle."""

from __future__ import annotations

import json
import math
import time
from pathlib import Path

import numpy as np
import pandas as pd

from build_analysis_dataset import normalize_seqn, read_xpt_columns
from nhanes_common import ensure_dir


ROOT = Path(__file__).resolve().parents[1]
TEMPLATE_PATH = Path(__file__).resolve().parent / "templates" / "aging_biomarkers_dashboard_template.html"

DEFAULT_TRIM_MODE = "trim_10_90"
BASELINE_AGE_BIN = "20-24"
ENDLINE_AGE_BIN = "80-84"

COLLECTION_ORDER = [
    "inflammation",
    "metabolism",
    "kidney_reserve",
    "blood_oxygen",
    "hormones_nutrient",
    "cardiovascular_stress",
]

COLLECTION_COPY = {
    "inflammation": {
        "title": "Inflammation",
        "kicker": "Quiet background stress",
        "summary": "Low-grade immune activation often shows up as wider, higher, or more skewed distributions with age.",
        "accent": "rust",
    },
    "metabolism": {
        "title": "Metabolism",
        "kicker": "Fuel handling and liver load",
        "summary": "Glucose control, triglycerides, and liver-linked markers reveal how metabolic regulation shifts across adulthood.",
        "accent": "amber",
    },
    "kidney_reserve": {
        "title": "Kidney & Reserve",
        "kicker": "Organ reserve under pressure",
        "summary": "Creatinine, cystatin C, BUN, and related chemistry show how homeostatic reserve narrows over time.",
        "accent": "teal",
    },
    "blood_oxygen": {
        "title": "Blood & Oxygen",
        "kicker": "Red-cell architecture",
        "summary": "Hemoglobin, RDW, ferritin, and blood-count markers capture oxygen transport and hematologic aging.",
        "accent": "rose",
    },
    "hormones_nutrient": {
        "title": "Hormones & Nutrient Sensing",
        "kicker": "Signals that retune with age",
        "summary": "Sex hormones, thyroid markers, and vitamin-linked signals reflect endocrine remodeling and nutrient sensing.",
        "accent": "violet",
    },
    "cardiovascular_stress": {
        "title": "Cardiovascular Stress",
        "kicker": "Strain, remodeling, risk",
        "summary": "Cardiac stress and vascular-risk markers highlight how the distribution shape can change before averages alone tell the story.",
        "accent": "blue",
    },
}

TEST_NAME_TO_COLLECTION = {
    "CRP": "inflammation",
    "hs-CRP": "inflammation",
    "White blood cell count": "inflammation",
    "Lymphocytes": "inflammation",
    "Lymphocyte percentage": "inflammation",
    "Neutrophils": "inflammation",
    "Basophils": "inflammation",
    "Eosinophils": "inflammation",
    "CD4 T cells": "inflammation",
    "CD8 T cells": "inflammation",
    "Beta-2 microglobulin": "inflammation",
    "HbA1c": "metabolism",
    "Glucose": "metabolism",
    "Fasting insulin": "metabolism",
    "Triglycerides": "metabolism",
    "HDL-C": "metabolism",
    "LDL-C": "metabolism",
    "Total cholesterol": "metabolism",
    "ApoB": "metabolism",
    "ALT": "metabolism",
    "AST": "metabolism",
    "GGT": "metabolism",
    "Lactate": "metabolism",
    "C-peptide": "metabolism",
    "Uric acid": "metabolism",
    "Creatinine": "kidney_reserve",
    "Cystatin C": "kidney_reserve",
    "Blood urea nitrogen": "kidney_reserve",
    "Albumin": "kidney_reserve",
    "Bicarbonate / CO2": "kidney_reserve",
    "Calcium": "kidney_reserve",
    "Chloride": "kidney_reserve",
    "Phosphate": "kidney_reserve",
    "Potassium": "kidney_reserve",
    "Sodium": "kidney_reserve",
    "Total protein": "kidney_reserve",
    "Total bilirubin": "kidney_reserve",
    "Hemoglobin": "blood_oxygen",
    "Hematocrit": "blood_oxygen",
    "RBC count": "blood_oxygen",
    "RDW": "blood_oxygen",
    "MCV": "blood_oxygen",
    "MCH": "blood_oxygen",
    "MCHC": "blood_oxygen",
    "Ferritin": "blood_oxygen",
    "Alkaline phosphatase": "blood_oxygen",
    "Bone-specific alkaline phosphatase": "blood_oxygen",
    "Creatine kinase": "blood_oxygen",
    "Mean platelet volume": "blood_oxygen",
    "Platelet count": "blood_oxygen",
    "25-OH vitamin D": "hormones_nutrient",
    "Testosterone": "hormones_nutrient",
    "Estradiol": "hormones_nutrient",
    "SHBG": "hormones_nutrient",
    "TSH": "hormones_nutrient",
    "Free T3": "hormones_nutrient",
    "PTH": "hormones_nutrient",
    "High-sensitivity troponin": "cardiovascular_stress",
    "NT-proBNP": "cardiovascular_stress",
    "Homocysteine": "cardiovascular_stress",
    "Fibrinogen": "cardiovascular_stress",
    "Neurofilament light chain": "cardiovascular_stress",
}

PRIORITY_MARKERS = {
    "Albumin",
    "CRP",
    "hs-CRP",
    "HbA1c",
    "Creatinine",
    "Cystatin C",
    "Blood urea nitrogen",
    "Hemoglobin",
    "RDW",
    "NT-proBNP",
    "25-OH vitamin D",
    "Testosterone",
    "SHBG",
    "Homocysteine",
    "Fibrinogen",
    "White blood cell count",
    "Lymphocytes",
    "Platelet count",
    "Triglycerides",
    "ALT",
    "AST",
}

DEFAULT_COMPARE_MARKERS = ["Albumin", "CRP", "Creatinine"]
DEFAULT_EXPLORE_MARKER = "Albumin"

AGE_BINS = list(np.arange(20, 90, 5))
AGE_LABELS = [f"{age}-{age + 4}" for age in range(20, 85, 5)]
AGE_MIDS = {label: age + 2.5 for label, age in zip(AGE_LABELS, range(20, 85, 5))}

PUBLIC_DISEASES = {
    "diabetes": {
        "title": "Diabetes",
        "kicker": "Diagnosed diabetes",
        "summary": "Self-reported doctor diagnosis in NHANES. This comparison intentionally reintroduces participants excluded from the healthy-aging baseline.",
        "accent": "rust",
        "source": "DIQ010 == 1",
    },
    "hypertension": {
        "title": "Hypertension",
        "kicker": "High blood pressure",
        "summary": "People reporting high blood pressure often show different kidney, inflammatory, and vascular biomarker patterns across adulthood.",
        "accent": "amber",
        "source": "BPQ020 == 1",
    },
    "cvd": {
        "title": "Cardiovascular disease",
        "kicker": "Heart and vascular disease history",
        "summary": "This NHANES flag combines reported congestive heart failure, coronary heart disease, angina, heart attack, and stroke diagnoses.",
        "accent": "blue",
        "source": "MCQ160B/C/D/E/F == 1",
    },
    "kidney": {
        "title": "Kidney disease",
        "kicker": "Weak or failing kidneys",
        "summary": "The disease group comes from the NHANES kidney-history question and can reveal how renal markers separate from the healthy baseline.",
        "accent": "teal",
        "source": "KIQ022 == 1",
    },
    "liver": {
        "title": "Liver disease",
        "kicker": "Liver condition history",
        "summary": "This comparison groups participants who reported liver disease on the standard NHANES medical history items.",
        "accent": "amber",
        "source": "MCQ160L or MCQ500/MCQ510A-F == 1",
    },
    "cancer": {
        "title": "Cancer history",
        "kicker": "Any reported malignancy",
        "summary": "This uses the broad cancer-history question, so the disease group is heterogeneous by design.",
        "accent": "rose",
        "source": "MCQ220 == 1",
    },
    "asthma": {
        "title": "Asthma",
        "kicker": "Respiratory disease",
        "summary": "Asthma is tracked in the processed participant file even though it is not part of the healthy-aging exclusion rule.",
        "accent": "violet",
        "source": "MCQ010 == 1",
    },
    "thyroid_problem": {
        "title": "Thyroid problem",
        "kicker": "Reported thyroid disorder",
        "summary": "This group captures participants who reported a thyroid problem in NHANES. It is useful for comparing endocrine markers against the healthy baseline.",
        "accent": "violet",
        "source": "MCQ160M == 1",
    },
    "stroke": {
        "title": "Stroke",
        "kicker": "Reported stroke history",
        "summary": "Stroke is shown as its own narrower disease comparison for biomarkers tied to vascular strain and inflammation.",
        "accent": "blue",
        "source": "MCQ160F == 1",
    },
}

DISEASE_TRIMS: dict[str, tuple[float, float] | None] = {
    "all": None,
    DEFAULT_TRIM_MODE: (0.10, 0.90),
}

DISEASE_LONG_COLUMNS = [
    "seqn",
    "cycle_start_year",
    "age_years",
    "sex",
    "biomarker_id",
    "value",
]


def _clean_text(value: object) -> str:
    if value is None:
        return ""
    if isinstance(value, float) and math.isnan(value):
        return ""
    return str(value).strip()


def _clean_slug(value: str) -> str:
    text = "".join(ch.lower() if ch.isalnum() else "-" for ch in value)
    while "--" in text:
        text = text.replace("--", "-")
    return text.strip("-")


def _is_number(value: object) -> bool:
    if value is None:
        return False
    if isinstance(value, bool):
        return False
    try:
        return math.isfinite(float(value))
    except (TypeError, ValueError):
        return False


def _number_or_none(value: object) -> float | None:
    if not _is_number(value):
        return None
    return float(value)


def _pct_change(start: object, end: object) -> float | None:
    start_value = _number_or_none(start)
    end_value = _number_or_none(end)
    if start_value is None or end_value is None:
        return None
    if abs(start_value) < 1e-12:
        return None
    return ((end_value - start_value) / abs(start_value)) * 100.0


def _point_lookup(points: list[dict]) -> dict[str, dict]:
    out: dict[str, dict] = {}
    for point in points or []:
        age_bin = _clean_text(point.get("age_bin"))
        if not age_bin:
            continue
        out[age_bin] = point
    return out


def _quantile_skewness_from_stats(q25: pd.Series, median: pd.Series, q75: pd.Series) -> pd.Series:
    denom = q75 - q25
    out = (q75 + q25 - 2.0 * median) / denom
    return out.where(denom.abs() > 1e-12, np.nan)


def load_public_disease_long(
    public_manifest: list[dict],
    participant_flags: pd.DataFrame | None,
    raw_dir: str | Path,
    screening_summary_path: str | Path,
    merge_map_path: str | Path,
) -> pd.DataFrame | None:
    if participant_flags is None or participant_flags.empty:
        return None

    raw_root = Path(raw_dir)
    screening_path = Path(screening_summary_path)
    merge_path = Path(merge_map_path)
    if not raw_root.exists() or not screening_path.exists() or not merge_path.exists():
        return None

    biomarker_ids = {str(entry["biomarker_id"]) for entry in public_manifest}
    if not biomarker_ids:
        return None

    screening = pd.read_csv(screening_path)
    screening = screening[screening["screen_result"].eq("kept")].copy()
    screening = screening[screening["pooled_id"].astype(str).isin(biomarker_ids)].copy()
    if screening.empty:
        return None

    merge_map = pd.read_csv(merge_path)
    merge_map = merge_map[
        ["pooled_id", "variable_name", "variable_desc", "conversion_factor_to_pooled_unit"]
    ].drop_duplicates(subset=["pooled_id", "variable_name", "variable_desc"])

    screening = screening.merge(
        merge_map,
        on=["pooled_id", "variable_name", "variable_desc"],
        how="left",
    )
    screening["conversion_factor_to_pooled_unit"] = pd.to_numeric(
        screening["conversion_factor_to_pooled_unit"],
        errors="coerce",
    ).fillna(1.0)

    people = participant_flags[["seqn", "cycle_start_year", "age_years", "sex"]].drop_duplicates(
        subset=["seqn", "cycle_start_year"]
    )

    frames: list[pd.DataFrame] = []
    for (cycle_start_year, data_file_name), group in screening.groupby(
        ["cycle_start_year", "data_file_name"],
        observed=True,
    ):
        year = int(cycle_start_year)
        xpt_path = raw_root / str(year) / f"{data_file_name}.xpt"
        if not xpt_path.exists():
            continue

        variable_names = sorted(group["variable_name"].dropna().astype(str).unique())
        if not variable_names:
            continue

        try:
            raw_df = read_xpt_columns(xpt_path, columns=["SEQN", *variable_names])
        except Exception:
            continue

        if "SEQN" not in raw_df.columns:
            continue

        base = pd.DataFrame(
            {
                "seqn": normalize_seqn(raw_df),
                "cycle_start_year": year,
            }
        )
        base = base.dropna(subset=["seqn"])
        base = base.merge(
            people[people["cycle_start_year"] == year],
            on=["seqn", "cycle_start_year"],
            how="inner",
        )
        if base.empty:
            continue

        for row in group.itertuples(index=False):
            variable_name = str(row.variable_name)
            if variable_name not in raw_df.columns:
                continue

            tmp = base.copy()
            tmp["value"] = pd.to_numeric(raw_df[variable_name], errors="coerce")
            tmp = tmp.dropna(subset=["value"])
            if tmp.empty:
                continue

            factor = float(row.conversion_factor_to_pooled_unit)
            if factor != 1.0:
                tmp["value"] = tmp["value"] * factor

            tmp["biomarker_id"] = str(row.pooled_id)
            frames.append(tmp[DISEASE_LONG_COLUMNS])

    if not frames:
        return None

    return pd.concat(frames, ignore_index=True)


def _compute_binned_long(
    df: pd.DataFrame,
    group_cols: list[str],
    trim_quantiles: tuple[float, float] | None = None,
) -> pd.DataFrame:
    if df is None or df.empty:
        return pd.DataFrame()

    tmp = df.copy()
    tmp["age_bin"] = pd.cut(tmp["age_years"], bins=AGE_BINS, labels=AGE_LABELS, right=False, include_lowest=True)
    tmp["age_mid"] = tmp["age_bin"].map(AGE_MIDS).astype(float)
    tmp = tmp.dropna(subset=["age_bin", "value"])
    if tmp.empty:
        return pd.DataFrame()

    keys = [*group_cols, "age_bin", "age_mid"]
    if trim_quantiles is not None:
        q_lo, q_hi = trim_quantiles
        quantiles = (
            tmp.groupby(keys, observed=True)["value"]
            .quantile([q_lo, q_hi])
            .unstack(level=-1)
            .rename(columns={q_lo: "trim_lo", q_hi: "trim_hi"})
            .reset_index()
        )
        tmp = tmp.merge(quantiles, on=keys, how="left")
        tmp = tmp[(tmp["value"] >= tmp["trim_lo"]) & (tmp["value"] <= tmp["trim_hi"])].copy()

    grouped = (
        tmp.groupby(keys, observed=True)["value"]
        .agg(
            n="count",
            mean="mean",
            std="std",
            median="median",
            q25=lambda series: float(np.nanpercentile(series.to_numpy(dtype=float), 25)),
            q75=lambda series: float(np.nanpercentile(series.to_numpy(dtype=float), 75)),
            p10=lambda series: float(np.nanpercentile(series.to_numpy(dtype=float), 10)),
            p90=lambda series: float(np.nanpercentile(series.to_numpy(dtype=float), 90)),
            skewness=lambda series: float(pd.Series(series.to_numpy(dtype=float)).skew()),
        )
        .reset_index()
    )
    if grouped.empty:
        return grouped

    grouped["cv"] = grouped["std"] / grouped["mean"].abs()
    grouped.loc[grouped["mean"].abs() < 1e-8, "cv"] = np.nan
    grouped["quantile_skewness"] = _quantile_skewness_from_stats(grouped["q25"], grouped["median"], grouped["q75"])
    grouped["passes_n_threshold"] = grouped["n"] >= 30
    return grouped.reset_index(drop=True)


def _grouped_to_points_map(df: pd.DataFrame) -> dict[str, list[dict]]:
    out: dict[str, list[dict]] = {}
    if df is None or df.empty:
        return out

    for biomarker_id, group_df in df.groupby("biomarker_id", observed=True):
        points: list[dict] = []
        for row in group_df.sort_values("age_mid").itertuples(index=False):
            points.append(
                {
                    "age_bin": str(row.age_bin),
                    "age_mid": float(row.age_mid),
                    "n": int(row.n),
                    "mean": _number_or_none(row.mean),
                    "std": _number_or_none(row.std),
                    "median": _number_or_none(row.median),
                    "q25": _number_or_none(row.q25),
                    "q75": _number_or_none(row.q75),
                    "p10": _number_or_none(row.p10),
                    "p90": _number_or_none(row.p90),
                    "skewness": _number_or_none(row.skewness),
                    "quantile_skewness": _number_or_none(row.quantile_skewness),
                    "cv": _number_or_none(row.cv),
                    "iqr": _number_or_none(row.q75) - _number_or_none(row.q25)
                    if _number_or_none(row.q75) is not None and _number_or_none(row.q25) is not None
                    else None,
                    "passes_n_threshold": bool(row.passes_n_threshold),
                }
            )
        out[str(biomarker_id)] = points
    return out


def _grouped_to_sex_points_map(df: pd.DataFrame) -> dict[str, dict[str, list[dict]]]:
    out: dict[str, dict[str, list[dict]]] = {}
    if df is None or df.empty:
        return out

    for (biomarker_id, sex_norm), group_df in df.groupby(["biomarker_id", "sex_norm"], observed=True):
        out.setdefault(str(biomarker_id), {})[str(sex_norm)] = _grouped_to_points_map(group_df)[str(biomarker_id)]
    return out


def _build_group_payload(long_df: pd.DataFrame) -> dict[str, dict]:
    empty_payload = {
        "points_by_mode": {mode: {} for mode in DISEASE_TRIMS},
        "sex_points_by_mode": {mode: {} for mode in DISEASE_TRIMS},
        "raw_counts": {},
        "raw_counts_by_sex": {},
    }
    if long_df is None or long_df.empty:
        return empty_payload

    use = long_df[["biomarker_id", "age_years", "value", "sex"]].dropna(subset=["biomarker_id", "age_years", "value"]).copy()
    if use.empty:
        return empty_payload

    use["sex_norm"] = use["sex"].astype(str).str.strip().str.lower()
    use.loc[~use["sex_norm"].isin(["female", "male"]), "sex_norm"] = "unknown"
    sex_use = use[use["sex_norm"].isin(["female", "male"])][["biomarker_id", "age_years", "value", "sex_norm"]].copy()

    points_by_mode: dict[str, dict[str, list[dict]]] = {}
    sex_points_by_mode: dict[str, dict[str, dict[str, list[dict]]]] = {}
    for mode, quantiles in DISEASE_TRIMS.items():
        pooled_binned = _compute_binned_long(
            use[["biomarker_id", "age_years", "value"]],
            group_cols=["biomarker_id"],
            trim_quantiles=quantiles,
        )
        sex_binned = _compute_binned_long(
            sex_use,
            group_cols=["biomarker_id", "sex_norm"],
            trim_quantiles=quantiles,
        )
        points_by_mode[mode] = _grouped_to_points_map(pooled_binned)
        sex_points_by_mode[mode] = _grouped_to_sex_points_map(sex_binned)

    raw_counts = use.groupby("biomarker_id", observed=True).size().astype(int).to_dict()
    raw_counts_by_sex: dict[str, dict[str, int]] = {}
    sex_counts = (
        use[use["sex_norm"].isin(["female", "male"])]
        .groupby(["biomarker_id", "sex_norm"], observed=True)
        .size()
        .reset_index(name="n")
    )
    for row in sex_counts.itertuples(index=False):
        raw_counts_by_sex.setdefault(str(row.biomarker_id), {})[str(row.sex_norm)] = int(row.n)

    return {
        "points_by_mode": points_by_mode,
        "sex_points_by_mode": sex_points_by_mode,
        "raw_counts": {str(key): int(value) for key, value in raw_counts.items()},
        "raw_counts_by_sex": raw_counts_by_sex,
    }


def build_disease_explorer_bundle(
    public_manifest: list[dict],
    long_df: pd.DataFrame | None,
    participant_flags: pd.DataFrame | None,
) -> dict[str, object]:
    if long_df is None or long_df.empty or participant_flags is None or participant_flags.empty:
        return {"conditions": [], "by_condition": {}}

    biomarker_ids = {str(entry["biomarker_id"]) for entry in public_manifest}
    manifest_by_id = {str(entry["biomarker_id"]): entry for entry in public_manifest}

    needed_flag_columns = ["seqn", "cycle_start_year", "healthy_flag", *PUBLIC_DISEASES.keys()]
    missing_columns = [column for column in needed_flag_columns if column not in participant_flags.columns]
    if missing_columns:
        raise KeyError(f"Participant flag table is missing required disease columns: {missing_columns}")

    use_long = long_df[long_df["biomarker_id"].astype(str).isin(biomarker_ids)].copy()
    use_long = use_long.dropna(subset=["seqn", "cycle_start_year", "biomarker_id", "age_years", "value"])
    if use_long.empty:
        return {"conditions": [], "by_condition": {}}

    flag_table = participant_flags[needed_flag_columns].drop_duplicates(subset=["seqn", "cycle_start_year"]).copy()
    merged = use_long.merge(flag_table, on=["seqn", "cycle_start_year"], how="left")
    merged = merged[merged["age_years"].between(20, 84.999999)].copy()
    if merged.empty:
        return {"conditions": [], "by_condition": {}}

    healthy_rows = merged[merged["healthy_flag"] == True].copy()
    healthy_payload = _build_group_payload(healthy_rows)

    condition_index: list[dict] = []
    by_condition: dict[str, dict] = {}

    participants = flag_table.copy()
    for condition_key, meta in PUBLIC_DISEASES.items():
        if condition_key not in merged.columns:
            continue

        disease_rows = merged[merged[condition_key] == True].copy()
        if disease_rows.empty:
            continue

        disease_participants = participants[participants[condition_key] == True].copy()
        if disease_participants.empty:
            continue

        disease_payload = _build_group_payload(disease_rows)
        condition_records: list[dict] = []
        for biomarker_id in sorted(biomarker_ids, key=lambda key: manifest_by_id[key]["display_name"]):
            entry = manifest_by_id[biomarker_id]
            condition_records.append(
                {
                    "biomarker_id": biomarker_id,
                    "display_name": entry["display_name"],
                    "chart_display_name": entry["chart_display_name"],
                    "unit": entry["unit"],
                    "featured_collection": entry["featured_collection"],
                    "featured_collection_title": entry["featured_collection_title"],
                    "aging_domain": entry["aging_domain"],
                    "groups": {
                        "healthy": {
                            "raw_total_n": int(healthy_payload["raw_counts"].get(biomarker_id, 0)),
                            "raw_total_n_by_sex": healthy_payload["raw_counts_by_sex"].get(biomarker_id, {}),
                            "points_by_filter": {
                                mode: healthy_payload["points_by_mode"].get(mode, {}).get(biomarker_id, [])
                                for mode in DISEASE_TRIMS
                            },
                            "sex_points_by_filter": {
                                mode: healthy_payload["sex_points_by_mode"].get(mode, {}).get(biomarker_id, {})
                                for mode in DISEASE_TRIMS
                            },
                        },
                        "condition": {
                            "raw_total_n": int(disease_payload["raw_counts"].get(biomarker_id, 0)),
                            "raw_total_n_by_sex": disease_payload["raw_counts_by_sex"].get(biomarker_id, {}),
                            "points_by_filter": {
                                mode: disease_payload["points_by_mode"].get(mode, {}).get(biomarker_id, [])
                                for mode in DISEASE_TRIMS
                            },
                            "sex_points_by_filter": {
                                mode: disease_payload["sex_points_by_mode"].get(mode, {}).get(biomarker_id, {})
                                for mode in DISEASE_TRIMS
                            },
                        },
                    },
                }
            )

        condition_detail_path = f"diseases/{condition_key}.json"
        condition_index.append(
            {
                "key": condition_key,
                "title": meta["title"],
                "kicker": meta["kicker"],
                "summary": meta["summary"],
                "accent": meta["accent"],
                "source": meta["source"],
                "detail_path": condition_detail_path,
                "participant_count": int(len(disease_participants)),
                "female_count": int(((disease_participants["sex"] == "female")).sum()) if "sex" in disease_participants.columns else 0,
                "male_count": int(((disease_participants["sex"] == "male")).sum()) if "sex" in disease_participants.columns else 0,
            }
        )
        by_condition[condition_key] = {
            "condition": condition_index[-1],
            "biomarkers": condition_records,
        }

    return {"conditions": condition_index, "by_condition": by_condition}


def _collection_rank(collection_key: str) -> int:
    try:
        return COLLECTION_ORDER.index(collection_key)
    except ValueError:
        return len(COLLECTION_ORDER)


def _humanize_token(token: str) -> str:
    if not token:
        return ""
    return token.replace("_", " ").strip()


def _infer_collection(row: pd.Series) -> str:
    test_name = _clean_text(row.get("test_name"))
    if test_name in TEST_NAME_TO_COLLECTION:
        return TEST_NAME_TO_COLLECTION[test_name]

    aging_domain = _clean_text(row.get("aging_domain")).lower()
    organ = _clean_text(row.get("primary_organ_system")).lower()

    if "vascular" in aging_domain or "cardio" in aging_domain or "heart" in organ:
        return "cardiovascular_stress"
    if "kidney" in organ or "reserve" in aging_domain or "homeostatic" in aging_domain:
        return "kidney_reserve"
    if "hemat" in organ or "blood" in organ or "oxygen" in aging_domain:
        return "blood_oxygen"
    if "hormone" in aging_domain or "endocrine" in aging_domain or "gonadal" in organ:
        return "hormones_nutrient"
    if "immune" in organ or "inflamm" in aging_domain:
        return "inflammation"
    return "metabolism"


def _compute_sex_divergence_score(payload: dict, trim_mode: str = DEFAULT_TRIM_MODE) -> float | None:
    points_by_filter = payload.get("sex_points_by_filter") or {}
    mode_points = points_by_filter.get(trim_mode) or {}
    female_lookup = _point_lookup(mode_points.get("female") or [])
    male_lookup = _point_lookup(mode_points.get("male") or [])

    female_base = _number_or_none((female_lookup.get(BASELINE_AGE_BIN) or {}).get("median"))
    male_base = _number_or_none((male_lookup.get(BASELINE_AGE_BIN) or {}).get("median"))
    if female_base is None or male_base is None:
        return None
    if abs(female_base) < 1e-12 or abs(male_base) < 1e-12:
        return None

    gaps: list[float] = []
    for age_bin in sorted(set(female_lookup) & set(male_lookup)):
        female_median = _number_or_none(female_lookup[age_bin].get("median"))
        male_median = _number_or_none(male_lookup[age_bin].get("median"))
        if female_median is None or male_median is None:
            continue
        female_shift = ((female_median / female_base) - 1.0) * 100.0
        male_shift = ((male_median / male_base) - 1.0) * 100.0
        gaps.append(abs(female_shift - male_shift))

    if not gaps:
        return None
    return float(sum(gaps) / len(gaps))


def _compute_public_metrics_from_points(
    points: list[dict],
    sample_count: int,
    sex_divergence_score: float | None,
) -> dict[str, float | int | None]:
    lookup = _point_lookup(points)
    start = lookup.get(BASELINE_AGE_BIN) or {}
    end = lookup.get(ENDLINE_AGE_BIN) or {}

    start_upper = None
    end_upper = None
    start_lower = None
    end_lower = None
    if start:
        start_p90 = _number_or_none(start.get("p90"))
        start_median = _number_or_none(start.get("median"))
        start_p10 = _number_or_none(start.get("p10"))
        if start_p90 is not None and start_median is not None:
            start_upper = start_p90 - start_median
        if start_median is not None and start_p10 is not None:
            start_lower = start_median - start_p10
    if end:
        end_p90 = _number_or_none(end.get("p90"))
        end_median = _number_or_none(end.get("median"))
        end_p10 = _number_or_none(end.get("p10"))
        if end_p90 is not None and end_median is not None:
            end_upper = end_p90 - end_median
        if end_median is not None and end_p10 is not None:
            end_lower = end_median - end_p10

    return {
        "available_age_bins": len(points),
        "sample_count": int(sample_count),
        "median_change_pct_20_24_to_80_84": _pct_change(start.get("median"), end.get("median")),
        "iqr_change_pct_20_24_to_80_84": _pct_change(start.get("iqr"), end.get("iqr")),
        "sd_change_pct_20_24_to_80_84": _pct_change(start.get("std"), end.get("std")),
        "cv_change_pct_20_24_to_80_84": _pct_change(start.get("cv"), end.get("cv")),
        "upper_tail_change_pct": _pct_change(start_upper, end_upper),
        "lower_tail_change_pct": _pct_change(start_lower, end_lower),
        "tail_asymmetry_change": (
            _number_or_none(end.get("quantile_skewness")) - _number_or_none(start.get("quantile_skewness"))
            if _is_number(end.get("quantile_skewness")) and _is_number(start.get("quantile_skewness"))
            else None
        ),
        "sex_divergence_score": sex_divergence_score,
    }


def _compute_context_metrics(payload: dict) -> dict[str, dict[str, dict[str, float | int | None]]]:
    sex_divergence_default = _compute_sex_divergence_score(payload, trim_mode=DEFAULT_TRIM_MODE)
    by_context: dict[str, dict[str, dict[str, float | int | None]]] = {}

    pooled_points_by_filter = payload.get("points_by_filter") or {}
    sex_points_by_filter = payload.get("sex_points_by_filter") or {}
    total_n = int(payload.get("raw_total_n") or 0)
    total_n_by_sex = payload.get("raw_total_n_by_sex") or {}

    for cohort in ["pooled", "female", "male"]:
        by_context[cohort] = {}
        for trim_mode in ["all", DEFAULT_TRIM_MODE]:
            if cohort == "pooled":
                points = pooled_points_by_filter.get(trim_mode) or []
                sample_count = total_n
            else:
                sex_payload = sex_points_by_filter.get(trim_mode) or {}
                points = sex_payload.get(cohort) or []
                sample_count = int(total_n_by_sex.get(cohort) or 0)

            by_context[cohort][trim_mode] = _compute_public_metrics_from_points(
                points=points,
                sample_count=sample_count,
                sex_divergence_score=sex_divergence_default,
            )

    return by_context


def build_public_manifest(
    metadata: pd.DataFrame,
    series_index: dict[str, str],
    series_payloads: dict[str, dict],
    aging_catalog_csv: str | Path,
) -> list[dict]:
    catalog_path = Path(aging_catalog_csv)
    if not catalog_path.exists():
        raise FileNotFoundError(f"Aging biomarker catalog not found: {catalog_path}")

    catalog_df = pd.read_csv(catalog_path)
    catalog_df = catalog_df[catalog_df["specimen"].astype(str).str.lower() == "blood"].copy()
    catalog_df = catalog_df[catalog_df["in_nhanes_dashboard"] == "Yes"].copy()
    catalog_df = catalog_df[~catalog_df["nhanes_match_name"].fillna("").str.contains(";")].copy()

    metadata_by_name = metadata.set_index("biomarker_name").to_dict(orient="index")
    manifest: list[dict] = []

    for row in catalog_df.sort_values("test_name").itertuples(index=False):
        source_name = _clean_text(getattr(row, "nhanes_match_name", ""))
        meta = metadata_by_name.get(source_name)
        if meta is None:
            continue

        biomarker_id = _clean_text(meta.get("biomarker_id"))
        rel_path = series_index.get(biomarker_id)
        if not rel_path:
            continue

        payload = series_payloads.get(rel_path)
        if not payload:
            continue

        collection_key = _infer_collection(pd.Series(row._asdict()))
        public_metrics_by_context = _compute_context_metrics(payload)
        public_metrics = public_metrics_by_context["pooled"][DEFAULT_TRIM_MODE]
        detail_series_path = f"data/{rel_path}"
        unit = _clean_text(meta.get("unit"))
        test_name = _clean_text(getattr(row, "test_name", source_name))
        has_clalit_overlay = bool(payload.get("clalit_data"))
        landing_score = (
            (1000 if test_name in PRIORITY_MARKERS else 0)
            + abs(public_metrics.get("median_change_pct_20_24_to_80_84") or 0.0)
            + abs(public_metrics.get("iqr_change_pct_20_24_to_80_84") or 0.0)
            + abs(public_metrics.get("sex_divergence_score") or 0.0)
        )

        manifest.append(
            {
                "biomarker_id": biomarker_id,
                "display_name": test_name,
                "chart_display_name": f"{test_name} ({unit})" if unit else test_name,
                "unit": unit,
                "aging_domain": _clean_text(getattr(row, "aging_domain", "")),
                "primary_organ_system": _clean_text(getattr(row, "primary_organ_system", "")),
                "measurement_class": _clean_text(getattr(row, "measurement_class", "")),
                "clock_relevance": _clean_text(getattr(row, "clock_relevance", "")),
                "featured_collection": collection_key,
                "featured_collection_title": COLLECTION_COPY[collection_key]["title"],
                "detail_series_path": detail_series_path,
                "has_clalit_overlay": has_clalit_overlay,
                "is_priority_marker": bool(test_name in PRIORITY_MARKERS),
                "landing_score": float(landing_score),
                "source_name": source_name,
                "source_display_name": _clean_text(meta.get("display_name") or meta.get("biomarker_name") or test_name),
                "source_variable_name": _clean_text(meta.get("variable_name")),
                "series_file": rel_path,
                "public_metrics": public_metrics,
                "public_metrics_by_context": public_metrics_by_context,
                "slug": _clean_slug(test_name),
            }
        )

    manifest.sort(
        key=lambda item: (
            _collection_rank(item["featured_collection"]),
            -int(item["is_priority_marker"]),
            -float(item["landing_score"]),
            item["display_name"],
        )
    )
    return manifest


def render_public_dashboard_html(data_base: str) -> str:
    data_version = str(int(time.time()))
    template = TEMPLATE_PATH.read_text(encoding="utf-8")
    return (
        template.replace("__DATA_VERSION__", data_version)
        .replace("__PUBLIC_DATA_BASE__", data_base)
        .replace("__COLLECTION_COPY__", json.dumps(COLLECTION_COPY, ensure_ascii=True))
        .replace("__DEFAULT_COMPARE_MARKERS__", json.dumps(DEFAULT_COMPARE_MARKERS, ensure_ascii=True))
        .replace("__DEFAULT_EXPLORE_MARKER__", json.dumps(DEFAULT_EXPLORE_MARKER, ensure_ascii=True))
    )


def write_public_dashboard_bundle(
    out_html: Path,
    out_json: Path,
    data_dir_name: str,
    manifest: list[dict],
    disease_bundle: dict[str, object] | None = None,
) -> None:
    data_dir = out_html.parent / data_dir_name
    disease_dir = data_dir / "diseases"

    ensure_dir(out_html.parent)
    ensure_dir(data_dir)
    ensure_dir(disease_dir)
    ensure_dir(out_json.parent)

    (data_dir / "manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=True, allow_nan=False, indent=2),
        encoding="utf-8",
    )
    disease_bundle = disease_bundle or {"conditions": [], "by_condition": {}}
    (data_dir / "disease_index.json").write_text(
        json.dumps(disease_bundle.get("conditions", []), ensure_ascii=True, allow_nan=False, indent=2),
        encoding="utf-8",
    )
    for condition_key, payload in (disease_bundle.get("by_condition", {}) or {}).items():
        (disease_dir / f"{condition_key}.json").write_text(
            json.dumps(payload, ensure_ascii=True, allow_nan=False, indent=2),
            encoding="utf-8",
        )
    out_html.write_text(render_public_dashboard_html(data_dir_name), encoding="utf-8")

    summary = {
        "manifest_count": len(manifest),
        "featured_collection_count": len(COLLECTION_COPY),
        "disease_condition_count": len(disease_bundle.get("conditions", [])),
        "default_trim_mode": DEFAULT_TRIM_MODE,
        "data_dir": str(data_dir),
    }
    out_json.write_text(json.dumps(summary, ensure_ascii=True, indent=2, allow_nan=False), encoding="utf-8")

    print(f"Wrote public manifest: {data_dir / 'manifest.json'}")
    print(f"Wrote public disease index: {data_dir / 'disease_index.json'}")
    print(f"Wrote public dashboard HTML: {out_html}")
    print(f"Wrote public dashboard summary JSON: {out_json}")
