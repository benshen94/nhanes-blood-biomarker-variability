#!/usr/bin/env python3
"""Build the public-facing aging biomarkers dashboard bundle."""

from __future__ import annotations

import json
import math
import time
from pathlib import Path

import pandas as pd

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
) -> None:
    data_dir = out_html.parent / data_dir_name

    ensure_dir(out_html.parent)
    ensure_dir(data_dir)
    ensure_dir(out_json.parent)

    (data_dir / "manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=True, allow_nan=False, indent=2),
        encoding="utf-8",
    )
    out_html.write_text(render_public_dashboard_html(data_dir_name), encoding="utf-8")

    summary = {
        "manifest_count": len(manifest),
        "featured_collection_count": len(COLLECTION_COPY),
        "default_trim_mode": DEFAULT_TRIM_MODE,
        "data_dir": str(data_dir),
    }
    out_json.write_text(json.dumps(summary, ensure_ascii=True, indent=2, allow_nan=False), encoding="utf-8")

    print(f"Wrote public manifest: {data_dir / 'manifest.json'}")
    print(f"Wrote public dashboard HTML: {out_html}")
    print(f"Wrote public dashboard summary JSON: {out_json}")
