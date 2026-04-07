#!/usr/bin/env python3
"""Robust aging biomarker trajectory analysis for the curated aging biomarker catalog."""

from __future__ import annotations

import argparse
import json
import math
import os
import re
import shutil
import tempfile
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterable

os.environ.setdefault(
    "MPLCONFIGDIR",
    str(Path(tempfile.gettempdir()) / "codex-mpl-aging-biomarker-analysis"),
)

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from pygam import LinearGAM, s
from scipy.cluster.hierarchy import dendrogram, fcluster, linkage
from scipy.interpolate import UnivariateSpline
from scipy.sparse import csc_matrix, csr_matrix, spmatrix
from scipy.stats import skew as scipy_skew
from scipy.stats import spearmanr, wasserstein_distance
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score

from nhanes_common import ensure_dir


for sparse_cls in (spmatrix, csr_matrix, csc_matrix):
    if not hasattr(sparse_cls, "A"):
        sparse_cls.A = property(lambda self: self.toarray())


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CATALOG = ROOT / "projects" / "aging_biomarkers" / "catalog" / "aging_biomarkers.csv"
DEFAULT_LONG = ROOT / "data" / "processed" / "biomarker_long.parquet"
DEFAULT_CLALIT_QUARTILES = ROOT / "data" / "clalit" / "clalit_quartiles.csv"
DEFAULT_CLALIT_F = ROOT / "data" / "clalit" / "females_all_statistics.csv"
DEFAULT_CLALIT_M = ROOT / "data" / "clalit" / "males_all_statistics.csv"
DEFAULT_CLALIT_MAP = ROOT / "data" / "clalit_mapping.json"
DEFAULT_OUT = ROOT / "projects" / "aging_biomarkers" / "analysis"
DEFAULT_REVIEW = [
    "Albumin",
    "CRP",
    "hs-CRP",
    "Hemoglobin",
    "RDW",
    "Cystatin C",
    "HbA1c",
    "Triglycerides",
    "Leukocyte telomere length",
]

COHORTS = ("pooled", "female", "male")
CLALIT_COHORTS = ("female", "male")
SUMMARY_METRICS = ("median", "std", "iqr", "cv", "skewness", "quantile_skewness")
CLALIT_SUMMARY_METRICS = ("median", "cv", "quantile_skewness")
CURVE_NORMALIZATIONS = ("raw", "young_ratio", "young_log_fold", "shape_z")
DISTRIBUTION_NORMALIZATIONS = ("young_z_raw", "young_z_log")
DISTRIBUTION_FEATURES = (
    "median_drift",
    "iqr_change",
    "upper_tail_change",
    "lower_tail_change",
    "tail_asymmetry",
)
POSITIVE_SMOOTH_METRICS = {"std", "iqr", "cv"}
AGE_BIN_EDGES = list(np.arange(20, 90, 5))
AGE_BIN_LABELS = [f"{a}-{a+4}" for a in range(20, 85, 5)]
AGE_BIN_MIDS = {lab: a + 2.5 for lab, a in zip(AGE_BIN_LABELS, range(20, 85, 5))}
AGE_GRID = np.arange(20, 85, dtype=float)
MIN_BIN_N = 30
MIN_VALID_BINS = 5
REFERENCE_BIN = "20-24"
REFERENCE_AGE = 22.5
TRIM_LO = 0.10
TRIM_HI = 0.90
EPS = 1e-12
FPCA_COMPONENTS = 4
FPCA_RETAIN_MIN = 2
FPCA_PC3_EV_THRESHOLD = 0.10
CLUSTER_MIN_BIOMARKERS = 6
CLUSTER_MIN_FEATURES = 2
CLUSTER_K_MIN = 2
CLUSTER_K_MAX = 8
PCA_VIS_COMPONENTS = 3

SUMMARY_METRIC_Y_LABELS = {
    "median": "Median",
    "std": "Standard deviation",
    "iqr": "Interquartile range",
    "cv": "Coefficient of variation",
    "skewness": "Skewness",
    "quantile_skewness": "Quantile skewness",
}
SUMMARY_RHO_COLUMNS = tuple(f"rho_{metric}" for metric in SUMMARY_METRICS)
CLALIT_SUMMARY_RHO_COLUMNS = tuple(f"rho_{metric}" for metric in CLALIT_SUMMARY_METRICS)
SUMMARY_RHO_PCA_METHODS = {
    "signed_all6": {
        "label": "Signed all 6",
        "description": "Signed rho_median, rho_std, rho_iqr, rho_cv, rho_skewness, rho_quantile_skewness.",
        "features": (
            ("rho_median", "rho_median", "Median rho", "identity"),
            ("rho_std", "rho_std", "Std rho", "identity"),
            ("rho_iqr", "rho_iqr", "IQR rho", "identity"),
            ("rho_cv", "rho_cv", "CV rho", "identity"),
            ("rho_skewness", "rho_skewness", "Skewness rho", "identity"),
            ("rho_quantile_skewness", "rho_quantile_skewness", "Quantile skewness rho", "identity"),
        ),
    },
    "abs_median_all6": {
        "label": "Abs median, all 6",
        "description": "abs(rho_median) with std, iqr, cv, skewness, and quantile skewness left signed.",
        "features": (
            ("rho_median", "abs_rho_median", "abs(rho_median)", "abs"),
            ("rho_std", "rho_std", "Std rho", "identity"),
            ("rho_iqr", "rho_iqr", "IQR rho", "identity"),
            ("rho_cv", "rho_cv", "CV rho", "identity"),
            ("rho_skewness", "rho_skewness", "Skewness rho", "identity"),
            ("rho_quantile_skewness", "rho_quantile_skewness", "Quantile skewness rho", "identity"),
        ),
    },
    "abs_median_cv_skew_only": {
        "label": "Abs median, no std/IQR",
        "description": "abs(rho_median) plus rho_cv, rho_skewness, and rho_quantile_skewness; std and iqr removed.",
        "features": (
            ("rho_median", "abs_rho_median", "abs(rho_median)", "abs"),
            ("rho_cv", "rho_cv", "CV rho", "identity"),
            ("rho_skewness", "rho_skewness", "Skewness rho", "identity"),
            ("rho_quantile_skewness", "rho_quantile_skewness", "Quantile skewness rho", "identity"),
        ),
    },
    "abs_median_abs_std_abs_iqr": {
        "label": "Abs median/std/IQR",
        "description": "abs(rho_median), abs(rho_std), abs(rho_iqr), with cv, skewness, and quantile skewness left signed.",
        "features": (
            ("rho_median", "abs_rho_median", "abs(rho_median)", "abs"),
            ("rho_std", "abs_rho_std", "abs(rho_std)", "abs"),
            ("rho_iqr", "abs_rho_iqr", "abs(rho_iqr)", "abs"),
            ("rho_cv", "rho_cv", "CV rho", "identity"),
            ("rho_skewness", "rho_skewness", "Skewness rho", "identity"),
            ("rho_quantile_skewness", "rho_quantile_skewness", "Quantile skewness rho", "identity"),
        ),
    },
}
CLALIT_SUMMARY_RHO_PCA_METHODS = {
    "signed_all3": {
        "label": "Signed all 3",
        "description": "Signed rho_median, rho_cv, and rho_quantile_skewness.",
        "features": (
            ("rho_median", "rho_median", "Median rho", "identity"),
            ("rho_cv", "rho_cv", "CV rho", "identity"),
            ("rho_quantile_skewness", "rho_quantile_skewness", "Quantile skewness rho", "identity"),
        ),
    },
    "abs_median_all3": {
        "label": "Abs median, all 3",
        "description": "abs(rho_median) with rho_cv and rho_quantile_skewness left signed.",
        "features": (
            ("rho_median", "abs_rho_median", "abs(rho_median)", "abs"),
            ("rho_cv", "rho_cv", "CV rho", "identity"),
            ("rho_quantile_skewness", "rho_quantile_skewness", "Quantile skewness rho", "identity"),
        ),
    },
    "abs_all3": {
        "label": "Abs all 3",
        "description": "abs(rho_median), abs(rho_cv), and abs(rho_quantile_skewness).",
        "features": (
            ("rho_median", "abs_rho_median", "abs(rho_median)", "abs"),
            ("rho_cv", "abs_rho_cv", "abs(rho_cv)", "abs"),
            ("rho_quantile_skewness", "abs_rho_quantile_skewness", "abs(rho_quantile_skewness)", "abs"),
        ),
    },
}
SUMMARY_FEATURE_METADATA_COLUMNS = (
    "analysis_id",
    "test_name",
    "category",
    "subcategory",
    "primary_organ_system",
    "aging_domain",
    "measurement_class",
    "target_kind",
)
SUMMARY_PCA_COLOR_COLUMNS = (
    "category",
    "subcategory",
    "primary_organ_system",
    "aging_domain",
    "measurement_class",
    "target_kind",
)
CLALIT_SCALE_COLUMNS = (
    "min",
    "q1",
    "q25",
    "median",
    "mad",
    "se",
    "q75",
    "q3",
    "max",
    "mean",
    "sd",
    "p10",
    "p25",
    "p75",
    "p90",
    "geom_mean",
)
CLALIT_QUARTILE_RAW_COLUMNS = ("raw_q0", "raw_q1", "raw_q2", "raw_q3", "raw_q4")
CLALIT_CANONICAL_OVERRIDES = {
    "25-hydroxyvitamin d3": "25_oh_vitamin_d",
    "alanine aminotransferase alt": "alt",
    "aspartate aminotransferase ast": "ast",
    "creatine phosphokinase cpk": "creatine_kinase",
    "direct hdl-cholesterol": "hdl_c",
    "eosinophils number": "eosinophils",
    "insulin si": "fasting_insulin",
    "lymphocyte number": "lymphocytes",
    "monocyte number": "monocytes",
    "monocyte percent": "monocytes",
    "parathyroid hormone elecys method": "pth",
    "segmented neutrophils num": "neutrophils",
    "segmented neutrophils percent": "neutrophils",
    "testosterone total": "testosterone",
    "thyroxine free": "free_t4",
    "total calcium": "calcium",
    "total t3": "free_t3",
}


@dataclass(frozen=True)
class AnalysisTarget:
    analysis_id: str
    test_name: str
    target_kind: str
    source_variables: tuple[str, ...]
    category: str
    subcategory: str
    primary_organ_system: str
    aging_domain: str
    measurement_class: str
    nhanes_presence_status: str
    nhanes_match_name: str
    notes: str


DERIVED_RULES: dict[str, dict[str, object]] = {
    "egfr": {
        "test_name": "eGFR",
        "components": ("LBXSCR", "LBDSCR", "LBDSCRSI", "SSCYST", "SSCYPC"),
        "notes": "Derived with the 2021 CKD-EPI creatinine-cystatin equation (race-free).",
    },
    "neutrophil_to_lymphocyte_ratio": {
        "test_name": "Neutrophil-to-lymphocyte ratio",
        "components": ("LBDNENO", "LBDLYMNO"),
        "notes": "Derived ratio from trimmed component trajectories.",
    },
    "platelet_to_lymphocyte_ratio": {
        "test_name": "Platelet-to-lymphocyte ratio",
        "components": ("LBXPLTSI", "LBDLYMNO"),
        "notes": "Derived ratio from trimmed component trajectories.",
    },
    "cd4_cd8_ratio": {
        "test_name": "CD4/CD8 ratio",
        "components": ("LBXCD4", "LBXCD8"),
        "notes": "Derived ratio from trimmed component trajectories.",
    },
}


def slugify(text: str) -> str:
    value = re.sub(r"[^a-z0-9]+", "_", str(text).strip().lower())
    return value.strip("_")


def split_source_variables(value: object) -> tuple[str, ...]:
    tokens = [
        tok.strip()
        for tok in re.split(r"[;|]", str(value or ""))
        if tok and str(tok).strip() and str(tok).strip().lower() != "nan"
    ]
    return tuple(dict.fromkeys(tokens))


def load_catalog_metadata(catalog_path: Path) -> pd.DataFrame:
    catalog = pd.read_csv(catalog_path).fillna("")
    catalog["analysis_id"] = catalog["test_name"].map(slugify)
    keep_cols = [
        "analysis_id",
        "test_name",
        "category",
        "subcategory",
        "primary_organ_system",
        "aging_domain",
        "measurement_class",
    ]
    out = catalog.reindex(columns=keep_cols).copy()
    out["target_kind"] = "catalog"
    return out.drop_duplicates(subset=["analysis_id"]).reset_index(drop=True)


def build_catalog_alias_index(catalog_df: pd.DataFrame) -> dict[str, set[str]]:
    alias_index: dict[str, set[str]] = {}
    for row in catalog_df.itertuples(index=False):
        analysis_id = str(getattr(row, "analysis_id"))
        tokens = [getattr(row, "test_name", "")]
        for extra_col in ("aliases", "nhanes_match_name"):
            if extra_col in catalog_df.columns:
                tokens.append(getattr(row, extra_col, ""))
        for token_group in tokens:
            for part in re.split(r"[;|]", str(token_group)):
                part = part.strip()
                if not part:
                    continue
                alias_index.setdefault(slugify(part), set()).add(analysis_id)
    return alias_index


def expand_clalit_mapping_target(target: object) -> list[dict[str, object]]:
    if isinstance(target, list):
        out: list[dict[str, object]] = []
        for item in target:
            out.extend(expand_clalit_mapping_target(item))
        return out
    if isinstance(target, dict):
        biomarker_id = str(target.get("biomarker_id") or target.get("id") or "").strip()
        if not biomarker_id:
            return []
        scale_factor = float(target.get("scale_factor", 1.0) or 1.0)
        scale_reason = str(target.get("scale_reason") or "").strip()
        return [{"biomarker_id": biomarker_id, "scale_factor": scale_factor, "scale_reason": scale_reason}]
    biomarker_id = str(target or "").strip()
    if not biomarker_id:
        return []
    return [{"biomarker_id": biomarker_id, "scale_factor": 1.0, "scale_reason": ""}]


def resolve_clalit_analysis_id(
    clalit_biomarker_id: object,
    alias_index: dict[str, set[str]],
    valid_analysis_ids: set[str],
) -> tuple[str | None, str]:
    raw_name = str(clalit_biomarker_id or "").strip()
    if not raw_name:
        return None, "empty"
    override_id = CLALIT_CANONICAL_OVERRIDES.get(raw_name.lower())
    if override_id:
        return (override_id, "override") if override_id in valid_analysis_ids else (None, "override_missing_in_catalog")

    slug = slugify(raw_name)
    if slug in valid_analysis_ids:
        return slug, "direct_slug"

    matched_ids = alias_index.get(slug, set())
    if len(matched_ids) == 1:
        analysis_id = next(iter(matched_ids))
        return analysis_id, "catalog_alias"
    if len(matched_ids) > 1:
        return None, "ambiguous_alias"
    return None, "unresolved"


def weighted_average_or_nan(values: pd.Series, weights: pd.Series) -> float:
    value_array = pd.to_numeric(values, errors="coerce").to_numpy(dtype=float)
    weight_array = pd.to_numeric(weights, errors="coerce").to_numpy(dtype=float)
    valid_mask = np.isfinite(value_array) & np.isfinite(weight_array) & (weight_array > 0)
    if not valid_mask.any():
        return np.nan
    return float(np.average(value_array[valid_mask], weights=weight_array[valid_mask]))


def _load_clalit_mapping(clalit_map_path: Path) -> dict[str, object]:
    with open(clalit_map_path) as fh:
        return json.load(fh)


def _attach_clalit_mapping(frame: pd.DataFrame, mapping: dict[str, object]) -> pd.DataFrame:
    clalit_df = frame.copy()
    clalit_df["mapped_targets"] = clalit_df["test"].map(mapping)
    clalit_df = clalit_df.dropna(subset=["mapped_targets"]).copy()
    clalit_df["mapped_targets"] = clalit_df["mapped_targets"].apply(expand_clalit_mapping_target)
    clalit_df = clalit_df.explode("mapped_targets").copy()
    clalit_df = clalit_df[clalit_df["mapped_targets"].notna()].copy()
    clalit_df["clalit_biomarker_id"] = clalit_df["mapped_targets"].apply(lambda value: str(value.get("biomarker_id") or "").strip())
    clalit_df["scale_factor"] = clalit_df["mapped_targets"].apply(lambda value: float(value.get("scale_factor", 1.0) or 1.0))
    clalit_df["scale_reason"] = clalit_df["mapped_targets"].apply(lambda value: str(value.get("scale_reason") or "").strip())
    return clalit_df.drop(columns=["mapped_targets"])


def _resolve_clalit_targets(
    clalit_df: pd.DataFrame,
    catalog_path: Path,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    catalog_df = pd.read_csv(catalog_path).fillna("")
    catalog_df["analysis_id"] = catalog_df["test_name"].map(slugify)
    alias_index = build_catalog_alias_index(catalog_df)
    valid_analysis_ids = set(catalog_df["analysis_id"].astype(str))

    resolution_rows: list[dict[str, object]] = []
    resolved_pairs: list[tuple[str | None, str]] = []
    for biomarker_id in clalit_df["clalit_biomarker_id"].astype(str):
        analysis_id, resolution_method = resolve_clalit_analysis_id(biomarker_id, alias_index, valid_analysis_ids)
        resolved_pairs.append((analysis_id, resolution_method))
        resolution_rows.append(
            {
                "clalit_biomarker_id": biomarker_id,
                "analysis_id": analysis_id,
                "resolution_method": resolution_method,
            }
        )

    resolved_df = clalit_df.copy()
    resolved_df["analysis_id"] = [pair[0] for pair in resolved_pairs]
    resolved_df["resolution_method"] = [pair[1] for pair in resolved_pairs]

    resolution_df = pd.DataFrame(resolution_rows).drop_duplicates().sort_values(
        ["resolution_method", "clalit_biomarker_id", "analysis_id"],
        ignore_index=True,
    )
    targets_df = load_catalog_metadata(catalog_path)
    resolved_df = resolved_df[resolved_df["analysis_id"].notna()].copy()
    resolved_df = resolved_df.merge(targets_df, on="analysis_id", how="left", suffixes=("", "_catalog"))
    resolved_df["test_name"] = resolved_df["test_name_catalog"].fillna(resolved_df["test_name"])
    return resolved_df, targets_df, resolution_df


def _summarize_clalit_quartile_group(group: pd.DataFrame) -> pd.Series:
    weights = pd.to_numeric(group["n"], errors="coerce")
    median = weighted_average_or_nan(group["median"], weights)
    q25 = weighted_average_or_nan(group["q25"], weights)
    q75 = weighted_average_or_nan(group["q75"], weights)

    valid_weights = pd.to_numeric(group["n"], errors="coerce").to_numpy(dtype=float)
    valid_weights = valid_weights[np.isfinite(valid_weights) & (valid_weights > 0)]
    total_n = float(valid_weights.sum()) if len(valid_weights) else np.nan

    return pd.Series(
        {
            "n": total_n,
            "median": median,
            "q25": q25,
            "q75": q75,
        }
    )


def _finalize_clalit_grouped_stats(grouped: pd.DataFrame, targets_df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    grouped = grouped.copy()
    grouped["iqr"] = grouped["q75"] - grouped["q25"]
    grouped["cv"] = grouped["iqr"] / grouped["median"]
    grouped.loc[grouped["median"].abs() < EPS, "cv"] = np.nan
    grouped["quantile_skewness"] = quantile_skewness_from_stats(grouped["q25"], grouped["median"], grouped["q75"])
    grouped["passes_n_threshold"] = grouped["n"] >= MIN_BIN_N
    grouped = grouped[
        [
            "analysis_id",
            "test_name",
            "cohort",
            "age_bin",
            "age_mid",
            "n",
            "median",
            "cv",
            "quantile_skewness",
            "passes_n_threshold",
        ]
    ].sort_values(["cohort", "analysis_id", "age_mid"], ignore_index=True)

    included_ids = set(grouped["analysis_id"].astype(str))
    filtered_targets = targets_df[targets_df["analysis_id"].isin(included_ids)].copy().sort_values(
        ["category", "test_name", "analysis_id"],
        ignore_index=True,
    )
    return grouped, filtered_targets


def _build_clalit_summary_stats_from_quartiles(
    catalog_path: Path,
    clalit_quartiles_path: Path,
    clalit_map_path: Path,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    mapping = _load_clalit_mapping(clalit_map_path)

    clalit_df = pd.read_csv(clalit_quartiles_path).copy()
    clalit_df["cohort"] = clalit_df["sex"].astype(str).str.lower()
    clalit_df = clalit_df[clalit_df["cohort"].isin(CLALIT_COHORTS)].copy()
    clalit_df = clalit_df[clalit_df["scale_type"] == "regular"].copy()
    clalit_df = clalit_df[clalit_df["age_bin"].isin(AGE_BIN_LABELS)].copy()
    clalit_df = _attach_clalit_mapping(clalit_df, mapping)

    clalit_df["n"] = pd.to_numeric(clalit_df["stats_n_total"], errors="coerce")
    for col in CLALIT_QUARTILE_RAW_COLUMNS:
        clalit_df[col] = pd.to_numeric(clalit_df[col], errors="coerce")

    clalit_df["q25"] = clalit_df["raw_q1"] * clalit_df["scale_factor"]
    clalit_df["median"] = clalit_df["raw_q2"] * clalit_df["scale_factor"]
    clalit_df["q75"] = clalit_df["raw_q3"] * clalit_df["scale_factor"]
    clalit_df["age_mid"] = clalit_df["age_bin"].map(AGE_BIN_MIDS).astype(float)

    clalit_resolved, targets_df, resolution_df = _resolve_clalit_targets(clalit_df, catalog_path)
    grouped = (
        clalit_resolved.groupby(["analysis_id", "test_name", "cohort", "age_bin", "age_mid"], observed=True)[
            ["n", "median", "q25", "q75"]
        ]
        .apply(_summarize_clalit_quartile_group)
        .reset_index()
    )
    grouped, targets_df = _finalize_clalit_grouped_stats(grouped, targets_df)
    return grouped, targets_df, resolution_df


def _build_clalit_summary_stats_from_legacy_stats(
    catalog_path: Path,
    clalit_f_path: Path,
    clalit_m_path: Path,
    clalit_map_path: Path,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    mapping = _load_clalit_mapping(clalit_map_path)

    sex_frames = []
    for cohort, path in (("female", clalit_f_path), ("male", clalit_m_path)):
        frame = pd.read_csv(path).copy()
        frame["cohort"] = cohort
        sex_frames.append(frame)

    clalit_df = pd.concat(sex_frames, ignore_index=True)
    clalit_df = _attach_clalit_mapping(clalit_df, mapping)
    clalit_df = clalit_df[(clalit_df["age"] >= 20) & (clalit_df["age"] < 85)].copy()

    for col in CLALIT_SCALE_COLUMNS:
        if col in clalit_df.columns:
            clalit_df[col] = pd.to_numeric(clalit_df[col], errors="coerce") * clalit_df["scale_factor"]
    clalit_df["median"] = pd.to_numeric(clalit_df["median"], errors="coerce")
    clalit_df["q25"] = pd.to_numeric(clalit_df["q25"], errors="coerce")
    clalit_df["q75"] = pd.to_numeric(clalit_df["q75"], errors="coerce")
    clalit_df["n"] = pd.to_numeric(clalit_df["n"], errors="coerce")
    clalit_df["age_mid"] = pd.to_numeric(clalit_df["age"], errors="coerce")
    clalit_df["age_bin"] = clalit_df["age_mid"].round().astype("Int64").astype(str)

    clalit_resolved, targets_df, resolution_df = _resolve_clalit_targets(clalit_df, catalog_path)
    grouped = (
        clalit_resolved.groupby(["analysis_id", "test_name", "cohort", "age_bin", "age_mid"], observed=True)[
            ["n", "median", "q25", "q75"]
        ]
        .apply(_summarize_clalit_quartile_group)
        .reset_index()
    )
    grouped, targets_df = _finalize_clalit_grouped_stats(grouped, targets_df)
    return grouped, targets_df, resolution_df


def build_clalit_summary_stats(
    catalog_path: Path,
    clalit_map_path: Path,
    *,
    clalit_quartiles_path: Path | None = None,
    clalit_f_path: Path | None = None,
    clalit_m_path: Path | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    if clalit_quartiles_path and Path(clalit_quartiles_path).exists():
        return _build_clalit_summary_stats_from_quartiles(
            catalog_path,
            Path(clalit_quartiles_path),
            clalit_map_path,
        )

    if clalit_f_path and clalit_m_path and Path(clalit_f_path).exists() and Path(clalit_m_path).exists():
        return _build_clalit_summary_stats_from_legacy_stats(
            catalog_path,
            Path(clalit_f_path),
            Path(clalit_m_path),
            clalit_map_path,
        )

    raise FileNotFoundError("Clalit summary analysis requires either clalit_quartiles.csv or both legacy female/male statistics CSVs.")


def quantile_skewness_from_stats(q25, median, q75):
    q25_s = pd.to_numeric(q25, errors="coerce")
    median_s = pd.to_numeric(median, errors="coerce")
    q75_s = pd.to_numeric(q75, errors="coerce")
    denom = q75_s - q25_s
    out = (q75_s + q25_s - 2.0 * median_s) / denom
    return out.where(denom.abs() > EPS, np.nan)


def tail_asymmetry_from_stats(q10, q50, q90):
    q10_s = pd.to_numeric(q10, errors="coerce")
    q50_s = pd.to_numeric(q50, errors="coerce")
    q90_s = pd.to_numeric(q90, errors="coerce")
    denom = q90_s - q10_s
    out = (q90_s + q10_s - 2.0 * q50_s) / denom
    return out.where(denom.abs() > EPS, np.nan)


def assign_age_bins(age: pd.Series) -> tuple[pd.Series, pd.Series]:
    age_bin = pd.cut(age, bins=AGE_BIN_EDGES, labels=AGE_BIN_LABELS, right=False, include_lowest=True)
    age_mid = age_bin.map(AGE_BIN_MIDS).astype(float)
    return age_bin, age_mid


def safe_skew(values: np.ndarray) -> float:
    if len(values) < 3:
        return np.nan
    value = float(scipy_skew(values, bias=False, nan_policy="omit"))
    return value if np.isfinite(value) else np.nan


def compute_egfr_2021(scr_mg_dl: np.ndarray, scys_mg_l: np.ndarray, age_years: np.ndarray, sex: np.ndarray) -> np.ndarray:
    sex_lower = np.asarray(pd.Series(sex).astype(str).str.lower())
    female_mask = sex_lower == "female"
    kappa = np.where(female_mask, 0.7, 0.9)
    alpha = np.where(female_mask, -0.219, -0.144)
    scr_ratio = scr_mg_dl / kappa
    scys_ratio = scys_mg_l / 0.8
    out = (
        135.0
        * np.minimum(scr_ratio, 1.0) ** alpha
        * np.maximum(scr_ratio, 1.0) ** (-0.544)
        * np.minimum(scys_ratio, 1.0) ** (-0.323)
        * np.maximum(scys_ratio, 1.0) ** (-0.778)
        * (0.9961 ** age_years)
        * np.where(female_mask, 0.963, 1.0)
    )
    out = np.where(np.isfinite(out), out, np.nan)
    return out


def cohort_subset(df: pd.DataFrame, cohort: str) -> pd.DataFrame:
    if cohort == "pooled":
        return df.copy()
    sex_name = "Female" if cohort == "female" else "Male"
    return df[df["sex"] == sex_name].copy()


def dedupe_long_rows(df: pd.DataFrame, group_cols: list[str]) -> pd.DataFrame:
    return (
        df.groupby(group_cols, observed=True)["value"]
        .mean()
        .reset_index()
        .replace([np.inf, -np.inf], np.nan)
        .dropna(subset=["value"])
    )


def load_targets(catalog_path: Path, long_df: pd.DataFrame, max_biomarkers: int | None = None) -> tuple[list[AnalysisTarget], pd.DataFrame]:
    catalog = pd.read_csv(catalog_path).fillna("")
    variable_names = set(long_df["variable_name"].dropna().astype(str).unique())
    targets: list[AnalysisTarget] = []
    manifest_rows: list[dict[str, object]] = []

    for row in catalog.itertuples(index=False):
        analysis_id = slugify(row.test_name)
        base = {
            "analysis_id": analysis_id,
            "test_name": str(row.test_name),
            "category": str(getattr(row, "category", "")),
            "subcategory": str(getattr(row, "subcategory", "")),
            "primary_organ_system": str(getattr(row, "primary_organ_system", "")),
            "aging_domain": str(getattr(row, "aging_domain", "")),
            "measurement_class": str(getattr(row, "measurement_class", "")),
            "nhanes_presence_status": str(getattr(row, "nhanes_presence_status", "")),
            "nhanes_match_name": str(getattr(row, "nhanes_match_name", "")),
            "notes": str(getattr(row, "notes", "")),
        }
        src_vars = split_source_variables(getattr(row, "nhanes_source_variables", ""))
        presence_status = str(getattr(row, "nhanes_presence_status", ""))

        include = False
        target_kind = "direct"
        drop_reason = ""
        source_vars = src_vars

        if analysis_id in DERIVED_RULES:
            target_kind = "derived"
            source_vars = tuple(DERIVED_RULES[analysis_id]["components"])
            include = all(v in variable_names for v in source_vars)
            if not include:
                drop_reason = "missing_derived_components"
        elif presence_status == "derived_from_nhanes_components":
            target_kind = "derived"
            include = False
            drop_reason = "unsupported_derived_concept_for_analysis"
        else:
            include = len(set(source_vars) & variable_names) > 0
            if not include:
                drop_reason = "no_matching_variable_name_in_long_table"

        if include:
            targets.append(
                AnalysisTarget(
                    analysis_id=analysis_id,
                    test_name=base["test_name"],
                    target_kind=target_kind,
                    source_variables=tuple(source_vars),
                    category=base["category"],
                    subcategory=base["subcategory"],
                    primary_organ_system=base["primary_organ_system"],
                    aging_domain=base["aging_domain"],
                    measurement_class=base["measurement_class"],
                    nhanes_presence_status=base["nhanes_presence_status"],
                    nhanes_match_name=base["nhanes_match_name"],
                    notes=DERIVED_RULES.get(analysis_id, {}).get("notes", base["notes"]),
                )
            )

        manifest_rows.append(
            {
                **base,
                "target_kind": target_kind,
                "source_variables": "|".join(source_vars),
                "included_from_catalog": bool(include),
                "drop_reason": drop_reason,
            }
        )

    manifest = pd.DataFrame(manifest_rows)
    if max_biomarkers is not None:
        keep_ids = {t.analysis_id for t in targets[:max_biomarkers]}
        over_limit = manifest["included_from_catalog"] & ~manifest["analysis_id"].isin(keep_ids)
        targets = [t for t in targets if t.analysis_id in keep_ids]
        manifest.loc[over_limit, "drop_reason"] = "max_biomarkers_limit"
        manifest["included_from_catalog"] = manifest["included_from_catalog"] & manifest["analysis_id"].isin(keep_ids)
    return targets, manifest


def build_direct_rows(long_df: pd.DataFrame, target: AnalysisTarget) -> pd.DataFrame:
    use = long_df[long_df["variable_name"].isin(target.source_variables)].copy()
    if use.empty:
        return pd.DataFrame(columns=["seqn", "cycle_start_year", "age_years", "sex", "analysis_id", "test_name", "value"])
    group_cols = ["seqn", "cycle_start_year", "age_years", "sex"]
    deduped = dedupe_long_rows(use[group_cols + ["value"]], group_cols)
    deduped["analysis_id"] = target.analysis_id
    deduped["test_name"] = target.test_name
    return deduped


def build_derived_rows(long_df: pd.DataFrame, target: AnalysisTarget) -> pd.DataFrame:
    base_cols = ["seqn", "cycle_start_year", "age_years", "sex"]

    def component(var_names: Iterable[str], label: str) -> pd.DataFrame:
        use = long_df[long_df["variable_name"].isin(list(var_names))][base_cols + ["value"]].copy()
        if use.empty:
            return pd.DataFrame(columns=base_cols + [label])
        out = dedupe_long_rows(use, base_cols).rename(columns={"value": label})
        return out

    if target.analysis_id == "neutrophil_to_lymphocyte_ratio":
        neut = component(["LBDNENO"], "neutrophils")
        lymph = component(["LBDLYMNO"], "lymphocytes")
        merged = neut.merge(lymph, on=base_cols, how="inner")
        merged = merged[merged["lymphocytes"] > 0].copy()
        merged["value"] = merged["neutrophils"] / merged["lymphocytes"]
    elif target.analysis_id == "platelet_to_lymphocyte_ratio":
        plate = component(["LBXPLTSI"], "platelets")
        lymph = component(["LBDLYMNO"], "lymphocytes")
        merged = plate.merge(lymph, on=base_cols, how="inner")
        merged = merged[merged["lymphocytes"] > 0].copy()
        merged["value"] = merged["platelets"] / merged["lymphocytes"]
    elif target.analysis_id == "cd4_cd8_ratio":
        cd4 = component(["LBXCD4"], "cd4")
        cd8 = component(["LBXCD8"], "cd8")
        merged = cd4.merge(cd8, on=base_cols, how="inner")
        merged = merged[merged["cd8"] > 0].copy()
        merged["value"] = merged["cd4"] / merged["cd8"]
    elif target.analysis_id == "egfr":
        scr = component(["LBXSCR", "LBDSCR", "LBDSCRSI"], "creatinine")
        scys = component(["SSCYST", "SSCYPC"], "cystatin_c")
        merged = scr.merge(scys, on=base_cols, how="inner")
        merged = merged[(merged["creatinine"] > 0) & (merged["cystatin_c"] > 0)].copy()
        merged["value"] = compute_egfr_2021(
            merged["creatinine"].to_numpy(dtype=float),
            merged["cystatin_c"].to_numpy(dtype=float),
            merged["age_years"].to_numpy(dtype=float),
            merged["sex"].to_numpy(),
        )
    else:
        merged = pd.DataFrame(columns=base_cols + ["value"])

    if merged.empty:
        return pd.DataFrame(columns=base_cols + ["analysis_id", "test_name", "value"])
    merged = merged.replace([np.inf, -np.inf], np.nan).dropna(subset=["value"]).copy()
    merged["analysis_id"] = target.analysis_id
    merged["test_name"] = target.test_name
    return merged[base_cols + ["analysis_id", "test_name", "value"]]


def build_analysis_rows(long_df: pd.DataFrame, targets: list[AnalysisTarget]) -> pd.DataFrame:
    rows: list[pd.DataFrame] = []
    keep_cols = ["seqn", "cycle_start_year", "age_years", "sex", "variable_name", "value"]
    use = long_df[keep_cols].copy()
    use = use[(use["age_years"] >= 20) & (use["age_years"] < 85)].dropna(subset=["value", "age_years", "sex"])
    for target in targets:
        target_rows = build_direct_rows(use, target) if target.target_kind == "direct" else build_derived_rows(use, target)
        if not target_rows.empty:
            rows.append(target_rows)
    if not rows:
        return pd.DataFrame(columns=["seqn", "cycle_start_year", "age_years", "sex", "analysis_id", "test_name", "value"])
    out = pd.concat(rows, ignore_index=True)
    out["analysis_id"] = out["analysis_id"].astype(str)
    out["test_name"] = out["test_name"].astype(str)
    return out


def trim_within_bins(df: pd.DataFrame, cohort: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    tmp = cohort_subset(df, cohort)
    if tmp.empty:
        empty_long = pd.DataFrame(
            columns=["analysis_id", "test_name", "cohort", "seqn", "cycle_start_year", "age_years", "sex", "age_bin", "age_mid", "value"]
        )
        empty_summary = pd.DataFrame(
            columns=[
                "analysis_id",
                "test_name",
                "cohort",
                "age_bin",
                "age_mid",
                "raw_n",
                "kept_n",
                "removed_n",
                "removed_pct",
                "trim_lo",
                "trim_hi",
            ]
        )
        return empty_long, empty_summary

    tmp = tmp.copy()
    tmp["age_bin"], tmp["age_mid"] = assign_age_bins(tmp["age_years"])
    tmp = tmp.dropna(subset=["age_bin", "value"]).copy()
    keys = ["analysis_id", "test_name", "age_bin", "age_mid"]
    quant = (
        tmp.groupby(keys, observed=True)["value"]
        .quantile([TRIM_LO, TRIM_HI])
        .unstack(level=-1)
        .rename(columns={TRIM_LO: "trim_lo", TRIM_HI: "trim_hi"})
        .reset_index()
    )
    raw_counts = tmp.groupby(keys, observed=True).size().reset_index(name="raw_n")
    merged = tmp.merge(quant, on=keys, how="left").merge(raw_counts, on=keys, how="left")
    kept = merged[(merged["value"] >= merged["trim_lo"]) & (merged["value"] <= merged["trim_hi"])].copy()
    kept_counts = kept.groupby(keys, observed=True).size().reset_index(name="kept_n")
    summary = raw_counts.merge(quant, on=keys, how="left").merge(kept_counts, on=keys, how="left")
    summary["kept_n"] = summary["kept_n"].fillna(0).astype(int)
    summary["removed_n"] = summary["raw_n"] - summary["kept_n"]
    summary["removed_pct"] = np.where(summary["raw_n"] > 0, summary["removed_n"] / summary["raw_n"], np.nan)
    summary["cohort"] = cohort
    kept["cohort"] = cohort
    kept = kept[
        ["analysis_id", "test_name", "cohort", "seqn", "cycle_start_year", "age_years", "sex", "age_bin", "age_mid", "value"]
    ].reset_index(drop=True)
    return kept, summary.reset_index(drop=True)


def compute_summary_stats(trimmed_df: pd.DataFrame) -> pd.DataFrame:
    if trimmed_df.empty:
        return pd.DataFrame(
            columns=[
                "analysis_id",
                "test_name",
                "cohort",
                "age_bin",
                "age_mid",
                "n",
                "mean",
                "std",
                "median",
                "q10",
                "q25",
                "q75",
                "q90",
                "skewness",
                "iqr",
                "cv",
                "quantile_skewness",
                "passes_n_threshold",
            ]
        )

    grouped = (
        trimmed_df.groupby(["analysis_id", "test_name", "cohort", "age_bin", "age_mid"], observed=True)["value"]
        .agg(
            n="count",
            mean="mean",
            std="std",
            median="median",
            q10=lambda s: float(np.nanpercentile(s.to_numpy(dtype=float), 10)),
            q25=lambda s: float(np.nanpercentile(s.to_numpy(dtype=float), 25)),
            q75=lambda s: float(np.nanpercentile(s.to_numpy(dtype=float), 75)),
            q90=lambda s: float(np.nanpercentile(s.to_numpy(dtype=float), 90)),
            skewness=lambda s: safe_skew(s.to_numpy(dtype=float)),
        )
        .reset_index()
    )
    grouped["iqr"] = grouped["q75"] - grouped["q25"]
    grouped["cv"] = grouped["std"] / grouped["mean"].abs()
    grouped.loc[grouped["mean"].abs() < EPS, "cv"] = np.nan
    grouped["quantile_skewness"] = quantile_skewness_from_stats(grouped["q25"], grouped["median"], grouped["q75"])
    grouped["passes_n_threshold"] = grouped["n"] >= MIN_BIN_N
    return grouped


def _fit_weighted_spline(x: np.ndarray, y: np.ndarray, weights: np.ndarray):
    order = np.argsort(x)
    x_sorted = x[order]
    y_sorted = y[order]
    w_sorted = weights[order]
    k = min(3, max(1, len(np.unique(x_sorted)) - 1))
    weighted_mean = np.average(y_sorted, weights=w_sorted)
    weighted_var = np.average((y_sorted - weighted_mean) ** 2, weights=w_sorted)
    smooth_strength = max(weighted_var * len(x_sorted), EPS)
    return UnivariateSpline(x_sorted, y_sorted, w=np.sqrt(w_sorted), k=k, s=smooth_strength)


def _fit_weighted_gam(x: np.ndarray, y: np.ndarray, weights: np.ndarray):
    unique_x = np.unique(x)
    n_splines = min(8, max(4, len(unique_x)))
    gam = LinearGAM(s(0, n_splines=n_splines, spline_order=3))
    lam_grid = np.logspace(-2, 2, 5)
    try:
        gam.gridsearch(
            x.reshape(-1, 1),
            y,
            weights=weights,
            lam=lam_grid,
            progress=False,
        )
        return lambda ages: gam.predict(np.asarray(ages, dtype=float).reshape(-1, 1))
    except Exception:
        try:
            gam.fit(x.reshape(-1, 1), y, weights=weights)
            return lambda ages: gam.predict(np.asarray(ages, dtype=float).reshape(-1, 1))
        except Exception:
            spline = _fit_weighted_spline(x, y, weights)
            return lambda ages: spline(np.asarray(ages, dtype=float))


def fit_smoothed_curve(age_mid: np.ndarray, values: np.ndarray, weights: np.ndarray, *, log_scale: bool) -> tuple[np.ndarray, float]:
    mask = np.isfinite(age_mid) & np.isfinite(values) & np.isfinite(weights) & (weights > 0)
    if log_scale:
        mask &= values > 0
    x = age_mid[mask]
    y = values[mask]
    w = weights[mask]
    if len(np.unique(x)) < MIN_VALID_BINS:
        return np.full(len(AGE_GRID), np.nan), np.nan
    if log_scale:
        y = np.log(y)
    if np.nanstd(y) < EPS:
        const = float(np.nanmean(y))
        grid_pred = np.full(len(AGE_GRID), const, dtype=float)
        ref_pred = const
        if log_scale:
            grid_pred = np.exp(grid_pred)
            ref_pred = math.exp(ref_pred)
        return grid_pred, ref_pred
    predictor = _fit_weighted_gam(x, y, w)
    grid_pred = np.asarray(predictor(AGE_GRID), dtype=float)
    ref_pred = float(np.asarray(predictor(np.array([REFERENCE_AGE], dtype=float)), dtype=float)[0])
    if log_scale:
        grid_pred = np.exp(grid_pred)
        ref_pred = math.exp(ref_pred)
    return grid_pred, ref_pred


def describe_curve_with_spearman(values: np.ndarray) -> float:
    mask = np.isfinite(values)
    if mask.sum() < 2:
        return np.nan
    if np.nanstd(values[mask]) < EPS:
        return 0.0
    rho, _ = spearmanr(AGE_GRID[mask], values[mask])
    return float(rho) if np.isfinite(rho) else np.nan


def normalize_curve(values: np.ndarray, reference: float) -> dict[str, np.ndarray]:
    out = {"raw": values.copy()}
    if np.isfinite(reference) and abs(reference) > EPS:
        ratio = values / reference
        out["young_ratio"] = ratio
    else:
        out["young_ratio"] = np.full_like(values, np.nan)

    ratio = out["young_ratio"]
    if np.all(np.isfinite(ratio)) and np.all(ratio > 0):
        out["young_log_fold"] = np.log(ratio)
    else:
        out["young_log_fold"] = np.full_like(values, np.nan)

    if np.isfinite(values).sum() >= 2:
        curve_mean = float(np.nanmean(values))
        curve_sd = float(np.nanstd(values))
        if np.isfinite(curve_sd) and curve_sd > EPS:
            out["shape_z"] = (values - curve_mean) / curve_sd
        elif np.isfinite(curve_mean):
            out["shape_z"] = np.zeros_like(values, dtype=float)
        else:
            out["shape_z"] = np.full_like(values, np.nan)
    else:
        out["shape_z"] = np.full_like(values, np.nan)
    return out


def build_empty_fpca_tables(ids: list[str] | None = None, test_names: list[str] | None = None) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    ids = ids or []
    test_names = test_names or []
    score_df = pd.DataFrame({"analysis_id": ids, "test_name": test_names})
    eig_df = pd.DataFrame({"age": AGE_GRID})
    ev_df = pd.DataFrame(
        {
            "pc": [f"pc{k}" for k in range(1, FPCA_COMPONENTS + 1)],
            "explained_variance_ratio": [np.nan] * FPCA_COMPONENTS,
        }
    )
    for k in range(1, FPCA_COMPONENTS + 1):
        score_df[f"pc{k}"] = np.nan
        eig_df[f"pc{k}"] = np.nan
    return score_df, eig_df, ev_df


def run_fpca_from_curve_map(curve_map: dict[str, dict[str, object]]) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    if not curve_map:
        return build_empty_fpca_tables()

    ids = sorted(curve_map)
    test_names = [str(curve_map[aid]["test_name"]) for aid in ids]
    score_df, eig_df, ev_df = build_empty_fpca_tables(ids, test_names)
    matrix = np.vstack([np.asarray(curve_map[aid]["values"], dtype=float) for aid in ids])

    if matrix.shape[0] < 2:
        return score_df, eig_df, ev_df
    if np.nanmax(np.abs(matrix - matrix[0])) < EPS:
        for k in range(1, FPCA_COMPONENTS + 1):
            score_df[f"pc{k}"] = 0.0
            eig_df[f"pc{k}"] = 0.0
        ev_df["explained_variance_ratio"] = 0.0
        return score_df, eig_df, ev_df

    n_components = min(FPCA_COMPONENTS, matrix.shape[0], matrix.shape[1])
    pca = PCA(n_components=n_components)
    scores = pca.fit_transform(matrix)
    for k in range(n_components):
        score_df[f"pc{k+1}"] = scores[:, k]
        eig_df[f"pc{k+1}"] = pca.components_[k]
        ev_df.loc[k, "explained_variance_ratio"] = float(pca.explained_variance_ratio_[k])
    return score_df, eig_df, ev_df


def get_retained_pc_columns(score_df: pd.DataFrame, ev_df: pd.DataFrame) -> list[str]:
    retained: list[str] = []
    for pc in ("pc1", "pc2"):
        if pc in score_df.columns and score_df[pc].notna().any():
            retained.append(pc)
    if "pc3" in score_df.columns and score_df["pc3"].notna().any() and len(ev_df) >= 3:
        pc3_ev = pd.to_numeric(ev_df["explained_variance_ratio"], errors="coerce").iloc[2]
        if np.isfinite(pc3_ev) and pc3_ev >= FPCA_PC3_EV_THRESHOLD:
            retained.append("pc3")
    return retained


def build_concatenated_score_tables(
    family_order: Iterable[str],
    family_scores: dict[str, pd.DataFrame],
    family_evs: dict[str, pd.DataFrame],
) -> tuple[pd.DataFrame, pd.DataFrame]:
    merged: pd.DataFrame | None = None
    retention_rows: list[dict[str, object]] = []

    for family in family_order:
        score_df = family_scores.get(family, pd.DataFrame())
        ev_df = family_evs.get(family, pd.DataFrame())
        retained_pcs = get_retained_pc_columns(score_df, ev_df)
        retention_rows.append(
            {
                "family": family,
                "retained_pcs": "|".join(retained_pcs),
                "n_retained_pcs": len(retained_pcs),
            }
        )
        if score_df.empty or not retained_pcs:
            continue
        keep_cols = ["analysis_id", "test_name"] + retained_pcs
        sub = score_df[keep_cols].copy()
        sub = sub.rename(columns={pc: f"{family}_{pc}" for pc in retained_pcs})
        merged = sub if merged is None else merged.merge(sub, on=["analysis_id", "test_name"], how="inner")

    if merged is None:
        merged = pd.DataFrame(columns=["analysis_id", "test_name"])

    score_cols = [c for c in merged.columns if c not in {"analysis_id", "test_name"}]
    if score_cols:
        merged = merged.dropna(subset=score_cols).reset_index(drop=True)
    return merged, pd.DataFrame(retention_rows)


def concat_nonempty_frames(frames: Iterable[pd.DataFrame]) -> pd.DataFrame:
    kept = [frame for frame in frames if frame is not None and not frame.empty]
    return pd.concat(kept, ignore_index=True) if kept else pd.DataFrame()


def standardize_score_table(raw_scores: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    std_df = raw_scores[["analysis_id", "test_name"]].copy() if not raw_scores.empty else pd.DataFrame(columns=["analysis_id", "test_name"])
    scaling_rows: list[dict[str, object]] = []
    for col in [c for c in raw_scores.columns if c not in {"analysis_id", "test_name"}]:
        series = pd.to_numeric(raw_scores[col], errors="coerce")
        mean = float(series.mean()) if len(series) else np.nan
        sd = float(series.std(ddof=0)) if len(series) else np.nan
        keep = bool(np.isfinite(sd) and sd > EPS)
        scaling_rows.append(
            {
                "score_column": col,
                "mean": mean,
                "sd": sd,
                "kept": keep,
            }
        )
        if keep:
            std_df[col] = (series - mean) / sd
    return std_df, pd.DataFrame(scaling_rows)


def compute_score_pca(standardized_scores: pd.DataFrame) -> pd.DataFrame:
    pca_df = standardized_scores[["analysis_id", "test_name"]].copy()
    score_cols = [c for c in standardized_scores.columns if c not in {"analysis_id", "test_name"}]
    for k in range(1, PCA_VIS_COMPONENTS + 1):
        pca_df[f"pc{k}"] = np.nan
    if len(standardized_scores) < 2 or not score_cols:
        return pca_df
    n_components = min(PCA_VIS_COMPONENTS, len(score_cols), len(standardized_scores))
    pca = PCA(n_components=n_components)
    coords = pca.fit_transform(standardized_scores[score_cols].to_numpy(dtype=float))
    for k in range(n_components):
        pca_df[f"pc{k+1}"] = coords[:, k]
    return pca_df


def linkage_to_frame(linkage_matrix: np.ndarray | None) -> pd.DataFrame:
    if linkage_matrix is None or len(linkage_matrix) == 0:
        return pd.DataFrame(columns=["cluster_1", "cluster_2", "distance", "sample_count"])
    return pd.DataFrame(
        linkage_matrix,
        columns=["cluster_1", "cluster_2", "distance", "sample_count"],
    )


def cluster_mean_curves(curves_df: pd.DataFrame, assignments: pd.DataFrame, family_col: str) -> pd.DataFrame:
    if curves_df.empty or assignments.empty or "cluster" not in assignments.columns:
        return pd.DataFrame(columns=["analysis_id", "test_name", "cluster", family_col, "age", "value"])
    merged = curves_df.merge(assignments[["analysis_id", "test_name", "cluster"]], on=["analysis_id", "test_name"], how="inner")
    if merged.empty:
        return pd.DataFrame(columns=["cluster", family_col, "age", "value", "n_biomarkers"])
    out = (
        merged.groupby(["cluster", family_col, "age"], observed=True)
        .agg(value=("value", "mean"), n_biomarkers=("analysis_id", "nunique"))
        .reset_index()
    )
    return out


def run_hierarchical_clustering(standardized_scores: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, dict[str, object]]:
    score_cols = [c for c in standardized_scores.columns if c not in {"analysis_id", "test_name"}]
    summary: dict[str, object] = {
        "n_biomarkers": int(len(standardized_scores)),
        "n_score_columns": int(len(score_cols)),
        "best_k": None,
        "best_silhouette": None,
        "skip_reason": "",
    }
    assignments = standardized_scores[["analysis_id", "test_name"]].copy() if not standardized_scores.empty else pd.DataFrame(columns=["analysis_id", "test_name"])
    if len(standardized_scores) < CLUSTER_MIN_BIOMARKERS:
        summary["skip_reason"] = "too_few_biomarkers"
        return pd.DataFrame(columns=["k", "silhouette"]), assignments, pd.DataFrame(columns=["cluster_1", "cluster_2", "distance", "sample_count"]), summary
    if len(score_cols) < CLUSTER_MIN_FEATURES:
        summary["skip_reason"] = "too_few_nonconstant_score_columns"
        return pd.DataFrame(columns=["k", "silhouette"]), assignments, pd.DataFrame(columns=["cluster_1", "cluster_2", "distance", "sample_count"]), summary

    matrix = standardized_scores[score_cols].to_numpy(dtype=float)
    linkage_matrix = linkage(matrix, method="ward", metric="euclidean")
    best: dict[str, object] | None = None
    rows: list[dict[str, object]] = []
    max_k = min(CLUSTER_K_MAX, len(standardized_scores) - 1)
    for k in range(CLUSTER_K_MIN, max_k + 1):
        labels = fcluster(linkage_matrix, t=k, criterion="maxclust")
        if len(np.unique(labels)) < 2:
            continue
        score = float(silhouette_score(matrix, labels, metric="euclidean"))
        rows.append({"k": int(k), "silhouette": score})
        if best is None or score > float(best["silhouette"]) + EPS or (abs(score - float(best["silhouette"])) <= EPS and k < int(best["k"])):
            best = {"k": int(k), "silhouette": score, "labels": labels}

    silhouette_df = pd.DataFrame(rows)
    linkage_df = linkage_to_frame(linkage_matrix)
    if best is None:
        summary["skip_reason"] = "no_valid_k_for_silhouette"
        return silhouette_df, assignments, linkage_df, summary

    assignments = assignments.copy()
    assignments["cluster"] = np.asarray(best["labels"], dtype=int)
    summary["best_k"] = int(best["k"])
    summary["best_silhouette"] = float(best["silhouette"])
    return silhouette_df, assignments, linkage_df, summary


def plot_dendrogram_figure(linkage_df: pd.DataFrame, out_path: Path, title: str) -> None:
    if linkage_df.empty:
        return
    fig, ax = plt.subplots(figsize=(10, 5))
    dendrogram(linkage_df.to_numpy(dtype=float), no_labels=True, ax=ax, color_threshold=None)
    ax.set_title(title)
    ax.set_xlabel("Biomarker")
    ax.set_ylabel("Ward distance")
    fig.tight_layout()
    fig.savefig(out_path, dpi=160)
    plt.close(fig)


def plot_cluster_pca_scatter(pca_df: pd.DataFrame, assignments: pd.DataFrame, out_path: Path, title: str) -> None:
    if pca_df.empty or assignments.empty or "cluster" not in assignments.columns:
        return
    merged = pca_df.merge(assignments[["analysis_id", "cluster"]], on="analysis_id", how="inner")
    if merged[["pc1", "pc2"]].dropna().shape[0] < 2:
        return
    fig, ax = plt.subplots(figsize=(7, 6))
    for cluster, sub in merged.groupby("cluster", observed=True):
        ax.scatter(sub["pc1"], sub["pc2"], s=28, alpha=0.85, label=f"Cluster {int(cluster)}")
    ax.set_xlabel("PCA 1")
    ax.set_ylabel("PCA 2")
    ax.set_title(title)
    ax.legend(loc="best", fontsize=8)
    fig.tight_layout()
    fig.savefig(out_path, dpi=160)
    plt.close(fig)


def plot_cluster_mean_panel(curves_df: pd.DataFrame, family_order: Iterable[str], family_col: str, out_path: Path, title: str) -> None:
    if curves_df.empty:
        return
    families = list(family_order)
    n_plots = len(families)
    ncols = 3
    nrows = int(math.ceil(n_plots / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(14, 4 * nrows), sharex=True)
    axes = np.atleast_1d(axes).ravel()
    any_plotted = False
    for ax, family in zip(axes, families):
        sub = curves_df[curves_df[family_col] == family]
        if sub.empty:
            ax.set_axis_off()
            continue
        for cluster, cluster_sub in sub.groupby("cluster", observed=True):
            ax.plot(cluster_sub["age"], cluster_sub["value"], lw=2, label=f"Cluster {int(cluster)}")
            any_plotted = True
        ax.set_title(str(family))
    for ax in axes[n_plots:]:
        ax.set_axis_off()
    if not any_plotted:
        plt.close(fig)
        return
    axes[0].legend(loc="best", fontsize=8)
    fig.suptitle(title)
    fig.tight_layout()
    fig.savefig(out_path, dpi=160)
    plt.close(fig)


def run_family_clustering_workflow(
    family_order: Iterable[str],
    family_scores: dict[str, pd.DataFrame],
    family_evs: dict[str, pd.DataFrame],
    curves_df: pd.DataFrame,
    family_col: str,
    out_dir: Path,
    mean_plot_name: str,
    branch_name: str,
    cohort: str,
    normalization: str,
) -> tuple[dict[str, object], pd.DataFrame, pd.DataFrame]:
    ensure_dir(out_dir)
    raw_scores, retention_df = build_concatenated_score_tables(family_order, family_scores, family_evs)
    raw_scores.to_csv(out_dir / "concatenated_scores_raw.csv", index=False)
    retention_df.to_csv(out_dir / "retained_fpca_components.csv", index=False)

    standardized_scores, scaling_df = standardize_score_table(raw_scores)
    standardized_scores.to_csv(out_dir / "concatenated_scores_standardized.csv", index=False)
    scaling_df.to_csv(out_dir / "score_scaling.csv", index=False)

    silhouette_df, assignments_df, linkage_df, summary = run_hierarchical_clustering(standardized_scores)
    silhouette_df.to_csv(out_dir / "silhouette_by_k.csv", index=False)
    assignments_df.to_csv(out_dir / "cluster_assignments.csv", index=False)
    linkage_df.to_csv(out_dir / "linkage_matrix.csv", index=False)

    pca_df = compute_score_pca(standardized_scores)
    pca_df.to_csv(out_dir / "score_pca.csv", index=False)

    cluster_curves = cluster_mean_curves(curves_df, assignments_df, family_col)
    cluster_curves.to_parquet(out_dir / "cluster_mean_curves.parquet", index=False)

    summary.update(
        {
            "branch": branch_name,
            "cohort": cohort,
            "normalization": normalization,
            "retained_family_count": int((retention_df["n_retained_pcs"] > 0).sum()) if not retention_df.empty else 0,
            "retained_components": retention_df.to_dict(orient="records"),
            "raw_score_columns": [c for c in raw_scores.columns if c not in {"analysis_id", "test_name"}],
            "standardized_score_columns": [c for c in standardized_scores.columns if c not in {"analysis_id", "test_name"}],
            "dropped_score_columns": scaling_df.loc[~scaling_df["kept"], "score_column"].tolist() if not scaling_df.empty else [],
        }
    )
    (out_dir / "clustering_summary.json").write_text(json.dumps(summary, indent=2))

    if not linkage_df.empty and not summary.get("skip_reason"):
        plot_dendrogram_figure(linkage_df, out_dir / "dendrogram.png", f"{branch_name}: {cohort} / {normalization}")
    if not summary.get("skip_reason"):
        plot_cluster_pca_scatter(
            pca_df,
            assignments_df,
            out_dir / "pca_scatter_pc1_pc2_clusters.png",
            f"{branch_name}: {cohort} / {normalization}",
        )
        plot_cluster_mean_panel(
            cluster_curves,
            family_order,
            family_col,
            out_dir / mean_plot_name,
            f"{branch_name}: cluster mean trajectories ({cohort} / {normalization})",
        )

    cluster_sizes = pd.DataFrame(columns=["branch", "cohort", "normalization", "cluster", "n_biomarkers"])
    if not assignments_df.empty and "cluster" in assignments_df.columns:
        cluster_sizes = (
            assignments_df.groupby("cluster", observed=True)
            .size()
            .reset_index(name="n_biomarkers")
            .assign(branch=branch_name, cohort=cohort, normalization=normalization)
        )
        cluster_sizes = cluster_sizes[["branch", "cohort", "normalization", "cluster", "n_biomarkers"]]
        cluster_sizes.to_csv(out_dir / "cluster_sizes.csv", index=False)
    else:
        cluster_sizes.to_csv(out_dir / "cluster_sizes.csv", index=False)

    return summary, cluster_sizes, silhouette_df.assign(branch=branch_name, cohort=cohort, normalization=normalization)


def summarize_rho_completeness(
    rho_df: pd.DataFrame,
    cohort_candidate_ids: dict[str, set[str]],
    *,
    cohorts: Iterable[str],
    normalizations: Iterable[str],
    rho_columns: Iterable[str],
) -> tuple[dict[tuple[str, str], set[str]], pd.DataFrame]:
    rho_columns = tuple(str(col) for col in rho_columns)
    if rho_df.empty:
        return {}, pd.DataFrame(
            columns=[
                "cohort",
                "normalization",
                "candidate_biomarkers",
                "biomarkers_with_any_rho",
                "complete_biomarkers",
                "dropped_for_pca",
                "biomarkers_with_no_rho_rows",
                "rows_with_any_na",
                "na_cells_before_complete_case",
                *[f"missing_{col}" for col in rho_columns],
                "note",
            ]
        )

    wide = (
        rho_df.assign(feature=lambda d: "rho_" + d["metric"].astype(str))
        .pivot_table(
            index=["analysis_id", "cohort", "normalization"],
            columns="feature",
            values="spearman_rho",
            aggfunc="first",
        )
        .reset_index()
    )
    wide.columns.name = None
    for col in rho_columns:
        if col not in wide.columns:
            wide[col] = np.nan

    complete_ids_by_group: dict[tuple[str, str], set[str]] = {}
    rows: list[dict[str, object]] = []
    feature_text = ", ".join(rho_columns)
    for cohort in cohorts:
        candidate_ids = set(cohort_candidate_ids.get(cohort, set()))
        candidate_n = len(candidate_ids)
        for normalization in normalizations:
            sub = wide[(wide["cohort"] == cohort) & (wide["normalization"] == normalization)].copy()
            present_ids = set(sub["analysis_id"].astype(str)) if not sub.empty else set()
            if sub.empty:
                complete_ids_by_group[(cohort, normalization)] = set()
                row = {
                    "cohort": cohort,
                    "normalization": normalization,
                    "candidate_biomarkers": candidate_n,
                    "biomarkers_with_any_rho": 0,
                    "complete_biomarkers": 0,
                    "dropped_for_pca": candidate_n,
                    "biomarkers_with_no_rho_rows": candidate_n,
                    "rows_with_any_na": 0,
                    "na_cells_before_complete_case": 0,
                }
                for col in rho_columns:
                    row[f"missing_{col}"] = 0
            else:
                complete_mask = sub[list(rho_columns)].notna().all(axis=1)
                complete_ids = set(sub.loc[complete_mask, "analysis_id"].astype(str))
                complete_ids_by_group[(cohort, normalization)] = complete_ids
                row = {
                    "cohort": cohort,
                    "normalization": normalization,
                    "candidate_biomarkers": candidate_n,
                    "biomarkers_with_any_rho": int(len(sub)),
                    "complete_biomarkers": int(complete_mask.sum()),
                    "dropped_for_pca": int(candidate_n - len(complete_ids)),
                    "biomarkers_with_no_rho_rows": int(len(candidate_ids - present_ids)),
                    "rows_with_any_na": int((~complete_mask).sum()),
                    "na_cells_before_complete_case": int(sub[list(rho_columns)].isna().sum().sum()),
                }
                for col in rho_columns:
                    row[f"missing_{col}"] = int(sub[col].isna().sum())

            if normalization == "young_log_fold":
                row["note"] = (
                    "young_log_fold requires a strictly positive smoothed young-reference ratio across the full age grid; "
                    "skewness-derived curves often cross zero and are dropped from the complete-case PCA matrix."
                )
            else:
                row["note"] = f"PCA uses the complete-case Spearman-rho matrix across {feature_text}."
            rows.append(row)

    diag_df = pd.DataFrame(rows).sort_values(["cohort", "normalization"]).reset_index(drop=True)
    return complete_ids_by_group, diag_df


def summarize_summary_rho_completeness(
    rho_df: pd.DataFrame,
    cohort_candidate_ids: dict[str, set[str]],
) -> tuple[dict[tuple[str, str], set[str]], pd.DataFrame]:
    return summarize_rho_completeness(
        rho_df,
        cohort_candidate_ids,
        cohorts=COHORTS,
        normalizations=CURVE_NORMALIZATIONS,
        rho_columns=SUMMARY_RHO_COLUMNS,
    )


def summarize_cohort_validity(summary_stats: pd.DataFrame, manifest: pd.DataFrame) -> pd.DataFrame:
    valid_counts = (
        summary_stats[summary_stats["passes_n_threshold"]]
        .groupby(["analysis_id", "cohort"], observed=True)
        .size()
        .reset_index(name="valid_bins")
    )
    for cohort in COHORTS:
        cohort_map = dict(
            zip(
                valid_counts.loc[valid_counts["cohort"] == cohort, "analysis_id"],
                valid_counts.loc[valid_counts["cohort"] == cohort, "valid_bins"],
            )
        )
        manifest[f"valid_bins_{cohort}"] = manifest["analysis_id"].map(cohort_map).fillna(0).astype(int)
        manifest[f"included_{cohort}"] = manifest[f"valid_bins_{cohort}"] >= MIN_VALID_BINS
    manifest["included_any_cohort"] = manifest[[f"included_{cohort}" for cohort in COHORTS]].any(axis=1)
    mask = manifest["included_from_catalog"] & ~manifest["included_any_cohort"]
    manifest.loc[mask, "drop_reason"] = "insufficient_post_trim_bins"
    return manifest


def annotate_normalization_validity(
    manifest: pd.DataFrame,
    smoothed_curves: pd.DataFrame,
    reference_df: pd.DataFrame,
) -> pd.DataFrame:
    out = manifest.copy()
    valid_summary = (
        smoothed_curves[smoothed_curves["value"].notna()][["analysis_id", "cohort", "normalization", "metric"]]
        .drop_duplicates()
    )
    summary_counts = (
        valid_summary.groupby(["analysis_id", "cohort", "normalization"], observed=True)
        .size()
        .reset_index(name="metric_count")
    )
    for cohort in COHORTS:
        for normalization in CURVE_NORMALIZATIONS:
            lookup = dict(
                zip(
                    summary_counts.loc[
                        (summary_counts["cohort"] == cohort) & (summary_counts["normalization"] == normalization),
                        "analysis_id",
                    ],
                    summary_counts.loc[
                        (summary_counts["cohort"] == cohort) & (summary_counts["normalization"] == normalization),
                        "metric_count",
                    ],
                )
            )
            out[f"summary_metrics_available_{normalization}_{cohort}"] = out["analysis_id"].map(lookup).fillna(0).astype(int)

    if reference_df.empty:
        for cohort in COHORTS:
            for normalization in DISTRIBUTION_NORMALIZATIONS:
                out[f"distribution_valid_{normalization}_{cohort}"] = False
        return out

    for cohort in COHORTS:
        cohort_ref = reference_df[reference_df["cohort"] == cohort].copy()
        raw_lookup = dict(zip(cohort_ref["analysis_id"], cohort_ref["raw_valid"]))
        log_lookup = dict(zip(cohort_ref["analysis_id"], cohort_ref["log_valid"]))
        out[f"distribution_valid_young_z_raw_{cohort}"] = (
            out["analysis_id"].map(raw_lookup).astype("boolean").fillna(False).astype(bool)
        )
        out[f"distribution_valid_young_z_log_{cohort}"] = (
            out["analysis_id"].map(log_lookup).astype("boolean").fillna(False).astype(bool)
        )
    return out


def build_summary_branch(
    summary_stats: pd.DataFrame,
    targets_df: pd.DataFrame,
    out_dir: Path,
) -> tuple[
    pd.DataFrame,
    dict[tuple[str, str, str], pd.DataFrame],
    dict[tuple[str, str, str], pd.DataFrame],
    pd.DataFrame,
    pd.DataFrame,
    pd.DataFrame,
    pd.DataFrame,
]:
    ensure_dir(out_dir)
    feature_dir = out_dir / "feature_matrices"
    fpca_dir = out_dir / "fpca"
    cluster_dir = out_dir / "clustering"
    ensure_dir(feature_dir)
    ensure_dir(fpca_dir)
    ensure_dir(cluster_dir)

    smoothed_rows: list[dict[str, object]] = []
    curve_store: dict[tuple[str, str, str], dict[str, dict[str, object]]] = {}
    rho_rows: list[dict[str, object]] = []

    valid_summary = summary_stats[summary_stats["passes_n_threshold"]].copy()
    for (analysis_id, test_name, cohort), group in valid_summary.groupby(["analysis_id", "test_name", "cohort"], observed=True):
        group = group.sort_values("age_mid")
        x = group["age_mid"].to_numpy(dtype=float)
        w = group["n"].to_numpy(dtype=float)
        for metric in SUMMARY_METRICS:
            y = group[metric].to_numpy(dtype=float)
            log_scale = metric in POSITIVE_SMOOTH_METRICS
            curve, reference = fit_smoothed_curve(x, y, w, log_scale=log_scale)
            if np.isfinite(curve).sum() < len(AGE_GRID):
                continue
            norm_curves = normalize_curve(curve, reference)
            for normalization, norm_values in norm_curves.items():
                if np.isfinite(norm_values).sum() < len(AGE_GRID):
                    continue
                curve_store.setdefault((cohort, normalization, metric), {})[analysis_id] = {
                    "test_name": test_name,
                    "values": norm_values,
                }
                rho_rows.append(
                    {
                        "analysis_id": analysis_id,
                        "test_name": test_name,
                        "cohort": cohort,
                        "metric": metric,
                        "normalization": normalization,
                        "spearman_rho": describe_curve_with_spearman(norm_values),
                    }
                )
                for age, value in zip(AGE_GRID, norm_values):
                    smoothed_rows.append(
                        {
                            "analysis_id": analysis_id,
                            "test_name": test_name,
                            "cohort": cohort,
                            "metric": metric,
                            "normalization": normalization,
                            "age": float(age),
                            "value": float(value) if np.isfinite(value) else np.nan,
                            "reference_value": float(reference) if np.isfinite(reference) else np.nan,
                        }
                    )

    smoothed_df = pd.DataFrame(smoothed_rows)
    smoothed_df.to_parquet(out_dir / "smoothed_curves_long.parquet", index=False)

    rho_df = pd.DataFrame(rho_rows)
    targets_lookup = targets_df[list(SUMMARY_FEATURE_METADATA_COLUMNS)].drop_duplicates()
    cohort_candidate_ids = {
        cohort: set(summary_stats.loc[summary_stats["cohort"] == cohort, "analysis_id"].astype(str).unique().tolist())
        for cohort in COHORTS
    }
    complete_ids_by_group, rho_diag_df = summarize_summary_rho_completeness(rho_df, cohort_candidate_ids)

    fpca_scores_written: dict[tuple[str, str, str], pd.DataFrame] = {}
    fpca_ev_written: dict[tuple[str, str, str], pd.DataFrame] = {}
    cluster_summary_rows: list[dict[str, object]] = []
    cluster_size_frames: list[pd.DataFrame] = []
    cluster_silhouette_frames: list[pd.DataFrame] = []

    for cohort in COHORTS:
        cohort_dir = feature_dir / cohort
        ensure_dir(cohort_dir)
        for normalization in CURVE_NORMALIZATIONS:
            complete_ids = complete_ids_by_group.get((cohort, normalization), set())
            base = targets_lookup[targets_lookup["analysis_id"].isin(complete_ids)].copy()
            feature_df = base.sort_values(["category", "test_name", "analysis_id"]).reset_index(drop=True)
            cohort_norm_dir = fpca_dir / cohort / normalization
            ensure_dir(cohort_norm_dir)
            for metric in SUMMARY_METRICS:
                rho_metric = rho_df[
                    (rho_df["cohort"] == cohort)
                    & (rho_df["normalization"] == normalization)
                    & (rho_df["metric"] == metric)
                ][["analysis_id", "spearman_rho"]].rename(columns={"spearman_rho": f"rho_{metric}"})
                feature_df = feature_df.merge(rho_metric, on="analysis_id", how="left")

                curve_map = curve_store.get((cohort, normalization, metric), {})
                if not curve_map:
                    empty_scores, eig_df, ev_df = build_empty_fpca_tables(
                        feature_df["analysis_id"].astype(str).tolist(),
                        feature_df["test_name"].astype(str).tolist(),
                    )
                    empty_scores.to_csv(cohort_norm_dir / f"{metric}_scores.csv", index=False)
                    ev_df.to_csv(cohort_norm_dir / f"{metric}_explained_variance.csv", index=False)
                    eig_df.to_csv(cohort_norm_dir / f"{metric}_eigenfunctions.csv", index=False)
                    fpca_scores_written[(cohort, normalization, metric)] = empty_scores
                    fpca_ev_written[(cohort, normalization, metric)] = ev_df
                    for k in range(1, FPCA_COMPONENTS + 1):
                        feature_df[f"pc{k}_{metric}"] = np.nan
                    continue

                score_df, eig_df, ev_df = run_fpca_from_curve_map(curve_map)
                for k in range(1, FPCA_COMPONENTS + 1):
                    feature_df = feature_df.merge(
                        score_df[["analysis_id", f"pc{k}"]].rename(columns={f"pc{k}": f"pc{k}_{metric}"}),
                        on="analysis_id",
                        how="left",
                    )
                score_df.to_csv(cohort_norm_dir / f"{metric}_scores.csv", index=False)
                eig_df.to_csv(cohort_norm_dir / f"{metric}_eigenfunctions.csv", index=False)
                ev_df.to_csv(cohort_norm_dir / f"{metric}_explained_variance.csv", index=False)
                fpca_scores_written[(cohort, normalization, metric)] = score_df
                fpca_ev_written[(cohort, normalization, metric)] = ev_df

            feature_df.to_csv(cohort_dir / f"{normalization}_summary_features.csv", index=False)
            family_scores = {metric: fpca_scores_written[(cohort, normalization, metric)] for metric in SUMMARY_METRICS}
            family_evs = {metric: fpca_ev_written[(cohort, normalization, metric)] for metric in SUMMARY_METRICS}
            curve_subset = smoothed_df[
                (smoothed_df["cohort"] == cohort) & (smoothed_df["normalization"] == normalization)
            ][["analysis_id", "test_name", "metric", "age", "value"]].copy()
            summary_row, cluster_sizes, silhouette_df = run_family_clustering_workflow(
                SUMMARY_METRICS,
                family_scores,
                family_evs,
                curve_subset,
                "metric",
                cluster_dir / cohort / normalization,
                "cluster_mean_summary_trajectories.png",
                "summary_stats",
                cohort,
                normalization,
            )
            cluster_summary_rows.append(summary_row)
            cluster_size_frames.append(cluster_sizes)
            cluster_silhouette_frames.append(silhouette_df)

    cluster_summary_df = pd.DataFrame(cluster_summary_rows)
    cluster_sizes_df = concat_nonempty_frames(cluster_size_frames)
    cluster_silhouette_df = concat_nonempty_frames(cluster_silhouette_frames)
    return smoothed_df, fpca_scores_written, fpca_ev_written, rho_diag_df, cluster_summary_df, cluster_sizes_df, cluster_silhouette_df


def build_summary_rho_feature_outputs(
    summary_stats: pd.DataFrame,
    targets_df: pd.DataFrame,
    out_dir: Path,
    *,
    metrics: Iterable[str],
    cohorts: Iterable[str],
    rho_columns: Iterable[str],
) -> tuple[pd.DataFrame, pd.DataFrame]:
    metrics = tuple(str(metric) for metric in metrics)
    cohorts = tuple(str(cohort) for cohort in cohorts)
    rho_columns = tuple(str(col) for col in rho_columns)

    ensure_dir(out_dir)
    feature_dir = out_dir / "feature_matrices"
    ensure_dir(feature_dir)

    smoothed_rows: list[dict[str, object]] = []
    rho_rows: list[dict[str, object]] = []
    valid_summary = summary_stats[summary_stats["passes_n_threshold"]].copy()

    for (analysis_id, test_name, cohort), group in valid_summary.groupby(["analysis_id", "test_name", "cohort"], observed=True):
        group = group.sort_values("age_mid")
        x = group["age_mid"].to_numpy(dtype=float)
        w = group["n"].to_numpy(dtype=float)
        for metric in metrics:
            y = pd.to_numeric(group[metric], errors="coerce").to_numpy(dtype=float)
            log_scale = metric in POSITIVE_SMOOTH_METRICS
            curve, reference = fit_smoothed_curve(x, y, w, log_scale=log_scale)
            if np.isfinite(curve).sum() < len(AGE_GRID):
                continue
            norm_curves = normalize_curve(curve, reference)
            for normalization, norm_values in norm_curves.items():
                if np.isfinite(norm_values).sum() < len(AGE_GRID):
                    continue
                rho_rows.append(
                    {
                        "analysis_id": analysis_id,
                        "test_name": test_name,
                        "cohort": cohort,
                        "metric": metric,
                        "normalization": normalization,
                        "spearman_rho": describe_curve_with_spearman(norm_values),
                    }
                )
                for age, value in zip(AGE_GRID, norm_values):
                    smoothed_rows.append(
                        {
                            "analysis_id": analysis_id,
                            "test_name": test_name,
                            "cohort": cohort,
                            "metric": metric,
                            "normalization": normalization,
                            "age": float(age),
                            "value": float(value) if np.isfinite(value) else np.nan,
                            "reference_value": float(reference) if np.isfinite(reference) else np.nan,
                        }
                    )

    smoothed_df = pd.DataFrame(smoothed_rows)
    smoothed_df.to_parquet(out_dir / "smoothed_curves_long.parquet", index=False)

    rho_df = pd.DataFrame(rho_rows)
    targets_lookup = targets_df[list(SUMMARY_FEATURE_METADATA_COLUMNS)].drop_duplicates()
    cohort_candidate_ids = {
        cohort: set(summary_stats.loc[summary_stats["cohort"] == cohort, "analysis_id"].astype(str).unique().tolist())
        for cohort in cohorts
    }
    complete_ids_by_group, rho_diag_df = summarize_rho_completeness(
        rho_df,
        cohort_candidate_ids,
        cohorts=cohorts,
        normalizations=CURVE_NORMALIZATIONS,
        rho_columns=rho_columns,
    )

    for cohort in cohorts:
        cohort_dir = feature_dir / cohort
        ensure_dir(cohort_dir)
        for normalization in CURVE_NORMALIZATIONS:
            complete_ids = complete_ids_by_group.get((cohort, normalization), set())
            base = targets_lookup[targets_lookup["analysis_id"].isin(complete_ids)].copy()
            feature_df = base.sort_values(["category", "test_name", "analysis_id"]).reset_index(drop=True)
            for metric in metrics:
                rho_metric = rho_df[
                    (rho_df["cohort"] == cohort)
                    & (rho_df["normalization"] == normalization)
                    & (rho_df["metric"] == metric)
                ][["analysis_id", "spearman_rho"]].rename(columns={"spearman_rho": f"rho_{metric}"})
                feature_df = feature_df.merge(rho_metric, on="analysis_id", how="left")
            feature_df.to_csv(cohort_dir / f"{normalization}_summary_features.csv", index=False)

    return smoothed_df, rho_diag_df


def build_distribution_branch(
    trimmed_df: pd.DataFrame,
    manifest: pd.DataFrame,
    out_dir: Path,
) -> tuple[pd.DataFrame, pd.DataFrame, dict[tuple[str, str], pd.DataFrame], pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    ensure_dir(out_dir)
    feature_dir = out_dir / "feature_matrices"
    fpca_dir = out_dir / "fpca"
    cluster_dir = out_dir / "clustering"
    wasserstein_dir = out_dir / "wasserstein"
    ensure_dir(feature_dir)
    ensure_dir(fpca_dir)
    ensure_dir(cluster_dir)
    ensure_dir(wasserstein_dir)

    reference_rows: list[dict[str, object]] = []
    normalized_rows: list[dict[str, object]] = []
    curve_rows: list[dict[str, object]] = []
    pairwise_matrices: dict[tuple[str, str], pd.DataFrame] = {}
    cluster_summary_rows: list[dict[str, object]] = []
    cluster_size_frames: list[pd.DataFrame] = []
    cluster_silhouette_frames: list[pd.DataFrame] = []

    for cohort in COHORTS:
        cohort_trimmed = trimmed_df[trimmed_df["cohort"] == cohort].copy()
        if cohort_trimmed.empty:
            continue

        refs = []
        for analysis_id, group in cohort_trimmed.groupby("analysis_id", observed=True):
            young = group[group["age_bin"] == REFERENCE_BIN]["value"].to_numpy(dtype=float)
            raw_valid = len(young) >= 2 and np.nanstd(young) > EPS
            log_valid = raw_valid and np.all(group["value"].to_numpy(dtype=float) > 0) and np.nanstd(np.log(young)) > EPS
            refs.append(
                {
                    "analysis_id": analysis_id,
                    "cohort": cohort,
                    "young_mu_raw": float(np.nanmean(young)) if len(young) else np.nan,
                    "young_sd_raw": float(np.nanstd(young)) if len(young) else np.nan,
                    "young_mu_log": float(np.nanmean(np.log(young))) if log_valid else np.nan,
                    "young_sd_log": float(np.nanstd(np.log(young))) if log_valid else np.nan,
                    "raw_valid": bool(raw_valid),
                    "log_valid": bool(log_valid),
                    "young_n": int(len(young)),
                }
            )
        refs_df = pd.DataFrame(refs)
        reference_rows.extend(refs_df.to_dict(orient="records"))
        cohort_trimmed = cohort_trimmed.merge(refs_df, on=["analysis_id", "cohort"], how="left")

        normalized_parts = []
        raw_part = cohort_trimmed[cohort_trimmed["raw_valid"]].copy()
        raw_part["normalization"] = "young_z_raw"
        raw_part["normalized_value"] = (raw_part["value"] - raw_part["young_mu_raw"]) / raw_part["young_sd_raw"]
        normalized_parts.append(raw_part)

        log_part = cohort_trimmed[cohort_trimmed["log_valid"]].copy()
        log_part["normalization"] = "young_z_log"
        log_part["normalized_value"] = (np.log(log_part["value"]) - log_part["young_mu_log"]) / log_part["young_sd_log"]
        normalized_parts.append(log_part)

        if not normalized_parts:
            continue
        cohort_norm = pd.concat(normalized_parts, ignore_index=True)
        cohort_norm = cohort_norm.replace([np.inf, -np.inf], np.nan).dropna(subset=["normalized_value"]).copy()
        normalized_rows.extend(
            cohort_norm[
                [
                    "analysis_id",
                    "test_name",
                    "cohort",
                    "normalization",
                    "seqn",
                    "cycle_start_year",
                    "age_years",
                    "age_bin",
                    "age_mid",
                    "value",
                    "normalized_value",
                ]
            ].to_dict(orient="records")
        )

        grouped = (
            cohort_norm.groupby(["analysis_id", "test_name", "cohort", "normalization", "age_bin", "age_mid"], observed=True)["normalized_value"]
            .agg(
                n="count",
                q10=lambda s: float(np.nanpercentile(s.to_numpy(dtype=float), 10)),
                q25=lambda s: float(np.nanpercentile(s.to_numpy(dtype=float), 25)),
                q50=lambda s: float(np.nanpercentile(s.to_numpy(dtype=float), 50)),
                q75=lambda s: float(np.nanpercentile(s.to_numpy(dtype=float), 75)),
                q90=lambda s: float(np.nanpercentile(s.to_numpy(dtype=float), 90)),
            )
            .reset_index()
        )
        grouped["median_drift"] = grouped["q50"]
        grouped["iqr_change"] = grouped["q75"] - grouped["q25"]
        grouped["upper_tail_change"] = grouped["q90"] - grouped["q50"]
        grouped["lower_tail_change"] = grouped["q50"] - grouped["q10"]
        grouped["tail_asymmetry"] = tail_asymmetry_from_stats(grouped["q10"], grouped["q50"], grouped["q90"])
        grouped["passes_n_threshold"] = grouped["n"] >= MIN_BIN_N

        for normalization in DISTRIBUTION_NORMALIZATIONS:
            norm_grouped = grouped[grouped["normalization"] == normalization].copy()
            feature_curve_store: dict[str, dict[str, dict[str, object]]] = {feature: {} for feature in DISTRIBUTION_FEATURES}
            feature_base = manifest[manifest[f"included_{cohort}"]].copy()
            feature_base = feature_base[feature_base["included_from_catalog"]].copy()
            feature_df = feature_base[
                [
                    "analysis_id",
                    "test_name",
                    "category",
                    "subcategory",
                    "primary_organ_system",
                    "aging_domain",
                    "measurement_class",
                    "target_kind",
                ]
            ].drop_duplicates()

            for feature in DISTRIBUTION_FEATURES:
                for (analysis_id, test_name), sub in norm_grouped.groupby(["analysis_id", "test_name"], observed=True):
                    valid = sub[sub["passes_n_threshold"]].sort_values("age_mid")
                    if len(valid) < MIN_VALID_BINS:
                        continue
                    curve, reference = fit_smoothed_curve(
                        valid["age_mid"].to_numpy(dtype=float),
                        valid[feature].to_numpy(dtype=float),
                        valid["n"].to_numpy(dtype=float),
                        log_scale=False,
                    )
                    if np.isfinite(curve).sum() < len(AGE_GRID):
                        continue
                    rho = describe_curve_with_spearman(curve)
                    feature_df.loc[feature_df["analysis_id"] == analysis_id, f"rho_{feature}"] = rho
                    feature_curve_store[feature][str(analysis_id)] = {
                        "test_name": str(test_name),
                        "values": curve,
                    }
                    for age, value in zip(AGE_GRID, curve):
                        curve_rows.append(
                            {
                                "analysis_id": analysis_id,
                                "test_name": test_name,
                                "cohort": cohort,
                                "normalization": normalization,
                                "feature": feature,
                                "age": float(age),
                                "value": float(value) if np.isfinite(value) else np.nan,
                                "reference_value": float(reference) if np.isfinite(reference) else np.nan,
                            }
                        )

            cohort_feature_dir = feature_dir / cohort
            ensure_dir(cohort_feature_dir)
            feature_df.to_csv(cohort_feature_dir / f"{normalization}_distribution_features.csv", index=False)

            family_scores: dict[str, pd.DataFrame] = {}
            family_evs: dict[str, pd.DataFrame] = {}
            norm_fpca_dir = fpca_dir / cohort / normalization
            ensure_dir(norm_fpca_dir)
            for feature in DISTRIBUTION_FEATURES:
                score_df, eig_df, ev_df = run_fpca_from_curve_map(feature_curve_store.get(feature, {}))
                score_df.to_csv(norm_fpca_dir / f"{feature}_scores.csv", index=False)
                eig_df.to_csv(norm_fpca_dir / f"{feature}_eigenfunctions.csv", index=False)
                ev_df.to_csv(norm_fpca_dir / f"{feature}_explained_variance.csv", index=False)
                family_scores[feature] = score_df
                family_evs[feature] = ev_df

            curves_subset = pd.DataFrame(curve_rows)
            if not curves_subset.empty:
                curves_subset = curves_subset[
                    (curves_subset["cohort"] == cohort) & (curves_subset["normalization"] == normalization)
                ][["analysis_id", "test_name", "feature", "age", "value"]].copy()
            summary_row, cluster_sizes, silhouette_df = run_family_clustering_workflow(
                DISTRIBUTION_FEATURES,
                family_scores,
                family_evs,
                curves_subset,
                "feature",
                cluster_dir / cohort / normalization,
                "cluster_mean_feature_trajectories.png",
                "full_distribution",
                cohort,
                normalization,
            )
            cluster_summary_rows.append(summary_row)
            cluster_size_frames.append(cluster_sizes)
            cluster_silhouette_frames.append(silhouette_df)

            dist_map: dict[tuple[str, str], np.ndarray] = {}
            for (analysis_id, age_bin), sub in cohort_norm[cohort_norm["normalization"] == normalization].groupby(
                ["analysis_id", "age_bin"], observed=True
            ):
                if len(sub) >= MIN_BIN_N:
                    dist_map[(str(analysis_id), str(age_bin))] = np.sort(sub["normalized_value"].to_numpy(dtype=float))

            biomarker_ids = sorted(
                {
                    aid
                    for aid in norm_grouped.loc[norm_grouped["passes_n_threshold"], "analysis_id"].astype(str).unique().tolist()
                    if manifest.loc[manifest["analysis_id"] == aid, f"included_{cohort}"].any()
                }
            )
            matrix = pd.DataFrame(np.nan, index=biomarker_ids, columns=biomarker_ids, dtype=float)
            long_rows = []
            for aid in biomarker_ids:
                matrix.loc[aid, aid] = 0.0
            for i, aid_1 in enumerate(biomarker_ids):
                for aid_2 in biomarker_ids[i + 1 :]:
                    shared_bins = sorted(
                        set(
                            b
                            for (aid, b) in dist_map
                            if aid == aid_1
                        ).intersection(
                            b for (aid, b) in dist_map if aid == aid_2
                        )
                    )
                    if not shared_bins:
                        continue
                    distances = [
                        wasserstein_distance(dist_map[(aid_1, b)], dist_map[(aid_2, b)])
                        for b in shared_bins
                    ]
                    mean_dist = float(np.mean(distances))
                    matrix.loc[aid_1, aid_2] = mean_dist
                    matrix.loc[aid_2, aid_1] = mean_dist
                    long_rows.append(
                        {
                            "analysis_id_1": aid_1,
                            "analysis_id_2": aid_2,
                            "cohort": cohort,
                            "normalization": normalization,
                            "distance": mean_dist,
                            "n_shared_bins": len(shared_bins),
                        }
                    )
            cohort_w_dir = wasserstein_dir / cohort
            ensure_dir(cohort_w_dir)
            matrix.to_parquet(cohort_w_dir / f"{normalization}_pairwise_distance_matrix.parquet", index=True)
            pd.DataFrame(long_rows).to_csv(cohort_w_dir / f"{normalization}_pairwise_distance_long.csv", index=False)
            pairwise_matrices[(cohort, normalization)] = matrix

    reference_df = pd.DataFrame(reference_rows)
    reference_df.to_csv(out_dir / "reference_stats.csv", index=False)
    normalized_df = pd.DataFrame(normalized_rows)
    normalized_df.to_parquet(out_dir / "age_bin_distributions_long.parquet", index=False)
    feature_curves_df = pd.DataFrame(curve_rows)
    feature_curves_df.to_parquet(out_dir / "quantile_feature_curves_long.parquet", index=False)
    cluster_summary_df = pd.DataFrame(cluster_summary_rows)
    cluster_sizes_df = concat_nonempty_frames(cluster_size_frames)
    cluster_silhouette_df = concat_nonempty_frames(cluster_silhouette_frames)
    return reference_df, feature_curves_df, pairwise_matrices, cluster_summary_df, cluster_sizes_df, cluster_silhouette_df


def write_root_readme(out_dir: Path) -> None:
    text = f"""# Aging Biomarker Analysis

This folder contains the robust offline aging-biomarker trajectory analysis for the curated `aging_biomarkers.csv` catalog.

## Shared design
- Cohorts: {", ".join(COHORTS)}
- Age bins: {", ".join(AGE_BIN_LABELS)}
- Age grid for smoothed curves: {int(AGE_GRID.min())}-{int(AGE_GRID.max())}
- Mandatory trimming: keep only the within-bin {int(TRIM_LO*100)}th-{int(TRIM_HI*100)}th percentile band before all downstream calculations
- Minimum analyzable bin size: `n >= {MIN_BIN_N}`
- Minimum valid bins per cohort: `{MIN_VALID_BINS}`
- Summary metrics: {", ".join(SUMMARY_METRICS)}
- Summary curve normalizations: {", ".join(CURVE_NORMALIZATIONS)}
- Full-distribution normalizations: {", ".join(DISTRIBUTION_NORMALIZATIONS)}

## Contents
- `analysis_manifest.csv`: biomarker-level inclusion table and drop reasons
- `PIPELINE.md`: visual flow of the preprocessing and analysis steps
- `run_manifest.json`: run configuration and file inventory
- `explorer/`: standalone master HTML explorer that unifies summary clustering, full-distribution clustering, family FPCA, and summary-rho PCA
  - `explorer/rho_pca_explorer.html`: canonical source-switchable rho-PCA dashboard for NHANES vs Clalit
- `clalit/`: Clalit-only summary-stat Spearman-rho PCA outputs and standalone explorer
- `summary_stats/`: robust summary-stat trajectories, metric-wise FPCA outputs, rho-based PCA, branch-owned review panels, and clustering on concatenated retained FPCA scores
- `full_distribution/`: normalized distribution features, feature-wise FPCA outputs, concatenated-score clustering, and Wasserstein distance outputs plus heatmap review plots
- `qc/`: run-integrity checks only: manifests, trimming summaries, inclusion counts, normalization validity counts, and trim-distribution panels
"""
    (out_dir / "README.md").write_text(text)


def write_branch_readmes(out_dir: Path) -> None:
    (out_dir / "summary_stats" / "README.md").write_text(
        """# Summary-Stat Branch

Outputs for the robust summary-stat trajectory analysis.

## Files
- `binned_stats_long.parquet`: trimmed age-bin summary statistics
- `smoothed_curves_long.parquet`: smoothed summary-stat trajectories by cohort, metric, and normalization
- `feature_matrices/<cohort>/<normalization>_summary_features.csv`: normalization-specific biomarker matrices with descriptive Spearman rhos and `pc1..pc4` per metric
- `fpca/<cohort>/<normalization>/...`: FPCA score tables, eigenfunctions, and explained-variance tables for each metric
- `fpca/explained_variance_overview.csv` and `fpca/plots/...`: cross-metric FPCA overview tables and PNGs for quick review
- `clustering/<cohort>/<normalization>/...`: concatenated retained FPCA scores, standardized score tables, silhouette search, cluster assignments, PCA coordinates, dendrograms, and cluster-mean summary trajectories
- `clustering/summary.csv`, `clustering/cluster_sizes.csv`, `clustering/silhouette_by_k.csv`: branch-level clustering summaries
- `rho_pca/...`: PCA on the six summary-metric Spearman-rho features, including matrices, scree plots, loadings, and the interactive explorer
- `review/...`: review panels for raw-vs-smoothed metrics, normalization comparisons, and cohort overlays

## Linked explorer
- The canonical interactive workbench lives at `../explorer/aging_biomarker_explorer.html`.
- The older `rho_pca/rho_pca_explorer.html` remains for backward compatibility but is superseded by the master explorer.
"""
    )
    (out_dir / "full_distribution" / "README.md").write_text(
        """# Full-Distribution Branch

Outputs for the normalized empirical-distribution analysis.

## Files
- `reference_stats.csv`: young-reference means/SDs and normalization validity flags
- `age_bin_distributions_long.parquet`: trimmed normalized participant-level values
- `quantile_feature_curves_long.parquet`: smoothed quantile-derived feature trajectories
- `feature_matrices/<cohort>/<normalization>_distribution_features.csv`: descriptive Spearman rho features per biomarker
- `fpca/<cohort>/<normalization>/...`: FPCA score tables, eigenfunctions, and explained-variance tables for each quantile-derived feature family
- `clustering/<cohort>/<normalization>/...`: concatenated retained FPCA scores, standardized score tables, silhouette search, cluster assignments, PCA coordinates, dendrograms, and cluster-mean feature trajectories
- `clustering/summary.csv`, `clustering/cluster_sizes.csv`, `clustering/silhouette_by_k.csv`: branch-level clustering summaries
- `wasserstein/<cohort>/<normalization>_pairwise_distance_matrix.parquet`: age-integrated pairwise Wasserstein matrix
- `wasserstein/<cohort>/<normalization>_pairwise_distance_long.csv`: long-form pairwise distances
- `wasserstein/heatmaps/...` and `wasserstein/sanity.csv`: visualization and sanity checks for pairwise Wasserstein matrices

## Linked explorer
- The canonical interactive workbench lives at `../explorer/aging_biomarker_explorer.html`.
"""
    )


def write_pipeline_md(out_dir: Path) -> None:
    text = """# Pipeline Overview

```mermaid
flowchart TD
    A["aging_biomarkers.csv"] --> B["Match direct biomarkers by NHANES variable_name tokens"]
    A --> C["Derive ratio / eGFR targets where component variables exist"]
    B --> D["Healthy blood long table\\n(data/processed/biomarker_long.parquet)"]
    C --> D
    D --> E["Split into pooled / female / male cohorts"]
    E --> F["Assign age bins 20-24 .. 80-84"]
    F --> G["Within each biomarker x cohort x age bin: keep only 10th-90th percentile values"]
    G --> H["Trim summary + inclusion manifest"]
    G --> I["Summary-stat branch"]
    G --> J["Full-distribution branch"]
    I --> K["Compute median, std, iqr, cv, skewness, quantile skewness"]
    K --> L["Weighted GAM smoothing on age bins"]
    L --> M["Curve normalizations: raw, young_ratio, young_log_fold, shape_z"]
    M --> N["Descriptive Spearman rho"]
    M --> AB["Summary-rho PCA explorer inputs"]
    M --> O["FPCA per metric family"]
    O --> O1["Retain PC1 and PC2 always; keep PC3 if EV >= 10%"]
    O1 --> O2["Concatenate retained scores across metric families"]
    O2 --> O3["Standardize concatenated score columns"]
    O3 --> O4["Ward hierarchical clustering; choose K by silhouette over 2..8"]
    O3 --> O5["Ordinary PCA map of concatenated standardized scores"]
    O4 --> O6["Cluster-mean summary trajectories"]
    O5 --> UX["Unified standalone explorer"]
    O6 --> UX
    J --> P["Young-reference normalization: raw-z and log-z"]
    P --> Q["Age-bin quantiles from normalized distributions"]
    Q --> R["Quantile-derived feature curves"]
    R --> S["Weighted GAM smoothing and descriptive Spearman rho"]
    S --> S1["FPCA per feature family"]
    S1 --> S2["Retain PC1 and PC2 always; keep PC3 if EV >= 10%"]
    S2 --> S3["Concatenate retained scores across feature families"]
    S3 --> S4["Standardize concatenated score columns"]
    S4 --> S5["Ward hierarchical clustering; choose K by silhouette over 2..8"]
    S4 --> S6["Ordinary PCA map of concatenated standardized scores"]
    S5 --> S7["Cluster-mean feature trajectories"]
    S6 --> UX
    S7 --> UX
    P --> T["Age-integrated pairwise Wasserstein distances"]
    H --> U["QC tables and trim checks"]
    O5 --> V["Summary-stat review outputs"]
    O6 --> V
    N --> V
    S6 --> W["Full-distribution review outputs"]
    S7 --> W
    T --> W
    AB --> UX
    U --> UX
```

## Notes
- Summary and full-distribution branches share the same trimmed participant rows.
- Smoothing is done with weighted GAMs on age-bin midpoints.
- Positive-only summary metrics are smoothed on the log scale before back-transformation.
- The full-distribution branch clusters biomarkers on retained FPCA scores from the five quantile-derived feature families.
- The summary-stat branch clusters biomarkers on retained FPCA scores from the six summary-metric families.
- The unified explorer is an offline standalone HTML with embedded Plotly JS and an embedded analysis bundle.
"""
    (out_dir / "PIPELINE.md").write_text(text)


def plot_before_after_trim(
    raw_rows: pd.DataFrame,
    trimmed_rows: pd.DataFrame,
    review_targets: list[str],
    out_dir: Path,
) -> None:
    ensure_dir(out_dir)
    pooled_raw = cohort_subset(raw_rows, "pooled")
    pooled_trim = trimmed_rows[trimmed_rows["cohort"] == "pooled"].copy()
    for name in review_targets:
        raw = pooled_raw[pooled_raw["test_name"] == name]["value"].to_numpy(dtype=float)
        trim = pooled_trim[pooled_trim["test_name"] == name]["value"].to_numpy(dtype=float)
        if len(raw) < 10 or len(trim) < 10:
            continue
        fig, axes = plt.subplots(1, 2, figsize=(11, 4))
        axes[0].hist(raw, bins=40, color="#8aa0b6", alpha=0.75)
        axes[0].set_title(f"{name}: raw pooled values")
        axes[1].hist(trim, bins=40, color="#2a9d8f", alpha=0.75)
        axes[1].set_title(f"{name}: trimmed pooled values")
        for ax in axes:
            ax.ticklabel_format(style="plain", axis="x")
        fig.tight_layout()
        fig.savefig(out_dir / f"{slugify(name)}_trim_before_after.png", dpi=160)
        plt.close(fig)


def plot_summary_metric_panels(
    summary_stats: pd.DataFrame,
    smoothed_curves: pd.DataFrame,
    review_targets: list[str],
    out_dir: Path,
) -> None:
    ensure_dir(out_dir)
    pooled_stats = summary_stats[(summary_stats["cohort"] == "pooled") & (summary_stats["passes_n_threshold"])].copy()
    pooled_smooth = smoothed_curves[
        (smoothed_curves["cohort"] == "pooled") & (smoothed_curves["normalization"] == "raw")
    ].copy()
    for name in review_targets:
        fig, axes = plt.subplots(2, 3, figsize=(14, 8), sharex=True)
        axes = axes.ravel()
        any_plotted = False
        for ax, metric in zip(axes, SUMMARY_METRICS):
            stats_sub = pooled_stats[(pooled_stats["test_name"] == name)]
            smooth_sub = pooled_smooth[(pooled_smooth["test_name"] == name) & (pooled_smooth["metric"] == metric)]
            if stats_sub.empty or smooth_sub.empty:
                ax.set_axis_off()
                continue
            ax.scatter(stats_sub["age_mid"], stats_sub[metric], s=18, color="#6c757d", alpha=0.75, label="binned")
            ax.plot(smooth_sub["age"], smooth_sub["value"], color="#d1495b", lw=2, label="GAM")
            ax.set_title(SUMMARY_METRIC_Y_LABELS[metric])
            any_plotted = True
        if not any_plotted:
            plt.close(fig)
            continue
        axes[0].legend(loc="best", fontsize=8)
        fig.suptitle(f"{name}: raw binned vs smoothed trajectories (pooled)")
        fig.tight_layout()
        fig.savefig(out_dir / f"{slugify(name)}_summary_metrics_pooled.png", dpi=160)
        plt.close(fig)


def plot_normalization_panels(smoothed_curves: pd.DataFrame, review_targets: list[str], out_dir: Path) -> None:
    ensure_dir(out_dir)
    pooled = smoothed_curves[smoothed_curves["cohort"] == "pooled"].copy()
    for name in review_targets:
        fig, axes = plt.subplots(2, 2, figsize=(12, 8), sharex=True)
        any_plotted = False
        pairs = [("median", "young_ratio"), ("median", "shape_z"), ("cv", "young_ratio"), ("cv", "young_log_fold")]
        for ax, (metric, normalization) in zip(axes.ravel(), pairs):
            sub = pooled[
                (pooled["test_name"] == name)
                & (pooled["metric"] == metric)
                & (pooled["normalization"] == normalization)
            ]
            if sub.empty or sub["value"].notna().sum() == 0:
                ax.set_axis_off()
                continue
            ax.plot(sub["age"], sub["value"], color="#264653", lw=2)
            ax.set_title(f"{metric} / {normalization}")
            any_plotted = True
        if not any_plotted:
            plt.close(fig)
            continue
        fig.suptitle(f"{name}: normalization comparison panels (pooled)")
        fig.tight_layout()
        fig.savefig(out_dir / f"{slugify(name)}_normalization_panels.png", dpi=160)
        plt.close(fig)


def plot_fpca_outputs(
    fpca_ev: dict[tuple[str, str, str], pd.DataFrame],
    fpca_scores: dict[tuple[str, str, str], pd.DataFrame],
    fpca_dir: Path,
) -> pd.DataFrame:
    plots_dir = fpca_dir / "plots"
    ensure_dir(plots_dir)
    summary_rows = []
    for (cohort, normalization, metric), ev_df in fpca_ev.items():
        score_df = fpca_scores.get((cohort, normalization, metric))
        eig_path = fpca_dir / cohort / normalization / f"{metric}_eigenfunctions.csv"
        if not eig_path.exists():
            continue
        eig_df = pd.read_csv(eig_path)
        summary_rows.append(
            {
                "cohort": cohort,
                "normalization": normalization,
                "metric": metric,
                "pc1_explained_variance_ratio": float(ev_df["explained_variance_ratio"].iloc[0]) if not ev_df.empty else np.nan,
                "pc2_explained_variance_ratio": float(ev_df["explained_variance_ratio"].iloc[1]) if len(ev_df) > 1 else np.nan,
            }
        )
        fig, axes = plt.subplots(1, 2, figsize=(11, 4))
        axes[0].bar(ev_df["pc"], ev_df["explained_variance_ratio"].fillna(0.0))
        axes[0].set_title("Scree")
        for k in range(1, 5):
            if f"pc{k}" in eig_df.columns:
                axes[1].plot(eig_df["age"], eig_df[f"pc{k}"], lw=1.5, label=f"PC{k}")
        axes[1].set_title("Eigenfunctions")
        axes[1].legend(loc="best", fontsize=8)
        fig.suptitle(f"{cohort} / {normalization} / {metric}")
        fig.tight_layout()
        fig.savefig(plots_dir / f"{cohort}_{normalization}_{metric}_fpca.png", dpi=160)
        plt.close(fig)
        if score_df is not None and {"pc1", "pc2"}.issubset(score_df.columns):
            sc = score_df.dropna(subset=["pc1", "pc2"])
            if len(sc) >= 2:
                fig, ax = plt.subplots(figsize=(6, 5))
                ax.scatter(sc["pc1"], sc["pc2"], s=18, alpha=0.8)
                ax.set_xlabel("PC1")
                ax.set_ylabel("PC2")
                ax.set_title(f"{cohort} / {normalization} / {metric}")
                fig.tight_layout()
                fig.savefig(plots_dir / f"{cohort}_{normalization}_{metric}_scores_pc1_pc2.png", dpi=160)
                plt.close(fig)
    summary_df = pd.DataFrame(summary_rows)
    summary_df.to_csv(fpca_dir / "explained_variance_overview.csv", index=False)
    return summary_df


def _save_placeholder_png(out_path: Path, title: str, message: str, *, figsize: tuple[float, float]) -> None:
    fig, ax = plt.subplots(figsize=figsize)
    ax.axis("off")
    ax.text(0.5, 0.60, title, ha="center", va="center", fontsize=13, weight="bold")
    ax.text(0.5, 0.40, message, ha="center", va="center", fontsize=10)
    fig.tight_layout()
    fig.savefig(out_path, dpi=160)
    plt.close(fig)


def _rho_feature_label(name: str) -> str:
    metric = str(name).replace("rho_", "")
    return {
        "median": "Median rho",
        "std": "Std rho",
        "iqr": "IQR rho",
        "cv": "CV rho",
        "skewness": "Skewness rho",
        "quantile_skewness": "Quantile skewness rho",
    }.get(metric, name)


def _rho_method_feature_rows(
    method_key: str,
    method_specs: dict[str, dict[str, object]],
) -> list[dict[str, object]]:
    spec = method_specs[method_key]
    rows: list[dict[str, object]] = []
    for order, (source_col, feature_id, feature_label, transform) in enumerate(spec["features"], start=1):
        rows.append(
            {
                "source_col": source_col,
                "feature_id": feature_id,
                "feature_label": feature_label,
                "transform": transform,
                "feature_order": order,
            }
        )
    return rows


def _summary_rho_method_feature_rows(method_key: str) -> list[dict[str, object]]:
    return _rho_method_feature_rows(method_key, SUMMARY_RHO_PCA_METHODS)


def _build_rho_method_matrix(
    feature_df: pd.DataFrame,
    method_key: str,
    *,
    method_specs: dict[str, dict[str, object]],
    rho_columns: Iterable[str],
) -> tuple[pd.DataFrame, list[dict[str, object]]]:
    feature_rows = _rho_method_feature_rows(method_key, method_specs)
    base_cols = [*SUMMARY_FEATURE_METADATA_COLUMNS, *list(rho_columns)]
    matrix_df = feature_df.reindex(columns=base_cols).copy()
    for row in feature_rows:
        values = pd.to_numeric(matrix_df[row["source_col"]], errors="coerce")
        if row["transform"] == "abs":
            values = values.abs()
        matrix_df[row["feature_id"]] = values
    return matrix_df, feature_rows


def _build_summary_rho_method_matrix(feature_df: pd.DataFrame, method_key: str) -> tuple[pd.DataFrame, list[dict[str, object]]]:
    return _build_rho_method_matrix(
        feature_df,
        method_key,
        method_specs=SUMMARY_RHO_PCA_METHODS,
        rho_columns=SUMMARY_RHO_COLUMNS,
    )


def plot_summary_rho_pca_scree(
    explained_df: pd.DataFrame,
    out_path: Path,
    *,
    method_label: str,
    cohort: str,
    normalization: str,
) -> None:
    if explained_df.empty:
        _save_placeholder_png(
            out_path,
            f"{method_label} | {cohort} / {normalization}",
            "No complete-case biomarkers available for rho PCA.",
            figsize=(7.0, 4.2),
        )
        return

    fig, ax1 = plt.subplots(figsize=(7.0, 4.2))
    x = explained_df["component_index"].to_numpy(dtype=int)
    ratios = explained_df["explained_variance_ratio"].fillna(0.0).to_numpy(dtype=float)
    cumulative = explained_df["cumulative_explained"].fillna(0.0).to_numpy(dtype=float)
    ax1.bar(x, ratios, color="#2563eb", alpha=0.86)
    ax1.set_xlabel("Principal component")
    ax1.set_ylabel("Explained variance ratio")
    ax1.set_xticks(x, explained_df["component"].tolist())
    ax1.set_ylim(0.0, max(0.35, float(ratios.max()) * 1.15 if len(ratios) else 0.35))
    ax1.grid(axis="y", alpha=0.22)

    ax2 = ax1.twinx()
    ax2.plot(x, cumulative, color="#dc2626", marker="o", linewidth=2.0)
    ax2.set_ylabel("Cumulative explained variance")
    ax2.set_ylim(0.0, 1.02)

    fig.suptitle(f"Summary-rho PCA scree: {method_label} | {cohort} / {normalization}")
    fig.tight_layout()
    fig.savefig(out_path, dpi=160)
    plt.close(fig)


def plot_summary_rho_pca_loadings(
    loadings_wide_df: pd.DataFrame,
    out_path: Path,
    *,
    method_label: str,
    cohort: str,
    normalization: str,
) -> None:
    pc_cols = [c for c in loadings_wide_df.columns if re.fullmatch(r"pc\d+", str(c))]
    shown = pc_cols[:3]
    if loadings_wide_df.empty or not shown:
        _save_placeholder_png(
            out_path,
            f"{method_label} | {cohort} / {normalization}",
            "No loadings available for rho PCA.",
            figsize=(10.0, 4.5),
        )
        return

    fig, axes = plt.subplots(1, len(shown), figsize=(4.4 * len(shown), 4.8), sharey=True)
    axes_arr = np.atleast_1d(axes)
    y = np.arange(len(loadings_wide_df))
    label_col = "feature_label" if "feature_label" in loadings_wide_df.columns else "feature"
    labels = [str(v) for v in loadings_wide_df[label_col].tolist()]
    for ax, col in zip(axes_arr, shown):
        vals = loadings_wide_df[col].fillna(0.0).to_numpy(dtype=float)
        colors = ["#0f766e" if v >= 0 else "#b45309" for v in vals]
        ax.barh(y, vals, color=colors, alpha=0.88)
        ax.axvline(0.0, color="#111827", linewidth=1.0)
        ax.set_title(col.upper())
        ax.set_xlabel("Loading")
        ax.set_xlim(-1.05, 1.05)
        ax.grid(axis="x", alpha=0.18)
        ax.set_yticks(y, labels)
    fig.suptitle(f"Summary-rho PCA loadings: {method_label} | {cohort} / {normalization}")
    fig.tight_layout()
    fig.savefig(out_path, dpi=160)
    plt.close(fig)


def _json_records(df: pd.DataFrame) -> str:
    return json.dumps(df.where(pd.notna(df), None).to_dict(orient="records"))


def render_summary_rho_pca_html(
    scores_df: pd.DataFrame,
    explained_df: pd.DataFrame,
    loadings_df: pd.DataFrame,
    diagnostics_df: pd.DataFrame,
    out_path: Path,
) -> None:
    score_json = _json_records(scores_df)
    explained_json = _json_records(explained_df)
    loadings_json = _json_records(loadings_df)
    diagnostics_json = _json_records(diagnostics_df)
    color_cols_json = json.dumps(["none", *SUMMARY_PCA_COLOR_COLUMNS])
    methods_json = json.dumps(
        [
            {"value": key, "label": spec["label"], "description": spec["description"]}
            for key, spec in SUMMARY_RHO_PCA_METHODS.items()
        ]
    )
    html = f"""<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="utf-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1" />
  <title>Summary-Rho PCA Explorer</title>
  <script src="https://cdn.plot.ly/plotly-2.35.2.min.js"></script>
  <style>
    :root {{
      --bg: #f5f1e8;
      --card: #fffdf8;
      --ink: #152036;
      --muted: #5b6475;
      --line: #d8cfbf;
      --accent: #0f766e;
      --accent-2: #b45309;
    }}
    * {{ box-sizing: border-box; }}
    body {{
      margin: 0;
      font-family: "Avenir Next", "Gill Sans", sans-serif;
      background:
        radial-gradient(circle at top left, rgba(15,118,110,0.10), transparent 34%),
        radial-gradient(circle at top right, rgba(180,83,9,0.10), transparent 30%),
        var(--bg);
      color: var(--ink);
    }}
    .wrap {{
      max-width: 1500px;
      margin: 0 auto;
      padding: 24px 20px 32px;
      transition: filter 180ms ease, opacity 180ms ease;
    }}
    .wrap.faded {{
      filter: saturate(0.78) brightness(0.96);
      opacity: 0.72;
    }}
    .hero {{
      margin-bottom: 16px;
      padding: 18px 20px;
      background: linear-gradient(135deg, rgba(255,255,255,0.94), rgba(250,246,237,0.98));
      border: 1px solid var(--line);
      border-radius: 18px;
      box-shadow: 0 18px 44px rgba(21,32,54,0.08);
    }}
    .hero h1 {{
      margin: 0 0 8px;
      font-size: clamp(1.5rem, 2vw, 2.1rem);
      letter-spacing: 0.01em;
    }}
    .hero p {{
      margin: 0;
      color: var(--muted);
      max-width: 980px;
      line-height: 1.45;
    }}
    .controls {{
      display: grid;
      grid-template-columns: repeat(7, minmax(140px, 1fr));
      gap: 12px;
      margin-bottom: 12px;
    }}
    .control {{
      background: var(--card);
      border: 1px solid var(--line);
      border-radius: 14px;
      padding: 10px 12px;
      box-shadow: 0 10px 24px rgba(21,32,54,0.05);
    }}
    .control label {{
      display: flex;
      align-items: center;
      gap: 6px;
      font-size: 0.8rem;
      text-transform: uppercase;
      letter-spacing: 0.08em;
      color: var(--muted);
      margin-bottom: 6px;
    }}
    .control select,
    .control input[type="range"] {{
      width: 100%;
      padding: 8px 10px;
      border-radius: 10px;
      border: 1px solid #cbbfa8;
      background: #fffdfa;
      color: var(--ink);
      font: inherit;
    }}
    .control input[type="range"] {{
      padding: 0;
    }}
    .icon-button {{
      display: inline-flex;
      align-items: center;
      justify-content: center;
      width: 20px;
      height: 20px;
      border-radius: 999px;
      border: 1px solid rgba(15,118,110,0.35);
      background: rgba(15,118,110,0.10);
      color: var(--accent);
      font-size: 0.78rem;
      font-weight: 700;
      cursor: pointer;
      padding: 0;
    }}
    .icon-button:hover {{
      background: rgba(15,118,110,0.18);
    }}
    .control-note {{
      font-size: 0.84rem;
      color: var(--muted);
      line-height: 1.35;
      min-height: 1.2em;
    }}
    .overlay {{
      position: fixed;
      inset: 0;
      display: none;
      align-items: center;
      justify-content: center;
      padding: 24px;
      background: rgba(21,32,54,0.18);
      backdrop-filter: blur(2px) saturate(0.9);
      z-index: 30;
    }}
    .overlay.open {{
      display: flex;
    }}
    .info-panel {{
      width: min(760px, calc(100vw - 32px));
      max-height: min(78vh, 760px);
      overflow: auto;
      padding: 18px 20px;
      background: linear-gradient(180deg, rgba(255,254,250,0.98), rgba(248,243,233,0.98));
      border: 1px solid rgba(216,207,191,0.85);
      border-radius: 18px;
      box-shadow: 0 28px 70px rgba(21,32,54,0.18);
    }}
    .info-panel h2 {{
      margin: 0 0 10px;
      font-size: 1rem;
    }}
    .info-panel p {{
      margin: 0 0 10px;
      color: var(--muted);
      line-height: 1.45;
    }}
    .info-grid {{
      display: grid;
      grid-template-columns: repeat(2, minmax(0, 1fr));
      gap: 10px;
    }}
    .info-item {{
      padding: 10px 12px;
      border: 1px solid var(--line);
      border-radius: 12px;
      background: rgba(255,255,255,0.72);
    }}
    .info-item strong {{
      display: block;
      margin-bottom: 4px;
    }}
    .summary {{
      margin-bottom: 14px;
      padding: 12px 14px;
      background: rgba(255,255,255,0.78);
      border: 1px solid var(--line);
      border-radius: 14px;
      color: var(--muted);
      line-height: 1.45;
    }}
    .grid {{
      display: grid;
      grid-template-columns: minmax(0, 1.7fr) minmax(340px, 0.9fr);
      gap: 14px;
      align-items: start;
    }}
    .stack {{
      display: grid;
      gap: 14px;
    }}
    .card {{
      background: var(--card);
      border: 1px solid var(--line);
      border-radius: 18px;
      padding: 10px;
      box-shadow: 0 14px 34px rgba(21,32,54,0.06);
    }}
    .plot {{
      width: 100%;
      min-height: 520px;
    }}
    .plot.small {{
      min-height: 300px;
    }}
    @media (max-width: 1100px) {{
      .controls {{ grid-template-columns: repeat(2, minmax(180px, 1fr)); }}
      .grid {{ grid-template-columns: 1fr; }}
      .info-grid {{ grid-template-columns: 1fr; }}
    }}
    @media (max-width: 680px) {{
      .wrap {{ padding: 16px 12px 24px; }}
      .controls {{ grid-template-columns: 1fr; }}
      .plot {{ min-height: 420px; }}
      .plot.small {{ min-height: 280px; }}
    }}
  </style>
</head>
<body>
  <div class="wrap">
    <section class="hero">
      <h1>Summary-Stat Spearman-Rho PCA</h1>
      <p>
        PCA on the complete-case matrix of biomarker-level Spearman rhos across median, spread, and skewness summaries.
        Use the controls to switch PCA method, cohort, normalization, marker coloring, and whether the scatter stays in PC1/PC2 or expands into PC1/PC2/PC3.
      </p>
    </section>
    <section class="controls">
      <div class="control">
        <label for="method">PCA method</label>
        <select id="method"></select>
      </div>
      <div class="control">
        <label for="cohort">Cohort</label>
        <select id="cohort"></select>
      </div>
      <div class="control">
        <label for="normalization">Normalization <button type="button" id="normInfoToggle" class="icon-button" aria-label="Explain normalization choices">i</button></label>
        <select id="normalization"></select>
      </div>
      <div class="control">
        <label for="colorBy">Color markers by</label>
        <select id="colorBy"></select>
      </div>
      <div class="control">
        <label for="zAxis">Z-axis</label>
        <select id="zAxis"></select>
      </div>
      <div class="control">
        <label for="callouts">Text callouts</label>
        <select id="callouts"></select>
        <div class="control-note" id="calloutModeNote">2D uses non-overlap placement with a fallback so every label still renders. 3D falls back to direct labels.</div>
      </div>
      <div class="control">
        <label for="calloutSize">Callout text size</label>
        <input id="calloutSize" type="range" min="8" max="22" step="1" value="11" />
        <div class="control-note" id="calloutSizeValue">11 px</div>
      </div>
    </section>
    <div class="overlay" id="normInfoOverlay" aria-hidden="true">
      <section class="info-panel" id="normInfoPanel" role="dialog" aria-modal="true" aria-label="Normalization explainer">
        <h2>What the normalization choices mean</h2>
        <p>
          The dropdown changes how each smoothed age trajectory is normalized before its Spearman rho is computed against age.
          The rho itself is not normalized afterward.
        </p>
        <div class="info-grid">
          <div class="info-item">
            <strong>raw</strong>
            Uses the smoothed metric as-is across age. Keeps absolute direction and scale.
          </div>
          <div class="info-item">
            <strong>young_ratio</strong>
            Divides the trajectory by its young-reference value, so age 20-24 acts as baseline.
          </div>
          <div class="info-item">
            <strong>young_log_fold</strong>
            Takes log of the young-reference ratio. This is only valid when the full smoothed trajectory stays positive.
          </div>
          <div class="info-item">
            <strong>shape_z</strong>
            Z-scores the trajectory across age points. This removes level and keeps only trajectory shape.
          </div>
        </div>
        <p>
          The PCA method selector changes which transformed rho features enter the PCA. The normalization selector still refers to the trajectory normalization used before those rhos were computed.
        </p>
      </section>
    </div>
    <div class="summary" id="summary"></div>
    <section class="grid">
      <div class="card">
        <div id="scatter" class="plot"></div>
      </div>
      <div class="stack">
        <div class="card">
          <div id="scree" class="plot small"></div>
        </div>
        <div class="card">
          <div id="loadings" class="plot small"></div>
        </div>
      </div>
    </section>
  </div>
  <script>
    const scores = {score_json};
    const explained = {explained_json};
    const loadings = {loadings_json};
    const diagnostics = {diagnostics_json};
    const colorColumns = {color_cols_json};
    const methodOptions = {methods_json};
    const palette = [
      '#2563eb', '#dc2626', '#0f766e', '#7c3aed', '#b45309', '#0891b2',
      '#be123c', '#4d7c0f', '#4338ca', '#ea580c', '#0d9488', '#9333ea'
    ];

    const wrapEl = document.querySelector('.wrap');
    const methodEl = document.getElementById('method');
    const cohortEl = document.getElementById('cohort');
    const normalizationEl = document.getElementById('normalization');
    const colorByEl = document.getElementById('colorBy');
    const zAxisEl = document.getElementById('zAxis');
    const calloutsEl = document.getElementById('callouts');
    const calloutSizeEl = document.getElementById('calloutSize');
    const calloutSizeValueEl = document.getElementById('calloutSizeValue');
    const calloutModeNoteEl = document.getElementById('calloutModeNote');
    const normInfoToggleEl = document.getElementById('normInfoToggle');
    const normInfoOverlayEl = document.getElementById('normInfoOverlay');
    const normInfoPanelEl = document.getElementById('normInfoPanel');
    const summaryEl = document.getElementById('summary');
    const scatterEl = document.getElementById('scatter');
    const screeEl = document.getElementById('scree');
    const loadingsEl = document.getElementById('loadings');

    function uniq(arr) {{
      return [...new Set(arr)];
    }}

    function fillSelect(el, values, preferred) {{
      el.innerHTML = '';
      values.forEach((value) => {{
        const option = document.createElement('option');
        option.value = value;
        option.textContent = value;
        el.appendChild(option);
      }});
      if (preferred && values.includes(preferred)) {{
        el.value = preferred;
      }}
    }}

    function fillOptionRecords(el, records, preferred) {{
      el.innerHTML = '';
      records.forEach((record) => {{
        const option = document.createElement('option');
        option.value = record.value;
        option.textContent = record.label;
        option.title = record.description || record.label;
        el.appendChild(option);
      }});
      if (preferred && records.some((record) => record.value === preferred)) {{
        el.value = preferred;
      }}
    }}

    function currentMethodRecord() {{
      return methodOptions.find((record) => record.value === methodEl.value) || null;
    }}

    function currentRows() {{
      return scores.filter(
        (row) => row.method === methodEl.value && row.cohort === cohortEl.value && row.normalization === normalizationEl.value
      );
    }}

    function currentExplained() {{
      return explained.filter(
        (row) => row.method === methodEl.value && row.cohort === cohortEl.value && row.normalization === normalizationEl.value
      );
    }}

    function currentLoadings() {{
      return loadings.filter(
        (row) => row.method === methodEl.value && row.cohort === cohortEl.value && row.normalization === normalizationEl.value
      );
    }}

    function currentDiagnostic() {{
      return diagnostics.find(
        (row) => row.method === methodEl.value && row.cohort === cohortEl.value && row.normalization === normalizationEl.value
      ) || null;
    }}

    function colorValue(row) {{
      const field = colorByEl.value;
      if (field === 'none') return 'All biomarkers';
      return row[field] || 'Missing';
    }}

    function hoverText(row, show3d) {{
      const pcs = [
        `PC1: ${{Number(row.pc1).toFixed(4)}}`,
        `PC2: ${{Number(row.pc2).toFixed(4)}}`,
      ];
      if (show3d && row.pc3 !== null && row.pc3 !== undefined) {{
        pcs.push(`PC3: ${{Number(row.pc3).toFixed(4)}}`);
      }}
      const rhoLines = [
        `rho_median: ${{Number(row.rho_median).toFixed(4)}}`,
        `rho_std: ${{Number(row.rho_std).toFixed(4)}}`,
        `rho_iqr: ${{Number(row.rho_iqr).toFixed(4)}}`,
        `rho_cv: ${{Number(row.rho_cv).toFixed(4)}}`,
        `rho_skewness: ${{Number(row.rho_skewness).toFixed(4)}}`,
        `rho_quantile_skewness: ${{Number(row.rho_quantile_skewness).toFixed(4)}}`,
      ];
      return [
        `<b>${{row.test_name}}</b>`,
        `id=${{row.analysis_id}}`,
        `category=${{row.category || 'NA'}}`,
        `subcategory=${{row.subcategory || 'NA'}}`,
        `system=${{row.primary_organ_system || 'NA'}}`,
        `domain=${{row.aging_domain || 'NA'}}`,
        `class=${{row.measurement_class || 'NA'}}`,
        `target=${{row.target_kind || 'NA'}}`,
        ...pcs,
        ...rhoLines,
      ].join('<br>');
    }}

    function labelText(row) {{
      return row.test_name || row.analysis_id;
    }}

    function extent(values) {{
      const finite = values.map(Number).filter((v) => Number.isFinite(v));
      if (!finite.length) return [-1, 1];
      const lo = Math.min(...finite);
      const hi = Math.max(...finite);
      if (Math.abs(hi - lo) < 1e-9) return [lo - 1, hi + 1];
      const pad = (hi - lo) * 0.08;
      return [lo - pad, hi + pad];
    }}

    function boxesOverlap(a, b) {{
      return !(a.x1 < b.x0 || a.x0 > b.x1 || a.y1 < b.y0 || a.y0 > b.y1);
    }}

    function buildRepelledAnnotations(rows, xRange, yRange, fontSize) {{
      const width = Math.max(scatterEl.clientWidth || 900, 420);
      const height = Math.max(scatterEl.clientHeight || 520, 360);
      const offsets = [
        {{ dx: 12, dy: -12, xanchor: 'left', yanchor: 'bottom' }},
        {{ dx: 12, dy: 12, xanchor: 'left', yanchor: 'top' }},
        {{ dx: -12, dy: -12, xanchor: 'right', yanchor: 'bottom' }},
        {{ dx: -12, dy: 12, xanchor: 'right', yanchor: 'top' }},
        {{ dx: 0, dy: -18, xanchor: 'center', yanchor: 'bottom' }},
        {{ dx: 0, dy: 18, xanchor: 'center', yanchor: 'top' }},
        {{ dx: 18, dy: 0, xanchor: 'left', yanchor: 'middle' }},
        {{ dx: -18, dy: 0, xanchor: 'right', yanchor: 'middle' }},
      ];
      const placed = [];
      const annotations = [];
      const ordered = [...rows].sort((a, b) => {{
        const ar = (Number(a.pc1) ** 2) + (Number(a.pc2) ** 2);
        const br = (Number(b.pc1) ** 2) + (Number(b.pc2) ** 2);
        return br - ar;
      }});
      ordered.forEach((row) => {{
        const label = labelText(row);
        const xNorm = (Number(row.pc1) - xRange[0]) / (xRange[1] - xRange[0]);
        const yNorm = (Number(row.pc2) - yRange[0]) / (yRange[1] - yRange[0]);
        const textWidth = ((label.length * fontSize * 0.62) + (fontSize * 1.4)) / width;
        const textHeight = (fontSize * 1.7) / height;
        for (const offset of offsets) {{
          let centerX = xNorm + (offset.dx / width);
          let centerY = yNorm + (-offset.dy / height);
          const box = {{
            x0: centerX - (textWidth / 2),
            x1: centerX + (textWidth / 2),
            y0: centerY - (textHeight / 2),
            y1: centerY + (textHeight / 2),
          }};
          if (box.x0 < 0.02 || box.x1 > 0.98 || box.y0 < 0.03 || box.y1 > 0.97) continue;
          if (placed.some((existing) => boxesOverlap(existing, box))) continue;
          placed.push(box);
          annotations.push({{
            x: Number(row.pc1),
            y: Number(row.pc2),
            xref: 'x',
            yref: 'y',
            text: label,
            showarrow: true,
            arrowhead: 0,
            arrowsize: 0.8,
            arrowwidth: 0.8,
            arrowcolor: 'rgba(21,32,54,0.28)',
            ax: offset.dx,
            ay: offset.dy,
            font: {{ size: fontSize, color: '#152036' }},
            bgcolor: 'rgba(255,253,248,0.82)',
            bordercolor: 'rgba(21,32,54,0.12)',
            borderwidth: 1,
            borderpad: 2,
            xanchor: offset.xanchor,
            yanchor: offset.yanchor,
          }});
          break;
        }}
      }});
      return annotations;
    }}

    function renderSummary(rows, diag, evRows) {{
      const complete = rows.length;
      const total = diag ? diag.candidate_biomarkers : complete;
      const dropped = diag ? diag.dropped_for_pca : 0;
      const pc1 = evRows.find((r) => r.component === 'PC1');
      const pc2 = evRows.find((r) => r.component === 'PC2');
      const pc3 = evRows.find((r) => r.component === 'PC3');
      const parts = [
        `<b>${{currentMethodRecord() ? currentMethodRecord().label : methodEl.value}}</b>`,
        `<b>${{cohortEl.value}}</b> / <b>${{normalizationEl.value}}</b>`,
        `Complete biomarkers in PCA: <b>${{complete}}</b> / ${{total}}`,
        `Dropped before PCA: <b>${{dropped}}</b>`,
      ];
      const methodRecord = currentMethodRecord();
      if (methodRecord && methodRecord.description) parts.push(methodRecord.description);
      if (pc1) parts.push(`PC1=${{(100 * Number(pc1.explained_variance_ratio)).toFixed(1)}}%`);
      if (pc2) parts.push(`PC2=${{(100 * Number(pc2.explained_variance_ratio)).toFixed(1)}}%`);
      if (pc3) parts.push(`PC3=${{(100 * Number(pc3.explained_variance_ratio)).toFixed(1)}}%`);
      if (diag && diag.note) parts.push(diag.note);
      summaryEl.innerHTML = parts.join(' | ');
    }}

    function renderScatter(rows, evRows) {{
      const show3d = zAxisEl.value === 'pc3';
      const showCallouts = calloutsEl.value === 'test_name';
      const calloutSize = Number(calloutSizeEl.value);
      calloutModeNoteEl.textContent = show3d
        ? '3D uses direct labels. Non-overlap placement is only available in 2D.'
        : '2D uses non-overlap placement.';
      const grouped = new Map();
      rows.forEach((row) => {{
        const key = colorValue(row);
        if (!grouped.has(key)) grouped.set(key, []);
        grouped.get(key).push(row);
      }});
      const traces = [...grouped.entries()].flatMap(([label, group], idx) => {{
        const base = {{
          name: label,
          mode: showCallouts && show3d ? 'markers+text' : 'markers',
          marker: {{
            size: 9,
            color: palette[idx % palette.length],
            opacity: 0.86,
            line: {{ width: 0.8, color: 'rgba(21,32,54,0.35)' }},
          }},
          textfont: {{
            size: calloutSize,
            color: '#152036',
          }},
          textposition: 'top center',
          text: showCallouts && show3d ? group.map((row) => labelText(row)) : undefined,
          customdata: group.map((row) => hoverText(row, show3d)),
          hovertemplate: '%{{customdata}}<extra></extra>',
        }};
        if (show3d) {{
          return [{{
            ...base,
            type: 'scatter3d',
            x: group.map((row) => row.pc1),
            y: group.map((row) => row.pc2),
            z: group.map((row) => row.pc3),
            textposition: 'top center',
          }}];
        }}
        return [{{
          ...base,
          type: 'scattergl',
          xaxis: 'x',
          yaxis: 'y',
          x: group.map((row) => row.pc1),
          y: group.map((row) => row.pc2),
        }}];
      }});
      if (!show3d) {{
        traces.push(
          {{
            type: 'histogram',
            x: rows.map((row) => row.pc1),
            xaxis: 'x',
            yaxis: 'y2',
            nbinsx: 18,
            marker: {{ color: 'rgba(21,32,54,0.42)', line: {{ width: 0 }} }},
            hovertemplate: 'PC1 bin count: %{{y}}<extra></extra>',
            showlegend: false,
          }},
          {{
            type: 'histogram',
            y: rows.map((row) => row.pc2),
            xaxis: 'x2',
            yaxis: 'y',
            nbinsy: 18,
            marker: {{ color: 'rgba(21,32,54,0.42)', line: {{ width: 0 }} }},
            hovertemplate: 'PC2 bin count: %{{x}}<extra></extra>',
            showlegend: false,
          }},
        );
      }}

      const titleBits = [
        `PC1 vs PC2`,
        show3d ? 'with PC3 z-axis' : '2D',
      ];
      const methodRecord = currentMethodRecord();
      if (methodRecord) {{
        titleBits.unshift(methodRecord.label);
      }}
      const pc1 = evRows.find((r) => r.component === 'PC1');
      const pc2 = evRows.find((r) => r.component === 'PC2');
      const pc3 = evRows.find((r) => r.component === 'PC3');
      if (pc1 && pc2) {{
        titleBits.push(
          `variance: PC1 ${{(100 * Number(pc1.explained_variance_ratio)).toFixed(1)}}%, PC2 ${{(100 * Number(pc2.explained_variance_ratio)).toFixed(1)}}%`
        );
      }}
      if (show3d && pc3) {{
        titleBits.push(`PC3 ${{(100 * Number(pc3.explained_variance_ratio)).toFixed(1)}}%`);
      }}

      const layout = show3d
        ? {{
            title: titleBits.join(' | '),
            paper_bgcolor: 'rgba(0,0,0,0)',
            plot_bgcolor: 'rgba(0,0,0,0)',
            legend: {{ orientation: 'h', y: -0.1 }},
            margin: {{ l: 0, r: 0, t: 56, b: 0 }},
            scene: {{
              xaxis: {{ title: 'PC1', zeroline: true, gridcolor: '#ebe7dd' }},
              yaxis: {{ title: 'PC2', zeroline: true, gridcolor: '#ebe7dd' }},
              zaxis: {{ title: 'PC3', zeroline: true, gridcolor: '#ebe7dd' }},
            }},
          }}
        : {{
            title: titleBits.join(' | '),
            paper_bgcolor: 'rgba(0,0,0,0)',
            plot_bgcolor: 'rgba(0,0,0,0)',
            legend: {{ orientation: 'h', y: -0.16 }},
            margin: {{ l: 58, r: 22, t: 56, b: 60 }},
            barmode: 'overlay',
            bargap: 0.05,
            xaxis: {{
              title: 'PC1',
              zeroline: true,
              gridcolor: '#ebe7dd',
              domain: [0.0, 0.84],
              anchor: 'y',
            }},
            yaxis: {{
              title: 'PC2',
              zeroline: true,
              gridcolor: '#ebe7dd',
              domain: [0.22, 1.0],
              anchor: 'x',
            }},
            xaxis2: {{
              domain: [0.87, 1.0],
              showgrid: false,
              zeroline: false,
              showticklabels: false,
            }},
            yaxis2: {{
              domain: [0.0, 0.17],
              showgrid: false,
              zeroline: false,
              showticklabels: false,
            }},
          }};
      if (!show3d && showCallouts) {{
        layout.annotations = buildRepelledAnnotations(rows, extent(rows.map((row) => row.pc1)), extent(rows.map((row) => row.pc2)), calloutSize);
      }}
      Plotly.react(scatterEl, traces, layout, {{ responsive: true, displaylogo: false }});
    }}

    function renderScree(evRows) {{
      const traces = [
        {{
          type: 'bar',
          x: evRows.map((row) => row.component),
          y: evRows.map((row) => row.explained_variance_ratio),
          marker: {{ color: '#2563eb' }},
          name: 'Explained variance',
        }},
        {{
          type: 'scatter',
          x: evRows.map((row) => row.component),
          y: evRows.map((row) => row.cumulative_explained),
          mode: 'lines+markers',
          marker: {{ color: '#dc2626' }},
          line: {{ color: '#dc2626', width: 2 }},
          yaxis: 'y2',
          name: 'Cumulative',
        }},
      ];
      const layout = {{
        title: 'Scree',
        paper_bgcolor: 'rgba(0,0,0,0)',
        plot_bgcolor: 'rgba(0,0,0,0)',
        margin: {{ l: 48, r: 48, t: 42, b: 42 }},
        xaxis: {{ title: 'Component' }},
        yaxis: {{ title: 'Explained variance', rangemode: 'tozero', gridcolor: '#ebe7dd' }},
        yaxis2: {{
          title: 'Cumulative',
          overlaying: 'y',
          side: 'right',
          range: [0, 1.02],
        }},
        legend: {{ orientation: 'h', y: -0.22 }},
      }};
      Plotly.react(screeEl, traces, layout, {{ responsive: true, displaylogo: false }});
    }}

    function renderLoadings(rows) {{
      const presentFeatures = uniq(
        rows
          .slice()
          .sort((a, b) => Number(a.feature_order || 0) - Number(b.feature_order || 0))
          .map((row) => row.feature)
      );
      const featureLabelMap = new Map(
        rows.map((row) => [row.feature, row.feature_label || row.feature])
      );
      const comps = ['PC1', 'PC2', 'PC3'];
      const traces = comps
        .filter((component) => rows.some((row) => row.component === component))
        .map((component, idx) => {{
          const subset = presentFeatures.map((feature) => {{
            const hit = rows.find((row) => row.feature === feature && row.component === component);
            return hit ? Number(hit.loading) : 0;
          }});
          return {{
            type: 'bar',
            orientation: 'h',
            y: presentFeatures.map((feature) => featureLabelMap.get(feature) || feature),
            x: subset,
            name: component,
            marker: {{ color: palette[idx] }},
          }};
        }});
      const layout = {{
        title: 'Loadings',
        barmode: 'group',
        paper_bgcolor: 'rgba(0,0,0,0)',
        plot_bgcolor: 'rgba(0,0,0,0)',
        margin: {{ l: 132, r: 22, t: 42, b: 42 }},
        xaxis: {{ title: 'Loading', range: [-1.05, 1.05], zeroline: true, gridcolor: '#ebe7dd' }},
        legend: {{ orientation: 'h', y: -0.22 }},
      }};
      Plotly.react(loadingsEl, traces, layout, {{ responsive: true, displaylogo: false }});
    }}

    function render() {{
      const rows = currentRows();
      const evRows = currentExplained();
      const loadingRows = currentLoadings();
      const diag = currentDiagnostic();
      renderSummary(rows, diag, evRows);
      renderScatter(rows, evRows);
      renderScree(evRows);
      renderLoadings(loadingRows);
    }}

    function updateNormalizationOptions() {{
      const norms = uniq(
        scores
          .filter((row) => row.method === methodEl.value && row.cohort === cohortEl.value)
          .map((row) => row.normalization)
      );
      const current = normalizationEl.value;
      fillSelect(normalizationEl, norms, norms.includes(current) ? current : 'raw');
    }}

    fillOptionRecords(methodEl, methodOptions, 'signed_all6');
    const cohorts = uniq(scores.map((row) => row.cohort));
    fillSelect(cohortEl, cohorts, 'female');
    updateNormalizationOptions();
    fillSelect(colorByEl, colorColumns, 'category');
    fillSelect(zAxisEl, ['none', 'pc3'], 'none');
    fillSelect(calloutsEl, ['none', 'test_name'], 'none');
    calloutSizeValueEl.textContent = `${{calloutSizeEl.value}} px`;

    methodEl.addEventListener('change', () => {{
      updateNormalizationOptions();
      render();
    }});
    cohortEl.addEventListener('change', () => {{
      updateNormalizationOptions();
      render();
    }});
    normalizationEl.addEventListener('change', render);
    colorByEl.addEventListener('change', render);
    zAxisEl.addEventListener('change', render);
    calloutsEl.addEventListener('change', render);
    calloutSizeEl.addEventListener('input', () => {{
      calloutSizeValueEl.textContent = `${{calloutSizeEl.value}} px`;
      render();
    }});
    function openNormInfo() {{
      wrapEl.classList.add('faded');
      normInfoOverlayEl.classList.add('open');
      normInfoOverlayEl.setAttribute('aria-hidden', 'false');
    }}

    function closeNormInfo() {{
      wrapEl.classList.remove('faded');
      normInfoOverlayEl.classList.remove('open');
      normInfoOverlayEl.setAttribute('aria-hidden', 'true');
    }}

    normInfoToggleEl.addEventListener('click', (event) => {{
      event.stopPropagation();
      if (normInfoOverlayEl.classList.contains('open')) {{
        closeNormInfo();
      }} else {{
        openNormInfo();
      }}
    }});
    normInfoOverlayEl.addEventListener('click', (event) => {{
      if (!normInfoPanelEl.contains(event.target)) {{
        closeNormInfo();
      }}
    }});
    normInfoPanelEl.addEventListener('click', (event) => {{
      event.stopPropagation();
    }});
    window.addEventListener('keydown', (event) => {{
      if (event.key === 'Escape') {{
        closeNormInfo();
      }}
    }});
    window.addEventListener('resize', () => {{
      Plotly.Plots.resize(scatterEl);
      Plotly.Plots.resize(screeEl);
      Plotly.Plots.resize(loadingsEl);
    }});

    render();
  </script>
</body>
</html>
"""
    out_path.write_text(html)


def render_clalit_rho_pca_html(
    scores_df: pd.DataFrame,
    explained_df: pd.DataFrame,
    loadings_df: pd.DataFrame,
    diagnostics_df: pd.DataFrame,
    out_path: Path,
) -> None:
    score_json = _json_records(scores_df)
    explained_json = _json_records(explained_df)
    loadings_json = _json_records(loadings_df)
    diagnostics_json = _json_records(diagnostics_df)
    color_cols_json = json.dumps(["none", *SUMMARY_PCA_COLOR_COLUMNS])
    rho_cols_json = json.dumps(list(CLALIT_SUMMARY_RHO_COLUMNS))
    rho_label_json = json.dumps({col: _rho_feature_label(col) for col in CLALIT_SUMMARY_RHO_COLUMNS})
    methods_json = json.dumps(
        [
            {"value": key, "label": spec["label"], "description": spec["description"]}
            for key, spec in CLALIT_SUMMARY_RHO_PCA_METHODS.items()
        ]
    )
    html = f"""<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="utf-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1" />
  <title>Clalit Summary-Rho PCA Explorer</title>
  <script src="https://cdn.plot.ly/plotly-2.35.2.min.js"></script>
  <style>
    :root {{
      --bg: #f5f1e8;
      --card: #fffdf8;
      --ink: #152036;
      --muted: #5b6475;
      --line: #d8cfbf;
      --accent: #0f766e;
    }}
    * {{ box-sizing: border-box; }}
    body {{
      margin: 0;
      font-family: "Avenir Next", "Gill Sans", sans-serif;
      background:
        radial-gradient(circle at top left, rgba(15,118,110,0.10), transparent 34%),
        radial-gradient(circle at top right, rgba(180,83,9,0.10), transparent 30%),
        var(--bg);
      color: var(--ink);
    }}
    .wrap {{
      max-width: 1540px;
      margin: 0 auto;
      padding: 24px 20px 32px;
      transition: filter 180ms ease, opacity 180ms ease;
    }}
    .wrap.faded {{
      filter: saturate(0.78) brightness(0.96);
      opacity: 0.72;
    }}
    .hero, .control, .card, .summary {{
      background: var(--card);
      border: 1px solid var(--line);
      box-shadow: 0 14px 34px rgba(21,32,54,0.06);
    }}
    .hero {{
      margin-bottom: 16px;
      padding: 18px 20px;
      border-radius: 18px;
    }}
    .hero h1 {{
      margin: 0 0 8px;
      font-size: clamp(1.5rem, 2vw, 2.1rem);
      letter-spacing: 0.01em;
    }}
    .hero p {{
      margin: 0;
      color: var(--muted);
      max-width: 1080px;
      line-height: 1.45;
    }}
    .controls {{
      display: grid;
      grid-template-columns: repeat(7, minmax(130px, 1fr));
      gap: 12px;
      margin-bottom: 12px;
    }}
    .control {{
      border-radius: 14px;
      padding: 10px 12px;
    }}
    .control label {{
      display: flex;
      align-items: center;
      gap: 6px;
      font-size: 0.8rem;
      text-transform: uppercase;
      letter-spacing: 0.08em;
      color: var(--muted);
      margin-bottom: 6px;
    }}
    .control select,
    .control input[type="range"],
    .tests-search {{
      width: 100%;
      padding: 8px 10px;
      border-radius: 10px;
      border: 1px solid #cbbfa8;
      background: #fffdfa;
      color: var(--ink);
      font: inherit;
    }}
    .control input[type="range"] {{ padding: 0; }}
    .icon-button {{
      display: inline-flex;
      align-items: center;
      justify-content: center;
      width: 20px;
      height: 20px;
      border-radius: 999px;
      border: 1px solid rgba(15,118,110,0.35);
      background: rgba(15,118,110,0.10);
      color: var(--accent);
      font-size: 0.78rem;
      font-weight: 700;
      cursor: pointer;
      padding: 0;
    }}
    .control-note, .tests-meta {{
      font-size: 0.84rem;
      color: var(--muted);
      line-height: 1.35;
      min-height: 1.2em;
    }}
    .overlay {{
      position: fixed;
      inset: 0;
      display: none;
      align-items: center;
      justify-content: center;
      padding: 24px;
      background: rgba(21,32,54,0.18);
      backdrop-filter: blur(2px) saturate(0.9);
      z-index: 30;
    }}
    .overlay.open {{ display: flex; }}
    .info-panel {{
      width: min(780px, calc(100vw - 32px));
      max-height: min(78vh, 760px);
      overflow: auto;
      padding: 18px 20px;
      background: linear-gradient(180deg, rgba(255,254,250,0.98), rgba(248,243,233,0.98));
      border: 1px solid rgba(216,207,191,0.85);
      border-radius: 18px;
      box-shadow: 0 28px 70px rgba(21,32,54,0.18);
    }}
    .info-panel h2 {{ margin: 0 0 10px; font-size: 1rem; }}
    .info-panel p {{ margin: 0 0 10px; color: var(--muted); line-height: 1.45; }}
    .info-grid {{
      display: grid;
      grid-template-columns: repeat(2, minmax(0, 1fr));
      gap: 10px;
    }}
    .info-item {{
      padding: 10px 12px;
      border: 1px solid var(--line);
      border-radius: 12px;
      background: rgba(255,255,255,0.72);
    }}
    .info-item strong {{
      display: block;
      margin-bottom: 4px;
    }}
    .summary {{
      margin-bottom: 14px;
      padding: 12px 14px;
      border-radius: 14px;
      color: var(--muted);
      line-height: 1.45;
    }}
    .grid {{
      display: grid;
      grid-template-columns: minmax(0, 1.7fr) minmax(340px, 0.9fr);
      gap: 14px;
      align-items: start;
    }}
    .stack {{
      display: grid;
      gap: 14px;
    }}
    .card {{
      border-radius: 18px;
      padding: 10px;
    }}
    .plot {{
      width: 100%;
      min-height: 520px;
    }}
    .plot.small {{ min-height: 300px; }}
    .tests-card {{ padding: 12px 14px; }}
    .tests-card h3 {{ margin: 0 0 8px; font-size: 1rem; }}
    .tests-search {{ margin-bottom: 8px; }}
    .tests-list {{
      max-height: 260px;
      overflow: auto;
      display: grid;
      gap: 6px;
      padding-right: 4px;
    }}
    .test-chip {{
      padding: 7px 9px;
      border: 1px solid rgba(21,32,54,0.10);
      border-radius: 10px;
      background: rgba(255,255,255,0.78);
      font-size: 0.92rem;
    }}
    @media (max-width: 1100px) {{
      .controls {{ grid-template-columns: repeat(2, minmax(180px, 1fr)); }}
      .grid {{ grid-template-columns: 1fr; }}
      .info-grid {{ grid-template-columns: 1fr; }}
    }}
    @media (max-width: 680px) {{
      .wrap {{ padding: 16px 12px 24px; }}
      .controls {{ grid-template-columns: 1fr; }}
      .plot {{ min-height: 420px; }}
      .plot.small {{ min-height: 280px; }}
    }}
  </style>
</head>
<body>
  <div class="wrap">
    <section class="hero">
      <h1>Clalit Summary-Stat Spearman-Rho PCA</h1>
      <p>
        PCA on complete-case Clalit biomarker Spearman rhos across quartile-derived median, IQR-over-median CV, and quantile-skewness summaries.
        The normalization selector applies to the smoothed trajectories before rho is computed; the PCA method selector changes which rho features enter the PCA.
      </p>
    </section>
    <section class="controls">
      <div class="control">
        <label for="method">PCA method <button type="button" id="methodInfoToggle" class="icon-button" aria-label="Explain PCA methods">i</button></label>
        <select id="method"></select>
      </div>
      <div class="control">
        <label for="cohort">Cohort</label>
        <select id="cohort"></select>
      </div>
      <div class="control">
        <label for="normalization">Normalization <button type="button" id="normInfoToggle" class="icon-button" aria-label="Explain normalization choices">i</button></label>
        <select id="normalization"></select>
      </div>
      <div class="control">
        <label for="colorBy">Color markers by</label>
        <select id="colorBy"></select>
      </div>
      <div class="control">
        <label for="zAxis">Z-axis</label>
        <select id="zAxis"></select>
      </div>
      <div class="control">
        <label for="callouts">Text callouts</label>
        <select id="callouts"></select>
        <div class="control-note" id="calloutModeNote">2D uses non-overlap placement. 3D falls back to direct labels.</div>
      </div>
      <div class="control">
        <label for="calloutSize">Callout text size</label>
        <input id="calloutSize" type="range" min="8" max="22" step="1" value="11" />
        <div class="control-note" id="calloutSizeValue">11 px</div>
      </div>
    </section>
    <div class="overlay" id="normInfoOverlay" aria-hidden="true">
      <section class="info-panel" id="normInfoPanel" role="dialog" aria-modal="true" aria-label="Normalization explainer">
        <h2>What the normalization choices mean</h2>
        <p>
          These options normalize each smoothed age trajectory before Spearman rho is computed against age. The rho values are not normalized afterward.
        </p>
        <div class="info-grid">
          <div class="info-item"><strong>raw</strong>Uses the smoothed metric as-is across age. Keeps absolute direction and scale.</div>
          <div class="info-item"><strong>young_ratio</strong>Divides the trajectory by its young-reference value, so age 20-24 acts as baseline.</div>
          <div class="info-item"><strong>young_log_fold</strong>Takes log of the young-reference ratio. Only valid when the full smoothed trajectory stays positive.</div>
          <div class="info-item"><strong>shape_z</strong>Z-scores the trajectory across age points. This removes level and keeps only trajectory shape.</div>
        </div>
      </section>
    </div>
    <div class="overlay" id="methodInfoOverlay" aria-hidden="true">
      <section class="info-panel" id="methodInfoPanel" role="dialog" aria-modal="true" aria-label="PCA method explainer">
        <h2>What the PCA methods change</h2>
        <p>
          Each method changes the feature matrix that enters PCA. `abs(...)` means the rho feature is transformed before PCA, not that a separate absolute trajectory was smoothed.
        </p>
        <div class="info-grid">
          <div class="info-item"><strong>Signed all 3</strong>Signed rho_median, rho_cv, and rho_quantile_skewness.</div>
          <div class="info-item"><strong>Abs median, all 3</strong>abs(rho_median) with rho_cv and rho_quantile_skewness left signed.</div>
          <div class="info-item"><strong>Abs all 3</strong>abs(rho_median), abs(rho_cv), and abs(rho_quantile_skewness).</div>
        </div>
      </section>
    </div>
    <div class="summary" id="summary"></div>
    <section class="grid">
      <div class="card">
        <div id="scatter" class="plot"></div>
      </div>
      <div class="stack">
        <div class="card">
          <div id="scree" class="plot small"></div>
        </div>
        <div class="card">
          <div id="loadings" class="plot small"></div>
        </div>
        <div class="card tests-card">
          <h3>Clalit Tests In Current View</h3>
          <input id="testSearch" class="tests-search" type="search" placeholder="Filter tests by name" />
          <div id="testsMeta" class="tests-meta"></div>
          <div id="testsList" class="tests-list"></div>
        </div>
      </div>
    </section>
  </div>
  <script>
    const scores = {score_json};
    const explained = {explained_json};
    const loadings = {loadings_json};
    const diagnostics = {diagnostics_json};
    const colorColumns = {color_cols_json};
    const rhoColumns = {rho_cols_json};
    const rhoLabels = {rho_label_json};
    const methodOptions = {methods_json};
    const palette = [
      '#2563eb', '#dc2626', '#0f766e', '#7c3aed', '#b45309', '#0891b2',
      '#be123c', '#4d7c0f', '#4338ca', '#ea580c', '#0d9488', '#9333ea'
    ];
    const wrapEl = document.querySelector('.wrap');
    const methodEl = document.getElementById('method');
    const cohortEl = document.getElementById('cohort');
    const normalizationEl = document.getElementById('normalization');
    const colorByEl = document.getElementById('colorBy');
    const zAxisEl = document.getElementById('zAxis');
    const calloutsEl = document.getElementById('callouts');
    const calloutSizeEl = document.getElementById('calloutSize');
    const calloutSizeValueEl = document.getElementById('calloutSizeValue');
    const calloutModeNoteEl = document.getElementById('calloutModeNote');
    const normInfoToggleEl = document.getElementById('normInfoToggle');
    const methodInfoToggleEl = document.getElementById('methodInfoToggle');
    const normInfoOverlayEl = document.getElementById('normInfoOverlay');
    const methodInfoOverlayEl = document.getElementById('methodInfoOverlay');
    const normInfoPanelEl = document.getElementById('normInfoPanel');
    const methodInfoPanelEl = document.getElementById('methodInfoPanel');
    const summaryEl = document.getElementById('summary');
    const scatterEl = document.getElementById('scatter');
    const screeEl = document.getElementById('scree');
    const loadingsEl = document.getElementById('loadings');
    const testSearchEl = document.getElementById('testSearch');
    const testsMetaEl = document.getElementById('testsMeta');
    const testsListEl = document.getElementById('testsList');

    function uniq(arr) {{ return [...new Set(arr)]; }}
    function fillSelect(el, values, preferred) {{
      el.innerHTML = '';
      values.forEach((value) => {{
        const option = document.createElement('option');
        option.value = value;
        option.textContent = value;
        el.appendChild(option);
      }});
      if (preferred && values.includes(preferred)) el.value = preferred;
    }}
    function fillOptionRecords(el, records, preferred) {{
      el.innerHTML = '';
      records.forEach((record) => {{
        const option = document.createElement('option');
        option.value = record.value;
        option.textContent = record.label;
        option.title = record.description || record.label;
        el.appendChild(option);
      }});
      if (preferred && records.some((record) => record.value === preferred)) el.value = preferred;
    }}
    function currentMethodRecord() {{
      return methodOptions.find((record) => record.value === methodEl.value) || null;
    }}
    function currentRows() {{
      return scores.filter((row) => row.method === methodEl.value && row.cohort === cohortEl.value && row.normalization === normalizationEl.value);
    }}
    function currentExplained() {{
      return explained.filter((row) => row.method === methodEl.value && row.cohort === cohortEl.value && row.normalization === normalizationEl.value);
    }}
    function currentLoadings() {{
      return loadings.filter((row) => row.method === methodEl.value && row.cohort === cohortEl.value && row.normalization === normalizationEl.value);
    }}
    function currentDiagnostic() {{
      return diagnostics.find((row) => row.method === methodEl.value && row.cohort === cohortEl.value && row.normalization === normalizationEl.value) || null;
    }}
    function colorValue(row) {{
      const field = colorByEl.value;
      if (field === 'none') return 'All biomarkers';
      return row[field] || 'Missing';
    }}
    function hoverText(row, show3d) {{
      const pcs = [`PC1: ${{Number(row.pc1).toFixed(4)}}`, `PC2: ${{Number(row.pc2).toFixed(4)}}`];
      if (show3d && row.pc3 !== null && row.pc3 !== undefined) pcs.push(`PC3: ${{Number(row.pc3).toFixed(4)}}`);
      const rhoLines = [];
      rhoColumns.forEach((col) => {{
        const value = Number(row[col]);
        if (Number.isFinite(value)) rhoLines.push(`${{rhoLabels[col] || col}}: ${{value.toFixed(4)}}`);
      }});
      return [
        `<b>${{row.test_name}}</b>`,
        `id=${{row.analysis_id}}`,
        `category=${{row.category || 'NA'}}`,
        `subcategory=${{row.subcategory || 'NA'}}`,
        `system=${{row.primary_organ_system || 'NA'}}`,
        `domain=${{row.aging_domain || 'NA'}}`,
        `class=${{row.measurement_class || 'NA'}}`,
        `target=${{row.target_kind || 'NA'}}`,
        ...pcs,
        ...rhoLines,
      ].join('<br>');
    }}
    function labelText(row) {{ return row.test_name || row.analysis_id; }}
    function extent(values) {{
      const finite = values.map(Number).filter((v) => Number.isFinite(v));
      if (!finite.length) return [-1, 1];
      const lo = Math.min(...finite);
      const hi = Math.max(...finite);
      if (Math.abs(hi - lo) < 1e-9) return [lo - 1, hi + 1];
      const pad = (hi - lo) * 0.08;
      return [lo - pad, hi + pad];
    }}
    function boxesOverlap(a, b) {{
      return !(a.x1 < b.x0 || a.x0 > b.x1 || a.y1 < b.y0 || a.y0 > b.y1);
    }}
    function buildRepelledAnnotations(rows, xRange, yRange, fontSize) {{
      const width = Math.max(scatterEl.clientWidth || 900, 420);
      const height = Math.max(scatterEl.clientHeight || 520, 360);
      const offsets = [];
      [12, 20, 28, 38, 52, 68].forEach((radius) => {{
        offsets.push(
          {{ dx: radius, dy: -radius, xanchor: 'left', yanchor: 'bottom' }},
          {{ dx: radius, dy: radius, xanchor: 'left', yanchor: 'top' }},
          {{ dx: -radius, dy: -radius, xanchor: 'right', yanchor: 'bottom' }},
          {{ dx: -radius, dy: radius, xanchor: 'right', yanchor: 'top' }},
          {{ dx: 0, dy: -(radius + 6), xanchor: 'center', yanchor: 'bottom' }},
          {{ dx: 0, dy: radius + 6, xanchor: 'center', yanchor: 'top' }},
          {{ dx: radius + 6, dy: 0, xanchor: 'left', yanchor: 'middle' }},
          {{ dx: -(radius + 6), dy: 0, xanchor: 'right', yanchor: 'middle' }}
        );
      }});
      const placed = [];
      const annotations = [];
      const ordered = [...rows].sort((a, b) => ((Number(b.pc1) ** 2) + (Number(b.pc2) ** 2)) - ((Number(a.pc1) ** 2) + (Number(a.pc2) ** 2)));
      ordered.forEach((row) => {{
        const label = labelText(row);
        const xNorm = (Number(row.pc1) - xRange[0]) / (xRange[1] - xRange[0]);
        const yNorm = (Number(row.pc2) - yRange[0]) / (yRange[1] - yRange[0]);
        const textWidth = ((label.length * fontSize * 0.62) + (fontSize * 1.4)) / width;
        const textHeight = (fontSize * 1.7) / height;
        let placedAnnotation = false;
        for (const offset of offsets) {{
          const centerX = xNorm + (offset.dx / width);
          const centerY = yNorm + (-offset.dy / height);
          const box = {{ x0: centerX - (textWidth / 2), x1: centerX + (textWidth / 2), y0: centerY - (textHeight / 2), y1: centerY + (textHeight / 2) }};
          if (box.x0 < 0.02 || box.x1 > 0.98 || box.y0 < 0.03 || box.y1 > 0.97) continue;
          if (placed.some((existing) => boxesOverlap(existing, box))) continue;
          placed.push(box);
          annotations.push({{
            x: Number(row.pc1), y: Number(row.pc2), xref: 'x', yref: 'y', text: label, showarrow: true, arrowhead: 0,
            arrowsize: 0.8, arrowwidth: 0.8, arrowcolor: 'rgba(21,32,54,0.28)', ax: offset.dx, ay: offset.dy,
            font: {{ size: fontSize, color: '#152036' }}, bgcolor: 'rgba(255,253,248,0.82)',
            bordercolor: 'rgba(21,32,54,0.12)', borderwidth: 1, borderpad: 2, xanchor: offset.xanchor, yanchor: offset.yanchor,
          }});
          placedAnnotation = true;
          break;
        }}
        if (!placedAnnotation) {{
          annotations.push({{
            x: Number(row.pc1), y: Number(row.pc2), xref: 'x', yref: 'y', text: label, showarrow: true, arrowhead: 0,
            arrowsize: 0.8, arrowwidth: 0.8, arrowcolor: 'rgba(21,32,54,0.28)', ax: 14, ay: -14,
            font: {{ size: fontSize, color: '#152036' }}, bgcolor: 'rgba(255,253,248,0.82)',
            bordercolor: 'rgba(21,32,54,0.12)', borderwidth: 1, borderpad: 2, xanchor: 'left', yanchor: 'bottom',
          }});
        }}
      }});
      return annotations;
    }}
    function renderSummary(rows, diag, evRows) {{
      const complete = rows.length;
      const total = diag ? diag.candidate_biomarkers : complete;
      const dropped = diag ? diag.dropped_for_pca : 0;
      const parts = [
        `<b>${{currentMethodRecord() ? currentMethodRecord().label : methodEl.value}}</b>`,
        `<b>${{cohortEl.value}}</b> / <b>${{normalizationEl.value}}</b>`,
        `Complete biomarkers in PCA: <b>${{complete}}</b> / ${{total}}`,
        `Dropped before PCA: <b>${{dropped}}</b>`,
      ];
      const methodRecord = currentMethodRecord();
      if (methodRecord && methodRecord.description) parts.push(methodRecord.description);
      ['PC1', 'PC2', 'PC3'].forEach((pc) => {{
        const row = evRows.find((item) => item.component === pc);
        if (row) parts.push(`${{pc}}=${{(100 * Number(row.explained_variance_ratio)).toFixed(1)}}%`);
      }});
      if (diag && diag.note) parts.push(diag.note);
      summaryEl.innerHTML = parts.join(' | ');
    }}
    function renderTestList(rows) {{
      const query = String(testSearchEl.value || '').trim().toLowerCase();
      const names = uniq(rows.map((row) => row.test_name || row.analysis_id)).sort((a, b) => a.localeCompare(b));
      const filtered = names.filter((name) => !query || name.toLowerCase().includes(query));
      testsMetaEl.textContent = `${{filtered.length}} shown of ${{names.length}} tests in the current PCA view`;
      testsListEl.innerHTML = filtered.length ? filtered.map((name) => `<div class="test-chip">${{name}}</div>`).join('') : '<div class="test-chip">No tests match the current filter.</div>';
    }}
    function renderScatter(rows, evRows) {{
      const show3d = zAxisEl.value === 'pc3';
      const showCallouts = calloutsEl.value === 'test_name';
      const calloutSize = Number(calloutSizeEl.value);
      calloutModeNoteEl.textContent = show3d ? '3D uses direct labels. Non-overlap placement is only available in 2D.' : '2D uses non-overlap placement first, then overlap fallback so every label still shows.';
      const grouped = new Map();
      rows.forEach((row) => {{
        const key = colorValue(row);
        if (!grouped.has(key)) grouped.set(key, []);
        grouped.get(key).push(row);
      }});
      const traces = [...grouped.entries()].flatMap(([label, group], idx) => {{
        const base = {{
          name: label,
          mode: showCallouts && show3d ? 'markers+text' : 'markers',
          marker: {{ size: 9, color: palette[idx % palette.length], opacity: 0.86, line: {{ width: 0.8, color: 'rgba(21,32,54,0.35)' }} }},
          textfont: {{ size: calloutSize, color: '#152036' }},
          textposition: 'top center',
          text: showCallouts && show3d ? group.map((row) => labelText(row)) : undefined,
          customdata: group.map((row) => hoverText(row, show3d)),
          hovertemplate: '%{{customdata}}<extra></extra>',
        }};
        if (show3d) {{
          return [{{ ...base, type: 'scatter3d', x: group.map((row) => row.pc1), y: group.map((row) => row.pc2), z: group.map((row) => row.pc3) }}];
        }}
        return [{{ ...base, type: 'scattergl', xaxis: 'x', yaxis: 'y', x: group.map((row) => row.pc1), y: group.map((row) => row.pc2) }}];
      }});
      if (!show3d) {{
        traces.push(
          {{ type: 'histogram', x: rows.map((row) => row.pc1), xaxis: 'x', yaxis: 'y2', nbinsx: 18, marker: {{ color: 'rgba(21,32,54,0.42)', line: {{ width: 0 }} }}, hovertemplate: 'PC1 bin count: %{{y}}<extra></extra>', showlegend: false }},
          {{ type: 'histogram', y: rows.map((row) => row.pc2), xaxis: 'x2', yaxis: 'y', nbinsy: 18, marker: {{ color: 'rgba(21,32,54,0.42)', line: {{ width: 0 }} }}, hovertemplate: 'PC2 bin count: %{{x}}<extra></extra>', showlegend: false }}
        );
      }}
      const pc1 = evRows.find((row) => row.component === 'PC1');
      const pc2 = evRows.find((row) => row.component === 'PC2');
      const pc3 = evRows.find((row) => row.component === 'PC3');
      const titleBits = [currentMethodRecord() ? currentMethodRecord().label : methodEl.value, 'PC1 vs PC2', show3d ? 'with PC3 z-axis' : '2D'];
      if (pc1 && pc2) titleBits.push(`variance: PC1 ${{(100 * Number(pc1.explained_variance_ratio)).toFixed(1)}}%, PC2 ${{(100 * Number(pc2.explained_variance_ratio)).toFixed(1)}}%`);
      if (show3d && pc3) titleBits.push(`PC3 ${{(100 * Number(pc3.explained_variance_ratio)).toFixed(1)}}%`);
      const layout = show3d ? {{
        title: titleBits.join(' | '),
        paper_bgcolor: 'rgba(0,0,0,0)', plot_bgcolor: 'rgba(0,0,0,0)', legend: {{ orientation: 'h', y: -0.1 }},
        margin: {{ l: 0, r: 0, t: 56, b: 0 }},
        scene: {{
          xaxis: {{ title: 'PC1', zeroline: true, gridcolor: '#ebe7dd' }},
          yaxis: {{ title: 'PC2', zeroline: true, gridcolor: '#ebe7dd' }},
          zaxis: {{ title: 'PC3', zeroline: true, gridcolor: '#ebe7dd' }},
        }},
      }} : {{
        title: titleBits.join(' | '),
        paper_bgcolor: 'rgba(0,0,0,0)', plot_bgcolor: 'rgba(0,0,0,0)', legend: {{ orientation: 'h', y: -0.16 }},
        margin: {{ l: 58, r: 22, t: 56, b: 60 }}, barmode: 'overlay', bargap: 0.05,
        xaxis: {{ title: 'PC1', zeroline: true, gridcolor: '#ebe7dd', domain: [0.0, 0.84], anchor: 'y' }},
        yaxis: {{ title: 'PC2', zeroline: true, gridcolor: '#ebe7dd', domain: [0.22, 1.0], anchor: 'x' }},
        xaxis2: {{ domain: [0.87, 1.0], showgrid: false, zeroline: false, showticklabels: false }},
        yaxis2: {{ domain: [0.0, 0.17], showgrid: false, zeroline: false, showticklabels: false }},
      }};
      if (!show3d && showCallouts) layout.annotations = buildRepelledAnnotations(rows, extent(rows.map((row) => row.pc1)), extent(rows.map((row) => row.pc2)), calloutSize);
      Plotly.react(scatterEl, traces, layout, {{ responsive: true, displaylogo: false }});
    }}
    function renderScree(evRows) {{
      const traces = [
        {{ type: 'bar', x: evRows.map((row) => row.component), y: evRows.map((row) => row.explained_variance_ratio), marker: {{ color: '#2563eb' }}, name: 'Explained variance' }},
        {{ type: 'scatter', x: evRows.map((row) => row.component), y: evRows.map((row) => row.cumulative_explained), mode: 'lines+markers', marker: {{ color: '#dc2626' }}, line: {{ color: '#dc2626', width: 2 }}, yaxis: 'y2', name: 'Cumulative' }},
      ];
      Plotly.react(screeEl, traces, {{
        title: 'Scree', paper_bgcolor: 'rgba(0,0,0,0)', plot_bgcolor: 'rgba(0,0,0,0)', margin: {{ l: 48, r: 48, t: 42, b: 42 }},
        xaxis: {{ title: 'Component' }}, yaxis: {{ title: 'Explained variance', rangemode: 'tozero', gridcolor: '#ebe7dd' }},
        yaxis2: {{ title: 'Cumulative', overlaying: 'y', side: 'right', range: [0, 1.02] }}, legend: {{ orientation: 'h', y: -0.22 }},
      }}, {{ responsive: true, displaylogo: false }});
    }}
    function renderLoadings(rows) {{
      const presentFeatures = uniq(rows.slice().sort((a, b) => Number(a.feature_order || 0) - Number(b.feature_order || 0)).map((row) => row.feature));
      const featureLabelMap = new Map(rows.map((row) => [row.feature, row.feature_label || row.feature]));
      const traces = ['PC1', 'PC2', 'PC3'].filter((component) => rows.some((row) => row.component === component)).map((component, idx) => {{
        const subset = presentFeatures.map((feature) => {{
          const hit = rows.find((row) => row.feature === feature && row.component === component);
          return hit ? Number(hit.loading) : 0;
        }});
        return {{ type: 'bar', orientation: 'h', y: presentFeatures.map((feature) => featureLabelMap.get(feature) || feature), x: subset, name: component, marker: {{ color: palette[idx] }} }};
      }});
      Plotly.react(loadingsEl, traces, {{
        title: 'Loadings', barmode: 'group', paper_bgcolor: 'rgba(0,0,0,0)', plot_bgcolor: 'rgba(0,0,0,0)',
        margin: {{ l: 148, r: 22, t: 42, b: 42 }}, xaxis: {{ title: 'Loading', range: [-1.05, 1.05], zeroline: true, gridcolor: '#ebe7dd' }},
        legend: {{ orientation: 'h', y: -0.22 }},
      }}, {{ responsive: true, displaylogo: false }});
    }}
    function render() {{
      const rows = currentRows();
      const evRows = currentExplained();
      renderSummary(rows, currentDiagnostic(), evRows);
      renderScatter(rows, evRows);
      renderScree(evRows);
      renderLoadings(currentLoadings());
      renderTestList(rows);
    }}
    function updateNormalizationOptions() {{
      const norms = uniq(scores.filter((row) => row.method === methodEl.value && row.cohort === cohortEl.value).map((row) => row.normalization));
      const current = normalizationEl.value;
      fillSelect(normalizationEl, norms, norms.includes(current) ? current : 'raw');
    }}
    function openOverlay(overlayEl) {{
      wrapEl.classList.add('faded');
      overlayEl.classList.add('open');
      overlayEl.setAttribute('aria-hidden', 'false');
    }}
    function closeOverlay(overlayEl) {{
      wrapEl.classList.remove('faded');
      overlayEl.classList.remove('open');
      overlayEl.setAttribute('aria-hidden', 'true');
    }}
    fillOptionRecords(methodEl, methodOptions, 'signed_all3');
    fillSelect(cohortEl, uniq(scores.map((row) => row.cohort)), 'female');
    updateNormalizationOptions();
    fillSelect(colorByEl, colorColumns, 'category');
    fillSelect(zAxisEl, ['none', 'pc3'], 'none');
    fillSelect(calloutsEl, ['none', 'test_name'], 'none');
    calloutSizeValueEl.textContent = `${{calloutSizeEl.value}} px`;
    methodEl.addEventListener('change', () => {{ updateNormalizationOptions(); render(); }});
    cohortEl.addEventListener('change', () => {{ updateNormalizationOptions(); render(); }});
    normalizationEl.addEventListener('change', render);
    colorByEl.addEventListener('change', render);
    zAxisEl.addEventListener('change', render);
    calloutsEl.addEventListener('change', render);
    calloutSizeEl.addEventListener('input', () => {{ calloutSizeValueEl.textContent = `${{calloutSizeEl.value}} px`; render(); }});
    testSearchEl.addEventListener('input', () => renderTestList(currentRows()));
    normInfoToggleEl.addEventListener('click', (event) => {{ event.stopPropagation(); normInfoOverlayEl.classList.contains('open') ? closeOverlay(normInfoOverlayEl) : openOverlay(normInfoOverlayEl); }});
    methodInfoToggleEl.addEventListener('click', (event) => {{ event.stopPropagation(); methodInfoOverlayEl.classList.contains('open') ? closeOverlay(methodInfoOverlayEl) : openOverlay(methodInfoOverlayEl); }});
    normInfoOverlayEl.addEventListener('click', (event) => {{ if (!normInfoPanelEl.contains(event.target)) closeOverlay(normInfoOverlayEl); }});
    methodInfoOverlayEl.addEventListener('click', (event) => {{ if (!methodInfoPanelEl.contains(event.target)) closeOverlay(methodInfoOverlayEl); }});
    window.addEventListener('keydown', (event) => {{ if (event.key === 'Escape') {{ closeOverlay(normInfoOverlayEl); closeOverlay(methodInfoOverlayEl); }} }});
    window.addEventListener('resize', () => {{ Plotly.Plots.resize(scatterEl); Plotly.Plots.resize(screeEl); Plotly.Plots.resize(loadingsEl); }});
    render();
  </script>
</body>
</html>
"""
    out_path.write_text(html)


def build_summary_rho_pca_outputs(
    summary_dir: Path,
    rho_diag_df: pd.DataFrame,
    *,
    method_specs: dict[str, dict[str, object]] = SUMMARY_RHO_PCA_METHODS,
    rho_columns: Iterable[str] = SUMMARY_RHO_COLUMNS,
    cohorts: Iterable[str] = COHORTS,
    render_html_fn=render_summary_rho_pca_html,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    outputs_dir = summary_dir / "rho_pca"
    scree_dir = outputs_dir / "scree"
    loadings_dir = outputs_dir / "loadings"
    matrices_dir = outputs_dir / "matrices"
    for path in (scree_dir, loadings_dir, matrices_dir):
        if path.exists():
            shutil.rmtree(path)
    ensure_dir(outputs_dir)
    ensure_dir(scree_dir)
    ensure_dir(loadings_dir)
    ensure_dir(matrices_dir)

    score_frames: list[pd.DataFrame] = []
    explained_frames: list[pd.DataFrame] = []
    loading_frames: list[pd.DataFrame] = []
    diagnostic_rows: list[dict[str, object]] = []

    rho_columns = tuple(str(col) for col in rho_columns)
    cohorts = tuple(str(cohort) for cohort in cohorts)
    max_components = max(len(spec["features"]) for spec in method_specs.values())
    base_diag_lookup: dict[tuple[str, str], dict[str, object]] = {}
    if not rho_diag_df.empty:
        base_diag_lookup = {
            (str(row["cohort"]), str(row["normalization"])): row
            for row in rho_diag_df.to_dict(orient="records")
        }
    for cohort in cohorts:
        for normalization in CURVE_NORMALIZATIONS:
            feature_path = summary_dir / "feature_matrices" / cohort / f"{normalization}_summary_features.csv"
            if feature_path.exists():
                feature_df = pd.read_csv(feature_path)
            else:
                feature_df = pd.DataFrame(columns=[*SUMMARY_FEATURE_METADATA_COLUMNS, *rho_columns])
            base_diag = base_diag_lookup.get((cohort, normalization), {})
            candidate_biomarkers = int(base_diag.get("candidate_biomarkers", len(feature_df)))

            for method_key, method_spec in method_specs.items():
                method_matrix_df, feature_rows = _build_rho_method_matrix(
                    feature_df,
                    method_key,
                    method_specs=method_specs,
                    rho_columns=rho_columns,
                )
                feature_ids = [str(row["feature_id"]) for row in feature_rows]
                feature_labels = [str(row["feature_label"]) for row in feature_rows]
                matrix_df = method_matrix_df.dropna(subset=feature_ids).copy()
                matrix_df.to_csv(matrices_dir / f"{cohort}_{normalization}_{method_key}_rho_matrix.csv", index=False)

                diagnostic_rows.append(
                    {
                        "method": method_key,
                        "method_label": method_spec["label"],
                        "cohort": cohort,
                        "normalization": normalization,
                        "candidate_biomarkers": candidate_biomarkers,
                        "complete_biomarkers": int(len(matrix_df)),
                        "dropped_for_pca": int(max(candidate_biomarkers - len(matrix_df), 0)),
                        "rows_with_any_na": int(method_matrix_df[feature_ids].isna().any(axis=1).sum()) if len(method_matrix_df) else 0,
                        "n_features": len(feature_ids),
                        "feature_ids": "|".join(feature_ids),
                        "feature_labels": "|".join(feature_labels),
                        "note": method_spec["description"],
                    }
                )

                if matrix_df.empty:
                    explained_df = pd.DataFrame(
                        {
                            "method": method_key,
                            "method_label": method_spec["label"],
                            "component": [f"PC{k}" for k in range(1, max_components + 1)],
                            "component_index": list(range(1, max_components + 1)),
                            "explained_variance_ratio": [0.0] * max_components,
                            "cumulative_explained": [0.0] * max_components,
                            "cohort": cohort,
                            "normalization": normalization,
                            "n_biomarkers": 0,
                            "n_features": len(feature_ids),
                        }
                    )
                    loadings_wide = pd.DataFrame(
                        {
                            "feature": feature_ids,
                            "feature_label": feature_labels,
                            "feature_order": list(range(1, len(feature_ids) + 1)),
                        }
                    )
                    for k in range(1, max_components + 1):
                        loadings_wide[f"pc{k}"] = 0.0
                    plot_summary_rho_pca_scree(
                        explained_df,
                        scree_dir / f"{cohort}_{normalization}_{method_key}_scree.png",
                        method_label=method_spec["label"],
                        cohort=cohort,
                        normalization=normalization,
                    )
                    plot_summary_rho_pca_loadings(
                        loadings_wide,
                        loadings_dir / f"{cohort}_{normalization}_{method_key}_loadings.png",
                        method_label=method_spec["label"],
                        cohort=cohort,
                        normalization=normalization,
                    )
                    explained_frames.append(explained_df)
                    loadings_long = loadings_wide.melt(
                        id_vars=["feature", "feature_label", "feature_order"],
                        var_name="component_raw",
                        value_name="loading",
                    )
                    loadings_long["component"] = loadings_long["component_raw"].str.upper()
                    loadings_long["cohort"] = cohort
                    loadings_long["normalization"] = normalization
                    loadings_long["method"] = method_key
                    loadings_long["method_label"] = method_spec["label"]
                    loading_frames.append(
                        loadings_long[
                            [
                                "method",
                                "method_label",
                                "cohort",
                                "normalization",
                                "feature",
                                "feature_label",
                                "feature_order",
                                "component",
                                "loading",
                            ]
                        ]
                    )
                    continue

                X = matrix_df[feature_ids].to_numpy(dtype=float)
                if matrix_df.shape[0] < 2 or np.nanmax(np.abs(X - X[0])) < EPS:
                    n_components = len(feature_ids)
                    scores = np.zeros((matrix_df.shape[0], max_components), dtype=float)
                    loadings = np.zeros((len(feature_ids), max_components), dtype=float)
                    explained = np.zeros(max_components, dtype=float)
                else:
                    n_components = min(max_components, matrix_df.shape[0], len(feature_ids))
                    pca = PCA(n_components=n_components)
                    scores_fit = pca.fit_transform(X)
                    scores = np.full((matrix_df.shape[0], max_components), np.nan, dtype=float)
                    scores[:, :n_components] = scores_fit
                    loadings = np.full((len(feature_ids), max_components), np.nan, dtype=float)
                    loadings[:, :n_components] = pca.components_.T
                    explained = np.full(max_components, np.nan, dtype=float)
                    explained[:n_components] = pca.explained_variance_ratio_

                score_df = matrix_df.copy()
                score_df.insert(0, "method_label", method_spec["label"])
                score_df.insert(0, "method", method_key)
                score_df.insert(0, "normalization", normalization)
                score_df.insert(0, "cohort", cohort)
                for k in range(1, max_components + 1):
                    score_df[f"pc{k}"] = scores[:, k - 1]
                score_frames.append(score_df)

                explained_df = pd.DataFrame(
                    {
                        "method": method_key,
                        "method_label": method_spec["label"],
                        "cohort": cohort,
                        "normalization": normalization,
                        "component": [f"PC{k}" for k in range(1, max_components + 1)],
                        "component_index": list(range(1, max_components + 1)),
                        "explained_variance_ratio": explained,
                        "cumulative_explained": np.nancumsum(np.nan_to_num(explained, nan=0.0)),
                        "n_biomarkers": int(len(matrix_df)),
                        "n_features": len(feature_ids),
                    }
                )
                explained_frames.append(explained_df)

                loadings_wide = pd.DataFrame(
                    {
                        "feature": feature_ids,
                        "feature_label": feature_labels,
                        "feature_order": list(range(1, len(feature_ids) + 1)),
                    }
                )
                for k in range(1, max_components + 1):
                    loadings_wide[f"pc{k}"] = loadings[:, k - 1]
                loadings_long = loadings_wide.melt(
                    id_vars=["feature", "feature_label", "feature_order"],
                    var_name="component_raw",
                    value_name="loading",
                )
                loadings_long["component"] = loadings_long["component_raw"].str.upper()
                loadings_long["cohort"] = cohort
                loadings_long["normalization"] = normalization
                loadings_long["method"] = method_key
                loadings_long["method_label"] = method_spec["label"]
                loading_frames.append(
                    loadings_long[
                        [
                            "method",
                            "method_label",
                            "cohort",
                            "normalization",
                            "feature",
                            "feature_label",
                            "feature_order",
                            "component",
                            "loading",
                        ]
                    ]
                )

                plot_summary_rho_pca_scree(
                    explained_df,
                    scree_dir / f"{cohort}_{normalization}_{method_key}_scree.png",
                    method_label=method_spec["label"],
                    cohort=cohort,
                    normalization=normalization,
                )
                plot_summary_rho_pca_loadings(
                    loadings_wide,
                    loadings_dir / f"{cohort}_{normalization}_{method_key}_loadings.png",
                    method_label=method_spec["label"],
                    cohort=cohort,
                    normalization=normalization,
                )

    scores_all = pd.concat(score_frames, ignore_index=True) if score_frames else pd.DataFrame()
    explained_all = pd.concat(explained_frames, ignore_index=True) if explained_frames else pd.DataFrame()
    loadings_all = pd.concat(loading_frames, ignore_index=True) if loading_frames else pd.DataFrame()
    diagnostics_all = pd.DataFrame(diagnostic_rows)
    scores_all.to_csv(outputs_dir / "rho_pca_scores.csv", index=False)
    explained_all.to_csv(outputs_dir / "rho_pca_explained_variance.csv", index=False)
    loadings_all.to_csv(outputs_dir / "rho_pca_loadings.csv", index=False)
    diagnostics_all.to_csv(outputs_dir / "rho_matrix_diagnostics.csv", index=False)
    render_html_fn(
        scores_all,
        explained_all,
        loadings_all,
        diagnostics_all,
        outputs_dir / "rho_pca_explorer.html",
    )
    (outputs_dir / "README.md").write_text(
        """# Summary-Rho PCA Outputs

This folder contains PCA outputs built from the complete-case summary-stat Spearman-rho matrices across alternative feature-set methods.

## Files
- `matrices/<cohort>_<normalization>_<method>_rho_matrix.csv`: method-specific complete-case rho matrix used for PCA
- `rho_pca_scores.csv`: biomarker scores and metadata, including `method` and `method_label`
- `rho_pca_explained_variance.csv`: explained-variance table per method, cohort, and normalization
- `rho_pca_loadings.csv`: loading table for interpreting the PCA axes, including transformed feature labels
- `rho_matrix_diagnostics.csv`: completeness diagnostics by method, cohort, and normalization
- `scree/<cohort>_<normalization>_<method>_scree.png`: scree plot PNG per method/cohort/normalization
- `loadings/<cohort>_<normalization>_<method>_loadings.png`: loadings PNG per method/cohort/normalization
- `rho_pca_explorer.html`: interactive PC scatter explorer with method, cohort, normalization, color, PC3, and test-list controls

## Notes
- Constant smoothed trajectories are encoded as Spearman rho `0.0` instead of missing.
- `young_log_fold` is stricter than the other normalizations because the full smoothed young-reference ratio must stay strictly positive.
- Each method is centered by the PCA implementation; no extra feature scaling is applied.
"""
    )
    return scores_all, explained_all, loadings_all


def write_clalit_branch_readmes(out_dir: Path) -> None:
    ensure_dir(out_dir)
    ensure_dir(out_dir / "summary_stats")
    (out_dir / "README.md").write_text(
        """# Clalit Aging-Biomarker Analysis

This folder contains the Clalit-only summary-stat Spearman-rho PCA analysis for the subset of mapped Clalit tests that resolve into the aging-biomarker catalog.

## Contents
- `summary_stats/smoothed_curves_long.parquet`: smoothed trajectories by sex, metric, and normalization
- `summary_stats/feature_matrices/<cohort>/<normalization>_summary_features.csv`: complete-case feature matrices used for the rho PCA inputs
- `summary_stats/mapping_diagnostics.csv`: Clalit mapping resolutions and unresolved targets
- `summary_stats/rho_pca/...`: PCA matrices, scree plots, loadings, diagnostics, and the standalone explorer
"""
    )
    (out_dir / "summary_stats" / "README.md").write_text(
        """# Clalit Summary-Stat Branch

Outputs for the Clalit summary-stat Spearman-rho PCA analysis.

## Files
- `smoothed_curves_long.parquet`: smoothed trajectories by cohort, metric, and normalization
- `feature_matrices/<cohort>/<normalization>_summary_features.csv`: normalization-specific biomarker rho matrices
- `mapping_diagnostics.csv`: resolution table for Clalit mapped targets
- `rho_pca/...`: PCA matrices, scree/loadings PNGs, diagnostics, and the interactive explorer
"""
    )


def run_clalit_summary_analysis(
    catalog_path: Path,
    out_dir: Path,
    *,
    clalit_quartiles_path: Path | None = None,
    clalit_f_path: Path | None = None,
    clalit_m_path: Path | None = None,
    clalit_map_path: Path,
) -> dict[str, object]:
    ensure_dir(out_dir)
    summary_dir = out_dir / "summary_stats"
    ensure_dir(summary_dir)

    summary_stats, targets_df, resolution_df = build_clalit_summary_stats(
        catalog_path,
        clalit_map_path,
        clalit_quartiles_path=clalit_quartiles_path,
        clalit_f_path=clalit_f_path,
        clalit_m_path=clalit_m_path,
    )
    resolution_df.to_csv(summary_dir / "mapping_diagnostics.csv", index=False)
    smoothed_curves, rho_diag_df = build_summary_rho_feature_outputs(
        summary_stats,
        targets_df,
        summary_dir,
        metrics=CLALIT_SUMMARY_METRICS,
        cohorts=CLALIT_COHORTS,
        rho_columns=CLALIT_SUMMARY_RHO_COLUMNS,
    )
    rho_pca_scores, rho_pca_explained, rho_pca_loadings = build_summary_rho_pca_outputs(
        summary_dir,
        rho_diag_df,
        method_specs=CLALIT_SUMMARY_RHO_PCA_METHODS,
        rho_columns=CLALIT_SUMMARY_RHO_COLUMNS,
        cohorts=CLALIT_COHORTS,
        render_html_fn=render_clalit_rho_pca_html,
    )
    write_clalit_branch_readmes(out_dir)
    run_manifest = {
        "catalog_path": str(catalog_path),
        "clalit_quartiles_path": str(clalit_quartiles_path) if clalit_quartiles_path else None,
        "clalit_f_path": str(clalit_f_path),
        "clalit_m_path": str(clalit_m_path),
        "clalit_map_path": str(clalit_map_path),
        "cohorts": list(CLALIT_COHORTS),
        "age_grid_start": int(AGE_GRID.min()),
        "age_grid_end": int(AGE_GRID.max()),
        "curve_normalizations": list(CURVE_NORMALIZATIONS),
        "summary_metrics": list(CLALIT_SUMMARY_METRICS),
        "summary_rho_pca_features": list(CLALIT_SUMMARY_RHO_COLUMNS),
        "summary_rho_pca_methods": {
            key: {
                "label": spec["label"],
                "description": spec["description"],
                "features": [feature_id for _, feature_id, _, _ in spec["features"]],
            }
            for key, spec in CLALIT_SUMMARY_RHO_PCA_METHODS.items()
        },
        "mapping_resolution_methods": sorted(resolution_df["resolution_method"].dropna().astype(str).unique().tolist()),
        "output_root": str(out_dir),
    }
    (out_dir / "run_manifest.json").write_text(json.dumps(run_manifest, indent=2))
    return {
        "summary_stats": summary_stats,
        "targets": targets_df,
        "mapping_diagnostics": resolution_df,
        "smoothed_curves": smoothed_curves,
        "rho_pca_scores": rho_pca_scores,
        "rho_pca_explained_variance": rho_pca_explained,
        "rho_pca_loadings": rho_pca_loadings,
    }


def write_combined_rho_pca_dashboard(
    out_path: Path,
    *,
    nhanes_rel_path: str,
    clalit_rel_path: str | None,
) -> None:
    clalit_option = (
        '<option value="clalit">Clalit</option>' if clalit_rel_path else ""
    )
    clalit_dataset = f'"clalit": "{clalit_rel_path}",' if clalit_rel_path else ""
    html = f"""<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="utf-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1" />
  <title>Spearman-Rho PCA Explorer</title>
  <style>
    :root {{
      --bg: #f5efe4;
      --card: rgba(255, 252, 246, 0.92);
      --ink: #162233;
      --muted: #5e6877;
      --line: rgba(22, 34, 51, 0.12);
      --accent: #0f766e;
      --accent-2: #b45309;
      --shadow: 0 24px 60px rgba(22, 34, 51, 0.12);
    }}
    * {{ box-sizing: border-box; }}
    body {{
      margin: 0;
      font-family: "Avenir Next", "Gill Sans", sans-serif;
      color: var(--ink);
      background:
        radial-gradient(circle at top left, rgba(15,118,110,0.12), transparent 32%),
        radial-gradient(circle at top right, rgba(180,83,9,0.10), transparent 28%),
        linear-gradient(180deg, #f8f3ea 0%, #f2ecdf 100%);
    }}
    .wrap {{
      max-width: 1640px;
      margin: 0 auto;
      padding: 24px 20px 28px;
    }}
    .hero {{
      display: grid;
      grid-template-columns: minmax(0, 1.3fr) minmax(280px, 0.7fr);
      gap: 18px;
      margin-bottom: 16px;
      padding: 20px 22px;
      border: 1px solid var(--line);
      border-radius: 24px;
      background: linear-gradient(140deg, rgba(255,255,255,0.95), rgba(249,244,234,0.92));
      box-shadow: var(--shadow);
    }}
    .hero h1 {{
      margin: 0 0 8px;
      font-size: clamp(1.7rem, 2.3vw, 2.5rem);
      letter-spacing: 0.01em;
    }}
    .hero p {{
      margin: 0;
      color: var(--muted);
      line-height: 1.5;
      max-width: 950px;
    }}
    .picker {{
      display: grid;
      gap: 10px;
      align-content: start;
      padding: 14px;
      border: 1px solid var(--line);
      border-radius: 18px;
      background: rgba(255,255,255,0.72);
    }}
    .picker label {{
      font-size: 0.78rem;
      letter-spacing: 0.08em;
      text-transform: uppercase;
      color: var(--muted);
    }}
    .picker select {{
      width: 100%;
      padding: 11px 12px;
      border: 1px solid rgba(22,34,51,0.16);
      border-radius: 12px;
      background: #fffdfa;
      color: var(--ink);
      font: inherit;
    }}
    .picker-note {{
      font-size: 0.9rem;
      color: var(--muted);
      line-height: 1.45;
    }}
    .frame-card {{
      border: 1px solid var(--line);
      border-radius: 24px;
      background: var(--card);
      box-shadow: var(--shadow);
      overflow: hidden;
    }}
    .frame-head {{
      display: flex;
      justify-content: space-between;
      align-items: center;
      gap: 12px;
      padding: 14px 18px;
      border-bottom: 1px solid var(--line);
      background: rgba(255,255,255,0.84);
    }}
    .frame-title {{
      font-size: 0.95rem;
      color: var(--muted);
    }}
    .frame-link {{
      color: var(--accent);
      text-decoration: none;
      font-size: 0.92rem;
      font-weight: 600;
    }}
    .frame-link:hover {{ text-decoration: underline; }}
    iframe {{
      width: 100%;
      height: 1550px;
      border: 0;
      display: block;
      background: #fff;
    }}
    @media (max-width: 980px) {{
      .hero {{ grid-template-columns: 1fr; }}
      iframe {{ height: 1680px; }}
    }}
  </style>
</head>
<body>
  <div class="wrap">
    <section class="hero">
      <div>
        <h1>Spearman-Rho PCA Explorer</h1>
        <p>
          This is the canonical rho dashboard for both datasets. Use the source selector to switch between the NHANES and Clalit PCA explorers without hunting through separate folders.
        </p>
      </div>
      <div class="picker">
        <label for="sourceSelect">Data Source</label>
        <select id="sourceSelect">
          <option value="nhanes">NHANES</option>
          {clalit_option}
        </select>
        <div class="picker-note" id="sourceNote">
          NHANES includes pooled, female, and male views plus the six-feature rho methods.
        </div>
      </div>
    </section>
    <section class="frame-card">
      <div class="frame-head">
        <div class="frame-title" id="frameTitle">Showing NHANES rho PCA explorer</div>
        <a id="frameLink" class="frame-link" href="{nhanes_rel_path}">Open current explorer directly</a>
      </div>
      <iframe id="explorerFrame" src="{nhanes_rel_path}" title="Spearman-rho PCA explorer"></iframe>
    </section>
  </div>
  <script>
    const datasets = {{
      nhanes: {{
        label: 'NHANES',
        path: '{nhanes_rel_path}',
        note: 'NHANES includes pooled, female, and male views plus the six-feature rho methods.',
      }},
      {clalit_dataset}
    }};
    if (datasets.clalit) {{
      datasets.clalit = {{
        label: 'Clalit',
        path: '{clalit_rel_path or ""}',
        note: 'Clalit includes female and male views plus the five-feature rho methods.',
      }};
    }}

    const sourceSelectEl = document.getElementById('sourceSelect');
    const frameEl = document.getElementById('explorerFrame');
    const frameTitleEl = document.getElementById('frameTitle');
    const frameLinkEl = document.getElementById('frameLink');
    const sourceNoteEl = document.getElementById('sourceNote');

    function setSource(source) {{
      const dataset = datasets[source] || datasets.nhanes;
      sourceSelectEl.value = source in datasets ? source : 'nhanes';
      frameEl.src = dataset.path;
      frameLinkEl.href = dataset.path;
      frameTitleEl.textContent = `Showing ${{dataset.label}} rho PCA explorer`;
      frameLinkEl.textContent = `Open ${{dataset.label}} explorer directly`;
      sourceNoteEl.textContent = dataset.note;
      try {{
        localStorage.setItem('rhoPcaSource', sourceSelectEl.value);
      }} catch (err) {{}}
    }}

    const remembered = (() => {{
      try {{
        return localStorage.getItem('rhoPcaSource');
      }} catch (err) {{
        return null;
      }}
    }})();
    setSource((remembered && datasets[remembered]) ? remembered : 'nhanes');
    sourceSelectEl.addEventListener('change', () => setSource(sourceSelectEl.value));
  </script>
</body>
</html>
"""
    out_path.write_text(html)


def plot_wasserstein_heatmaps(pairwise_matrices: dict[tuple[str, str], pd.DataFrame], out_dir: Path) -> pd.DataFrame:
    ensure_dir(out_dir)
    rows = []
    for (cohort, normalization), matrix in pairwise_matrices.items():
        if matrix.empty:
            continue
        arr = matrix.to_numpy(dtype=float)
        diag = np.nan_to_num(np.diag(arr), nan=0.0)
        symmetry_diff = float(np.nanmax(np.abs(arr - arr.T))) if arr.size else 0.0
        finite_fraction = float(np.isfinite(arr).mean()) if arr.size else np.nan
        rows.append(
            {
                "cohort": cohort,
                "normalization": normalization,
                "n_biomarkers": len(matrix),
                "max_abs_symmetry_diff": symmetry_diff,
                "diag_max_abs": float(np.max(np.abs(diag))) if len(diag) else 0.0,
                "finite_fraction": finite_fraction,
            }
        )
        fig, ax = plt.subplots(figsize=(8, 7))
        im = ax.imshow(arr, aspect="auto", interpolation="nearest")
        ax.set_title(f"Wasserstein heatmap: {cohort} / {normalization}")
        ax.set_xlabel("Biomarker")
        ax.set_ylabel("Biomarker")
        fig.colorbar(im, ax=ax, shrink=0.8)
        fig.tight_layout()
        fig.savefig(out_dir / f"{cohort}_{normalization}_wasserstein_heatmap.png", dpi=160)
        plt.close(fig)
    sanity_df = pd.DataFrame(rows)
    sanity_df.to_csv(out_dir.parent / "sanity.csv", index=False)
    return sanity_df


def plot_cohort_comparison_panels(
    smoothed_curves: pd.DataFrame,
    review_targets: list[str],
    out_dir: Path,
) -> None:
    ensure_dir(out_dir)
    use = smoothed_curves[smoothed_curves["normalization"] == "raw"].copy()
    colors = {"pooled": "#264653", "female": "#d1495b", "male": "#2563eb"}
    metrics = ("median", "iqr", "cv")
    for name in review_targets:
        fig, axes = plt.subplots(1, 3, figsize=(13, 4), sharex=True)
        any_plotted = False
        for ax, metric in zip(axes, metrics):
            for cohort in COHORTS:
                sub = use[(use["test_name"] == name) & (use["metric"] == metric) & (use["cohort"] == cohort)]
                if sub.empty or sub["value"].notna().sum() == 0:
                    continue
                ax.plot(sub["age"], sub["value"], color=colors[cohort], lw=2, label=cohort)
                any_plotted = True
            ax.set_title(SUMMARY_METRIC_Y_LABELS[metric])
        if not any_plotted:
            plt.close(fig)
            continue
        axes[0].legend(loc="best", fontsize=8)
        fig.suptitle(f"{name}: cohort comparison")
        fig.tight_layout()
        fig.savefig(out_dir / f"{slugify(name)}_cohort_comparison.png", dpi=160)
        plt.close(fig)


def _format_cluster_combo_rows(df: pd.DataFrame, limit: int = 4) -> str:
    if df.empty:
        return "None"
    labels = [
        f"{row.branch} / {row.cohort} / {row.normalization}"
        + (f" (silhouette={row.best_silhouette:.2f}, K={int(row.best_k)})" if pd.notna(row.best_silhouette) and pd.notna(row.best_k) else "")
        for row in df.head(limit).itertuples(index=False)
    ]
    return "; ".join(labels)


def write_qc_readme(out_dir: Path, manifest: pd.DataFrame, review_targets: list[str], cluster_summary_df: pd.DataFrame) -> None:
    included = manifest[manifest["included_any_cohort"]]
    dropped = manifest[manifest["included_from_catalog"] & ~manifest["included_any_cohort"]]
    missing_reviews = [name for name in review_targets if name not in included["test_name"].tolist()]
    text = f"""# QC Notes

## What looked plausible
- Included biomarkers with at least one analyzable cohort: {len(included)}
- The pipeline used the mandatory within-bin 10-90 trim before every downstream step.
- This folder is limited to run-integrity checks rather than exploratory branch outputs.
- Branch-specific FPCA, clustering, rho-PCA, and Wasserstein interpretation artifacts now live under `summary_stats/` or `full_distribution/`.

## What looked unstable
- Catalog biomarkers matched to the NHANES workspace but dropped after trimming / bin thresholds: {len(dropped)}
- Review biomarkers unavailable in the final analyzable set: {", ".join(missing_reviews) if missing_reviews else "None"}

## Caution flags
- Biomarkers with very sparse age-bin coverage after trimming should be treated cautiously.
- `young_z_log` and `young_log_fold` outputs exist only where every retained value stayed strictly positive and the young-bin SD was non-zero.
- Review the analysis-arm folders for interpretation plots and downstream exploratory summaries.
"""
    (out_dir / "README.md").write_text(text)


def collect_file_inventory(out_dir: Path) -> list[str]:
    return [str(path.relative_to(out_dir)) for path in sorted(out_dir.rglob("*")) if path.is_file()]


def remove_path(path: Path) -> None:
    if path.is_dir():
        shutil.rmtree(path)
    elif path.exists():
        path.unlink()


def cleanup_legacy_output_layout(out_dir: Path) -> None:
    legacy_paths = [
        out_dir / "summary_stats" / "outputs",
        out_dir / "qc" / "summary_metric_panels",
        out_dir / "qc" / "normalization_panels",
        out_dir / "qc" / "cohort_comparison",
        out_dir / "qc" / "fpca",
        out_dir / "qc" / "wasserstein",
        out_dir / "qc" / "clustering_summary.csv",
        out_dir / "qc" / "cluster_size_summary.csv",
        out_dir / "qc" / "clustering_silhouette_by_k.csv",
        out_dir / "qc" / "wasserstein_sanity.csv",
        out_dir / "qc" / "fpca_explained_variance_summary.csv",
    ]
    for path in legacy_paths:
        remove_path(path)


def write_branch_clustering_overview(
    branch_dir: Path,
    cluster_summary_df: pd.DataFrame,
    cluster_sizes_df: pd.DataFrame,
    cluster_silhouette_df: pd.DataFrame,
) -> None:
    cluster_dir = branch_dir / "clustering"
    ensure_dir(cluster_dir)
    cluster_summary_df.to_csv(cluster_dir / "summary.csv", index=False)
    cluster_sizes_df.to_csv(cluster_dir / "cluster_sizes.csv", index=False)
    cluster_silhouette_df.to_csv(cluster_dir / "silhouette_by_k.csv", index=False)


def build_summary_review_outputs(
    summary_stats: pd.DataFrame,
    smoothed_curves: pd.DataFrame,
    fpca_ev: dict[tuple[str, str, str], pd.DataFrame],
    fpca_scores: dict[tuple[str, str, str], pd.DataFrame],
    review_targets: list[str],
    out_dir: Path,
) -> None:
    review_dir = out_dir / "review"
    ensure_dir(review_dir)
    plot_summary_metric_panels(summary_stats, smoothed_curves, review_targets, review_dir / "summary_metric_panels")
    plot_normalization_panels(smoothed_curves, review_targets, review_dir / "normalization_panels")
    plot_cohort_comparison_panels(smoothed_curves, review_targets, review_dir / "cohort_comparison")
    plot_fpca_outputs(fpca_ev, fpca_scores, out_dir / "fpca")


def build_distribution_review_outputs(
    pairwise_matrices: dict[tuple[str, str], pd.DataFrame],
    out_dir: Path,
) -> None:
    wasserstein_dir = out_dir / "wasserstein"
    ensure_dir(wasserstein_dir)
    plot_wasserstein_heatmaps(pairwise_matrices, wasserstein_dir / "heatmaps")


def build_qc_outputs(
    raw_rows: pd.DataFrame,
    trimmed_rows: pd.DataFrame,
    trim_summary: pd.DataFrame,
    manifest: pd.DataFrame,
    review_targets: list[str],
    out_dir: Path,
) -> None:
    ensure_dir(out_dir)
    manifest.to_csv(out_dir / "analysis_manifest_snapshot.csv", index=False)
    trim_summary.to_csv(out_dir / "trim_summary_by_age_bin.csv", index=False)
    trim_agg = (
        trim_summary.groupby(["analysis_id", "test_name", "cohort"], observed=True)[["raw_n", "kept_n", "removed_n"]]
        .sum()
        .reset_index()
    )
    trim_agg["removed_pct"] = np.where(trim_agg["raw_n"] > 0, trim_agg["removed_n"] / trim_agg["raw_n"], np.nan)
    trim_agg.to_csv(out_dir / "trim_removal_summary.csv", index=False)

    inclusion_counts = []
    for cohort in COHORTS:
        inclusion_counts.append(
            {
                "cohort": cohort,
                "included_biomarkers": int(manifest[f"included_{cohort}"].sum()),
                "dropped_after_trim_or_bins": int((manifest["included_from_catalog"] & ~manifest[f"included_{cohort}"]).sum()),
            }
        )
    pd.DataFrame(inclusion_counts).to_csv(out_dir / "biomarker_inclusion_counts.csv", index=False)

    validity_rows = []
    for cohort in COHORTS:
        cohort_manifest = manifest[manifest[f"included_{cohort}"]].copy()
        row = {
            "cohort": cohort,
            "included_biomarkers": len(cohort_manifest),
        }
        for normalization in CURVE_NORMALIZATIONS:
            row[f"summary_{normalization}_biomarkers"] = int(
                (cohort_manifest[f"summary_metrics_available_{normalization}_{cohort}"] > 0).sum()
            )
        row["summary_young_log_fold_all_6_metrics"] = int(
            (cohort_manifest[f"summary_metrics_available_young_log_fold_{cohort}"] == len(SUMMARY_METRICS)).sum()
        )
        row["distribution_young_z_raw_biomarkers"] = int(cohort_manifest[f"distribution_valid_young_z_raw_{cohort}"].sum())
        row["distribution_young_z_log_biomarkers"] = int(cohort_manifest[f"distribution_valid_young_z_log_{cohort}"].sum())
        validity_rows.append(row)
    pd.DataFrame(validity_rows).to_csv(out_dir / "normalization_validity_counts.csv", index=False)

    plot_before_after_trim(raw_rows, trimmed_rows, review_targets, out_dir / "trim_distributions")
    write_qc_readme(out_dir, manifest, review_targets, pd.DataFrame())


def run_analysis(
    catalog_path: Path,
    long_path: Path,
    out_dir: Path,
    *,
    review_targets: list[str] | None = None,
    max_biomarkers: int | None = None,
    skip_qc: bool = False,
    skip_explorer: bool = False,
    clalit_quartiles_path: Path | None = None,
    clalit_f_path: Path | None = None,
    clalit_m_path: Path | None = None,
    clalit_map_path: Path | None = None,
) -> dict[str, object]:
    review_targets = review_targets or DEFAULT_REVIEW
    ensure_dir(out_dir)
    ensure_dir(out_dir / "summary_stats")
    ensure_dir(out_dir / "full_distribution")
    ensure_dir(out_dir / "qc")
    ensure_dir(out_dir / "explorer")
    cleanup_legacy_output_layout(out_dir)

    long_df = pd.read_parquet(long_path)
    long_df = long_df[(long_df["age_years"] >= 20) & (long_df["age_years"] < 85)].copy()
    targets, manifest = load_targets(catalog_path, long_df, max_biomarkers=max_biomarkers)
    targets_df = pd.DataFrame([asdict(t) for t in targets])

    raw_rows = build_analysis_rows(long_df, targets)
    all_trimmed = []
    all_trim_summaries = []
    for cohort in COHORTS:
        trimmed, trim_summary = trim_within_bins(raw_rows, cohort)
        all_trimmed.append(trimmed)
        all_trim_summaries.append(trim_summary)
    trimmed_rows = pd.concat(all_trimmed, ignore_index=True) if all_trimmed else pd.DataFrame()
    trim_summary_df = pd.concat(all_trim_summaries, ignore_index=True) if all_trim_summaries else pd.DataFrame()

    summary_stats = compute_summary_stats(trimmed_rows)
    summary_stats.to_parquet(out_dir / "summary_stats" / "binned_stats_long.parquet", index=False)

    manifest = summarize_cohort_validity(summary_stats, manifest)

    smoothed_curves, fpca_scores, fpca_ev, rho_diag_df, summary_cluster_summary, summary_cluster_sizes, summary_cluster_silhouette = build_summary_branch(
        summary_stats[summary_stats["analysis_id"].isin(manifest.loc[manifest["included_any_cohort"], "analysis_id"])].copy(),
        targets_df,
        out_dir / "summary_stats",
    )
    rho_pca_scores, rho_pca_explained, rho_pca_loadings = build_summary_rho_pca_outputs(
        out_dir / "summary_stats",
        rho_diag_df,
    )

    reference_df, feature_curves_df, pairwise_matrices, distribution_cluster_summary, distribution_cluster_sizes, distribution_cluster_silhouette = build_distribution_branch(
        trimmed_rows[trimmed_rows["analysis_id"].isin(manifest.loc[manifest["included_any_cohort"], "analysis_id"])].copy(),
        manifest,
        out_dir / "full_distribution",
    )
    cluster_summary_df = concat_nonempty_frames([summary_cluster_summary, distribution_cluster_summary])
    cluster_sizes_df = concat_nonempty_frames([summary_cluster_sizes, distribution_cluster_sizes])
    cluster_silhouette_df = concat_nonempty_frames([summary_cluster_silhouette, distribution_cluster_silhouette])
    write_branch_clustering_overview(
        out_dir / "summary_stats",
        summary_cluster_summary,
        summary_cluster_sizes,
        summary_cluster_silhouette,
    )
    write_branch_clustering_overview(
        out_dir / "full_distribution",
        distribution_cluster_summary,
        distribution_cluster_sizes,
        distribution_cluster_silhouette,
    )

    manifest = annotate_normalization_validity(manifest, smoothed_curves, reference_df)
    if not targets_df.empty:
        targets_df = targets_df.merge(
            manifest[
                [
                    "analysis_id",
                    "included_any_cohort",
                    "included_pooled",
                    "included_female",
                    "included_male",
                    "drop_reason",
                ]
            ],
            on="analysis_id",
            how="left",
        )
    manifest.to_csv(out_dir / "analysis_manifest.csv", index=False)

    write_root_readme(out_dir)
    write_branch_readmes(out_dir)
    write_pipeline_md(out_dir)
    build_summary_review_outputs(
        summary_stats,
        smoothed_curves,
        fpca_ev,
        fpca_scores,
        review_targets,
        out_dir / "summary_stats",
    )
    build_distribution_review_outputs(pairwise_matrices, out_dir / "full_distribution")

    if not skip_qc:
        build_qc_outputs(
            raw_rows,
            trimmed_rows,
            trim_summary_df,
            manifest,
            review_targets,
            out_dir / "qc",
        )

    explorer_result: dict[str, object] | None = None
    if not skip_explorer:
        from build_aging_biomarker_explorer import build_aging_biomarker_explorer

        explorer_result = build_aging_biomarker_explorer(out_dir)

    clalit_result: dict[str, object] | None = None
    has_clalit_quartiles = bool(clalit_quartiles_path and Path(clalit_quartiles_path).exists())
    has_clalit_legacy = bool(
        clalit_f_path and clalit_m_path and Path(clalit_f_path).exists() and Path(clalit_m_path).exists()
    )
    if clalit_map_path and Path(clalit_map_path).exists():
        if has_clalit_quartiles or has_clalit_legacy:
            clalit_result = run_clalit_summary_analysis(
                catalog_path,
                out_dir / "clalit",
                clalit_quartiles_path=Path(clalit_quartiles_path) if clalit_quartiles_path else None,
                clalit_f_path=Path(clalit_f_path) if clalit_f_path else None,
                clalit_m_path=Path(clalit_m_path) if clalit_m_path else None,
                clalit_map_path=Path(clalit_map_path),
            )

    write_combined_rho_pca_dashboard(
        out_dir / "explorer" / "rho_pca_explorer.html",
        nhanes_rel_path="../summary_stats/rho_pca/rho_pca_explorer.html",
        clalit_rel_path="../clalit/summary_stats/rho_pca/rho_pca_explorer.html" if clalit_result else None,
    )

    run_manifest = {
        "catalog_path": str(catalog_path),
        "long_path": str(long_path),
        "output_root": str(out_dir),
        "cohorts": list(COHORTS),
        "age_bins": AGE_BIN_LABELS,
        "age_grid_start": int(AGE_GRID.min()),
        "age_grid_end": int(AGE_GRID.max()),
        "trim_rule": {"within_age_bin": True, "lo": TRIM_LO, "hi": TRIM_HI},
        "smoothing_method": "weighted_gam_on_age_bins",
        "summary_metrics": list(SUMMARY_METRICS),
        "summary_rho_pca_features": list(SUMMARY_RHO_COLUMNS),
        "summary_rho_pca_methods": {
            key: {
                "label": spec["label"],
                "description": spec["description"],
                "features": [feature_id for _, feature_id, _, _ in spec["features"]],
            }
            for key, spec in SUMMARY_RHO_PCA_METHODS.items()
        },
        "curve_normalizations": list(CURVE_NORMALIZATIONS),
        "distribution_normalizations": list(DISTRIBUTION_NORMALIZATIONS),
        "distribution_features": list(DISTRIBUTION_FEATURES),
        "clustering_method": "hierarchical_ward",
        "cluster_selection_rule": "silhouette_best_k_2_8",
        "fpca_retention_rule": "keep_pc3_if_ev_ge_0.10",
        "pca_visualization_components": PCA_VIS_COMPONENTS,
        "explorer_output_path": str(out_dir / "explorer" / "aging_biomarker_explorer.html") if explorer_result else None,
        "rho_pca_combined_output_path": str(out_dir / "explorer" / "rho_pca_explorer.html"),
        "clalit_output_path": str(out_dir / "clalit") if clalit_result else None,
        "explorer_build_mode": "standalone_html_embedded_json" if explorer_result else None,
        "explorer_tabs": ["Overview", "Included Tests", "Summary Clustering", "Full-Distribution Clustering", "Family FPCA", "Summary-Rho PCA"] if explorer_result else [],
        "explorer_plotly_delivery": "inline_embedded_js" if explorer_result else None,
        "minimum_bin_n": MIN_BIN_N,
        "minimum_valid_bins": MIN_VALID_BINS,
        "review_targets": review_targets,
    }
    (out_dir / "run_manifest.json").write_text(json.dumps(run_manifest, indent=2))
    run_manifest["file_inventory"] = collect_file_inventory(out_dir)
    (out_dir / "run_manifest.json").write_text(json.dumps(run_manifest, indent=2))

    return {
        "manifest": manifest,
        "summary_stats": summary_stats,
        "smoothed_curves": smoothed_curves,
        "rho_pca_scores": rho_pca_scores,
        "rho_pca_explained_variance": rho_pca_explained,
        "rho_pca_loadings": rho_pca_loadings,
        "reference_stats": reference_df,
        "feature_curves": feature_curves_df,
        "clustering_summary": cluster_summary_df,
        "explorer": explorer_result,
        "clalit": clalit_result,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--catalog", type=Path, default=DEFAULT_CATALOG)
    parser.add_argument("--input", dest="input_path", type=Path, default=DEFAULT_LONG)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--review-biomarkers", default=",".join(DEFAULT_REVIEW))
    parser.add_argument("--max-biomarkers", type=int, default=None)
    parser.add_argument("--skip-qc", action="store_true")
    parser.add_argument("--skip-explorer", action="store_true")
    parser.add_argument("--clalit-quartiles", type=Path, default=DEFAULT_CLALIT_QUARTILES)
    parser.add_argument("--clalit-f", type=Path, default=DEFAULT_CLALIT_F)
    parser.add_argument("--clalit-m", type=Path, default=DEFAULT_CLALIT_M)
    parser.add_argument("--clalit-map", type=Path, default=DEFAULT_CLALIT_MAP)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    review_targets = [item.strip() for item in str(args.review_biomarkers).split(",") if item.strip()]
    result = run_analysis(
        args.catalog,
        args.input_path,
        args.out,
        review_targets=review_targets,
        max_biomarkers=args.max_biomarkers,
        skip_qc=args.skip_qc,
        skip_explorer=args.skip_explorer,
        clalit_quartiles_path=args.clalit_quartiles,
        clalit_f_path=args.clalit_f,
        clalit_m_path=args.clalit_m,
        clalit_map_path=args.clalit_map,
    )
    manifest = result["manifest"]
    included = int(manifest["included_any_cohort"].sum())
    print(f"Targets in catalog: {len(manifest):,}")
    print(f"Analyzable targets in at least one cohort: {included:,}")
    print(f"Output root: {args.out}")


if __name__ == "__main__":
    main()
