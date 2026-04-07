#!/usr/bin/env python3
"""Aggregate Clalit ridgeline densities into NHANES-style age-bin quartiles."""

from __future__ import annotations

import argparse
import json
import math
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from nhanes_common import ensure_dir, parse_unit_from_label


ROOT = Path(__file__).resolve().parents[1]

DEFAULT_RIDGELINE_F = ROOT / "data" / "clalit" / "all_ridgeline_data_females.csv"
DEFAULT_RIDGELINE_M = ROOT / "data" / "clalit" / "all_ridgeline_data_males.csv"
DEFAULT_STATS_F = ROOT / "data" / "clalit" / "females_all_statistics.csv"
DEFAULT_STATS_M = ROOT / "data" / "clalit" / "males_all_statistics.csv"
DEFAULT_AVAILABILITY = ROOT / "data" / "data_availability.csv"
DEFAULT_MAPPING = ROOT / "data" / "clalit_mapping.json"
DEFAULT_OUT = ROOT / "data" / "clalit" / "clalit_quartiles.csv"

RAW_SCALE = "regular"
LOG_SCALE = "log"
SCALE_ORDER = [RAW_SCALE, LOG_SCALE]


@dataclass(frozen=True)
class AgeBin:
    label: str
    lo: int
    hi: int
    mid: float


AGE_BINS = [
    AgeBin(label=f"{start}-{min(start + 4, 99)}", lo=start, hi=min(start + 5, 100), mid=(start + min(start + 4, 99)) / 2.0)
    for start in range(20, 100, 5)
]
AGE_BIN_BY_YEAR = {age: age_bin for age_bin in AGE_BINS for age in range(age_bin.lo, age_bin.hi)}

UNIT_TOKEN_NORMALIZATION = {
    "iu/l": "IU/L",
    "u/l": "U/L",
    "miu/l": "mIU/L",
    "uiu/ml": "uIU/mL",
    "uu/ml": "uIU/mL",
    "pg/ml": "pg/mL",
    "ng/ml": "ng/mL",
    "ug/ml": "ug/mL",
    "pg/dl": "pg/dL",
    "ng/dl": "ng/dL",
    "ug/dl": "ug/dL",
    "mg/dl": "mg/dL",
    "mg/l": "mg/L",
    "g/dl": "g/dL",
    "mmol/l": "mmol/L",
    "umol/l": "umol/L",
    "nmol/l": "nmol/L",
    "pmol/l": "pmol/L",
    "ug/l": "ug/L",
    "pg/l": "pg/L",
    "fl": "fL",
    "mm/hr": "mm/h",
    "mm/h": "mm/h",
    "%": "%",
}

EXPLICIT_UNIT_OVERRIDES: dict[str, dict[str, str]] = {
    "lab.101.no_meds": {"unit": "10^3/uL", "source": "manual_clinical_override", "confidence": "medium"},
    "lab.102.no_meds": {"unit": "10^6/uL", "source": "manual_clinical_override", "confidence": "medium"},
    "lab.103.no_meds": {"unit": "g/dL", "source": "manual_clinical_override", "confidence": "high"},
    "lab.104.no_meds": {"unit": "%", "source": "manual_clinical_override", "confidence": "high"},
    "lab.105.no_meds": {"unit": "10^3/uL", "source": "manual_clinical_override", "confidence": "medium"},
    "lab.106.no_meds": {"unit": "fL", "source": "manual_clinical_override", "confidence": "high"},
    "lab.107.no_meds": {"unit": "pg", "source": "manual_clinical_override", "confidence": "high"},
    "lab.108.no_meds": {"unit": "g/dL", "source": "manual_clinical_override", "confidence": "high"},
    "lab.109.no_meds": {"unit": "%", "source": "manual_clinical_override", "confidence": "medium"},
    "lab.110.no_meds": {"unit": "fL", "source": "manual_clinical_override", "confidence": "high"},
    "lab.111.no_meds": {"unit": "ng/mL", "source": "manual_clinical_override", "confidence": "medium"},
    "lab.112.no_meds": {"unit": "fL", "source": "manual_clinical_override", "confidence": "medium"},
    "lab.117.no_meds": {"unit": "g/dL", "source": "manual_clinical_override", "confidence": "low"},
    "lab.122.no_meds": {"unit": "%", "source": "pattern_percent", "confidence": "high"},
    "lab.123.no_meds": {"unit": "10^3/uL", "source": "manual_clinical_override", "confidence": "low"},
    "lab.124.no_meds": {"unit": "index", "source": "manual_clinical_override", "confidence": "medium"},
    "lab.126.no_meds": {"unit": "%", "source": "pattern_percent", "confidence": "high"},
    "lab.127.no_meds": {"unit": "%", "source": "pattern_percent", "confidence": "high"},
    "lab.128.no_meds": {"unit": "%", "source": "pattern_percent", "confidence": "high"},
    "lab.129.no_meds": {"unit": "%", "source": "pattern_percent", "confidence": "high"},
    "lab.140.no_meds": {"unit": "10^3/uL", "source": "manual_clinical_override", "confidence": "medium"},
    "lab.141.no_meds": {"unit": "%", "source": "pattern_percent", "confidence": "high"},
    "lab.142.no_meds": {"unit": "10^3/uL", "source": "manual_clinical_override", "confidence": "medium"},
    "lab.143.no_meds": {"unit": "%", "source": "pattern_percent", "confidence": "high"},
    "lab.144.no_meds": {"unit": "10^3/uL", "source": "manual_clinical_override", "confidence": "medium"},
    "lab.145.no_meds": {"unit": "%", "source": "pattern_percent", "confidence": "high"},
    "lab.146.no_meds": {"unit": "10^3/uL", "source": "manual_clinical_override", "confidence": "medium"},
    "lab.147.no_meds": {"unit": "%", "source": "pattern_percent", "confidence": "high"},
    "lab.148.no_meds": {"unit": "10^3/uL", "source": "manual_clinical_override", "confidence": "medium"},
    "lab.149.no_meds": {"unit": "%", "source": "pattern_percent", "confidence": "high"},
    "lab.150.no_meds": {"unit": "ratio", "source": "manual_clinical_override", "confidence": "high"},
    "lab.151.no_meds": {"unit": "fL", "source": "manual_clinical_override", "confidence": "high"},
    "lab.152.no_meds": {"unit": "%", "source": "manual_clinical_override", "confidence": "high"},
    "lab.158.no_meds": {"unit": "%", "source": "pattern_percent", "confidence": "high"},
    "lab.166.no_meds": {"unit": "pg", "source": "manual_clinical_override", "confidence": "medium"},
    "lab.172.no_meds": {"unit": "ratio", "source": "manual_clinical_override", "confidence": "medium"},
    "lab.17601.no_meds": {"unit": "INR", "source": "manual_clinical_override", "confidence": "high"},
    "lab.17602.no_meds": {"unit": "sec", "source": "manual_clinical_override", "confidence": "high"},
    "lab.17603.no_meds": {"unit": "%", "source": "pattern_percent", "confidence": "high"},
    "lab.17702.no_meds": {"unit": "sec", "source": "manual_clinical_override", "confidence": "high"},
    "lab.17703.no_meds": {"unit": "ratio", "source": "manual_clinical_override", "confidence": "high"},
    "lab.20008.no_meds": {"unit": "sec", "source": "manual_clinical_override", "confidence": "medium"},
    "lab.21000.no_meds": {"unit": "g/dL", "source": "manual_clinical_override", "confidence": "high"},
    "lab.21100.no_meds": {"unit": "g/dL", "source": "manual_clinical_override", "confidence": "high"},
    "lab.21200.no_meds": {"unit": "mg/dL", "source": "manual_clinical_override", "confidence": "high"},
    "lab.21400.no_meds": {"unit": "mg/dL", "source": "manual_clinical_override", "confidence": "high"},
    "lab.21500.no_meds": {"unit": "mg/dL", "source": "manual_clinical_override", "confidence": "high"},
    "lab.22500.no_meds": {"unit": "IU/L", "source": "manual_clinical_override", "confidence": "high"},
    "lab.23000.no_meds": {"unit": "U/L", "source": "manual_clinical_override", "confidence": "high"},
    "lab.24500.no_meds": {"unit": "mg/dL", "source": "manual_clinical_override", "confidence": "high"},
    "lab.25203.no_meds": {"unit": "%", "source": "manual_clinical_override", "confidence": "high"},
    "lab.33300.no_meds": {"unit": "ratio", "source": "manual_clinical_override", "confidence": "high"},
    "lab.35000.no_meds": {"unit": "mg/g creatinine", "source": "manual_clinical_override", "confidence": "medium"},
    "lab.36400.no_meds": {"unit": "g/dL", "source": "manual_clinical_override", "confidence": "high"},
    "lab.400.no_meds": {"unit": "mm/h", "source": "manual_clinical_override", "confidence": "high"},
    "lab.41500.no_meds": {"unit": "mg/dL", "source": "manual_clinical_override", "confidence": "high"},
    "lab.46700.no_meds": {"unit": "mg/L", "source": "manual_clinical_override", "confidence": "medium"},
    "lab.520706.no_meds": {"unit": "mg/dL", "source": "manual_clinical_override", "confidence": "high"},
    "lab.60200.no_meds": {"unit": "mg/L", "source": "manual_clinical_override", "confidence": "high"},
    "lab.101000.no_meds": {"unit": "mIU/L", "source": "manual_clinical_override", "confidence": "high"},
    "lab.101500.no_meds": {"unit": "umol/L", "source": "manual_clinical_override", "confidence": "high"},
    "lab.101600.no_meds": {"unit": "nmol/L", "source": "manual_clinical_override", "confidence": "medium"},
    "lab.101701.no_meds": {"unit": "ug/24h", "source": "manual_clinical_override", "confidence": "low"},
    "lab.101702.no_meds": {"unit": "", "source": "manual_clinical_override", "confidence": "low"},
    "lab.101800.no_meds": {"unit": "uIU/mL", "source": "manual_clinical_override", "confidence": "high"},
    "lab.102100.no_meds": {"unit": "mIU/L", "source": "manual_clinical_override", "confidence": "high"},
    "lab.102200.no_meds": {"unit": "nmol/L", "source": "manual_clinical_override", "confidence": "high"},
    "lab.102300.no_meds": {"unit": "pmol/L", "source": "manual_clinical_override", "confidence": "high"},
    "lab.102500.no_meds": {"unit": "pmol/L", "source": "clalit_scale_reason", "confidence": "high"},
    "lab.102600.no_meds": {"unit": "ug/L", "source": "manual_clinical_override", "confidence": "medium"},
    "lab.102800.no_meds": {"unit": "nmol/L", "source": "manual_clinical_override", "confidence": "high"},
    "lab.103000.no_meds": {"unit": "U/L", "source": "manual_clinical_override", "confidence": "high"},
    "lab.103200.no_meds": {"unit": "pmol/L", "source": "manual_clinical_override", "confidence": "medium"},
    "lab.103500.no_meds": {"unit": "pg/mL", "source": "manual_clinical_override", "confidence": "high"},
    "lab.104300.no_meds": {"unit": "", "source": "manual_clinical_override", "confidence": "low"},
    "lab.104400.no_meds": {"unit": "pg/mL", "source": "manual_clinical_override", "confidence": "medium"},
    "lab.100500.no_meds": {"unit": "pmol/L", "source": "manual_clinical_override", "confidence": "high"},
    "lab.100600.no_meds": {"unit": "nmol/L", "source": "manual_clinical_override", "confidence": "high"},
    "lab.100700.no_meds": {"unit": "nmol/L", "source": "manual_clinical_override", "confidence": "high"},
    "lab.100800.no_meds": {"unit": "IU/L", "source": "manual_clinical_override", "confidence": "high"},
    "lab.100900.no_meds": {"unit": "IU/L", "source": "manual_clinical_override", "confidence": "high"},
    "lab.101100.no_meds": {"unit": "nmol/L", "source": "manual_clinical_override", "confidence": "high"},
    "lab.101300.no_meds": {"unit": "nmol/L", "source": "manual_clinical_override", "confidence": "high"},
    "lab.31400.no_meds": {"unit": "nmol/L", "source": "nhanes_mapping_label", "confidence": "medium"},
    "lab.9904.no_meds": {"unit": "specific gravity", "source": "manual_clinical_override", "confidence": "high"},
    "lab.9907.no_meds": {"unit": "", "source": "manual_clinical_override", "confidence": "low"},
    "lab.9909.no_meds": {"unit": "", "source": "manual_clinical_override", "confidence": "low"},
    "lab.9911.no_meds": {"unit": "pH", "source": "manual_clinical_override", "confidence": "high"},
    "marker.BMI.no_meds": {"unit": "kg/m^2", "source": "manual_marker_convention", "confidence": "high"},
    "marker.bp.high.no_meds": {"unit": "mmHg", "source": "manual_marker_convention", "confidence": "high"},
    "marker.bp.low.no_meds": {"unit": "mmHg", "source": "manual_marker_convention", "confidence": "high"},
    "marker.height.no_meds": {"unit": "m", "source": "manual_marker_convention", "confidence": "high"},
    "marker.weight.no_meds": {"unit": "kg", "source": "manual_marker_convention", "confidence": "high"},
}


def age_bin_for_year(age: int) -> AgeBin | None:
    return AGE_BIN_BY_YEAR.get(int(age))


def parse_age_from_group(value: Any) -> int | None:
    text = str(value or "").strip()
    if not text:
        return None
    head = text.split("-", 1)[0]
    if not head.isdigit():
        return None
    return int(head)


def clean_unit_text(value: str) -> str:
    text = str(value or "").strip()
    if not text:
        return ""

    text = text.replace("μ", "u").replace("µ", "u")
    text = re.sub(r"\s+", "", text)
    token = text.lower()
    return UNIT_TOKEN_NORMALIZATION.get(token, value.strip())


def parse_unit_from_nhanes_name(label: Any) -> str:
    text = str(label or "").strip()
    if not text:
        return ""

    paren_matches = re.findall(r"\(([^)]+)\)", text)
    for candidate in reversed(paren_matches):
        unit = clean_unit_text(candidate)
        if unit in UNIT_TOKEN_NORMALIZATION.values():
            return unit

    if "__" in text:
        candidate = clean_unit_text(text.split("__")[-1])
        if candidate in UNIT_TOKEN_NORMALIZATION.values():
            return candidate

    m = re.search(
        r"(?i)(mIU/L|uIU/mL|IU/L|U/L|pg/mL|ng/mL|ug/mL|pg/dL|ng/dL|ug/dL|mg/dL|mg/L|g/dL|mmol/L|umol/L|nmol/L|pmol/L|ug/L|fL|mm/h|sec|%)\s*$",
        text,
    )
    if m:
        return clean_unit_text(m.group(1))

    return ""


def unit_override_from_mapping_reason(test: str, mapping: dict[str, Any]) -> dict[str, str] | None:
    payload = mapping.get(test)
    if not isinstance(payload, dict):
        return None

    reason = str(payload.get("scale_reason") or "")
    m = re.search(r"stored in ([A-Za-z/%^0-9.]+(?:/[A-Za-z0-9^]+)?)", reason)
    if not m:
        return None

    unit = clean_unit_text(m.group(1))
    if not unit:
        return None

    return {"unit": unit, "source": "clalit_scale_reason", "confidence": "high"}


def pattern_unit_from_row(row: pd.Series) -> dict[str, str] | None:
    long_name = str(row.get("long_name") or "")
    test_name = str(row.get("test_name") or "")
    short_name = str(row.get("short_name") or "")

    text = " ".join([long_name, test_name, short_name]).lower()
    if "%" in long_name or "%" in short_name or "percent" in text:
        return {"unit": "%", "source": "pattern_percent", "confidence": "high"}
    if "ratio" in text:
        return {"unit": "ratio", "source": "pattern_ratio", "confidence": "high"}
    if re.search(r"\bsec\b", text):
        return {"unit": "sec", "source": "pattern_seconds", "confidence": "high"}
    return None


def build_unit_metadata(tests_df: pd.DataFrame, availability_path: Path, mapping_path: Path) -> pd.DataFrame:
    availability = pd.read_csv(availability_path)
    availability = availability.rename(columns={"Clalit ID": "test", "Test Name (NHANES)": "nhanes_name"})
    availability = availability[["test", "nhanes_name"]].drop_duplicates()

    mapping = json.loads(mapping_path.read_text(encoding="utf-8"))

    merged = tests_df.merge(availability, on="test", how="left")

    units: list[dict[str, str]] = []
    for row in merged.itertuples(index=False):
        row_series = pd.Series(row._asdict())
        explicit = EXPLICIT_UNIT_OVERRIDES.get(row.test)
        if explicit is not None:
            units.append(explicit)
            continue

        from_mapping = unit_override_from_mapping_reason(row.test, mapping)
        if from_mapping is not None:
            units.append(from_mapping)
            continue

        pattern = pattern_unit_from_row(row_series)
        if pattern is not None:
            units.append(pattern)
            continue

        parsed = parse_unit_from_nhanes_name(getattr(row, "nhanes_name", ""))
        if parsed:
            units.append({"unit": parsed, "source": "nhanes_mapping_label", "confidence": "medium"})
            continue

        units.append({"unit": "", "source": "unknown", "confidence": "low"})

    unit_df = pd.DataFrame(units)
    return pd.concat([merged.reset_index(drop=True), unit_df], axis=1)


def density_curve_to_cdf(axis: np.ndarray, density: np.ndarray) -> tuple[np.ndarray, np.ndarray] | None:
    x = np.asarray(axis, dtype=float)
    y = np.asarray(density, dtype=float)

    mask = np.isfinite(x) & np.isfinite(y)
    x = x[mask]
    y = y[mask]
    if x.size < 2:
        return None

    order = np.argsort(x)
    x = x[order]
    y = y[order]

    dx = np.diff(x)
    valid_step = dx > 0
    if not valid_step.any():
        return None

    keep = np.concatenate([[True], valid_step])
    x = x[keep]
    y = y[keep]
    if x.size < 2:
        return None

    increments = np.diff(x) * (y[:-1] + y[1:]) / 2.0
    total_area = float(increments.sum())
    if not np.isfinite(total_area) or total_area <= 0:
        return None

    cdf = np.concatenate([[0.0], np.cumsum(increments) / total_area])
    cdf[-1] = 1.0
    return x, cdf


def quantiles_from_cdf(grid: np.ndarray, cdf: np.ndarray, probabilities: np.ndarray) -> np.ndarray:
    x = np.asarray(grid, dtype=float)
    y = np.asarray(cdf, dtype=float)
    p = np.asarray(probabilities, dtype=float)

    y = np.maximum.accumulate(y)
    y[0] = 0.0
    y[-1] = 1.0

    return np.interp(p, y, x)


def combine_age_curves(
    age_curves: list[tuple[np.ndarray, np.ndarray, float]],
) -> tuple[np.ndarray, np.ndarray] | None:
    if not age_curves:
        return None

    grid = np.unique(np.concatenate([axis for axis, _, _ in age_curves]))
    if grid.size < 2:
        return None

    total_weight = float(sum(weight for _, _, weight in age_curves))
    if total_weight <= 0:
        return None

    mixed = np.zeros(grid.shape[0], dtype=float)
    for axis, cdf, weight in age_curves:
        mixed += weight * np.interp(grid, axis, cdf, left=0.0, right=1.0)

    mixed /= total_weight
    mixed[0] = 0.0
    mixed[-1] = 1.0
    mixed = np.maximum.accumulate(mixed)
    return grid, mixed


def regular_or_log_axis(curve: pd.DataFrame, scale_type: str) -> np.ndarray:
    if scale_type == LOG_SCALE:
        return curve["log_x"].to_numpy(dtype=float)
    return curve["x"].to_numpy(dtype=float)


def active_scale_min_max(
    scale_type: str,
    raw_min: float | None,
    raw_max: float | None,
    grid: np.ndarray,
) -> tuple[float | None, float | None]:
    if scale_type == RAW_SCALE:
        return raw_min, raw_max

    log_min = math.log(raw_min) if raw_min is not None and raw_min > 0 else None
    log_max = math.log(raw_max) if raw_max is not None and raw_max > 0 else None

    if log_min is None and grid.size:
        log_min = float(np.nanmin(grid))
    if log_max is None and grid.size:
        log_max = float(np.nanmax(grid))
    return log_min, log_max


def raw_support_min_max(scale_type: str, grid: np.ndarray) -> tuple[float | None, float | None]:
    if grid.size == 0:
        return None, None

    if scale_type == RAW_SCALE:
        return float(np.nanmin(grid)), float(np.nanmax(grid))

    return float(math.exp(np.nanmin(grid))), float(math.exp(np.nanmax(grid)))


def float_or_none(value: Any) -> float | None:
    if value is None:
        return None
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    if not np.isfinite(number):
        return None
    return number


def group_stats_for_bin(stats_slice: pd.DataFrame) -> tuple[float | None, float | None, float | None]:
    if stats_slice.empty:
        return None, None, None

    total_n = float_or_none(stats_slice["n"].sum())

    raw_min = float_or_none(stats_slice["min"].min())
    raw_max = float_or_none(stats_slice["max"].max())
    return total_n, raw_min, raw_max


def aggregate_one_bin(
    sex: str,
    test_meta: dict[str, Any],
    scale_type: str,
    age_bin: AgeBin,
    ridgeline_groups: dict[tuple[str, int, str], pd.DataFrame],
    stats_lookup: dict[tuple[str, int], dict[str, float]],
    stats_by_test_and_bin: dict[tuple[str, str], tuple[float | None, float | None, float | None]],
) -> dict[str, Any]:
    test = str(test_meta["test"])
    ages_in_bin = list(range(age_bin.lo, age_bin.hi))
    age_curves: list[tuple[np.ndarray, np.ndarray, float]] = []
    age_years_present: list[int] = []
    density_points_used = 0
    used_stats_weights = False
    used_equal_fallback = False

    for age in ages_in_bin:
        curve = ridgeline_groups.get((test, age, scale_type))
        if curve is None or curve.empty:
            continue

        axis = regular_or_log_axis(curve, scale_type)
        density = curve["y"].to_numpy(dtype=float)
        cdf_parts = density_curve_to_cdf(axis, density)
        if cdf_parts is None:
            continue

        stats_row = stats_lookup.get((test, age))
        if stats_row is not None and float_or_none(stats_row.get("n")) is not None and float(stats_row["n"]) > 0:
            weight = float(stats_row["n"])
            used_stats_weights = True
        else:
            weight = 1.0
            used_equal_fallback = True

        age_curves.append((cdf_parts[0], cdf_parts[1], weight))
        age_years_present.append(age)
        density_points_used += int(curve.shape[0])

    stats_n_total, raw_min, raw_max = stats_by_test_and_bin.get((test, age_bin.label), (None, None, None))

    row = {
        "sex": sex,
        "test": test,
        "test_name": test_meta["test_name"],
        "short_name": test_meta["short_name"],
        "long_name": test_meta["long_name"],
        "system": test_meta["system"],
        "unit": test_meta["unit"],
        "unit_source": test_meta["unit_source"],
        "unit_confidence": test_meta["unit_confidence"],
        "nhanes_name": test_meta["nhanes_name"],
        "age_bin": age_bin.label,
        "age_lo": age_bin.lo,
        "age_hi": age_bin.hi - 1,
        "age_mid": age_bin.mid,
        "scale_type": scale_type,
        "axis_column": "log_x" if scale_type == LOG_SCALE else "x",
        "is_log_scale": scale_type == LOG_SCALE,
        "stats_n_total": stats_n_total,
        "n_age_years_present": len(age_years_present),
        "age_years_present": "|".join(str(age) for age in age_years_present),
        "density_points_used": density_points_used,
        "weighting_method": "stats_n_with_equal_fallback"
        if used_stats_weights and used_equal_fallback
        else "stats_n"
        if used_stats_weights
        else "equal_by_age"
        if age_curves
        else "none",
        "has_density_data": bool(age_curves),
        "raw_min": raw_min,
        "raw_max": raw_max,
        "q0": np.nan,
        "q1": np.nan,
        "q2": np.nan,
        "q3": np.nan,
        "q4": np.nan,
        "raw_q0": np.nan,
        "raw_q1": np.nan,
        "raw_q2": np.nan,
        "raw_q3": np.nan,
        "raw_q4": np.nan,
    }

    combined = combine_age_curves(age_curves)
    if combined is None:
        return row

    grid, mixed_cdf = combined
    q25, q50, q75 = quantiles_from_cdf(grid, mixed_cdf, np.array([0.25, 0.50, 0.75], dtype=float))

    if scale_type == RAW_SCALE:
        raw_q1 = float(q25)
        raw_q2 = float(q50)
        raw_q3 = float(q75)
    else:
        raw_q1 = float(math.exp(q25))
        raw_q2 = float(math.exp(q50))
        raw_q3 = float(math.exp(q75))

    support_min, support_max = raw_support_min_max(scale_type, grid)
    raw_q0 = min(v for v in [raw_min, support_min, raw_q1] if v is not None)
    raw_q4 = max(v for v in [raw_max, support_max, raw_q3] if v is not None)

    active_min, active_max = active_scale_min_max(scale_type, raw_q0, raw_q4, grid)
    row["q0"] = active_min if active_min is not None else np.nan
    row["q1"] = float(q25)
    row["q2"] = float(q50)
    row["q3"] = float(q75)
    row["q4"] = active_max if active_max is not None else np.nan
    row["raw_q0"] = raw_q0
    row["raw_q1"] = raw_q1
    row["raw_q2"] = raw_q2
    row["raw_q3"] = raw_q3
    row["raw_q4"] = raw_q4

    return row


def prepare_ridgeline(path: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    ridgeline = pd.read_csv(path)
    ridgeline["age"] = ridgeline["age_group"].map(parse_age_from_group)
    ridgeline = ridgeline.dropna(subset=["age"]).copy()
    ridgeline["age"] = ridgeline["age"].astype(int)
    ridgeline["age_bin"] = ridgeline["age"].map(lambda age: (age_bin_for_year(age) or AgeBin("", -1, -1, np.nan)).label)
    ridgeline = ridgeline[ridgeline["age_bin"] != ""].copy()

    tests_df = ridgeline[["test", "test_name", "short_name", "long_name", "system"]].drop_duplicates().sort_values(
        ["system", "long_name", "test"]
    )
    return ridgeline, tests_df


def prepare_stats(path: Path) -> pd.DataFrame:
    stats = pd.read_csv(path, usecols=["age", "n", "min", "max", "test"])
    stats = stats.dropna(subset=["age", "test"]).copy()
    stats["age"] = stats["age"].astype(int)
    stats["age_bin"] = stats["age"].map(lambda age: (age_bin_for_year(age) or AgeBin("", -1, -1, np.nan)).label)
    stats = stats[stats["age_bin"] != ""].copy()
    return stats


def build_export_for_sex(
    sex: str,
    ridgeline_path: Path,
    stats_path: Path,
    availability_path: Path,
    mapping_path: Path,
) -> pd.DataFrame:
    ridgeline, tests_df = prepare_ridgeline(ridgeline_path)
    stats = prepare_stats(stats_path)

    unit_meta = build_unit_metadata(tests_df, availability_path=availability_path, mapping_path=mapping_path)
    test_meta_lookup = {
        row.test: {
            "test": row.test,
            "test_name": row.test_name,
            "short_name": row.short_name,
            "long_name": row.long_name,
            "system": row.system,
            "unit": row.unit,
            "unit_source": row.source,
            "unit_confidence": row.confidence,
            "nhanes_name": row.nhanes_name,
        }
        for row in unit_meta.itertuples(index=False)
    }

    ridgeline_groups = {
        (str(test), int(age), str(scale_type)): group.copy()
        for (test, age, scale_type), group in ridgeline.groupby(["test", "age", "scale_type"], observed=True)
    }

    stats_lookup = {
        (str(row.test), int(row.age)): {"n": float(row.n), "min": float(row.min), "max": float(row.max)}
        for row in stats.itertuples(index=False)
    }
    stats_by_test_and_bin = {
        (str(test), str(age_bin)): group_stats_for_bin(group)
        for (test, age_bin), group in stats.groupby(["test", "age_bin"], observed=True)
    }

    rows: list[dict[str, Any]] = []
    for test in unit_meta["test"].tolist():
        test_meta = test_meta_lookup[test]
        for age_bin in AGE_BINS:
            for scale_type in SCALE_ORDER:
                rows.append(
                    aggregate_one_bin(
                        sex=sex,
                        test_meta=test_meta,
                        scale_type=scale_type,
                        age_bin=age_bin,
                        ridgeline_groups=ridgeline_groups,
                        stats_lookup=stats_lookup,
                        stats_by_test_and_bin=stats_by_test_and_bin,
                    )
                )

    out = pd.DataFrame(rows)
    return out.sort_values(["sex", "system", "long_name", "age_lo", "scale_type"]).reset_index(drop=True)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--ridgeline-f", type=Path, default=DEFAULT_RIDGELINE_F)
    parser.add_argument("--ridgeline-m", type=Path, default=DEFAULT_RIDGELINE_M)
    parser.add_argument("--stats-f", type=Path, default=DEFAULT_STATS_F)
    parser.add_argument("--stats-m", type=Path, default=DEFAULT_STATS_M)
    parser.add_argument("--availability", type=Path, default=DEFAULT_AVAILABILITY)
    parser.add_argument("--mapping", type=Path, default=DEFAULT_MAPPING)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    args = parser.parse_args()

    frames = [
        build_export_for_sex(
            sex="female",
            ridgeline_path=args.ridgeline_f,
            stats_path=args.stats_f,
            availability_path=args.availability,
            mapping_path=args.mapping,
        ),
        build_export_for_sex(
            sex="male",
            ridgeline_path=args.ridgeline_m,
            stats_path=args.stats_m,
            availability_path=args.availability,
            mapping_path=args.mapping,
        ),
    ]

    out = pd.concat(frames, ignore_index=True)
    ensure_dir(args.out.parent)
    out.to_csv(args.out, index=False)

    unit_missing = int(out[["test", "unit"]].drop_duplicates()["unit"].eq("").sum())
    print(f"Wrote {len(out):,} rows to {args.out}")
    print(f"Distinct tests: {out['test'].nunique():,}")
    print(f"Age bins: {len(AGE_BINS)}")
    print(f"Tests with missing unit labels: {unit_missing:,}")


if __name__ == "__main__":
    main()
