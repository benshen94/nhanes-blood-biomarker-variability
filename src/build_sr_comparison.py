#!/usr/bin/env python3
"""Build SR-vs-NHANES QQ comparison payloads for the blood dashboard."""

from __future__ import annotations

import argparse
import importlib.util
import json
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from scipy.stats import wasserstein_distance

from nhanes_common import ensure_dir


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_LONG = ROOT / "data" / "processed" / "biomarker_long.parquet"
DEFAULT_OUT_ROOT = ROOT / "projects" / "sr_comparison" / "blood"
DEFAULT_SR_SCRIPT = Path(
    "/Users/benshenhar/Library/CloudStorage/GoogleDrive-benshenhar@gmail.com/My Drive/Weizmann/Alon Lab/Aging/python/notebooks/SR_general/usa_2019_waterfall.py"
)
DEFAULT_SR_PACKAGE_ROOT = Path(
    "/Users/benshenhar/Library/CloudStorage/GoogleDrive-benshenhar@gmail.com/My Drive/Weizmann/Alon Lab/Aging/python"
)

AGE_BIN_EDGES = list(np.arange(20, 90, 5))
AGE_BIN_LABELS = [f"{start}-{start + 4}" for start in range(20, 85, 5)]
AGE_BIN_MIDS = {label: start + 2.5 for label, start in zip(AGE_BIN_LABELS, range(20, 85, 5))}

MIN_BIN_N = 30
QQ_PROBABILITIES = np.linspace(0.01, 0.99, 99)
WATERFALL_SAMPLE_PROBABILITIES = np.linspace(0.001, 0.999, 801)
SR_N_SIM = 100_000
SR_TMAX = 120
SR_SAVE_TIMES = 1
EPS = 1e-12
SR_TRIM_MODES = [
    {"key": "all", "label": "0% each tail", "tail_pct": 0, "lo": 0.0, "hi": 1.0},
    {"key": "trim_3_97", "label": "3% each tail", "tail_pct": 3, "lo": 0.03, "hi": 0.97},
    {"key": "trim_5_95", "label": "5% each tail", "tail_pct": 5, "lo": 0.05, "hi": 0.95},
    {"key": "trim_10_90", "label": "10% each tail", "tail_pct": 10, "lo": 0.10, "hi": 0.90},
]
DEFAULT_SR_TRIM_MODE = "trim_3_97"
SR_TRIM_MODE_BY_KEY = {mode["key"]: mode for mode in SR_TRIM_MODES}


def assign_age_bins(age: pd.Series) -> tuple[pd.Series, pd.Series]:
    age_bin = pd.cut(age, bins=AGE_BIN_EDGES, labels=AGE_BIN_LABELS, right=False, include_lowest=True)
    age_mid = age_bin.map(AGE_BIN_MIDS).astype(float)
    return age_bin, age_mid


def safe_float(value: Any) -> float | None:
    if value is None:
        return None
    number = float(value)
    if not np.isfinite(number):
        return None
    return number


def rounded_list(values: np.ndarray | list[float], digits: int = 6) -> list[float]:
    arr = np.asarray(values, dtype=float)
    if arr.size == 0:
        return []
    return [round(float(v), digits) for v in arr.tolist()]


def clean_sorted_values(values: np.ndarray | pd.Series) -> np.ndarray:
    arr = np.asarray(values, dtype=float)
    arr = arr[np.isfinite(arr)]
    if arr.size == 0:
        return np.array([], dtype=float)
    return np.sort(arr.astype(float))


def trim_distribution(values: np.ndarray | pd.Series, lo: float = 0.0, hi: float = 1.0) -> np.ndarray:
    arr = clean_sorted_values(values)
    if arr.size == 0:
        return np.array([], dtype=float)
    if lo <= 0 and hi >= 1:
        return arr
    if lo >= hi:
        return np.array([], dtype=float)

    lo_cut = float(np.quantile(arr, lo))
    hi_cut = float(np.quantile(arr, hi))
    kept = arr[(arr >= lo_cut) & (arr <= hi_cut)]
    if kept.size == 0:
        return np.array([], dtype=float)
    return np.sort(kept.astype(float))


def zscore_values(values: np.ndarray | pd.Series) -> np.ndarray:
    arr = clean_sorted_values(values)
    if arr.size == 0:
        return np.array([], dtype=float)

    mean = float(np.mean(arr))
    sd = float(np.std(arr, ddof=0))
    if sd <= EPS:
        return np.zeros(arr.shape[0], dtype=float)
    return (arr - mean) / sd


def quartiles(values: np.ndarray) -> tuple[float | None, float | None, float | None]:
    if values.size == 0:
        return None, None, None
    q1, median, q3 = np.quantile(values, [0.25, 0.50, 0.75])
    return safe_float(q1), safe_float(median), safe_float(q3)


def build_quantile_sample(
    values: np.ndarray,
    probabilities: np.ndarray = WATERFALL_SAMPLE_PROBABILITIES,
) -> list[float]:
    if values.size == 0:
        return []
    sample = np.quantile(values, probabilities)
    return rounded_list(sample)


def sr_trim_mode(trim_mode_key: str) -> dict[str, Any]:
    return SR_TRIM_MODE_BY_KEY.get(trim_mode_key, SR_TRIM_MODE_BY_KEY[DEFAULT_SR_TRIM_MODE])


def rounded_trim_modes() -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for trim_mode in SR_TRIM_MODES:
        rows.append(
            {
                "key": str(trim_mode["key"]),
                "label": str(trim_mode["label"]),
                "tail_pct": int(trim_mode["tail_pct"]),
                "lo": float(trim_mode["lo"]),
                "hi": float(trim_mode["hi"]),
            }
        )
    return rows


def compute_qq_fit(
    sr_values: np.ndarray,
    biomarker_values: np.ndarray,
    trim_mode_key: str = DEFAULT_SR_TRIM_MODE,
    probabilities: np.ndarray = QQ_PROBABILITIES,
) -> dict[str, Any]:
    trim_mode = sr_trim_mode(trim_mode_key)
    sr_clean = clean_sorted_values(sr_values)
    biomarker_trimmed = trim_distribution(
        biomarker_values,
        lo=float(trim_mode["lo"]),
        hi=float(trim_mode["hi"]),
    )

    sr_q1, sr_median, sr_q3 = quartiles(sr_clean)
    nhanes_q1, nhanes_median, nhanes_q3 = quartiles(biomarker_trimmed)

    result: dict[str, Any] = {
        "trim_mode": str(trim_mode["key"]),
        "trim_label": str(trim_mode["label"]),
        "trim_rule": {
            "lo": float(trim_mode["lo"]),
            "hi": float(trim_mode["hi"]),
        },
        "r2": None,
        "slope_m": None,
        "intercept_c": None,
        "wasserstein_z": None,
        "sr_n": int(sr_clean.size),
        "nhanes_n": int(biomarker_trimmed.size),
        "sr_q1": sr_q1,
        "sr_median": sr_median,
        "sr_q3": sr_q3,
        "nhanes_q1": nhanes_q1,
        "nhanes_median": nhanes_median,
        "nhanes_q3": nhanes_q3,
        "qq_sr_values": [],
        "qq_biomarker_values": [],
    }
    if sr_clean.size < MIN_BIN_N or biomarker_trimmed.size < MIN_BIN_N:
        return result

    sr_z = zscore_values(sr_clean)
    biomarker_z = zscore_values(biomarker_trimmed)
    if sr_z.size and biomarker_z.size:
        result["wasserstein_z"] = safe_float(wasserstein_distance(sr_z, biomarker_z))

    sr_quantiles = np.quantile(sr_clean, probabilities)
    biomarker_quantiles = np.quantile(biomarker_trimmed, probabilities)
    mask = np.isfinite(sr_quantiles) & np.isfinite(biomarker_quantiles)
    if mask.sum() < 2:
        return result

    x = sr_quantiles[mask]
    y = biomarker_quantiles[mask]
    if np.nanstd(x) <= EPS or np.nanstd(y) <= EPS:
        return result

    slope_m, intercept_c = np.polyfit(x, y, 1)
    y_hat = slope_m * x + intercept_c
    ss_res = float(np.sum((y - y_hat) ** 2))
    ss_tot = float(np.sum((y - np.mean(y)) ** 2))
    r2 = None if ss_tot <= EPS else safe_float(1.0 - (ss_res / ss_tot))

    result["r2"] = r2
    result["slope_m"] = safe_float(slope_m)
    result["intercept_c"] = safe_float(intercept_c)
    result["qq_sr_values"] = rounded_list(x)
    result["qq_biomarker_values"] = rounded_list(y)
    return result


def summarize_biomarker_bins(rows: list[dict[str, Any]]) -> dict[str, Any]:
    r2_values = [row["r2"] for row in rows if row.get("r2") is not None]
    slope_values = [row["slope_m"] for row in rows if row.get("slope_m") is not None]
    intercept_values = [row["intercept_c"] for row in rows if row.get("intercept_c") is not None]
    wasserstein_values = [row["wasserstein_z"] for row in rows if row.get("wasserstein_z") is not None]

    def mean_or_none(values: list[float]) -> float | None:
        if not values:
            return None
        return safe_float(np.mean(values))

    def std_or_none(values: list[float]) -> float | None:
        if not values:
            return None
        return safe_float(np.std(values, ddof=0))

    return {
        "mean_r2": mean_or_none(r2_values),
        "min_r2": safe_float(np.min(r2_values)) if r2_values else None,
        "median_r2": safe_float(np.median(r2_values)) if r2_values else None,
        "valid_bin_count": int(len(r2_values)),
        "mean_slope_m": mean_or_none(slope_values),
        "slope_m_sd": std_or_none(slope_values),
        "mean_intercept_c": mean_or_none(intercept_values),
        "intercept_c_sd": std_or_none(intercept_values),
        "mean_wasserstein_z": mean_or_none(wasserstein_values),
        "min_wasserstein_z": safe_float(np.min(wasserstein_values)) if wasserstein_values else None,
        "median_wasserstein_z": safe_float(np.median(wasserstein_values)) if wasserstein_values else None,
        "valid_wasserstein_bin_count": int(len(wasserstein_values)),
        "r2_by_age_bin": [
            {
                "age_bin": row["age_bin"],
                "age_mid": row["age_mid"],
                "r2": row.get("r2"),
            }
            for row in rows
        ],
        "wasserstein_z_by_age_bin": [
            {
                "age_bin": row["age_bin"],
                "age_mid": row["age_mid"],
                "wasserstein_z": row.get("wasserstein_z"),
            }
            for row in rows
        ],
    }


def load_sr_module(sr_script_path: Path, sr_package_root: Path):
    if str(sr_package_root) not in sys.path:
        sys.path.insert(0, str(sr_package_root))

    spec = importlib.util.spec_from_file_location("sr_usa_2019_waterfall", sr_script_path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Failed to create import spec for {sr_script_path}")

    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def run_sr_simulation(
    sr_script_path: Path,
    sr_package_root: Path,
    n_sim: int = SR_N_SIM,
    tmax: int = SR_TMAX,
    save_times: int = SR_SAVE_TIMES,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    sr_module = load_sr_module(sr_script_path, sr_package_root)
    params_dict, usa_cal = sr_module.build_usa_2019_params(n_sim)
    sim = sr_module.run_simulation(params_dict, usa_cal, n_sim, tmax=tmax, save_times=save_times)
    return np.asarray(sim.tspan), np.asarray(sim.paths), np.asarray(sim.death_times)


def load_or_build_sr_cache(
    out_root: Path,
    sr_script_path: Path,
    sr_package_root: Path,
    force_rerun: bool,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, str]:
    cache_path = out_root / "sr_usa_2019_trajectories.npz"
    if cache_path.exists() and not force_rerun:
        cached = np.load(cache_path)
        return np.asarray(cached["tspan"]), np.asarray(cached["paths"]), np.asarray(cached["death_times"]), "local_cache"

    tspan, paths, death_times = run_sr_simulation(
        sr_script_path=sr_script_path,
        sr_package_root=sr_package_root,
    )
    np.savez_compressed(cache_path, tspan=tspan, paths=paths, death_times=death_times)
    return tspan, paths, death_times, "rerun"


def extract_sr_alive_distributions(
    tspan: np.ndarray,
    paths: np.ndarray,
    death_times: np.ndarray,
) -> dict[str, np.ndarray]:
    distributions: dict[str, np.ndarray] = {}

    for age_bin in AGE_BIN_LABELS:
        age_mid = AGE_BIN_MIDS[age_bin]
        t_idx = int(np.argmin(np.abs(tspan - age_mid)))
        alive_mask = death_times > age_mid
        alive_values = clean_sorted_values(paths[alive_mask, t_idx])
        distributions[age_bin] = alive_values

    return distributions


def build_sr_reference_rows(
    tspan: np.ndarray,
    paths: np.ndarray,
    death_times: np.ndarray,
) -> tuple[list[dict[str, Any]], dict[str, np.ndarray]]:
    rows: list[dict[str, Any]] = []
    distributions: dict[str, np.ndarray] = {}
    alive_distributions = extract_sr_alive_distributions(tspan, paths, death_times)

    for age_bin in AGE_BIN_LABELS:
        age_mid = AGE_BIN_MIDS[age_bin]
        alive_values = alive_distributions[age_bin]
        q1, median, q3 = quartiles(alive_values)
        distributions[age_bin] = alive_values
        rows.append(
            {
                "age_bin": age_bin,
                "age_mid": age_mid,
                "sr_n": int(alive_values.size),
                "sr_q1": q1,
                "sr_median": median,
                "sr_q3": q3,
            }
        )

    return rows, distributions


def build_sr_waterfall_reference(
    tspan: np.ndarray,
    paths: np.ndarray,
    death_times: np.ndarray,
) -> dict[str, Any]:
    alive_distributions = extract_sr_alive_distributions(tspan, paths, death_times)
    bins: list[dict[str, Any]] = []

    for age_bin in AGE_BIN_LABELS:
        age_mid = AGE_BIN_MIDS[age_bin]
        alive_values = alive_distributions[age_bin]
        q1, median, q3 = quartiles(alive_values)
        bins.append(
            {
                "age_bin": age_bin,
                "age_mid": age_mid,
                "sr_n": int(alive_values.size),
                "sr_q1": q1,
                "sr_median": median,
                "sr_q3": q3,
                "values_sample": build_quantile_sample(alive_values),
            }
        )

    return {
        "age_bins": AGE_BIN_LABELS,
        "sample_probabilities": rounded_list(WATERFALL_SAMPLE_PROBABILITIES),
        "bins": bins,
    }


def build_dashboard_payload(
    long_df: pd.DataFrame,
    sr_reference_rows: list[dict[str, Any]],
    sr_distributions: dict[str, np.ndarray],
    sr_waterfall_reference: dict[str, Any],
) -> tuple[dict[str, Any], pd.DataFrame, pd.DataFrame]:
    use = long_df[["biomarker_id", "biomarker_name", "age_years", "value"]].copy()
    use = use.dropna(subset=["biomarker_id", "value", "age_years"])
    use = use[(use["age_years"] >= 20) & (use["age_years"] < 85)].copy()
    use["age_bin"], use["age_mid"] = assign_age_bins(use["age_years"])
    use = use.dropna(subset=["age_bin"]).copy()

    biomarker_names = (
        use[["biomarker_id", "biomarker_name"]]
        .dropna(subset=["biomarker_id"])
        .drop_duplicates()
        .sort_values(["biomarker_name", "biomarker_id"])
    )

    grouped_rows: dict[tuple[str, str], dict[str, Any]] = {}
    grouped = use.groupby(["biomarker_id", "biomarker_name", "age_bin", "age_mid"], observed=True)
    for (biomarker_id, biomarker_name, age_bin, age_mid), group in grouped:
        grouped_rows[(str(biomarker_id), str(age_bin))] = {
            "biomarker_id": str(biomarker_id),
            "biomarker_name": str(biomarker_name),
            "age_bin": str(age_bin),
            "age_mid": float(age_mid),
            "values": clean_sorted_values(group["value"].to_numpy(dtype=float)),
        }

    sr_reference_by_bin = {row["age_bin"]: row for row in sr_reference_rows}
    summary_by_biomarker: dict[str, dict[str, Any]] = {}
    detail_by_biomarker: dict[str, dict[str, Any]] = {}
    summary_rows: list[dict[str, Any]] = []
    detail_rows: list[dict[str, Any]] = []

    for row in biomarker_names.itertuples(index=False):
        biomarker_id = str(row.biomarker_id)
        biomarker_name = str(row.biomarker_name)
        trim_summaries: dict[str, dict[str, Any]] = {}
        trim_details: dict[str, dict[str, Any]] = {}

        for trim_mode in SR_TRIM_MODES:
            trim_key = str(trim_mode["key"])
            per_bin_rows: list[dict[str, Any]] = []

            for age_bin in AGE_BIN_LABELS:
                age_mid = float(AGE_BIN_MIDS[age_bin])
                base = grouped_rows.get((biomarker_id, age_bin))
                sr_row = sr_reference_by_bin[age_bin]

                if base is None:
                    detail = {
                        "biomarker_id": biomarker_id,
                        "biomarker_name": biomarker_name,
                        "age_bin": age_bin,
                        "age_mid": age_mid,
                        "trim_mode": trim_key,
                        "trim_label": str(trim_mode["label"]),
                        "trim_rule": {"lo": float(trim_mode["lo"]), "hi": float(trim_mode["hi"])},
                        "r2": None,
                        "slope_m": None,
                        "intercept_c": None,
                        "wasserstein_z": None,
                        "nhanes_n": 0,
                        "sr_n": sr_row["sr_n"],
                        "nhanes_q1": None,
                        "nhanes_median": None,
                        "nhanes_q3": None,
                        "sr_q1": sr_row["sr_q1"],
                        "sr_median": sr_row["sr_median"],
                        "sr_q3": sr_row["sr_q3"],
                        "qq_sr_values": [],
                        "qq_biomarker_values": [],
                    }
                else:
                    fit = compute_qq_fit(
                        sr_distributions[age_bin],
                        base["values"],
                        trim_mode_key=trim_key,
                    )
                    detail = {
                        "biomarker_id": biomarker_id,
                        "biomarker_name": biomarker_name,
                        "age_bin": age_bin,
                        "age_mid": age_mid,
                        **fit,
                        "sr_n": sr_row["sr_n"],
                        "sr_q1": sr_row["sr_q1"],
                        "sr_median": sr_row["sr_median"],
                        "sr_q3": sr_row["sr_q3"],
                    }

                per_bin_rows.append(detail)
                detail_rows.append(
                    {
                        "biomarker_id": biomarker_id,
                        "biomarker_name": biomarker_name,
                        "trim_mode": trim_key,
                        "trim_label": str(trim_mode["label"]),
                        "age_bin": age_bin,
                        "age_mid": age_mid,
                        "r2": detail["r2"],
                        "slope_m": detail["slope_m"],
                        "intercept_c": detail["intercept_c"],
                        "wasserstein_z": detail["wasserstein_z"],
                        "nhanes_n": detail["nhanes_n"],
                        "sr_n": detail["sr_n"],
                        "nhanes_q1": detail["nhanes_q1"],
                        "nhanes_median": detail["nhanes_median"],
                        "nhanes_q3": detail["nhanes_q3"],
                        "sr_q1": detail["sr_q1"],
                        "sr_median": detail["sr_median"],
                        "sr_q3": detail["sr_q3"],
                    }
                )

            summary = summarize_biomarker_bins(per_bin_rows)
            summary["trim_mode"] = trim_key
            summary["trim_label"] = str(trim_mode["label"])
            summary["trim_rule"] = {"lo": float(trim_mode["lo"]), "hi": float(trim_mode["hi"])}
            trim_summaries[trim_key] = summary
            trim_details[trim_key] = {
                "trim_mode": trim_key,
                "trim_label": str(trim_mode["label"]),
                "trim_rule": {"lo": float(trim_mode["lo"]), "hi": float(trim_mode["hi"])},
                "bins": per_bin_rows,
            }
            summary_rows.append(
                {
                    "biomarker_id": biomarker_id,
                    "biomarker_name": biomarker_name,
                    **summary,
                }
            )

        default_summary = trim_summaries[DEFAULT_SR_TRIM_MODE]
        default_detail = trim_details[DEFAULT_SR_TRIM_MODE]

        summary_by_biomarker[biomarker_id] = {
            **default_summary,
            "default_trim_mode": DEFAULT_SR_TRIM_MODE,
            "trim_modes": rounded_trim_modes(),
            "trim_summaries": trim_summaries,
        }
        detail_by_biomarker[biomarker_id] = {
            "biomarker_id": biomarker_id,
            "biomarker_name": biomarker_name,
            "age_bins": AGE_BIN_LABELS,
            "default_trim_mode": DEFAULT_SR_TRIM_MODE,
            "trim_modes": rounded_trim_modes(),
            "trim_rule": default_detail["trim_rule"],
            "trim_label": default_detail["trim_label"],
            "quantile_grid": rounded_list(QQ_PROBABILITIES),
            "reference_bins": sr_reference_rows,
            "bins": default_detail["bins"],
            "trim_details": trim_details,
        }

    payload = {
        "meta": {
            "age_bins": AGE_BIN_LABELS,
            "age_mids": AGE_BIN_MIDS,
            "default_trim_mode": DEFAULT_SR_TRIM_MODE,
            "trim_modes": rounded_trim_modes(),
            "sr_reference_tail_policy": "alive_only_no_tail_trimming",
            "min_bin_n": MIN_BIN_N,
            "qq_probabilities": rounded_list(QQ_PROBABILITIES),
        },
        "summary_by_biomarker": summary_by_biomarker,
        "detail_by_biomarker": detail_by_biomarker,
        "sr_reference_bins": sr_reference_rows,
        "sr_waterfall_reference": sr_waterfall_reference,
    }
    return payload, pd.DataFrame(summary_rows), pd.DataFrame(detail_rows)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--long", default=str(DEFAULT_LONG))
    ap.add_argument("--out-root", default=str(DEFAULT_OUT_ROOT))
    ap.add_argument("--sr-script-path", default=str(DEFAULT_SR_SCRIPT))
    ap.add_argument("--sr-package-root", default=str(DEFAULT_SR_PACKAGE_ROOT))
    ap.add_argument("--force-rerun", action="store_true")
    args = ap.parse_args()

    long_path = Path(args.long)
    out_root = Path(args.out_root)
    sr_script_path = Path(args.sr_script_path)
    sr_package_root = Path(args.sr_package_root)

    ensure_dir(out_root)

    long_df = pd.read_parquet(long_path, columns=["biomarker_id", "biomarker_name", "age_years", "value"])

    started_at = time.time()
    tspan, paths, death_times, sr_source = load_or_build_sr_cache(
        out_root=out_root,
        sr_script_path=sr_script_path,
        sr_package_root=sr_package_root,
        force_rerun=args.force_rerun,
    )
    sr_reference_rows, sr_distributions = build_sr_reference_rows(tspan, paths, death_times)
    sr_waterfall_reference = build_sr_waterfall_reference(tspan, paths, death_times)
    payload, summary_df, detail_df = build_dashboard_payload(
        long_df,
        sr_reference_rows,
        sr_distributions,
        sr_waterfall_reference,
    )

    summary_path = out_root / "biomarker_qq_summary.csv"
    detail_path = out_root / "biomarker_qq_detail.csv"
    payload_path = out_root / "dashboard_payload.json"
    manifest_path = out_root / "run_manifest.json"

    summary_df.to_csv(summary_path, index=False)
    detail_df.to_csv(detail_path, index=False)
    payload_path.write_text(json.dumps(payload, ensure_ascii=True, allow_nan=False), encoding="utf-8")

    manifest = {
        "built_at_unix": int(time.time()),
        "duration_seconds": round(time.time() - started_at, 2),
        "long_path": str(long_path),
        "out_root": str(out_root),
        "sr_script_path": str(sr_script_path),
        "sr_package_root": str(sr_package_root),
        "sr_source": sr_source,
        "sr_model": {
            "n_sim": SR_N_SIM,
            "tmax": SR_TMAX,
            "save_times": SR_SAVE_TIMES,
        },
        "waterfall_reference_sample_n": int(len(WATERFALL_SAMPLE_PROBABILITIES)),
        "age_bins": AGE_BIN_LABELS,
        "default_trim_mode": DEFAULT_SR_TRIM_MODE,
        "trim_modes": rounded_trim_modes(),
        "sr_reference_tail_policy": "alive_only_no_tail_trimming",
        "min_bin_n": MIN_BIN_N,
        "biomarker_count": int(summary_df["biomarker_id"].nunique()),
    }
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=True, indent=2, allow_nan=False), encoding="utf-8")

    print(f"Wrote SR summary CSV: {summary_path}")
    print(f"Wrote SR detail CSV: {detail_path}")
    print(f"Wrote SR dashboard payload: {payload_path}")
    print(f"Wrote SR run manifest: {manifest_path}")


if __name__ == "__main__":
    main()
