#!/usr/bin/env python3
"""Build harmonized biomarker long dataset from downloaded NHANES files.

Key behaviors:
- Pools same biomarker variable across NHANES files/cycles (e.g., LBXSAL across BIOPRO_I/J/P_BIOPRO).
- Excludes non-analytic fields (comment codes, questionnaire-style fields, categorical code variables).
- Keeps a pragmatic healthy adult cohort (age>=20, major pathology exclusions).
"""

from __future__ import annotations

import argparse
import re
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
import pyreadstat

from nhanes_common import ensure_dir


def read_xpt_columns(path: Path, columns: Optional[List[str]] = None) -> pd.DataFrame:
    df, _ = pyreadstat.read_xport(str(path), usecols=columns)
    return df


def collect_demo_files(raw_dir: Path) -> List[Path]:
    pats = ["DEMO*.xpt", "P_DEMO.xpt", "DEMO_L.xpt", "CDEMO*.xpt"]
    files: List[Path] = []
    for pat in pats:
        files.extend(raw_dir.rglob(pat))
    return sorted(set(files))


def collect_questionnaire_files(raw_dir: Path) -> List[Path]:
    pats = ["DIQ*.xpt", "MCQ*.xpt", "KIQ_U*.xpt"]
    files: List[Path] = []
    for pat in pats:
        files.extend(raw_dir.rglob(pat))
    return sorted(set(files))


def normalize_seqn(df: pd.DataFrame) -> pd.Series:
    return pd.to_numeric(df["SEQN"], errors="coerce").astype("Int64")


def load_demographics(raw_dir: Path) -> pd.DataFrame:
    rows = []
    for p in collect_demo_files(raw_dir):
        cycle_year = int(p.parent.name)
        try:
            df = read_xpt_columns(p)
        except Exception:
            continue
        if "SEQN" not in df.columns or "RIDAGEYR" not in df.columns:
            continue

        out = pd.DataFrame(
            {
                "seqn": normalize_seqn(df),
                "age_years": pd.to_numeric(df.get("RIDAGEYR"), errors="coerce"),
                "sex_code": pd.to_numeric(df.get("RIAGENDR"), errors="coerce"),
                "pregnant": pd.to_numeric(df.get("RIDEXPRG"), errors="coerce").eq(1),
                "cycle_start_year": cycle_year,
            }
        )
        rows.append(out)

    if not rows:
        raise RuntimeError("No demographics files loaded")

    demo = pd.concat(rows, ignore_index=True)
    demo = demo.dropna(subset=["seqn", "age_years"]).drop_duplicates(["seqn", "cycle_start_year"], keep="last")
    demo["sex"] = demo["sex_code"].map({1.0: "Male", 2.0: "Female"}).fillna("Unknown")
    return demo[["seqn", "age_years", "sex", "pregnant", "cycle_start_year"]]


def detect_any_yes(df: pd.DataFrame, candidates: List[str]) -> pd.Series:
    cols = [c for c in df.columns if c in candidates]
    if not cols:
        return pd.Series([pd.NA] * len(df), index=df.index, dtype="boolean")

    hit = pd.Series(False, index=df.index)
    for c in cols:
        hit = hit | pd.to_numeric(df[c], errors="coerce").eq(1)
    return hit.astype("boolean")


def load_health_flags(raw_dir: Path) -> Tuple[pd.DataFrame, pd.DataFrame]:
    files = collect_questionnaire_files(raw_dir)
    per_cycle = []
    avail_rows = []

    for p in files:
        cycle_year = int(p.parent.name)
        try:
            df = read_xpt_columns(p)
        except Exception:
            continue
        if "SEQN" not in df.columns:
            continue

        lower_map = {c.lower(): c for c in df.columns}

        def pick_cols(names: List[str]) -> List[str]:
            cols = []
            for n in names:
                c = lower_map.get(n.lower())
                if c:
                    cols.append(c)
            return sorted(set(cols))

        diabetes_cols = pick_cols(["DIQ010"])
        cancer_cols = pick_cols(["MCQ220"])
        kidney_cols = pick_cols(["KIQ022"])
        cvd_cols = pick_cols([
            "MCQ160B",
            "MCQ160C",
            "MCQ160D",
            "MCQ160E",
            "MCQ160F",
            "MCQ160b",
            "MCQ160c",
            "MCQ160d",
            "MCQ160e",
            "MCQ160f",
        ])

        tmp = pd.DataFrame({"seqn": normalize_seqn(df), "cycle_start_year": cycle_year})
        if diabetes_cols:
            tmp["diabetes"] = detect_any_yes(df, diabetes_cols)
        if cancer_cols:
            tmp["cancer"] = detect_any_yes(df, cancer_cols)
        if kidney_cols:
            tmp["kidney"] = detect_any_yes(df, kidney_cols)
        if cvd_cols:
            tmp["cvd"] = detect_any_yes(df, cvd_cols)

        per_cycle.append(tmp)
        avail_rows.append(
            {
                "cycle_start_year": cycle_year,
                "file": p.name,
                "diabetes_cols": "|".join(diabetes_cols),
                "cvd_cols": "|".join(cvd_cols),
                "cancer_cols": "|".join(cancer_cols),
                "kidney_cols": "|".join(kidney_cols),
            }
        )

    if not per_cycle:
        empty = pd.DataFrame(columns=["seqn", "cycle_start_year", "diabetes", "cvd", "cancer", "kidney"])
        return empty, pd.DataFrame(avail_rows)

    flags = pd.concat(per_cycle, ignore_index=True)
    agg = flags.groupby(["seqn", "cycle_start_year"], as_index=False).agg(
        {
            "diabetes": "max",
            "cvd": "max",
            "cancer": "max",
            "kidney": "max",
        }
    )

    availability = pd.DataFrame(avail_rows)
    return agg, availability


def build_participant_table(raw_dir: Path) -> Tuple[pd.DataFrame, pd.DataFrame]:
    demo = load_demographics(raw_dir)
    health, availability = load_health_flags(raw_dir)

    p = demo.merge(health, on=["seqn", "cycle_start_year"], how="left")

    def row_reason(r: pd.Series) -> str:
        def is_yes(v: object) -> bool:
            return False if pd.isna(v) else bool(v)

        reasons = []
        if is_yes(r.get("pregnant", False)):
            reasons.append("pregnant")
        if is_yes(r.get("diabetes", False)):
            reasons.append("diabetes")
        if is_yes(r.get("cvd", False)):
            reasons.append("cvd")
        if is_yes(r.get("cancer", False)):
            reasons.append("cancer")
        if is_yes(r.get("kidney", False)):
            reasons.append("kidney")
        return "|".join(reasons)

    p["exclusion_reason"] = p.apply(row_reason, axis=1)
    p["healthy_flag"] = p["exclusion_reason"].eq("")
    p = p[p["age_years"] >= 20].copy()
    return p, availability


PREFIX_SCALE = {"": 1.0, "p": 1e-12, "n": 1e-9, "u": 1e-6, "m": 1e-3, "c": 1e-2, "d": 1e-1}
DEN_SCALE = {"l": 1.0, "dl": 1e-1, "ml": 1e-3, "ul": 1e-6}


def parse_terminal_unit(label: str) -> tuple[str, str]:
    s = str(label or "").strip()
    m = re.search(r"\(([^()]*)\)\s*$", s)
    if not m:
        return s, ""
    unit = m.group(1).strip()
    base = s[: m.start()].strip().rstrip(",")
    return base, unit


def normalize_base_name(name: str) -> str:
    s = str(name or "").lower()
    repl = {"α": "a", "β": "b", "γ": "g", "δ": "d", "µ": "u", "μ": "u", "–": "-", "—": "-"}
    for k, v in repl.items():
        s = s.replace(k, v)
    s = re.sub(r"\s+", " ", s)
    s = re.sub(r"[^a-z0-9 %/+-]", " ", s)
    s = re.sub(r"\s+", " ", s).strip()
    return s


def normalize_unit(unit: str) -> str:
    u = str(unit or "").strip().lower().replace("μ", "u").replace("µ", "u")
    u = re.sub(r"\s+", "", u)
    return u


def parse_unit_signature(unit: str) -> Optional[dict]:
    u = normalize_unit(unit)
    if not u:
        return None
    m = re.match(r"^([pnumcd]?)(g|mol|iu|u|eq|kat)/(l|dl|ml|ul)$", u)
    if not m:
        return None
    pfx, base, den = m.groups()
    if pfx not in PREFIX_SCALE or den not in DEN_SCALE:
        return None
    return {"num_base": base, "num_scale": PREFIX_SCALE[pfx], "den_scale": DEN_SCALE[den], "unit_norm": u}


def conversion_factor(src_unit: str, dst_unit: str) -> Optional[float]:
    src = parse_unit_signature(src_unit)
    dst = parse_unit_signature(dst_unit)
    if src is None or dst is None:
        return None
    if src["num_base"] != dst["num_base"]:
        return None
    src_density = src["num_scale"] / src["den_scale"]
    dst_density = dst["num_scale"] / dst["den_scale"]
    if dst_density == 0:
        return None
    return float(src_density / dst_density)


def build_pooling_map(lab_manifest: pd.DataFrame) -> pd.DataFrame:
    blood = lab_manifest[lab_manifest["is_blood_candidate"]].copy()
    var_counts = (
        blood.groupby(["variable_name", "variable_desc"], as_index=False)
        .size()
        .sort_values(["variable_name", "size"], ascending=[True, False])
        .drop_duplicates(subset=["variable_name"], keep="first")
    )
    var_counts["base_name"], var_counts["unit_raw"] = zip(*var_counts["variable_desc"].map(parse_terminal_unit))
    var_counts["base_key"] = var_counts["base_name"].map(normalize_base_name)
    var_counts["unit_norm"] = var_counts["unit_raw"].map(normalize_unit)
    var_counts["unit_sig"] = var_counts["unit_raw"].map(parse_unit_signature)
    var_counts["compat_key"] = var_counts["unit_sig"].map(lambda x: f"{x['num_base']}/vol" if isinstance(x, dict) else "")
    var_counts.loc[var_counts["compat_key"].eq(""), "compat_key"] = "raw:" + var_counts["unit_norm"].fillna("")

    rows = []
    for base_key, g_base in var_counts.groupby("base_key", observed=True):
        multi_compat = g_base["compat_key"].nunique() > 1
        for compat_key, g in g_base.groupby("compat_key", observed=True):
            ref_row = g.sort_values("size", ascending=False).iloc[0]
            ref_unit = str(ref_row["unit_raw"] or "").strip()
            ref_base_name = str(ref_row["base_name"] or "").strip()

            pooled_id = base_key
            if multi_compat:
                suffix = normalize_unit(ref_unit) or compat_key.replace(":", "_")
                pooled_id = f"{base_key}__{suffix}"

            pooled_name = ref_base_name
            if ref_unit:
                pooled_name = f"{ref_base_name} ({ref_unit})"

            for _, r in g.iterrows():
                src_unit = str(r["unit_raw"] or "").strip()
                factor = 1.0
                if ref_unit and src_unit and normalize_unit(src_unit) != normalize_unit(ref_unit):
                    f = conversion_factor(src_unit, ref_unit)
                    if f is not None:
                        factor = float(f)
                rows.append(
                    {
                        "variable_name": str(r["variable_name"]),
                        "variable_desc": str(r["variable_desc"]),
                        "base_key": base_key,
                        "compat_key": compat_key,
                        "pooled_id": pooled_id,
                        "pooled_name": pooled_name,
                        "pooled_unit": ref_unit,
                        "conversion_factor_to_pooled_unit": factor,
                    }
                )

    return pd.DataFrame(rows)


def is_comment_or_code_variable(variable_name: str, variable_desc: str) -> bool:
    v = f"{variable_name} {variable_desc}".lower()
    patterns = [
        r"\bcomment\b",
        r"\bcomment code\b",
        r"\bresult code\b",
        r"\bstatus code\b",
        r"\bquality control\b",
        r"\bdetection limit\b",
        r"\bdo you\b",
        r"\bdid you\b",
        r"\bhow often\b",
        r"\bquestionnaire\b",
        r"\bdup\b",
        r"\bduplicate\b",
        r"\bab con\b",
        r"\bantibody con",
        r"\bod in dup",
        r"od_dup",
        r"\bmean ab conc",
    ]
    return any(re.search(p, v) is not None for p in patterns)


def is_continuous_numeric(s: pd.Series) -> bool:
    x = pd.to_numeric(s, errors="coerce").dropna()
    n = len(x)
    if n < 30:
        return False

    nunique = int(x.nunique(dropna=True))
    if nunique < 8:
        return False

    frac_unique = nunique / max(n, 1)
    integer_like = np.isclose(x.to_numpy(dtype=float), np.round(x.to_numpy(dtype=float)), atol=1e-12)
    integer_like_frac = float(integer_like.mean()) if len(integer_like) else 1.0

    if integer_like_frac > 0.995 and nunique <= 12:
        return False

    if frac_unique < 0.01 and nunique < 20:
        return False

    return True


def write_long_dataset(
    raw_dir: Path,
    processed_dir: Path,
    lab_manifest: pd.DataFrame,
    participants: pd.DataFrame,
) -> Tuple[int, int, int]:
    blood = lab_manifest[lab_manifest["is_blood_candidate"]].copy()
    blood = blood.drop_duplicates(subset=["xpt_url", "variable_name"]).reset_index(drop=True)

    file_meta = (
        blood[["data_file_name", "cycle_label", "cycle_start_year", "cycle_end_year", "xpt_url", "data_file_desc"]]
        .drop_duplicates(subset=["xpt_url"])
        .set_index("xpt_url")
    )

    vars_by_url: Dict[str, pd.DataFrame] = {
        url: g[["variable_name", "variable_desc"]].drop_duplicates().reset_index(drop=True)
        for url, g in blood.groupby("xpt_url")
    }

    pooling_map_df = build_pooling_map(lab_manifest)
    pooling_map = pooling_map_df.set_index("variable_name").to_dict(orient="index")

    out_path = processed_dir / "biomarker_long.parquet"
    ensure_dir(processed_dir)

    writer: Optional[pq.ParquetWriter] = None
    n_rows = 0
    n_files = 0
    kept_variables: set[str] = set()
    kept_pooled_ids: set[str] = set()
    screen_rows: List[dict] = []

    for url, vars_df in vars_by_url.items():
        m = file_meta.loc[url]
        year = int(re.search(r"/Public/(\d{4})/DataFiles/", url).group(1))
        fname = Path(url).name
        xpt_path = raw_dir / str(year) / fname
        if not xpt_path.exists():
            continue

        try:
            df = read_xpt_columns(xpt_path)
        except Exception:
            continue

        if "SEQN" not in df.columns:
            continue

        df["seqn"] = normalize_seqn(df)
        people = participants[participants["cycle_start_year"] == year][
            ["seqn", "age_years", "sex", "healthy_flag", "exclusion_reason"]
        ]
        if people.empty:
            continue

        for _, v in vars_df.iterrows():
            var = str(v["variable_name"])
            vdesc = str(v["variable_desc"])

            reason = ""
            if var not in df.columns:
                reason = "missing_in_file"
            elif var == "SEQN" or var.startswith("WT"):
                reason = "id_or_weight"
            elif is_comment_or_code_variable(var, vdesc):
                reason = "comment_or_code"
            elif var not in pooling_map:
                reason = "no_pool_map"
            else:
                if not is_continuous_numeric(df[var]):
                    reason = "non_continuous_numeric"

            if reason:
                screen_rows.append(
                    {
                        "cycle_start_year": year,
                        "data_file_name": m["data_file_name"],
                        "variable_name": var,
                        "variable_desc": vdesc,
                        "screen_result": "excluded",
                        "reason": reason,
                    }
                )
                continue

            tmp = pd.DataFrame({"seqn": df["seqn"], "value": pd.to_numeric(df[var], errors="coerce")})
            tmp = tmp.dropna(subset=["seqn", "value"])
            tmp = tmp.merge(people, on="seqn", how="inner")
            tmp = tmp[tmp["healthy_flag"]].copy()

            if tmp.empty:
                screen_rows.append(
                    {
                        "cycle_start_year": year,
                        "data_file_name": m["data_file_name"],
                        "variable_name": var,
                        "variable_desc": vdesc,
                        "screen_result": "excluded",
                        "reason": "no_healthy_data",
                    }
                )
                continue

            pool = pooling_map[var]
            factor = float(pool.get("conversion_factor_to_pooled_unit", 1.0))
            if factor != 1.0:
                tmp["value"] = tmp["value"] * factor

            biomarker_id = str(pool["pooled_id"])
            biomarker_name = str(pool["pooled_name"])
            pooled_unit = str(pool["pooled_unit"] or "")

            tmp["cycle_label"] = m["cycle_label"]
            tmp["cycle_start_year"] = int(m["cycle_start_year"])
            tmp["cycle_end_year"] = int(m["cycle_end_year"])
            tmp["biomarker_id"] = biomarker_id
            tmp["variable_name"] = var
            tmp["biomarker_name"] = biomarker_name
            tmp["unit"] = pooled_unit
            tmp["source_data_file"] = m["data_file_name"]

            keep_cols = [
                "seqn",
                "age_years",
                "sex",
                "cycle_label",
                "cycle_start_year",
                "cycle_end_year",
                "biomarker_id",
                "variable_name",
                "biomarker_name",
                "source_data_file",
                "value",
                "unit",
                "healthy_flag",
                "exclusion_reason",
            ]
            tmp = tmp[keep_cols]

            table = pa.Table.from_pandas(tmp, preserve_index=False)
            if writer is None:
                writer = pq.ParquetWriter(str(out_path), table.schema)
            writer.write_table(table)
            n_rows += len(tmp)
            kept_variables.add(var)
            kept_pooled_ids.add(biomarker_id)

            screen_rows.append(
                {
                    "cycle_start_year": year,
                    "data_file_name": m["data_file_name"],
                    "variable_name": var,
                    "variable_desc": vdesc,
                    "screen_result": "kept",
                    "reason": "",
                    "pooled_id": biomarker_id,
                }
            )

        n_files += 1

    if writer is not None:
        writer.close()
    else:
        pd.DataFrame(
            columns=[
                "seqn",
                "age_years",
                "sex",
                "cycle_label",
                "cycle_start_year",
                "cycle_end_year",
                "biomarker_id",
                "variable_name",
                "biomarker_name",
                "source_data_file",
                "value",
                "unit",
                "healthy_flag",
                "exclusion_reason",
            ]
        ).to_parquet(out_path, index=False)

    screen_df = pd.DataFrame(screen_rows)
    screen_df.to_csv(processed_dir / "variable_screening_summary.csv", index=False)

    kept_from_manifest = blood[blood["variable_name"].isin(kept_variables)].copy()
    kept_from_manifest = kept_from_manifest.merge(
        pooling_map_df[["variable_name", "pooled_id", "pooled_name", "pooled_unit"]],
        on="variable_name",
        how="left",
    )
    kept_from_manifest = kept_from_manifest[kept_from_manifest["pooled_id"].isin(kept_pooled_ids)].copy()

    catalog = (
        kept_from_manifest.groupby("pooled_id", as_index=False)
        .agg(
            biomarker_name=("pooled_name", "first"),
            unit=("pooled_unit", "first"),
            source_file_count=("data_file_name", lambda x: int(pd.Series(x).nunique())),
            source_files=("data_file_name", lambda x: "|".join(sorted(pd.Series(x).dropna().astype(str).unique()))),
            source_variable_count=("variable_name", lambda x: int(pd.Series(x).nunique())),
            source_variables=("variable_name", lambda x: "|".join(sorted(pd.Series(x).dropna().astype(str).unique()))),
        )
        .rename(columns={"pooled_id": "biomarker_id"})
    )
    catalog["variable_name"] = catalog["biomarker_id"]
    catalog["biomarker_name"] = catalog["biomarker_name"].fillna(catalog["variable_name"])
    catalog["unit"] = catalog["unit"].fillna("")
    catalog = catalog[
        [
            "biomarker_id",
            "variable_name",
            "biomarker_name",
            "unit",
            "source_file_count",
            "source_files",
            "source_variable_count",
            "source_variables",
        ]
    ]
    catalog = catalog.sort_values("biomarker_name").reset_index(drop=True)
    catalog.to_parquet(processed_dir / "biomarker_catalog.parquet", index=False)
    catalog.to_csv(processed_dir / "biomarker_catalog.csv", index=False)

    return n_rows, n_files, len(kept_pooled_ids)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--raw", default="data/raw")
    ap.add_argument("--manifest", default="data/processed/lab_variable_manifest.parquet")
    ap.add_argument("--out", default="data/processed")
    args = ap.parse_args()

    raw_dir = Path(args.raw)
    out_dir = Path(args.out)
    ensure_dir(out_dir)

    lab_manifest = pd.read_parquet(args.manifest)
    participants, availability = build_participant_table(raw_dir)

    participants.to_parquet(out_dir / "participant_health_flags.parquet", index=False)
    availability.to_csv(out_dir / "health_rule_availability_by_cycle.csv", index=False)

    n_rows, n_files, n_vars = write_long_dataset(
        raw_dir=raw_dir,
        processed_dir=out_dir,
        lab_manifest=lab_manifest,
        participants=participants,
    )

    print(f"Participant rows (age>=20): {len(participants):,}")
    print(f"Processed blood lab files: {n_files:,}")
    print(f"Pooled biomarkers kept: {n_vars:,}")
    print(f"Long biomarker rows written: {n_rows:,}")
    print(f"Dataset: {out_dir / 'biomarker_long.parquet'}")


if __name__ == "__main__":
    main()
