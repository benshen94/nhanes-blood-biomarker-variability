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
from collections import defaultdict
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
    pats = ["DIQ*.xpt", "MCQ*.xpt", "KIQ*.xpt", "BPQ*.xpt", "OSQ*.xpt", "VIQ*.xpt", "PFQ*.xpt", "HUQ*.xpt"]
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

        fixed_conditions = {
            "diabetes": ["DIQ010"],
            "asthma": ["MCQ010"],
            "cancer": ["MCQ220"],
            "kidney": ["KIQ022"],
            "liver": ["MCQ160L", "MCQ500", "MCQ510A", "MCQ510B", "MCQ510C", "MCQ510D", "MCQ510E", "MCQ510F"],
            "cvd": ["MCQ160B", "MCQ160C", "MCQ160D", "MCQ160E", "MCQ160F"],
            "hypertension": ["BPQ020"],
            "osteoporosis": ["OSQ060"],
            "cataract_operation": ["VIQ070"],
            "adl_disability": ["ADDLDIS", "ADLDIS"],
            "iadl_disability": ["IADLDIS"],
        }
        mcq_condition_labels = {
            "MCQ160A": "arthritis",
            "MCQ160B": "heart_failure",
            "MCQ160C": "coronary_heart_disease",
            "MCQ160D": "angina",
            "MCQ160E": "heart_attack",
            "MCQ160F": "stroke",
            "MCQ160G": "emphysema",
            "MCQ160J": "overweight",
            "MCQ160K": "chronic_bronchitis",
            "MCQ160L": "liver_condition",
            "MCQ160M": "thyroid_problem",
            "MCQ170K": "still_chronic_bronchitis",
            "MCQ170L": "still_liver_condition",
            "MCQ170M": "still_thyroid_problem",
        }
        dynamic_mcq_codes = sorted(
            {
                c.upper()
                for c in df.columns
                if re.fullmatch(r"MCQ160[A-Z]", c.upper()) or re.fullmatch(r"MCQ170[A-Z]", c.upper())
            }
        )

        tmp = pd.DataFrame({"seqn": normalize_seqn(df), "cycle_start_year": cycle_year})
        avail = {"cycle_start_year": cycle_year, "file": p.name}
        for name, codes in fixed_conditions.items():
            cols = pick_cols(codes)
            avail[f"{name}_cols"] = "|".join(cols)
            if cols:
                tmp[name] = detect_any_yes(df, cols)
        for code in dynamic_mcq_codes:
            alias = mcq_condition_labels.get(code, f"{code.lower()}_condition")
            cols = pick_cols([code])
            avail[f"{alias}_cols"] = "|".join(cols)
            if cols:
                tmp[alias] = detect_any_yes(df, cols)
        avail["mcq_condition_codes"] = "|".join(dynamic_mcq_codes)

        per_cycle.append(tmp)
        avail_rows.append(avail)

    if not per_cycle:
        empty = pd.DataFrame(columns=["seqn", "cycle_start_year", "diabetes", "asthma", "cvd", "cancer", "kidney", "liver"])
        return empty, pd.DataFrame(avail_rows)

    flags = pd.concat(per_cycle, ignore_index=True)
    for c in ["diabetes", "asthma", "cvd", "cancer", "kidney", "liver"]:
        if c not in flags.columns:
            flags[c] = pd.Series([pd.NA] * len(flags), dtype="boolean")
    value_cols = [c for c in flags.columns if c not in {"seqn", "cycle_start_year"}]
    agg = flags.groupby(["seqn", "cycle_start_year"], as_index=False)[value_cols].max()

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
        if is_yes(r.get("liver", False)):
            reasons.append("liver")
        return "|".join(reasons)

    p["exclusion_reason"] = p.apply(row_reason, axis=1)
    p["healthy_flag"] = p["exclusion_reason"].eq("")
    p = p[p["age_years"] >= 20].copy()
    return p, availability


PREFIX_SCALE = {"": 1.0, "p": 1e-12, "n": 1e-9, "u": 1e-6, "m": 1e-3, "c": 1e-2, "d": 1e-1}
DEN_SCALE = {"l": 1.0, "dl": 1e-1, "ml": 1e-3, "ul": 1e-6}
UNIT_CONTEXT_SUFFIXES = ("serumlipid", "serum", "plasma", "blood", "rbc")
TERMINAL_UNIT_RE = re.compile(
    r"(?i)(?P<base>.*?)(?:\(|,|:|\s)\s*(?P<unit>[pnumcd]?(?:g|mol|iu|u|eq|kat)/(?:l|dl|ml|ul)(?:\s*(?:serum lipid|serum|plasma|blood|rbc))?)\s*\)?\s*$"
)
BASE_KEY_ALIASES = {
    "cadmium": "blood cadmium",
    "cholesterol": "total cholesterol",
    "cholesterol total": "total cholesterol",
    "folate rbc": "rbc folate",
    "folate serum": "serum folate",
    "lead": "blood lead",
    "selenium": "serum selenium",
    "testosterone": "testosterone total",
    "thyroid stimulating hormone tsh": "thyroid stimulating hormone",
    "thyroxine total t4": "thyroxine t4",
    "total cis- and trans- lycopene": "total lycopene",
    "total folate": "serum total folate",
    "triglycerides": "triglyceride",
    "vitamin b12 serum": "vitamin b12",
}
MISSING_UNIT_OVERRIDES = {
    "albumin": "g/dL",
    "thyroid stimulating hormone": "uIU/mL",
    "total protein": "g/dL",
}
SPECIAL_UNIT_FACTORS = {
    "dehydroepiandrosterone sulfate": {
        ("umol/l", "ug/dl"): 36.847,
        ("ug/dl", "umol/l"): 1.0 / 36.847,
    }
}


def parse_terminal_unit(label: str) -> tuple[str, str]:
    s = str(label or "").strip()
    m = re.search(r"\(([^()]*)\)\s*$", s)
    if m:
        inner = m.group(1).strip()
        inner_match = TERMINAL_UNIT_RE.match(inner)
        if inner_match:
            unit = str(inner_match.group("unit") or "").strip()
            inner_base = str(inner_match.group("base") or "").strip().rstrip(",:")
            base = s[: m.start()].strip().rstrip(",:")
            if inner_base:
                base = f"{base} {inner_base}".strip()
        else:
            unit = inner
            base = s[: m.start()].strip().rstrip(",:")
        return base, unit

    m = TERMINAL_UNIT_RE.match(s)
    if m:
        base = str(m.group("base") or "").strip().rstrip(",:")
        unit = str(m.group("unit") or "").strip()
        return base, unit
    return s, ""


def normalize_base_name(name: str) -> str:
    s = str(name or "").lower()
    repl = {"α": "a", "β": "b", "γ": "g", "δ": "d", "µ": "u", "μ": "u", "–": "-", "—": "-"}
    for k, v in repl.items():
        s = s.replace(k, v)
    s = re.sub(r"\s+", " ", s)
    s = re.sub(r"[^a-z0-9 %/+-]", " ", s)
    s = re.sub(r"\brefigerated\b", "refrigerated", s)
    s = re.sub(r"\brefrig\b", "refrigerated", s)
    s = re.sub(r"\brefrigerated serum\b", "", s)
    s = re.sub(r"\brefrigerated\b", "", s)
    s = re.sub(r"\bserum\b", "serum", s)
    s = re.sub(r"\s+", " ", s).strip()
    return s


def normalize_unit(unit: str) -> str:
    u = str(unit or "").strip().lower().replace("μ", "u").replace("µ", "u")
    u = u.replace(",", "")
    u = re.sub(r"\s+", "", u)
    u = re.sub(r"^u/(l|dl|ml|ul)$", r"iu/\1", u)
    u = u.replace("mosm/kg", "mmol/kg")
    for suffix in UNIT_CONTEXT_SUFFIXES:
        if u.endswith(suffix):
            u = u[: -len(suffix)]
    u = re.sub(r"[^a-z0-9/%]+", "", u)
    return u


def canonical_pool_group_key(base_key: str) -> str:
    key = normalize_base_name(base_key)
    return BASE_KEY_ALIASES.get(key, key)


def preferred_unit_rank(unit: str) -> tuple[int, int, str]:
    norm = normalize_unit(unit)
    sig = parse_unit_signature(norm)
    if sig is None:
        return (3, 3, norm)

    if sig["num_base"] == "iu":
        cls = 0
    elif sig["num_base"] == "g":
        cls = 1
    else:
        cls = 2
    den_rank = {"dl": 0, "ml": 1, "l": 2, "ul": 3}.get(re.search(r"(l|dl|ml|ul)$", norm).group(1), 9)
    return (cls, den_rank, norm)


def sort_group_for_reference(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out["_unit_rank"] = out["unit_effective"].map(preferred_unit_rank)
    out = out.sort_values(["_unit_rank", "size"], ascending=[True, False]).drop(columns="_unit_rank")
    return out.reset_index(drop=True)


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


def infer_empirical_conversion_factor(
    group_rows: pd.DataFrame,
    src_unit: str,
    dst_unit: str,
    raw_dir: Optional[Path],
) -> tuple[Optional[float], str]:
    if raw_dir is None:
        return None, ""

    src_norm = normalize_unit(src_unit)
    dst_norm = normalize_unit(dst_unit)
    factors: List[float] = []
    evidences: List[str] = []

    for url, g_url in group_rows.groupby("xpt_url", observed=True):
        src_vars = sorted(g_url.loc[g_url["unit_effective_norm"] == src_norm, "variable_name"].astype(str).unique())
        dst_vars = sorted(g_url.loc[g_url["unit_effective_norm"] == dst_norm, "variable_name"].astype(str).unique())
        if not src_vars or not dst_vars:
            continue

        year_match = re.search(r"/Public/(\d{4})/DataFiles/", str(url))
        if year_match is None:
            continue
        xpt_path = raw_dir / str(int(year_match.group(1))) / Path(str(url)).name
        if not xpt_path.exists():
            continue

        cols = ["SEQN", *src_vars, *dst_vars]
        try:
            df = read_xpt_columns(xpt_path, columns=cols)
        except Exception:
            continue

        for src_var in src_vars:
            for dst_var in dst_vars:
                if src_var not in df.columns or dst_var not in df.columns:
                    continue
                pair = (
                    pd.DataFrame(
                        {
                            "src": pd.to_numeric(df[src_var], errors="coerce"),
                            "dst": pd.to_numeric(df[dst_var], errors="coerce"),
                        }
                    )
                    .replace([np.inf, -np.inf], np.nan)
                    .dropna()
                )
                pair = pair[pair["src"] != 0]
                if len(pair) < 25:
                    continue
                ratio = (pair["dst"] / pair["src"]).replace([np.inf, -np.inf], np.nan).dropna()
                if len(ratio) < 25:
                    continue
                q10 = float(ratio.quantile(0.10))
                q90 = float(ratio.quantile(0.90))
                if q10 == 0 or not np.isfinite(q10) or not np.isfinite(q90):
                    continue
                if abs(q90 / q10 - 1.0) > 0.05:
                    continue
                factors.append(float(ratio.median()))
                evidences.append(f"{xpt_path.name}:{src_var}->{dst_var}:n={len(ratio)}")

    if not factors:
        return None, ""
    return float(np.median(np.asarray(factors, dtype=float))), "; ".join(evidences[:3])


def assume_missing_unit(pool_group_key: str, explicit_units: List[str]) -> str:
    override = MISSING_UNIT_OVERRIDES.get(pool_group_key)
    if override:
        return override

    explicit_non_empty = [u for u in explicit_units if str(u).strip()]
    if not explicit_non_empty:
        return ""

    unique_norm = {normalize_unit(u) for u in explicit_non_empty}
    if len(unique_norm) == 1:
        return explicit_non_empty[0]

    ref = explicit_non_empty[0]
    if all(conversion_factor(u, ref) is not None for u in explicit_non_empty):
        return ref
    return ""


def build_unit_conversion_edges(group_rows: pd.DataFrame, raw_dir: Optional[Path]) -> Dict[tuple[str, str], tuple[float, str, str]]:
    unit_pairs: Dict[tuple[str, str], tuple[float, str, str]] = {}
    units = sorted({str(u) for u in group_rows["unit_effective"].astype(str).unique() if str(u).strip()})
    norm_to_display = {normalize_unit(u): u for u in units}

    for src_norm, src_unit in norm_to_display.items():
        unit_pairs[(src_norm, src_norm)] = (1.0, "identity", "")

    for src_norm, src_unit in norm_to_display.items():
        for dst_norm, dst_unit in norm_to_display.items():
            if src_norm == dst_norm:
                continue
            generic = conversion_factor(src_unit, dst_unit)
            if generic is not None:
                unit_pairs[(src_norm, dst_norm)] = (float(generic), "unit_scale", "")
                continue

            special = SPECIAL_UNIT_FACTORS.get(str(group_rows["pool_group_key"].iloc[0]), {}).get((src_norm, dst_norm))
            if special is not None:
                unit_pairs[(src_norm, dst_norm)] = (float(special), "manual_clinical_conversion", "")
                continue

            inferred, evidence = infer_empirical_conversion_factor(
                group_rows,
                src_unit=src_unit,
                dst_unit=dst_unit,
                raw_dir=raw_dir,
            )
            if inferred is not None:
                unit_pairs[(src_norm, dst_norm)] = (float(inferred), "paired_si_conversion", evidence)

    return unit_pairs


def resolve_unit_path(
    src_norm: str,
    dst_norm: str,
    edges: Dict[tuple[str, str], tuple[float, str, str]],
) -> tuple[Optional[float], str, str]:
    if src_norm == dst_norm:
        return 1.0, "identity", ""
    if (src_norm, dst_norm) in edges:
        factor, method, notes = edges[(src_norm, dst_norm)]
        return float(factor), method, notes

    neighbors: Dict[str, List[tuple[str, float, str, str]]] = defaultdict(list)
    for (src, dst), (factor, method, notes) in edges.items():
        neighbors[src].append((dst, float(factor), method, notes))

    queue: List[tuple[str, float, List[str], List[str]]] = [(src_norm, 1.0, [], [])]
    seen = {src_norm}
    while queue:
        node, factor_so_far, methods, notes = queue.pop(0)
        for nxt, edge_factor, edge_method, edge_note in neighbors.get(node, []):
            if nxt in seen:
                continue
            next_factor = factor_so_far * edge_factor
            next_methods = methods + ([edge_method] if edge_method != "identity" else [])
            next_notes = notes + ([edge_note] if edge_note else [])
            if nxt == dst_norm:
                method = " > ".join(next_methods) if next_methods else "identity"
                return next_factor, method, "; ".join(next_notes[:3])
            seen.add(nxt)
            queue.append((nxt, next_factor, next_methods, next_notes))
    return None, "", ""


def build_pooling_map(
    lab_manifest: pd.DataFrame,
    raw_dir: Optional[Path] = None,
    candidate_column: str = "is_blood_candidate",
) -> pd.DataFrame:
    if candidate_column not in lab_manifest.columns:
        raise ValueError(f"Candidate column not found in manifest: {candidate_column}")
    selected = lab_manifest[lab_manifest[candidate_column]].copy()
    var_counts = (
        selected.groupby(["variable_name", "variable_desc", "xpt_url"], as_index=False)
        .size()
        .sort_values(["variable_name", "size"], ascending=[True, False])
    )
    var_counts["base_name"], var_counts["unit_raw"] = zip(*var_counts["variable_desc"].map(parse_terminal_unit))
    var_counts["base_key"] = var_counts["base_name"].map(normalize_base_name)
    var_counts["pool_group_key"] = var_counts["base_key"].map(canonical_pool_group_key)
    explicit_units_by_group = (
        var_counts.loc[var_counts["unit_raw"].astype(str).str.strip().ne(""), ["pool_group_key", "unit_raw"]]
        .drop_duplicates()
        .groupby("pool_group_key", observed=True)["unit_raw"]
        .agg(list)
        .to_dict()
    )
    var_counts["unit_effective"] = var_counts.apply(
        lambda r: str(r["unit_raw"]).strip()
        or assume_missing_unit(str(r["pool_group_key"]), explicit_units_by_group.get(str(r["pool_group_key"]), [])),
        axis=1,
    )
    var_counts["unit_effective_norm"] = var_counts["unit_effective"].map(normalize_unit)

    aggregated = (
        var_counts.groupby(
            ["variable_name", "variable_desc", "base_name", "base_key", "pool_group_key", "unit_raw", "unit_effective", "unit_effective_norm"],
            as_index=False,
        )
        .agg(size=("size", "sum"), xpt_urls=("xpt_url", lambda s: "|".join(sorted(set(map(str, s))))))
    )

    rows = []
    for pool_group_key, g_base in aggregated.groupby("pool_group_key", observed=True):
        g_base = sort_group_for_reference(g_base)
        explicit_first = g_base[g_base["unit_effective_norm"].ne("")]
        ref_row = explicit_first.iloc[0] if not explicit_first.empty else g_base.iloc[0]
        ref_unit = str(ref_row["unit_effective"] or "").strip()
        ref_base_name = str(ref_row["base_name"] or "").strip()
        if not ref_base_name:
            ref_base_name = str(pool_group_key)

        conversion_edges = build_unit_conversion_edges(
            var_counts[var_counts["pool_group_key"] == pool_group_key],
            raw_dir=raw_dir,
        )
        compatible_rows = []
        incompatible_buckets: Dict[str, List[pd.Series]] = defaultdict(list)

        for _, r in g_base.iterrows():
            src_unit = str(r["unit_effective"] or "").strip()
            src_norm = normalize_unit(src_unit)
            factor = 1.0
            method = "identity"
            notes = ""

            if ref_unit and src_unit and src_norm != normalize_unit(ref_unit):
                resolved, resolved_method, resolved_notes = resolve_unit_path(
                    src_norm,
                    normalize_unit(ref_unit),
                    conversion_edges,
                )
                if resolved is None:
                    bucket = src_norm or f"raw:{normalize_base_name(str(r['variable_desc']))}"
                    incompatible_buckets[bucket].append(r)
                    continue
                factor = float(resolved)
                method = resolved_method
                notes = resolved_notes
            elif not src_unit and ref_unit:
                bucket = f"raw:{normalize_base_name(str(r['variable_desc']))}"
                incompatible_buckets[bucket].append(r)
                continue

            compatible_rows.append((r, factor, method, notes))

        pooled_groups: List[tuple[str, str, str, List[tuple[pd.Series, float, str, str]]]] = []
        pooled_id = str(pool_group_key)
        pooled_name = ref_base_name if not ref_unit else f"{ref_base_name} ({ref_unit})"
        pooled_groups.append((pooled_id, pooled_name, ref_unit, compatible_rows))

        for bucket, bucket_rows in sorted(incompatible_buckets.items()):
            bucket_df = pd.DataFrame(bucket_rows)
            bucket_ref = sort_group_for_reference(bucket_df).iloc[0]
            bucket_unit = str(bucket_ref["unit_effective"] or "").strip()
            bucket_name = str(bucket_ref["base_name"] or pool_group_key).strip()
            suffix = normalize_unit(bucket_unit) or bucket.replace(":", "_")
            bucket_id = f"{pool_group_key}__{suffix}"
            bucket_display = bucket_name if not bucket_unit else f"{bucket_name} ({bucket_unit})"
            pooled_groups.append(
                (
                    bucket_id,
                    bucket_display,
                    bucket_unit,
                    [(r, 1.0, "incompatible_unit_kept_separate", "") for r in bucket_rows],
                )
            )

        for group_id, group_name, group_unit, members in pooled_groups:
            for r, factor, method, notes in members:
                rows.append(
                    {
                        "variable_name": str(r["variable_name"]),
                        "variable_desc": str(r["variable_desc"]),
                        "base_key": str(r["base_key"]),
                        "pool_group_key": str(pool_group_key),
                        "source_unit_raw": str(r["unit_raw"] or "").strip(),
                        "source_unit_effective": str(r["unit_effective"] or "").strip(),
                        "pooled_id": group_id,
                        "pooled_name": group_name,
                        "pooled_unit": group_unit,
                        "conversion_factor_to_pooled_unit": float(factor),
                        "conversion_method": method,
                        "conversion_notes": notes,
                    }
                )

    return pd.DataFrame(rows)


def write_pooling_documentation(pooling_map_df: pd.DataFrame, processed_dir: Path) -> None:
    if pooling_map_df.empty:
        return

    ensure_dir(processed_dir)
    detailed_path = processed_dir / "duplicate_merge_map.csv"
    summary_path = processed_dir / "duplicate_merge_summary.csv"
    report_path = processed_dir / "duplicate_merge_report.md"

    detailed = pooling_map_df.copy()
    detailed["is_identity"] = np.isclose(detailed["conversion_factor_to_pooled_unit"], 1.0)
    detailed["had_missing_unit"] = detailed["source_unit_raw"].fillna("").astype(str).str.strip().eq("") & detailed[
        "source_unit_effective"
    ].fillna("").astype(str).str.strip().ne("")
    detailed = detailed.sort_values(["pool_group_key", "pooled_id", "variable_desc", "variable_name"]).reset_index(drop=True)

    summary_rows: List[dict] = []
    report_lines = [
        "# Duplicate Merge Report",
        "",
        "This report documents only the non-trivial duplicate-handling decisions: alias merges, unit conversions, missing-unit assumptions, and groups intentionally left separate.",
        "",
    ]

    for pool_group_key, g in detailed.groupby("pool_group_key", observed=True):
        pooled_ids = sorted(g["pooled_id"].astype(str).unique())
        base_keys = sorted(g["base_key"].astype(str).unique())
        source_units = sorted({u for u in g["source_unit_raw"].astype(str).unique() if u})
        effective_units = sorted({u for u in g["source_unit_effective"].astype(str).unique() if u})
        methods = sorted(g["conversion_method"].astype(str).unique())
        notes = [n for n in g["conversion_notes"].astype(str).unique() if n]

        if len(pooled_ids) == 1:
            decision = "merged"
        elif len(base_keys) > 1 or any(m != "incompatible_unit_kept_separate" for m in methods):
            decision = "partially_merged"
        else:
            decision = "kept_separate"

        reasons = []
        if len(base_keys) > 1:
            reasons.append("name aliases")
        if any(m == "unit_scale" for m in methods):
            reasons.append("scaled units")
        if any(m == "paired_si_conversion" for m in methods):
            reasons.append("paired SI/conventional conversion")
        if g["had_missing_unit"].any():
            reasons.append("missing-unit assumption")
        if decision != "merged" and len(pooled_ids) > 1:
            reasons.append("incompatible units/assays left separate")
        reason_text = ", ".join(reasons) if reasons else "exact-name pooling"

        is_interesting = (
            len(base_keys) > 1
            or len(source_units) > 1
            or any(m != "identity" for m in methods)
            or g["had_missing_unit"].any()
            or len(pooled_ids) > 1
        )
        if not is_interesting:
            continue

        summary_rows.append(
            {
                "pool_group_key": pool_group_key,
                "decision": decision,
                "pooled_ids": "|".join(pooled_ids),
                "base_keys": "|".join(base_keys),
                "source_units_raw": "|".join(source_units),
                "source_units_effective": "|".join(effective_units),
                "pooled_units": "|".join(sorted({u for u in g['pooled_unit'].astype(str).unique() if u})),
                "conversion_methods": "|".join(methods),
                "reason": reason_text,
                "conversion_notes": " | ".join(notes[:3]),
            }
        )

        report_lines.append(f"## {pool_group_key}")
        report_lines.append(f"- Decision: `{decision}`")
        report_lines.append(f"- Why: {reason_text}")
        report_lines.append(f"- Base names seen: `{', '.join(base_keys)}`")
        report_lines.append(f"- Source units: `{', '.join(source_units) if source_units else 'none recorded'}`")
        report_lines.append(f"- Effective units used: `{', '.join(effective_units) if effective_units else 'none'}`")
        report_lines.append(f"- Output IDs: `{', '.join(pooled_ids)}`")
        if notes:
            report_lines.append(f"- Evidence: `{notes[0]}`")
        report_lines.append("")

    filtered = detailed[detailed["pool_group_key"].isin({row["pool_group_key"] for row in summary_rows})].copy()
    filtered.to_csv(detailed_path, index=False)
    pd.DataFrame(summary_rows).sort_values(["decision", "pool_group_key"]).to_csv(summary_path, index=False)
    report_path.write_text("\n".join(report_lines).rstrip() + "\n", encoding="utf-8")


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
    candidate_column: str = "is_blood_candidate",
) -> Tuple[int, int, int]:
    if candidate_column not in lab_manifest.columns:
        raise ValueError(f"Candidate column not found in manifest: {candidate_column}")
    selected = lab_manifest[lab_manifest[candidate_column]].copy()
    selected = selected.drop_duplicates(subset=["xpt_url", "variable_name"]).reset_index(drop=True)

    file_meta = (
        selected[["data_file_name", "cycle_label", "cycle_start_year", "cycle_end_year", "xpt_url", "data_file_desc"]]
        .drop_duplicates(subset=["xpt_url"])
        .set_index("xpt_url")
    )

    vars_by_url: Dict[str, pd.DataFrame] = {
        url: g[["variable_name", "variable_desc"]].drop_duplicates().reset_index(drop=True)
        for url, g in selected.groupby("xpt_url")
    }

    pooling_map_df = build_pooling_map(lab_manifest, raw_dir=raw_dir, candidate_column=candidate_column)
    write_pooling_documentation(pooling_map_df, processed_dir)
    pooling_map = pooling_map_df.set_index(["variable_name", "variable_desc"]).to_dict(orient="index")

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
            elif (var, vdesc) not in pooling_map:
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

            pool = pooling_map[(var, vdesc)]
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

    kept_from_manifest = selected[selected["variable_name"].isin(kept_variables)].copy()
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
    ap.add_argument("--candidate-column", default="is_blood_candidate")
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
        candidate_column=args.candidate_column,
    )

    specimen_label = "blood"
    if args.candidate_column == "is_urine_candidate":
        specimen_label = "urine"
    elif args.candidate_column not in {"is_blood_candidate", "is_urine_candidate"}:
        specimen_label = args.candidate_column

    print(f"Participant rows (age>=20): {len(participants):,}")
    print(f"Processed {specimen_label} lab files: {n_files:,}")
    print(f"Pooled biomarkers kept: {n_vars:,}")
    print(f"Long biomarker rows written: {n_rows:,}")
    print(f"Dataset: {out_dir / 'biomarker_long.parquet'}")


if __name__ == "__main__":
    main()
