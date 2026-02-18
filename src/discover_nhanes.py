#!/usr/bin/env python3
"""Discover NHANES laboratory variables and build blood biomarker manifest."""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd
import urllib3

from nhanes_common import apply_blood_candidate_rule, ensure_dir, parse_component_datapage, parse_variablelist


def build_manifest(component: str, verify_ssl: bool = False) -> pd.DataFrame:
    files_df = parse_component_datapage(component=component, verify_ssl=verify_ssl)
    var_df = parse_variablelist(component=component, verify_ssl=verify_ssl)

    var_df = var_df.rename(
        columns={
            "Variable Name": "variable_name",
            "Variable Description": "variable_desc",
            "Data File Name": "data_file_name",
            "Data File Description": "data_file_desc_from_varlist",
            "Begin Year": "begin_year",
            "EndYear": "end_year",
            "Use Constraints": "use_constraints",
        }
    )

    var_df["begin_year"] = pd.to_numeric(var_df["begin_year"], errors="coerce").astype("Int64")
    var_df["end_year"] = pd.to_numeric(var_df["end_year"], errors="coerce").astype("Int64")

    merged = var_df.merge(
        files_df,
        on="data_file_name",
        how="left",
        suffixes=("", "_from_datapage"),
    )

    merged["data_file_desc"] = merged["data_file_desc"].fillna(merged["data_file_desc_from_varlist"])

    merged["is_blood_candidate"] = merged.apply(
        lambda r: apply_blood_candidate_rule(
            data_file_desc=str(r.get("data_file_desc", "")),
            variable_desc=str(r.get("variable_desc", "")),
            use_constraints=str(r.get("use_constraints", "")),
            variable_name=str(r.get("variable_name", "")),
        ),
        axis=1,
    )

    cols = [
        "cycle_start_year",
        "cycle_end_year",
        "cycle_label",
        "data_file_name",
        "data_file_desc",
        "xpt_url",
        "doc_url",
        "date_published",
        "variable_name",
        "variable_desc",
        "use_constraints",
        "is_blood_candidate",
    ]

    out = merged[cols].copy()
    out = out.dropna(subset=["xpt_url", "variable_name", "data_file_name"])
    out = out.sort_values(["data_file_name", "variable_name", "cycle_start_year"]).reset_index(drop=True)
    return out


def verify_sample_urls(df: pd.DataFrame, n: int, verify_ssl: bool = False) -> pd.DataFrame:
    import requests

    sampled = df[["xpt_url"]].drop_duplicates().head(n).copy()
    statuses = []
    for url in sampled["xpt_url"]:
        try:
            resp = requests.head(url, timeout=30, verify=verify_ssl, allow_redirects=True)
            statuses.append(resp.status_code)
        except Exception:
            statuses.append(None)
    sampled["http_status"] = statuses
    return sampled


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--component", default="Laboratory")
    ap.add_argument("--out", default="data/processed/lab_variable_manifest.parquet")
    ap.add_argument("--out-csv", default="data/processed/lab_variable_manifest.csv")
    ap.add_argument("--verify-ssl", action="store_true", help="Enable TLS cert verification")
    ap.add_argument("--verify-urls", action="store_true", help="HEAD check sample URLs")
    ap.add_argument("--verify-sample-size", type=int, default=200)
    args = ap.parse_args()
    if not args.verify_ssl:
        urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)

    out_path = Path(args.out)
    csv_path = Path(args.out_csv)
    ensure_dir(out_path.parent)

    manifest = build_manifest(component=args.component, verify_ssl=args.verify_ssl)
    manifest.to_parquet(out_path, index=False)
    manifest.to_csv(csv_path, index=False)

    print(f"Saved manifest: {out_path} ({len(manifest):,} rows)")
    print(f"Blood candidates: {manifest['is_blood_candidate'].sum():,}")

    rdc_included = manifest.loc[
        manifest["use_constraints"].fillna("").str.contains("RDC", case=False) & manifest["is_blood_candidate"]
    ]
    print(f"RDC rows included as blood candidates: {len(rdc_included)}")

    if args.verify_urls:
        sample_df = verify_sample_urls(manifest, n=args.verify_sample_size, verify_ssl=args.verify_ssl)
        sample_out = out_path.parent / "manifest_url_verification_sample.csv"
        sample_df.to_csv(sample_out, index=False)
        ok = sample_df["http_status"].eq(200).sum()
        print(f"URL sample verification saved: {sample_out} ({ok}/{len(sample_df)} HTTP 200)")


if __name__ == "__main__":
    main()
