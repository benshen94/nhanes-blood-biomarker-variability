#!/usr/bin/env python3
"""Download selected NHANES files for biomarker + covariate processing."""

from __future__ import annotations

import argparse
import re
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd
import requests
import urllib3

from nhanes_common import cycle_year_from_url, ensure_dir, parse_component_datapage, sha256_file


def select_download_urls(manifest_df: pd.DataFrame, verify_ssl: bool = False) -> pd.DataFrame:
    blood_files = manifest_df.loc[manifest_df["is_blood_candidate"], ["xpt_url", "cycle_start_year"]].drop_duplicates()
    blood_files["source"] = "laboratory"

    demo_df = parse_component_datapage(component="Demographics", verify_ssl=verify_ssl)
    q_df = parse_component_datapage(component="Questionnaire", verify_ssl=verify_ssl)

    demo_sel = demo_df.loc[
        demo_df["data_file_name"].str.match(r"^(DEMO.*|P_DEMO|DEMO_L|CDEMO.*)$"),
        ["xpt_url", "cycle_start_year", "data_file_name"],
    ].copy()
    demo_sel["source"] = "demographics"

    q_sel = q_df.loc[
        q_df["data_file_name"].str.match(r"^(DIQ.*|MCQ.*|KIQ_U.*)$"),
        ["xpt_url", "cycle_start_year", "data_file_name"],
    ].copy()
    q_sel["source"] = "questionnaire"

    blood_files = blood_files.assign(data_file_name=blood_files["xpt_url"].str.extract(r"/([^/]+)\.xpt$", expand=False))

    all_files = pd.concat([blood_files, demo_sel, q_sel], ignore_index=True)
    all_files["cycle_start_year"] = all_files["cycle_start_year"].fillna(
        all_files["xpt_url"].map(cycle_year_from_url)
    )
    all_files = all_files.drop_duplicates(subset=["xpt_url"]).sort_values(["source", "cycle_start_year", "xpt_url"])
    return all_files.reset_index(drop=True)


def download_one(url: str, out_path: Path, verify_ssl: bool = False, timeout: int = 180) -> tuple[int, int, str]:
    ensure_dir(out_path.parent)

    if out_path.exists() and out_path.stat().st_size > 0:
        checksum = sha256_file(out_path)
        return 200, out_path.stat().st_size, checksum

    resp = requests.get(url, timeout=timeout, verify=verify_ssl)
    status = resp.status_code
    if status != 200:
        return status, 0, ""

    out_path.write_bytes(resp.content)
    checksum = sha256_file(out_path)
    return status, out_path.stat().st_size, checksum


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest", default="data/processed/lab_variable_manifest.parquet")
    ap.add_argument("--out", default="data/raw")
    ap.add_argument("--download-manifest", default="data/processed/download_manifest.csv")
    ap.add_argument("--verify-ssl", action="store_true")
    args = ap.parse_args()
    if not args.verify_ssl:
        urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)

    manifest_path = Path(args.manifest)
    out_dir = Path(args.out)
    dl_manifest_path = Path(args.download_manifest)
    ensure_dir(out_dir)
    ensure_dir(dl_manifest_path.parent)

    manifest_df = pd.read_parquet(manifest_path)
    urls_df = select_download_urls(manifest_df, verify_ssl=args.verify_ssl)

    records = []
    total = len(urls_df)
    for i, (_, row) in enumerate(urls_df.iterrows(), start=1):
        url = row["xpt_url"]
        year = int(row["cycle_start_year"])
        fname = Path(url).name
        out_path = out_dir / str(year) / fname

        status, n_bytes, checksum = download_one(url, out_path=out_path, verify_ssl=args.verify_ssl)
        records.append(
            {
                "timestamp_utc": datetime.now(timezone.utc).isoformat(),
                "source": row["source"],
                "cycle_start_year": year,
                "data_file_name": row.get("data_file_name", Path(fname).stem),
                "url": url,
                "status": status,
                "bytes": n_bytes,
                "sha256": checksum,
                "output_path": str(out_path),
            }
        )
        if i % 50 == 0 or i == total:
            ok_so_far = sum(1 for r in records if r["status"] == 200)
            print(f"[{i}/{total}] ok={ok_so_far}")

    out_df = pd.DataFrame(records)
    out_df.to_csv(dl_manifest_path, index=False)

    ok = int((out_df["status"] == 200).sum())
    print(f"Downloaded/verified {ok}/{len(out_df)} files")
    print(f"Manifest: {dl_manifest_path}")


if __name__ == "__main__":
    main()
