#!/usr/bin/env python3
"""Common helpers for NHANES scraping, parsing, and IO."""

from __future__ import annotations

import hashlib
import io
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional

import pandas as pd
import requests
from bs4 import BeautifulSoup
import urllib3

BASE = "https://wwwn.cdc.gov"


@dataclass(frozen=True)
class ComponentRow:
    cycle_label: str
    cycle_start_year: int
    cycle_end_year: int
    data_file_name: str
    data_file_desc: str
    doc_url: str
    xpt_url: str
    date_published: str


def fetch_html(url: str, verify_ssl: bool = False, timeout: int = 120) -> str:
    if not verify_ssl:
        urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)
    resp = requests.get(url, timeout=timeout, verify=verify_ssl)
    resp.raise_for_status()
    return resp.text


def parse_cycle_years(cycle_label: str) -> tuple[int, int]:
    years = [int(y) for y in re.findall(r"(19\d{2}|20\d{2})", cycle_label)]
    if len(years) >= 2:
        return years[0], years[-1]
    if len(years) == 1:
        return years[0], years[0]
    raise ValueError(f"Could not parse cycle years from: {cycle_label!r}")


def filename_from_url(url: str) -> str:
    return Path(url.split("?")[0]).name


def file_stem_from_url(url: str) -> str:
    return Path(filename_from_url(url)).stem


def cycle_year_from_url(url: str) -> Optional[int]:
    m = re.search(r"/Public/(\d{4})/DataFiles/", url)
    return int(m.group(1)) if m else None


def parse_component_datapage(component: str, verify_ssl: bool = False) -> pd.DataFrame:
    url = f"{BASE}/nchs/nhanes/search/DataPage.aspx?Component={component}"
    html = fetch_html(url, verify_ssl=verify_ssl)
    soup = BeautifulSoup(html, "html.parser")

    table = soup.find("table", {"id": "GridView1"})
    if table is None:
        raise RuntimeError(f"GridView1 table not found for component={component}")

    rows: List[ComponentRow] = []
    for tr in table.find_all("tr"):
        tds = tr.find_all("td")
        if len(tds) != 5:
            continue

        cycle_label = tds[0].get_text(" ", strip=True)
        data_file_desc = tds[1].get_text(" ", strip=True)
        doc_anchor = tds[2].find("a")
        data_anchor = tds[3].find("a")
        date_published = tds[4].get_text(" ", strip=True)

        if doc_anchor is None or data_anchor is None:
            continue

        doc_href = doc_anchor.get("href", "")
        xpt_href = data_anchor.get("href", "")
        doc_url = doc_href if doc_href.startswith("http") else f"{BASE}{doc_href}"
        xpt_url = xpt_href if xpt_href.startswith("http") else f"{BASE}{xpt_href}"

        start, end = parse_cycle_years(cycle_label)
        data_file_name = file_stem_from_url(xpt_url)

        rows.append(
            ComponentRow(
                cycle_label=cycle_label,
                cycle_start_year=start,
                cycle_end_year=end,
                data_file_name=data_file_name,
                data_file_desc=data_file_desc,
                doc_url=doc_url,
                xpt_url=xpt_url,
                date_published=date_published,
            )
        )

    if not rows:
        raise RuntimeError(f"No data rows parsed for component={component}")

    return pd.DataFrame([r.__dict__ for r in rows])


def parse_variablelist(component: str, verify_ssl: bool = False) -> pd.DataFrame:
    url = f"{BASE}/nchs/nhanes/search/variablelist.aspx?Component={component}&Cycle=%20"
    html = fetch_html(url, verify_ssl=verify_ssl)
    df = pd.read_html(io.StringIO(html))[0]
    expected = {
        "Variable Name",
        "Variable Description",
        "Data File Name",
        "Data File Description",
        "Begin Year",
        "EndYear",
        "Component",
        "Use Constraints",
    }
    missing = expected - set(df.columns)
    if missing:
        raise RuntimeError(f"Missing expected variable list columns: {sorted(missing)}")
    return df


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def normalize_numeric_series(series: pd.Series) -> pd.Series:
    return pd.to_numeric(series, errors="coerce")


def parse_unit_from_label(label: str) -> str:
    m = re.search(r"\(([^)]+)\)", label or "")
    return m.group(1).strip() if m else ""


def apply_blood_candidate_rule(data_file_desc: str, variable_desc: str, use_constraints: str, variable_name: str = "") -> bool:
    txt = f"{data_file_desc} {variable_desc} {variable_name}".lower()
    use_txt = (use_constraints or "").lower()

    include_tokens = ["blood", "serum", "plasma", "whole blood", "rbc", "wbc"]
    exclude_tokens = ["urine", "urinary", "saliva", "oral", "vaginal", "semen", "hair", "nail", "milk", "csf"]

    has_include = any(tok in txt for tok in include_tokens)
    has_exclude = any(tok in txt for tok in exclude_tokens)
    # Many blood analytes are coded with LBX/LBD-style names and do not mention "blood/serum" in every description.
    has_lab_marker = bool(re.search(r"\b(lbx[a-z0-9]*|lbd[a-z0-9]*|sst[a-z0-9]*|ss[a-z0-9]+)\b", txt))
    is_rdc = "rdc" in use_txt
    return bool((has_include or has_lab_marker) and not has_exclude and not is_rdc)


def list_xpt_files(raw_dir: Path, pattern: str = "*.xpt") -> List[Path]:
    return sorted(raw_dir.rglob(pattern))


def decode_filename_to_cycle(filename: str) -> str:
    stem = Path(filename).stem
    if stem.startswith("P_"):
        return "2017-March 2020 Pre-Pandemic"
    return stem
