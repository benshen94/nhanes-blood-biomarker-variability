#!/usr/bin/env python3
"""Export the audience-facing dashboard as a minimal static site bundle."""

from __future__ import annotations

import json
import shutil
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
DASHBOARD_DIR = ROOT / "dashboard"
AGING_DIR = ROOT.parent
OUTPUT_DIR = AGING_DIR / "biomarker_dashboard"
PRESERVED_FILES = {".git", "README.md", "AGENTS.md"}


def ensure_clean_export_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)

    for child in path.iterdir():
        if child.name in PRESERVED_FILES:
            continue
        if child.is_dir():
            shutil.rmtree(child)
            continue
        child.unlink()


def copy_file(src: Path, dst: Path) -> None:
    dst.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(src, dst)


def manifest_series_paths() -> list[str]:
    manifest_path = DASHBOARD_DIR / "aging_biomarkers_public" / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    return sorted({str(item["detail_series_path"]) for item in manifest})


def write_index_redirect(out_dir: Path) -> None:
    source = (DASHBOARD_DIR / "longevity-explorer.html").read_text(encoding="utf-8")
    (out_dir / "index.html").write_text(source, encoding="utf-8")


def export_site() -> Path:
    ensure_clean_export_dir(OUTPUT_DIR)

    copy_file(DASHBOARD_DIR / "aging_biomarkers_dashboard.html", OUTPUT_DIR / "aging_biomarkers_dashboard.html")
    copy_file(DASHBOARD_DIR / "longevity-explorer.html", OUTPUT_DIR / "longevity-explorer.html")
    copy_file(DASHBOARD_DIR / "dashboard_data_aging_biomarkers.json", OUTPUT_DIR / "dashboard_data_aging_biomarkers.json")
    copy_file(DASHBOARD_DIR / ".nojekyll", OUTPUT_DIR / ".nojekyll") if (DASHBOARD_DIR / ".nojekyll").exists() else (OUTPUT_DIR / ".nojekyll").write_text("", encoding="utf-8")

    write_index_redirect(OUTPUT_DIR)

    public_data_dir = DASHBOARD_DIR / "aging_biomarkers_public"
    shutil.copytree(public_data_dir, OUTPUT_DIR / "aging_biomarkers_public", dirs_exist_ok=True)

    for rel_path in manifest_series_paths():
        copy_file(DASHBOARD_DIR / rel_path, OUTPUT_DIR / rel_path)

    return OUTPUT_DIR


def main() -> None:
    out_dir = export_site()
    print(f"Exported public dashboard site to: {out_dir}")


if __name__ == "__main__":
    main()
