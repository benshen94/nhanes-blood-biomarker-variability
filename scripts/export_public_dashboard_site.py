#!/usr/bin/env python3
"""Export the audience-facing dashboard as a minimal static site bundle."""

from __future__ import annotations

import json
import shutil
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
DASHBOARD_DIR = ROOT / "dashboard"
OUTPUT_DIR = ROOT / "output" / "public_dashboard_site"


def ensure_clean_dir(path: Path) -> None:
    if path.exists():
        shutil.rmtree(path)
    path.mkdir(parents=True, exist_ok=True)


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


def write_export_readme(out_dir: Path) -> None:
    text = """# Public Dashboard Site Export

This folder is ready to become a separate public GitHub Pages repository.

## Files included
- `index.html`
- `longevity-explorer.html`
- `aging_biomarkers_dashboard.html`
- `aging_biomarkers_public/`
- `data/series/` for the curated biomarker subset only

## Suggested deployment
1. Create a new GitHub repository for the public site only.
2. Copy the contents of this folder into that repo root.
3. Push the repo.
4. Enable GitHub Pages from the repo root branch.

## Notes
- This export includes only the files needed by the audience-facing dashboard.
- It does not include the full analysis repo, source notebooks, or unrelated dashboard assets.
- The dashboard expects `aging_biomarkers_public/` and `data/series/` to live next to the HTML files.
"""
    (out_dir / "README.md").write_text(text, encoding="utf-8")


def export_site() -> Path:
    ensure_clean_dir(OUTPUT_DIR)

    copy_file(DASHBOARD_DIR / "aging_biomarkers_dashboard.html", OUTPUT_DIR / "aging_biomarkers_dashboard.html")
    copy_file(DASHBOARD_DIR / "longevity-explorer.html", OUTPUT_DIR / "longevity-explorer.html")
    copy_file(DASHBOARD_DIR / "dashboard_data_aging_biomarkers.json", OUTPUT_DIR / "dashboard_data_aging_biomarkers.json")
    copy_file(DASHBOARD_DIR / ".nojekyll", OUTPUT_DIR / ".nojekyll") if (DASHBOARD_DIR / ".nojekyll").exists() else (OUTPUT_DIR / ".nojekyll").write_text("", encoding="utf-8")

    write_index_redirect(OUTPUT_DIR)
    write_export_readme(OUTPUT_DIR)

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
