#!/usr/bin/env python3

import json
import sys
import tempfile
import unittest
from pathlib import Path

import pandas as pd

sys.path.insert(0, str((Path(__file__).resolve().parents[1] / "src")))

from build_aging_biomarkers_dashboard import (
    build_public_manifest,
    render_public_dashboard_html,
    write_public_dashboard_bundle,
)


def make_point(age_bin, age_mid, median, q25, q75, p10, p90, std, cv, qskew=0.0):
    return {
        "age_bin": age_bin,
        "age_mid": age_mid,
        "median": median,
        "q25": q25,
        "q75": q75,
        "p10": p10,
        "p90": p90,
        "iqr": q75 - q25,
        "std": std,
        "cv": cv,
        "quantile_skewness": qskew,
    }


class TestBuildAgingBiomarkersDashboard(unittest.TestCase):
    def test_build_public_manifest_includes_context_metrics_and_catalog_copy(self):
        metadata = pd.DataFrame(
            [
                {
                    "biomarker_id": "c-reactive protein",
                    "biomarker_name": "C-reactive protein (mg/dL)",
                    "display_name": "C-reactive protein (mg/dL)",
                    "unit": "mg/dL",
                    "variable_name": "LBXCRP",
                }
            ]
        )
        series_index = {"c-reactive protein": "series/c-reactive_protein.json"}
        series_payloads = {
            "series/c-reactive_protein.json": {
                "raw_total_n": 1200,
                "raw_total_n_by_sex": {"female": 650, "male": 550},
                "points_by_filter": {
                    "all": [
                        make_point("20-24", 22.5, 1.0, 0.8, 1.2, 0.6, 1.4, 0.3, 0.30, 0.10),
                        make_point("80-84", 82.5, 3.0, 2.2, 4.0, 1.4, 5.2, 0.9, 0.60, 0.35),
                    ],
                    "trim_10_90": [
                        make_point("20-24", 22.5, 1.0, 0.8, 1.2, 0.6, 1.4, 0.3, 0.30, 0.10),
                        make_point("80-84", 82.5, 3.0, 2.2, 4.0, 1.4, 5.2, 0.9, 0.60, 0.35),
                    ],
                },
                "sex_points_by_filter": {
                    "all": {
                        "female": [
                            make_point("20-24", 22.5, 1.0, 0.8, 1.2, 0.6, 1.4, 0.2, 0.20, 0.08),
                            make_point("80-84", 82.5, 2.0, 1.5, 2.6, 1.0, 3.1, 0.4, 0.40, 0.15),
                        ],
                        "male": [
                            make_point("20-24", 22.5, 1.0, 0.8, 1.2, 0.6, 1.4, 0.2, 0.20, 0.08),
                            make_point("80-84", 82.5, 3.0, 2.4, 3.8, 1.8, 4.9, 0.6, 0.50, 0.22),
                        ],
                    },
                    "trim_10_90": {
                        "female": [
                            make_point("20-24", 22.5, 1.0, 0.8, 1.2, 0.6, 1.4, 0.2, 0.20, 0.08),
                            make_point("80-84", 82.5, 2.0, 1.5, 2.6, 1.0, 3.1, 0.4, 0.40, 0.15),
                        ],
                        "male": [
                            make_point("20-24", 22.5, 1.0, 0.8, 1.2, 0.6, 1.4, 0.2, 0.20, 0.08),
                            make_point("80-84", 82.5, 3.0, 2.4, 3.8, 1.8, 4.9, 0.6, 0.50, 0.22),
                        ],
                    },
                },
                "clalit_data": {"pooled": [{"age_bin": "20-24", "median": 0.9}]},
            }
        }

        with tempfile.TemporaryDirectory() as tmpdir:
            catalog_path = Path(tmpdir) / "aging_biomarkers.csv"
            pd.DataFrame(
                [
                    {
                        "test_name": "CRP",
                        "aliases": "",
                        "category": "Inflammation",
                        "subcategory": "Inflammation",
                        "primary_organ_system": "immune / liver",
                        "aging_domain": "inflammaging",
                        "measurement_class": "routine_clinical",
                        "specimen": "blood",
                        "clock_relevance": "PhenoAge",
                        "in_nhanes_workspace": "Yes",
                        "in_nhanes_dashboard": "Yes",
                        "in_nhanes_manifest": "Yes",
                        "nhanes_presence_status": "dashboard_biomarker",
                        "nhanes_match_name": "C-reactive protein (mg/dL)",
                        "nhanes_dashboard_category": "Specialized - Inflammatory",
                        "nhanes_source_variables": "LBXCRP",
                        "nhanes_source_files": "TEST",
                        "nhanes_specimen_match": "blood",
                        "nhanes_match_source": "dashboard",
                        "nhanes_match_method": "manual",
                        "notes": "",
                    }
                ]
            ).to_csv(catalog_path, index=False)

            manifest = build_public_manifest(
                metadata=metadata,
                series_index=series_index,
                series_payloads=series_payloads,
                aging_catalog_csv=catalog_path,
            )

        self.assertEqual(len(manifest), 1)
        row = manifest[0]
        self.assertEqual(row["display_name"], "CRP")
        self.assertEqual(row["featured_collection"], "inflammation")
        self.assertEqual(row["detail_series_path"], "data/series/c-reactive_protein.json")
        self.assertTrue(row["has_clalit_overlay"])
        self.assertEqual(row["public_metrics"]["sample_count"], 1200)
        self.assertAlmostEqual(row["public_metrics"]["median_change_pct_20_24_to_80_84"], 200.0, places=6)
        self.assertAlmostEqual(row["public_metrics"]["iqr_change_pct_20_24_to_80_84"], 350.0, places=6)
        self.assertAlmostEqual(row["public_metrics"]["sd_change_pct_20_24_to_80_84"], 200.0, places=6)
        self.assertAlmostEqual(row["public_metrics"]["upper_tail_change_pct"], 450.0, places=6)
        self.assertAlmostEqual(row["public_metrics"]["lower_tail_change_pct"], 300.0, places=6)
        self.assertAlmostEqual(row["public_metrics"]["tail_asymmetry_change"], 0.25, places=6)
        self.assertAlmostEqual(row["public_metrics"]["sex_divergence_score"], 50.0, places=6)
        self.assertEqual(row["public_metrics_by_context"]["female"]["trim_10_90"]["sample_count"], 650)
        self.assertEqual(row["public_metrics_by_context"]["male"]["trim_10_90"]["sample_count"], 550)
        self.assertAlmostEqual(
            row["public_metrics_by_context"]["female"]["trim_10_90"]["median_change_pct_20_24_to_80_84"],
            100.0,
            places=6,
        )

    def test_render_public_dashboard_html_contains_public_controls(self):
        html = render_public_dashboard_html("aging_biomarkers_public")

        self.assertIn('const DATA_BASE = "aging_biomarkers_public";', html)
        self.assertIn('id="tab-start"', html)
        self.assertIn('id="tab-explore"', html)
        self.assertIn('id="tab-rankings"', html)
        self.assertIn('id="tab-compare"', html)
        self.assertIn('id="explore-search"', html)
        self.assertIn('id="show-clalit" type="checkbox"', html)
        self.assertIn('id="rankings-metric"', html)
        self.assertIn('id="compare-mode"', html)
        self.assertIn('id="copy-link-btn"', html)
        self.assertIn('id="save-chart-btn"', html)

    def test_write_public_dashboard_bundle_writes_manifest_html_and_summary(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            out_html = root / "dashboard" / "aging_biomarkers_dashboard.html"
            out_json = root / "dashboard" / "dashboard_data_aging_biomarkers.json"
            manifest = [
                {
                    "biomarker_id": "albumin",
                    "display_name": "Albumin",
                    "public_metrics": {"sample_count": 100},
                }
            ]

            write_public_dashboard_bundle(
                out_html=out_html,
                out_json=out_json,
                data_dir_name="aging_biomarkers_public",
                manifest=manifest,
            )

            manifest_path = root / "dashboard" / "aging_biomarkers_public" / "manifest.json"
            self.assertTrue(out_html.exists())
            self.assertTrue(out_json.exists())
            self.assertTrue(manifest_path.exists())

            written_manifest = json.loads(manifest_path.read_text())
            summary = json.loads(out_json.read_text())
            html = out_html.read_text()

            self.assertEqual(written_manifest[0]["display_name"], "Albumin")
            self.assertEqual(summary["manifest_count"], 1)
            self.assertIn("aging_biomarkers_public", summary["data_dir"])
            self.assertIn("Aging Biomarkers Dashboard", html)


if __name__ == "__main__":
    unittest.main()
