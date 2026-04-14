#!/usr/bin/env python3

import json
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import pandas as pd

sys.path.insert(0, str((Path(__file__).resolve().parents[1] / "src")))

from build_aging_biomarkers_dashboard import (
    _points_for_mode,
    _select_disease_default_biomarker_ids,
    build_surprising_groups,
    build_disease_explorer_bundle,
    build_public_manifest,
    load_public_disease_long,
    render_public_dashboard_html,
    write_public_dashboard_bundle,
)


def make_point(age_bin, age_mid, median, q25, q75, p10, p90, std, cv, qskew=0.0):
    return {
        "age_bin": age_bin,
        "age_mid": age_mid,
        "n": 40,
        "median": median,
        "q25": q25,
        "q75": q75,
        "p10": p10,
        "p90": p90,
        "iqr": q75 - q25,
        "std": std,
        "cv": cv,
        "quantile_skewness": qskew,
        "passes_n_threshold": True,
    }


class TestBuildAgingBiomarkersDashboard(unittest.TestCase):
    def test_points_for_mode_keeps_only_threshold_passing_points(self):
        group = {
            "points_by_filter": {
                "trim_5_95": [
                    {"age_bin": "20-24", "median": 1.0, "passes_n_threshold": True},
                    {"age_bin": "25-29", "median": 2.0, "passes_n_threshold": False},
                ]
            }
        }

        filtered = _points_for_mode(group, "trim_5_95")

        self.assertEqual(len(filtered), 1)
        self.assertEqual(filtered[0]["age_bin"], "20-24")

    def test_select_disease_default_biomarker_ids_preserves_familiar_order(self):
        def disease_points(start, end):
            return {
                "points_by_filter": {
                    "trim_5_95": [
                        {"age_bin": "20-24", "median": start, "passes_n_threshold": True},
                        {"age_bin": "40-44", "median": (start + end) / 2.0, "passes_n_threshold": True},
                        {"age_bin": "80-84", "median": end, "passes_n_threshold": True},
                    ]
                }
            }

        records = [
            {
                "biomarker_id": "glycohemoglobin",
                "display_name": "HbA1c",
                "groups": {
                    "healthy": disease_points(5.2, 5.6),
                    "condition": disease_points(6.3, 6.9),
                },
            },
            {
                "biomarker_id": "plasma glucose",
                "display_name": "Glucose",
                "groups": {
                    "healthy": disease_points(90.0, 96.0),
                    "condition": disease_points(118.0, 140.0),
                },
            },
            {
                "biomarker_id": "c-peptide",
                "display_name": "C-peptide",
                "groups": {
                    "healthy": disease_points(1.0, 1.2),
                    "condition": disease_points(1.4, 1.8),
                },
            },
        ]

        selected = _select_disease_default_biomarker_ids("diabetes", records, limit=3)

        self.assertEqual(selected, ["glycohemoglobin", "plasma glucose", "c-peptide"])

    def test_select_disease_default_biomarker_ids_does_not_pad_with_unrelated_fallbacks(self):
        def disease_points():
            return {
                "points_by_filter": {
                    "trim_5_95": [
                        {"age_bin": "20-24", "median": 1.0, "passes_n_threshold": True},
                        {"age_bin": "40-44", "median": 1.2, "passes_n_threshold": True},
                        {"age_bin": "80-84", "median": 1.4, "passes_n_threshold": True},
                    ]
                }
            }

        records = [
            {
                "biomarker_id": "albumin",
                "display_name": "Albumin",
                "groups": {
                    "healthy": disease_points(),
                    "condition": disease_points(),
                },
            },
            {
                "biomarker_id": "random-marker",
                "display_name": "Random marker",
                "groups": {
                    "healthy": disease_points(),
                    "condition": disease_points(),
                },
            },
        ]

        selected = _select_disease_default_biomarker_ids("liver", records, limit=6)

        self.assertEqual(selected, ["albumin"])

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
        html = render_public_dashboard_html("aging_biomarkers_public", ga4_measurement_id="G-TEST123")

        self.assertIn('const DATA_BASE = "aging_biomarkers_public";', html)
        self.assertIn('window.PUBLIC_GA4_MEASUREMENT_ID = "G-TEST123";', html)
        self.assertIn('https://www.googletagmanager.com/gtag/js?id=', html)
        self.assertIn('id="tab-start"', html)
        self.assertIn('id="tab-explore"', html)
        self.assertIn('id="tab-disease"', html)
        self.assertIn('id="tab-rankings"', html)
        self.assertIn('id="tab-surprising"', html)
        self.assertIn('id="tab-compare"', html)
        self.assertIn('id="tab-calculator"', html)
        self.assertIn('id="explore-search"', html)
        self.assertIn('id="disease-condition-list"', html)
        self.assertIn('id="disease-plot"', html)
        self.assertIn("Public research explorer only.", html)
        self.assertNotIn('id="show-clalit" type="checkbox"', html)
        self.assertIn('id="rankings-metric"', html)
        self.assertIn('id="compare-mode"', html)
        self.assertIn('id="bioage-form"', html)
        self.assertIn('class="metric-help"', html)
        self.assertIn('id="bioage-birth-date"', html)
        self.assertIn('id="bioage-calc-btn"', html)
        self.assertIn("Calculate PhenoAge", html)
        self.assertIn('id="copy-link-btn"', html)
        self.assertIn('id="save-chart-btn"', html)

    def test_build_disease_explorer_bundle_emits_condition_payloads(self):
        public_manifest = [
            {
                "biomarker_id": "creatinine",
                "display_name": "Creatinine",
                "chart_display_name": "Creatinine (mg/dL)",
                "unit": "mg/dL",
                "featured_collection": "kidney_reserve",
                "featured_collection_title": "Kidney & Reserve",
                "aging_domain": "organ reserve",
            }
        ]
        long_df = pd.DataFrame(
            [
                {"seqn": 5, "cycle_start_year": 2001, "biomarker_id": "creatinine", "age_years": 82, "value": 0.9, "sex": "female"},
                {"seqn": 6, "cycle_start_year": 2001, "biomarker_id": "creatinine", "age_years": 82, "value": 1.6, "sex": "male"},
                {"seqn": 1, "cycle_start_year": 2001, "biomarker_id": "creatinine", "age_years": 42, "value": 0.8, "sex": "female"},
                {"seqn": 2, "cycle_start_year": 2001, "biomarker_id": "creatinine", "age_years": 42, "value": 1.3, "sex": "female"},
                {"seqn": 3, "cycle_start_year": 2001, "biomarker_id": "creatinine", "age_years": 62, "value": 0.9, "sex": "male"},
                {"seqn": 4, "cycle_start_year": 2001, "biomarker_id": "creatinine", "age_years": 62, "value": 1.5, "sex": "male"},
            ]
            * 40
        )
        participant_flags = pd.DataFrame(
            [
                {"seqn": 1, "cycle_start_year": 2001, "sex": "female", "healthy_flag": True, "diabetes": False, "hypertension": False, "cvd": False, "kidney": False, "liver": False, "cancer": False, "asthma": False, "thyroid_problem": False, "stroke": False},
                {"seqn": 2, "cycle_start_year": 2001, "sex": "female", "healthy_flag": False, "diabetes": True, "hypertension": False, "cvd": False, "kidney": False, "liver": False, "cancer": False, "asthma": False, "thyroid_problem": False, "stroke": False},
                {"seqn": 3, "cycle_start_year": 2001, "sex": "male", "healthy_flag": True, "diabetes": False, "hypertension": False, "cvd": False, "kidney": False, "liver": False, "cancer": False, "asthma": False, "thyroid_problem": False, "stroke": False},
                {"seqn": 4, "cycle_start_year": 2001, "sex": "male", "healthy_flag": False, "diabetes": True, "hypertension": False, "cvd": False, "kidney": False, "liver": False, "cancer": False, "asthma": False, "thyroid_problem": False, "stroke": False},
                {"seqn": 5, "cycle_start_year": 2001, "sex": "female", "healthy_flag": True, "diabetes": False, "hypertension": False, "cvd": False, "kidney": False, "liver": False, "cancer": False, "asthma": False, "thyroid_problem": False, "stroke": False},
                {"seqn": 6, "cycle_start_year": 2001, "sex": "male", "healthy_flag": False, "diabetes": True, "hypertension": False, "cvd": False, "kidney": False, "liver": False, "cancer": False, "asthma": False, "thyroid_problem": False, "stroke": False},
            ]
        )

        bundle = build_disease_explorer_bundle(
            public_manifest=public_manifest,
            long_df=long_df,
            participant_flags=participant_flags,
        )

        self.assertTrue(bundle["conditions"])
        diabetes = next(item for item in bundle["conditions"] if item["key"] == "diabetes")
        self.assertEqual(diabetes["title"], "Diabetes")
        self.assertIn("diseases/diabetes.json", diabetes["detail_path"])
        self.assertTrue(diabetes["default_biomarker_ids"])
        self.assertEqual(diabetes["default_title"], "Start with familiar metabolic markers")

        detail = bundle["by_condition"]["diabetes"]
        self.assertEqual(detail["condition"]["title"], "Diabetes")
        biomarker = detail["biomarkers"][0]
        self.assertEqual(biomarker["display_name"], "Creatinine")
        self.assertEqual(biomarker["groups"]["healthy"]["raw_total_n"], 120)
        self.assertEqual(biomarker["groups"]["condition"]["raw_total_n"], 120)
        self.assertIn("trim_10_90", biomarker["groups"]["healthy"]["points_by_filter"])
        self.assertIn("female", biomarker["groups"]["condition"]["sex_points_by_filter"]["all"])

    def test_build_surprising_groups_returns_three_non_empty_sections(self):
        manifest = [
            {
                "biomarker_id": "testosterone",
                "display_name": "Testosterone",
                "featured_collection_title": "Hormones & Nutrient Sensing",
                "public_metrics": {},
                "public_metrics_by_context": {
                    "pooled": {
                        "trim_5_95": {
                            "median_change_pct_20_24_to_80_84": -70.0,
                            "cv_change_pct_20_24_to_80_84": 5.0,
                            "upper_tail_change_pct": -5.0,
                            "sex_divergence_score": 40.0,
                        }
                    }
                },
            },
            {
                "biomarker_id": "rdw",
                "display_name": "RDW",
                "featured_collection_title": "Blood & Oxygen",
                "public_metrics": {},
                "public_metrics_by_context": {
                    "pooled": {
                        "trim_5_95": {
                            "median_change_pct_20_24_to_80_84": 4.0,
                            "cv_change_pct_20_24_to_80_84": 28.0,
                            "upper_tail_change_pct": 35.0,
                            "sex_divergence_score": 12.0,
                        }
                    }
                },
            },
            {
                "biomarker_id": "ntprobnp",
                "display_name": "NT-proBNP",
                "featured_collection_title": "Cardiovascular Stress",
                "public_metrics": {},
                "public_metrics_by_context": {
                    "pooled": {
                        "trim_5_95": {
                            "median_change_pct_20_24_to_80_84": 120.0,
                            "cv_change_pct_20_24_to_80_84": 12.0,
                            "upper_tail_change_pct": 20.0,
                            "sex_divergence_score": 90.0,
                        }
                    }
                },
            },
        ]

        surprising = build_surprising_groups(manifest, limit=4)

        self.assertEqual(len(surprising["groups"]), 3)
        groups = {group["key"]: group for group in surprising["groups"]}
        self.assertEqual(groups["falls_with_age"]["items"][0]["display_name"], "Testosterone")
        self.assertEqual(groups["stable_center_wild_distribution"]["items"][0]["display_name"], "RDW")
        self.assertEqual(groups["sex_divergence"]["items"][0]["display_name"], "NT-proBNP")

    def test_load_public_disease_long_reads_selected_public_biomarkers(self):
        public_manifest = [
            {"biomarker_id": "albumin"},
            {"biomarker_id": "creatinine"},
        ]
        participant_flags = pd.DataFrame(
            [
                {"seqn": 1, "cycle_start_year": 2001, "age_years": 42, "sex": "female"},
                {"seqn": 2, "cycle_start_year": 2001, "age_years": 64, "sex": "male"},
            ]
        )

        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            raw_dir = root / "raw" / "2001"
            raw_dir.mkdir(parents=True)
            (raw_dir / "BIOPRO_B.xpt").write_text("")

            screening_path = root / "variable_screening_summary.csv"
            pd.DataFrame(
                [
                    {
                        "cycle_start_year": 2001,
                        "data_file_name": "BIOPRO_B",
                        "variable_name": "LBXSAL",
                        "variable_desc": "Albumin (g/dL)",
                        "screen_result": "kept",
                        "pooled_id": "albumin",
                    },
                    {
                        "cycle_start_year": 2001,
                        "data_file_name": "BIOPRO_B",
                        "variable_name": "LBXSCR",
                        "variable_desc": "Creatinine (mg/dL)",
                        "screen_result": "kept",
                        "pooled_id": "creatinine",
                    },
                ]
            ).to_csv(screening_path, index=False)

            merge_map_path = root / "duplicate_merge_map.csv"
            pd.DataFrame(
                [
                    {
                        "pooled_id": "albumin",
                        "variable_name": "LBXSAL",
                        "variable_desc": "Albumin (g/dL)",
                        "conversion_factor_to_pooled_unit": 1.0,
                    },
                    {
                        "pooled_id": "creatinine",
                        "variable_name": "LBXSCR",
                        "variable_desc": "Creatinine (mg/dL)",
                        "conversion_factor_to_pooled_unit": 1.0,
                    },
                ]
            ).to_csv(merge_map_path, index=False)

            fake_xpt = pd.DataFrame(
                [
                    {"SEQN": 1, "LBXSAL": 4.2, "LBXSCR": 0.9},
                    {"SEQN": 2, "LBXSAL": 3.9, "LBXSCR": 1.3},
                ]
            )

            with patch("build_aging_biomarkers_dashboard.read_xpt_columns", return_value=fake_xpt):
                long_df = load_public_disease_long(
                    public_manifest=public_manifest,
                    participant_flags=participant_flags,
                    raw_dir=root / "raw",
                    screening_summary_path=screening_path,
                    merge_map_path=merge_map_path,
                )

        self.assertIsNotNone(long_df)
        self.assertEqual(set(long_df["biomarker_id"]), {"albumin", "creatinine"})
        self.assertEqual(len(long_df), 4)
        self.assertEqual(set(long_df["sex"]), {"female", "male"})

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
            disease_bundle = {
                "conditions": [
                    {
                        "key": "diabetes",
                        "title": "Diabetes",
                        "default_biomarker_ids": ["albumin"],
                        "detail_path": "diseases/diabetes.json",
                    }
                ],
                "by_condition": {
                    "diabetes": {
                        "condition": {"key": "diabetes", "title": "Diabetes"},
                        "biomarkers": [],
                    }
                },
            }
            surprising_bundle = {
                "groups": [
                    {
                        "key": "falls_with_age",
                        "title": "Markers that fall with age",
                        "items": [{"biomarker_id": "albumin", "display_name": "Albumin"}],
                    }
                ]
            }

            write_public_dashboard_bundle(
                out_html=out_html,
                out_json=out_json,
                data_dir_name="aging_biomarkers_public",
                manifest=manifest,
                disease_bundle=disease_bundle,
                surprising_bundle=surprising_bundle,
                ga4_measurement_id="G-TEST123",
            )

            manifest_path = root / "dashboard" / "aging_biomarkers_public" / "manifest.json"
            disease_index_path = root / "dashboard" / "aging_biomarkers_public" / "disease_index.json"
            disease_detail_path = root / "dashboard" / "aging_biomarkers_public" / "diseases" / "diabetes.json"
            surprising_path = root / "dashboard" / "aging_biomarkers_public" / "surprising.json"
            self.assertTrue(out_html.exists())
            self.assertTrue(out_json.exists())
            self.assertTrue(manifest_path.exists())
            self.assertTrue(disease_index_path.exists())
            self.assertTrue(disease_detail_path.exists())
            self.assertTrue(surprising_path.exists())

            written_manifest = json.loads(manifest_path.read_text())
            summary = json.loads(out_json.read_text())
            html = out_html.read_text()

            self.assertEqual(written_manifest[0]["display_name"], "Albumin")
            self.assertEqual(summary["manifest_count"], 1)
            self.assertEqual(summary["disease_condition_count"], 1)
            self.assertEqual(summary["surprising_group_count"], 1)
            self.assertEqual(summary["ga4_measurement_id"], "G-TEST123")
            self.assertIn("aging_biomarkers_public", summary["data_dir"])
            self.assertIn("Aging Biomarkers Dashboard", html)


if __name__ == "__main__":
    unittest.main()
