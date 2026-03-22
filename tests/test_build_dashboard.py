#!/usr/bin/env python3

import sys
import unittest
from pathlib import Path

import pandas as pd

sys.path.insert(0, str((Path(__file__).resolve().parents[1] / "src")))

from build_dashboard import build_outputs, process_clalit_data, render_dashboard_html


class TestBuildDashboard(unittest.TestCase):
    def test_build_outputs_emits_std_trends(self):
        rows = []
        specs = [
            (22, 1.0, 0.10),
            (32, 1.0, 0.20),
            (42, 1.0, 0.30),
            (52, 1.0, 0.40),
            (62, 1.0, 0.50),
        ]
        for age, mean, spread in specs:
            for i in range(30):
                rows.append(
                    {
                        "seqn": age * 1000 + i,
                        "cycle_start_year": 2001,
                        "biomarker_id": "synthetic-marker",
                        "age_years": age,
                        "value": mean + (spread if i % 2 else -spread),
                        "sex": "female",
                    }
                )
        long_df = pd.DataFrame(rows)
        catalog_df = pd.DataFrame(
            [
                {
                    "biomarker_id": "synthetic-marker",
                    "variable_name": "LBXTEST",
                    "biomarker_name": "Synthetic marker",
                    "unit": "mg/dL",
                    "source_file_count": 1,
                    "source_files": "TEST",
                    "source_variable_count": 1,
                    "source_variables": "LBXTEST",
                }
            ]
        )

        metadata, metrics, _, series_payloads = build_outputs(
            cv_df=pd.DataFrame(columns=["biomarker_id", "biomarker_name", "variable_name", "unit"]),
            metrics_df=pd.DataFrame(),
            catalog_df=catalog_df,
            long_df=long_df,
            raw_sample_n=50,
            random_seed=42,
            specimen_kind="blood",
        )

        self.assertEqual(len(metadata), 1)
        metric = metrics[0]
        std_trend = metric["trends_by_stat"]["std"]["all"]
        qskew_trend = metric["trends_by_stat"]["quantile_skewness"]["all"]
        self.assertEqual(std_trend["n_bins"], 5)
        self.assertGreater(std_trend["spearman_rho"], 0.9)
        self.assertEqual(qskew_trend["n_bins"], 5)
        self.assertIn("std_trends", metric)
        self.assertIn("sex_std_metrics", metric)
        self.assertIn("quantile_skewness_trends", metric)
        self.assertIn("sex_quantile_skewness_metrics", metric)

        payload = next(iter(series_payloads.values()))
        self.assertIn("std_trends", payload)
        self.assertIn("std", payload["trends_by_stat"])
        self.assertIn("sex_std_metrics", payload)
        self.assertIn("quantile_skewness_trends", payload)
        self.assertIn("quantile_skewness", payload["trends_by_stat"])
        self.assertIn("sex_quantile_skewness_metrics", payload)

    def test_process_clalit_data_supports_scaled_targets(self):
        clalit = pd.DataFrame(
            [
                {
                    "test": "lab.102500.no_meds",
                    "age": 40,
                    "n": 40,
                    "mean": 14.5,
                    "sd": 2.2,
                    "median": 14.3,
                    "q25": 13.0,
                    "q75": 15.7,
                }
            ]
        )
        mapping = {
            "lab.102500.no_meds": {
                "biomarker_id": "thyroxine free",
                "scale_factor": 0.07767,
            }
        }

        payload = process_clalit_data(clalit, clalit, mapping)
        pooled = payload["thyroxine free"]["pooled"][0]

        self.assertAlmostEqual(pooled["mean"], 14.5 * 0.07767, places=6)
        self.assertAlmostEqual(pooled["std"], 2.2 * 0.07767, places=6)
        self.assertAlmostEqual(pooled["median"], 14.3 * 0.07767, places=6)
        self.assertAlmostEqual(pooled["q25"], 13.0 * 0.07767, places=6)
        self.assertAlmostEqual(pooled["q75"], 15.7 * 0.07767, places=6)
        self.assertAlmostEqual(pooled["quantile_skewness"], (15.7 + 13.0 - 2 * 14.3) / (15.7 - 13.0), places=6)

    def test_rendered_dashboard_disables_native_search_autocomplete(self):
        html = render_dashboard_html(
            data_base="data",
            specimen_title="Blood",
            specimen_lower="blood",
            has_clalit=True,
            has_sr_comparison=True,
            specimen_switch_link="urinary.html",
        )

        self.assertIn('id="search" list="biomarker-options" placeholder="Type name, code, file..." autocomplete="off"', html)
        self.assertIn('id="waterfall-search" list="waterfall-biomarker-options" placeholder="Type biomarker name..." autocomplete="off"', html)
        self.assertIn('id="tab-sr-comparison"', html)
        self.assertIn("const HAS_SR_COMPARISON = true;", html)
        self.assertIn('id="full-view-skew-stat"', html)
        self.assertIn('id="filter-tests-full-skew-stat"', html)


    def test_build_outputs_merges_sr_comparison_payload(self):
        long_df = pd.DataFrame(
            [
                {
                    "seqn": 1001,
                    "cycle_start_year": 2001,
                    "biomarker_id": "sr-marker",
                    "age_years": 42.0,
                    "value": 1.0,
                    "sex": "female",
                }
            ]
        )
        catalog_df = pd.DataFrame(
            [
                {
                    "biomarker_id": "sr-marker",
                    "variable_name": "LBXSR",
                    "biomarker_name": "SR marker",
                    "unit": "mg/dL",
                    "source_file_count": 1,
                    "source_files": "TEST",
                    "source_variable_count": 1,
                    "source_variables": "LBXSR",
                }
            ]
        )
        sr_bundle = {
            "summary_by_biomarker": {
                "sr-marker": {
                    "mean_r2": 0.91,
                    "min_r2": 0.84,
                    "median_r2": 0.90,
                    "valid_bin_count": 7,
                    "mean_slope_m": 1.1,
                    "slope_m_sd": 0.2,
                    "mean_intercept_c": 0.4,
                    "intercept_c_sd": 0.1,
                    "r2_by_age_bin": [{"age_bin": "40-44", "age_mid": 42.5, "r2": 0.88}],
                }
            },
            "detail_by_biomarker": {
                "sr-marker": {
                    "bins": [
                        {
                            "age_bin": "40-44",
                            "age_mid": 42.5,
                            "r2": 0.88,
                            "slope_m": 1.2,
                            "intercept_c": 0.3,
                            "nhanes_n": 55,
                            "sr_n": 1000,
                            "nhanes_q1": 0.8,
                            "nhanes_median": 1.0,
                            "nhanes_q3": 1.2,
                            "sr_q1": 0.2,
                            "sr_median": 0.5,
                            "sr_q3": 0.9,
                            "qq_sr_values": [0.2, 0.5, 0.9],
                            "qq_biomarker_values": [0.8, 1.0, 1.2],
                        }
                    ]
                }
            },
        }

        _, metrics, _, series_payloads = build_outputs(
            cv_df=pd.DataFrame(columns=["biomarker_id", "biomarker_name", "variable_name", "unit"]),
            metrics_df=pd.DataFrame(),
            catalog_df=catalog_df,
            long_df=long_df,
            raw_sample_n=50,
            random_seed=42,
            specimen_kind="blood",
            sr_comparison_bundle=sr_bundle,
        )

        self.assertEqual(metrics[0]["sr_comparison_summary"]["mean_r2"], 0.91)
        payload = next(iter(series_payloads.values()))
        self.assertIn("sr_comparison", payload)
        self.assertEqual(payload["sr_comparison"]["bins"][0]["age_bin"], "40-44")

    def test_build_outputs_drops_85_plus_age_bin(self):
        rows = []
        for age in [22, 32, 42, 52, 62, 72, 82, 87]:
            for i in range(30):
                rows.append(
                    {
                        "seqn": age * 1000 + i,
                        "cycle_start_year": 2001,
                        "biomarker_id": "bin-check",
                        "age_years": age,
                        "value": float(age) + (0.5 if i % 2 else -0.5),
                        "sex": "female",
                    }
                )
        long_df = pd.DataFrame(rows)
        catalog_df = pd.DataFrame(
            [
                {
                    "biomarker_id": "bin-check",
                    "variable_name": "LBXBIN",
                    "biomarker_name": "Bin check",
                    "unit": "mg/dL",
                    "source_file_count": 1,
                    "source_files": "TEST",
                    "source_variable_count": 1,
                    "source_variables": "LBXBIN",
                }
            ]
        )

        _, metrics, _, series_payloads = build_outputs(
            cv_df=pd.DataFrame(columns=["biomarker_id", "biomarker_name", "variable_name", "unit"]),
            metrics_df=pd.DataFrame(),
            catalog_df=catalog_df,
            long_df=long_df,
            raw_sample_n=50,
            random_seed=42,
            specimen_kind="blood",
        )

        metric = metrics[0]
        payload = next(iter(series_payloads.values()))
        self.assertEqual(metric["trends_by_stat"]["std"]["all"]["n_bins"], 7)
        age_bins = [point["age_bin"] for point in payload["points_by_filter"]["all"]]
        self.assertNotIn("85+", age_bins)
        self.assertNotIn("85-89", age_bins)
        self.assertIn("80-84", age_bins)


if __name__ == "__main__":
    unittest.main()
