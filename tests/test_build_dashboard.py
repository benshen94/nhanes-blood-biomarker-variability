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
        self.assertEqual(std_trend["n_bins"], 5)
        self.assertGreater(std_trend["spearman_rho"], 0.9)
        self.assertIn("std_trends", metric)
        self.assertIn("sex_std_metrics", metric)

        payload = next(iter(series_payloads.values()))
        self.assertIn("std_trends", payload)
        self.assertIn("std", payload["trends_by_stat"])
        self.assertIn("sex_std_metrics", payload)

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

    def test_rendered_dashboard_disables_native_search_autocomplete(self):
        html = render_dashboard_html(
            data_base="data",
            specimen_title="Blood",
            specimen_lower="blood",
            has_clalit=True,
            specimen_switch_link="urinary.html",
        )

        self.assertIn('id="search" list="biomarker-options" placeholder="Type name, code, file..." autocomplete="off"', html)
        self.assertIn('id="waterfall-search" list="waterfall-biomarker-options" placeholder="Type biomarker name..." autocomplete="off"', html)


if __name__ == "__main__":
    unittest.main()
