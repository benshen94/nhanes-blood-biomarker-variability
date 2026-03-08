#!/usr/bin/env python3

import sys
import unittest
from pathlib import Path

import pandas as pd

sys.path.insert(0, str((Path(__file__).resolve().parents[1] / "src")))

from build_dashboard import process_clalit_data, render_dashboard_html


class TestBuildDashboard(unittest.TestCase):
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
