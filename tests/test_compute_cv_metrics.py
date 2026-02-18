#!/usr/bin/env python3

import math
import sys
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str((Path(__file__).resolve().parents[1] / "src")))

from compute_cv_metrics import assign_age_bins, compute_binned, compute_trends


class TestComputeCVMetrics(unittest.TestCase):
    def test_assign_age_bins(self):
        s = pd.Series([20, 24, 25, 84, 85, 99])
        b, m = assign_age_bins(s)
        self.assertEqual(list(b.astype(str)), ["20-24", "20-24", "25-29", "80-84", "85+", "85+"])
        self.assertEqual(list(m.values), [22.5, 22.5, 27.5, 82.5, 87.5, 87.5])

    def test_cv_formula_and_threshold(self):
        df = pd.DataFrame(
            {
                "biomarker_id": ["A::X"] * 6,
                "biomarker_name": ["X"] * 6,
                "age_years": [20, 21, 22, 23, 24, 25],
                "value": [10, 11, 9, 10, 10, 10],
            }
        )
        out = compute_binned(df)
        row = out.loc[out["age_bin"].astype(str) == "20-24"].iloc[0]
        expected_cv = np.std([10, 11, 9, 10, 10], ddof=1) / abs(np.mean([10, 11, 9, 10, 10]))
        self.assertTrue(math.isclose(row["cv"], expected_cv, rel_tol=1e-8))

    def test_decline_flag_logic(self):
        rows = []
        mids = [22.5, 27.5, 32.5, 37.5, 42.5, 47.5]
        cvs = [0.30, 0.26, 0.22, 0.18, 0.16, 0.14]
        for i, (m, c) in enumerate(zip(mids, cvs)):
            rows.append(
                {
                    "biomarker_id": "B::Y",
                    "biomarker_name": "Y",
                    "age_bin": f"bin{i}",
                    "age_mid": m,
                    "n": 50,
                    "mean": 100.0,
                    "std": 100.0 * c,
                    "cv": c,
                    "passes_n_threshold": True,
                }
            )
        cv_df = pd.DataFrame(rows)
        trends = compute_trends(cv_df)
        t = trends.iloc[0]
        self.assertEqual(t["n_bins"], 6)
        self.assertLess(t["spearman_rho"], 0)
        self.assertLess(t["linear_slope_cv_per_year"], 0)
        self.assertTrue(bool(t["decline_flag"]))


if __name__ == "__main__":
    unittest.main()
