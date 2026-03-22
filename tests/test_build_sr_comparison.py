#!/usr/bin/env python3

import sys
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str((Path(__file__).resolve().parents[1] / "src")))

from build_sr_comparison import (
    AGE_BIN_LABELS,
    AGE_BIN_MIDS,
    assign_age_bins,
    build_sr_reference_rows,
    build_sr_waterfall_reference,
    compute_qq_fit,
    summarize_biomarker_bins,
    trim_distribution,
)


class TestBuildSrComparison(unittest.TestCase):
    def test_assign_age_bins_matches_expected_labels(self):
        ages = pd.Series([20, 24, 25, 29, 80, 84, 85], dtype=float)
        age_bin, age_mid = assign_age_bins(ages)

        self.assertEqual(age_bin.astype(str).tolist()[:6], ["20-24", "20-24", "25-29", "25-29", "80-84", "80-84"])
        self.assertTrue(np.isnan(age_mid.iloc[6]))

    def test_trim_distribution_removes_symmetric_tails(self):
        values = np.arange(100, dtype=float)
        trimmed = trim_distribution(values)

        self.assertEqual(int(trimmed[0]), 3)
        self.assertEqual(int(trimmed[-1]), 96)
        self.assertEqual(len(trimmed), 94)

    def test_compute_qq_fit_recovers_affine_relationship(self):
        sr_values = np.linspace(1.0, 120.0, 300)
        biomarker_values = (2.5 * sr_values) + 7.0

        fit = compute_qq_fit(sr_values, biomarker_values)

        self.assertAlmostEqual(fit["r2"], 1.0, places=8)
        self.assertAlmostEqual(fit["slope_m"], 2.5, places=6)
        self.assertAlmostEqual(fit["intercept_c"], 7.0, places=6)
        self.assertGreaterEqual(fit["nhanes_n"], 30)
        self.assertGreaterEqual(fit["sr_n"], 30)
        self.assertEqual(len(fit["qq_sr_values"]), len(fit["qq_biomarker_values"]))

    def test_summarize_biomarker_bins_ignores_missing_rows(self):
        rows = [
            {"age_bin": "20-24", "age_mid": 22.5, "r2": 0.90, "slope_m": 1.20, "intercept_c": 0.30},
            {"age_bin": "25-29", "age_mid": 27.5, "r2": None, "slope_m": None, "intercept_c": None},
            {"age_bin": "30-34", "age_mid": 32.5, "r2": 0.80, "slope_m": 1.40, "intercept_c": 0.10},
        ]

        summary = summarize_biomarker_bins(rows)

        self.assertEqual(summary["valid_bin_count"], 2)
        self.assertAlmostEqual(summary["mean_r2"], 0.85, places=8)
        self.assertAlmostEqual(summary["min_r2"], 0.80, places=8)
        self.assertAlmostEqual(summary["mean_slope_m"], 1.30, places=8)
        self.assertAlmostEqual(summary["mean_intercept_c"], 0.20, places=8)
        self.assertEqual(len(summary["r2_by_age_bin"]), 3)

    def test_build_sr_reference_rows_emits_all_nhanes_bins(self):
        tspan = np.array([AGE_BIN_MIDS[label] for label in AGE_BIN_LABELS], dtype=float)
        base = np.linspace(0.5, 50.0, 60, dtype=float)
        paths = np.column_stack([base + idx for idx, _ in enumerate(AGE_BIN_LABELS)])
        death_times = np.full(len(base), np.inf, dtype=float)

        rows, distributions = build_sr_reference_rows(tspan, paths, death_times)

        self.assertEqual([row["age_bin"] for row in rows], AGE_BIN_LABELS)
        self.assertEqual(set(distributions.keys()), set(AGE_BIN_LABELS))
        self.assertTrue(all(distributions[label].size > 0 for label in AGE_BIN_LABELS))

    def test_build_sr_waterfall_reference_emits_quantile_samples(self):
        tspan = np.array([AGE_BIN_MIDS[label] for label in AGE_BIN_LABELS], dtype=float)
        base = np.linspace(0.5, 50.0, 60, dtype=float)
        paths = np.column_stack([base + idx for idx, _ in enumerate(AGE_BIN_LABELS)])
        death_times = np.full(len(base), np.inf, dtype=float)

        payload = build_sr_waterfall_reference(tspan, paths, death_times)

        self.assertEqual(payload["age_bins"], AGE_BIN_LABELS)
        self.assertEqual(len(payload["bins"]), len(AGE_BIN_LABELS))
        self.assertEqual(len(payload["sample_probabilities"]), len(payload["bins"][0]["values_sample"]))
        self.assertGreater(payload["bins"][0]["sr_n"], 0)


if __name__ == "__main__":
    unittest.main()
