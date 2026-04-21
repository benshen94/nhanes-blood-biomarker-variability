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
    compress_rank_rows_for_payload,
    build_reference_peer_records,
    build_reference_rows_from_distributions,
    build_sr_reference_rows,
    build_sr_waterfall_reference,
    build_rank_bin_distributions,
    compute_qq_fit,
    compute_rank_bin_rows,
    extract_raw_reference_distributions,
    percentile_ranks_with_tie_breaks,
    summarize_biomarker_bins,
    summarize_rank_bins,
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
        trimmed = trim_distribution(values, lo=0.03, hi=0.97)

        self.assertEqual(int(trimmed[0]), 3)
        self.assertEqual(int(trimmed[-1]), 96)
        self.assertEqual(len(trimmed), 94)

    def test_compute_qq_fit_recovers_affine_relationship(self):
        sr_values = np.linspace(1.0, 120.0, 300)
        biomarker_values = (2.5 * sr_values) + 7.0

        fit = compute_qq_fit(sr_values, biomarker_values, trim_mode_key="all")

        self.assertAlmostEqual(fit["r2"], 1.0, places=8)
        self.assertAlmostEqual(fit["slope_m"], 2.5, places=6)
        self.assertAlmostEqual(fit["intercept_c"], 7.0, places=6)
        self.assertAlmostEqual(fit["wasserstein_z"], 0.0, places=8)
        self.assertGreaterEqual(fit["nhanes_n"], 30)
        self.assertGreaterEqual(fit["sr_n"], 30)
        self.assertEqual(len(fit["qq_sr_values"]), len(fit["qq_biomarker_values"]))

    def test_compute_qq_fit_trims_biomarker_only(self):
        sr_values = np.arange(100, dtype=float)
        biomarker_values = np.arange(100, dtype=float)

        fit = compute_qq_fit(sr_values, biomarker_values, trim_mode_key="trim_10_90")

        self.assertEqual(fit["sr_n"], 100)
        self.assertEqual(fit["nhanes_n"], 80)
        self.assertEqual(fit["sr_q1"], 24.75)
        self.assertEqual(fit["sr_q3"], 74.25)

    def test_summarize_biomarker_bins_ignores_missing_rows(self):
        rows = [
            {"age_bin": "20-24", "age_mid": 22.5, "r2": 0.90, "slope_m": 1.20, "intercept_c": 0.30, "wasserstein_z": 0.20},
            {"age_bin": "25-29", "age_mid": 27.5, "r2": None, "slope_m": None, "intercept_c": None, "wasserstein_z": None},
            {"age_bin": "30-34", "age_mid": 32.5, "r2": 0.80, "slope_m": 1.40, "intercept_c": 0.10, "wasserstein_z": 0.10},
        ]

        summary = summarize_biomarker_bins(rows)

        self.assertEqual(summary["valid_bin_count"], 2)
        self.assertAlmostEqual(summary["mean_r2"], 0.85, places=8)
        self.assertAlmostEqual(summary["min_r2"], 0.80, places=8)
        self.assertAlmostEqual(summary["mean_slope_m"], 1.30, places=8)
        self.assertAlmostEqual(summary["mean_intercept_c"], 0.20, places=8)
        self.assertAlmostEqual(summary["mean_wasserstein_z"], 0.15, places=8)
        self.assertAlmostEqual(summary["min_wasserstein_z"], 0.10, places=8)
        self.assertEqual(len(summary["r2_by_age_bin"]), 3)
        self.assertEqual(len(summary["wasserstein_z_by_age_bin"]), 3)

    def test_percentile_ranks_with_tie_breaks_is_stable(self):
        values = np.array([1.0, 1.0, 1.0, 2.0, 2.0, 3.0], dtype=float)

        first = percentile_ranks_with_tie_breaks(values, "stable-seed")
        second = percentile_ranks_with_tie_breaks(values, "stable-seed")

        self.assertTrue(np.allclose(first, second))
        self.assertEqual(sorted(np.round(first, 6).tolist()), sorted(np.round(second, 6).tolist()))

    def test_build_rank_bin_distributions_trims_each_age_bin_before_pooling(self):
        values_by_age_bin = {
            "20-24": np.arange(10, dtype=float),
            "25-29": np.arange(100, 110, dtype=float),
        }

        rank_bins = build_rank_bin_distributions(values_by_age_bin, trim_mode_key="trim_10_90", seed_key="trim-test")

        kept = np.concatenate([rank_bins["20-24"], rank_bins["25-29"]])
        self.assertEqual(len(kept), 16)
        self.assertEqual(len(rank_bins["20-24"]), 8)
        self.assertEqual(len(rank_bins["25-29"]), 8)
        self.assertTrue(np.all((kept >= 1) & (kept <= 100)))

    def test_build_rank_bin_distributions_can_skip_trim(self):
        values_by_age_bin = {
            "20-24": np.arange(10, dtype=float),
            "25-29": np.arange(100, 110, dtype=float),
        }

        rank_bins = build_rank_bin_distributions(
            values_by_age_bin,
            trim_mode_key="trim_10_90",
            seed_key="no-trim-test",
            trim_each_bin=False,
        )

        kept = np.concatenate([rank_bins["20-24"], rank_bins["25-29"]])
        self.assertEqual(len(kept), 20)
        self.assertEqual(len(rank_bins["20-24"]), 10)
        self.assertEqual(len(rank_bins["25-29"]), 10)
        self.assertTrue(np.all((kept >= 1) & (kept <= 100)))

    def test_compute_rank_bin_rows_detects_shifted_age_localization(self):
        biomarker_values_by_age_bin = {
            age_bin: np.array([], dtype=float) for age_bin in AGE_BIN_LABELS
        }
        sr_values_by_age_bin = {
            age_bin: np.array([], dtype=float) for age_bin in AGE_BIN_LABELS
        }

        sr_values_by_age_bin["20-24"] = np.linspace(0.0, 1.0, 40)
        sr_values_by_age_bin["25-29"] = np.linspace(2.0, 3.0, 40)
        biomarker_values_by_age_bin["20-24"] = np.linspace(2.0, 3.0, 40)
        biomarker_values_by_age_bin["25-29"] = np.linspace(0.0, 1.0, 40)

        sr_rank_bins = build_rank_bin_distributions(sr_values_by_age_bin, trim_mode_key="all", seed_key="sr")
        rows = compute_rank_bin_rows(
            biomarker_id="marker-a",
            biomarker_name="Marker A",
            biomarker_values_by_age_bin=biomarker_values_by_age_bin,
            sr_rank_bins=sr_rank_bins,
            trim_mode_key="all",
        )

        shifted_20 = next(row for row in rows if row["age_bin"] == "20-24")
        shifted_25 = next(row for row in rows if row["age_bin"] == "25-29")
        self.assertGreater(shifted_20["wasserstein_rank"], 40.0)
        self.assertGreater(shifted_25["wasserstein_rank"], 40.0)

    def test_summarize_rank_bins_ignores_missing_rows(self):
        rows = [
            {"age_bin": "20-24", "age_mid": 22.5, "wasserstein_rank": 0.10},
            {"age_bin": "25-29", "age_mid": 27.5, "wasserstein_rank": None},
            {"age_bin": "30-34", "age_mid": 32.5, "wasserstein_rank": 0.30},
        ]

        summary = summarize_rank_bins(rows)

        self.assertEqual(summary["valid_rank_bin_count"], 2)
        self.assertAlmostEqual(summary["mean_wasserstein_rank"], 0.20, places=8)
        self.assertAlmostEqual(summary["min_wasserstein_rank"], 0.10, places=8)
        self.assertAlmostEqual(summary["median_wasserstein_rank"], 0.20, places=8)
        self.assertEqual(len(summary["wasserstein_rank_by_age_bin"]), 3)

    def test_build_sr_reference_rows_emits_all_nhanes_bins(self):
        tspan = np.array([AGE_BIN_MIDS[label] for label in AGE_BIN_LABELS], dtype=float)
        base = np.linspace(0.5, 50.0, 60, dtype=float)
        paths = np.column_stack([base + idx for idx, _ in enumerate(AGE_BIN_LABELS)])
        death_times = np.full(len(base), np.inf, dtype=float)
        death_times[:15] = 55.0
        death_times[15:30] = 35.0

        rows, distributions = build_sr_reference_rows(tspan, paths, death_times)

        self.assertEqual([row["age_bin"] for row in rows], AGE_BIN_LABELS)
        self.assertEqual(set(distributions.keys()), set(AGE_BIN_LABELS))
        self.assertGreater(distributions["20-24"].size, distributions["60-64"].size)
        self.assertEqual(rows[0]["sr_n"], int(distributions["20-24"].size))
        self.assertEqual(rows[-1]["sr_n"], int(distributions["80-84"].size))

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

    def test_extract_raw_reference_distributions_uses_existing_age_bin_labels(self):
        frame = pd.DataFrame(
            [
                {"age": 44.0, "age_bin": np.nan, "FI_hrs": 0.10},
                {"age": 52.0, "age_bin": "50-54", "FI_hrs": 0.20},
                {"age": 53.0, "age_bin": "50-54", "FI_hrs": 0.30},
                {"age": 67.0, "age_bin": "65-69", "FI_hrs": 0.40},
                {"age": 88.0, "age_bin": "85-89", "FI_hrs": 0.50},
            ]
        )

        distributions = extract_raw_reference_distributions(
            frame,
            value_column="FI_hrs",
            age_column="age",
            age_bin_column="age_bin",
        )

        self.assertEqual(distributions["50-54"].tolist(), [0.2, 0.3])
        self.assertEqual(distributions["65-69"].tolist(), [0.4])
        self.assertEqual(distributions["20-24"].tolist(), [])
        self.assertEqual(distributions["80-84"].tolist(), [])

    def test_build_reference_rows_from_distributions_keeps_empty_bins(self):
        distributions = {age_bin: np.array([], dtype=float) for age_bin in AGE_BIN_LABELS}
        distributions["50-54"] = np.array([0.2, 0.3, 0.4], dtype=float)

        rows = build_reference_rows_from_distributions(distributions)

        self.assertEqual([row["age_bin"] for row in rows], AGE_BIN_LABELS)
        fifty_row = next(row for row in rows if row["age_bin"] == "50-54")
        early_row = next(row for row in rows if row["age_bin"] == "20-24")
        self.assertEqual(fifty_row["sr_n"], 3)
        self.assertEqual(early_row["sr_n"], 0)
        self.assertIsNone(early_row["sr_q1"])

    def test_build_reference_peer_records_excludes_identity_fit(self):
        fit_builds = [
            {
                "fit_key": "original",
                "fit_label": "SR original fit",
                "sr_reference_rows": build_reference_rows_from_distributions({
                    age_bin: np.linspace(0.0, 1.0, 40, dtype=float) for age_bin in AGE_BIN_LABELS
                }),
                "sr_distributions": {
                    age_bin: np.linspace(0.0, 1.0, 40, dtype=float) for age_bin in AGE_BIN_LABELS
                },
            },
            {
                "fit_key": "fi_hrs",
                "fit_label": "FI (HRS)",
                "sr_reference_rows": build_reference_rows_from_distributions({
                    age_bin: np.linspace(1.0, 2.0, 40, dtype=float) for age_bin in AGE_BIN_LABELS
                }),
                "sr_distributions": {
                    age_bin: np.linspace(1.0, 2.0, 40, dtype=float) for age_bin in AGE_BIN_LABELS
                },
            },
        ]
        fit_options = [
            {"key": "original", "kind": "usa_2019_script", "reference_population_label": "alive-only reference"},
            {"key": "fi_hrs", "kind": "raw_reference_csv", "reference_population_label": "participant reference"},
        ]

        payload = build_reference_peer_records(
            fit_builds=fit_builds,
            fit_options=fit_options,
            default_fit_key="original",
        )

        self.assertEqual(payload["category"], "User distributions")
        self.assertEqual(len(payload["records"]), 2)
        original_record = next(record for record in payload["records"] if record["reference_key"] == "original")
        self.assertNotIn("original", original_record["sr_comparison_summary_by_fit"])
        self.assertIn("fi_hrs", original_record["sr_comparison_summary_by_fit"])

    def test_compress_rank_rows_for_payload_limits_rank_array_size(self):
        rows = [
            {
                "age_bin": "50-54",
                "age_mid": 52.5,
                "nhanes_rank_values": list(range(1, 501)),
            }
        ]

        compressed = compress_rank_rows_for_payload(rows, max_values=51)

        self.assertEqual(len(compressed), 1)
        self.assertEqual(len(compressed[0]["nhanes_rank_values"]), 51)


if __name__ == "__main__":
    unittest.main()
