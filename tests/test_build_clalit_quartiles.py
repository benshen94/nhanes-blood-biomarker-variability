#!/usr/bin/env python3
"""Focused checks for the Clalit quartile builder."""

from __future__ import annotations

import importlib.util
import sys
import unittest
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
MODULE_PATH = ROOT / "src" / "build_clalit_quartiles.py"
sys.path.insert(0, str(ROOT / "src"))

SPEC = importlib.util.spec_from_file_location("build_clalit_quartiles", MODULE_PATH)
assert SPEC is not None and SPEC.loader is not None
MODULE = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = MODULE
SPEC.loader.exec_module(MODULE)


class BuildClalitQuartilesTests(unittest.TestCase):
    def test_age_bin_extension_matches_requested_tail_bins(self) -> None:
        self.assertEqual(MODULE.age_bin_for_year(20).label, "20-24")
        self.assertEqual(MODULE.age_bin_for_year(84).label, "80-84")
        self.assertEqual(MODULE.age_bin_for_year(85).label, "85-89")
        self.assertEqual(MODULE.age_bin_for_year(94).label, "90-94")
        self.assertEqual(MODULE.age_bin_for_year(99).label, "95-99")
        self.assertIsNone(MODULE.age_bin_for_year(19))

    def test_uniform_density_produces_expected_quartiles(self) -> None:
        curve = MODULE.density_curve_to_cdf(
            axis=np.array([0.0, 1.0], dtype=float),
            density=np.array([1.0, 1.0], dtype=float),
        )
        self.assertIsNotNone(curve)
        grid, cdf = curve

        q25, q50, q75 = MODULE.quantiles_from_cdf(grid, cdf, np.array([0.25, 0.50, 0.75], dtype=float))

        self.assertAlmostEqual(float(q25), 0.25, places=6)
        self.assertAlmostEqual(float(q50), 0.50, places=6)
        self.assertAlmostEqual(float(q75), 0.75, places=6)

    def test_equal_curve_mixture_stays_monotone(self) -> None:
        left = MODULE.density_curve_to_cdf(
            axis=np.array([0.0, 1.0], dtype=float),
            density=np.array([1.0, 1.0], dtype=float),
        )
        right = MODULE.density_curve_to_cdf(
            axis=np.array([0.0, 1.0], dtype=float),
            density=np.array([1.0, 1.0], dtype=float),
        )
        assert left is not None and right is not None

        combined = MODULE.combine_age_curves(
            [
                (left[0], left[1], 10.0),
                (right[0], right[1], 5.0),
            ]
        )
        self.assertIsNotNone(combined)
        grid, cdf = combined

        self.assertTrue(np.all(np.diff(grid) >= 0))
        self.assertTrue(np.all(np.diff(cdf) >= -1e-12))
        self.assertAlmostEqual(float(cdf[0]), 0.0, places=8)
        self.assertAlmostEqual(float(cdf[-1]), 1.0, places=8)


if __name__ == "__main__":
    unittest.main()
