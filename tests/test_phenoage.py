#!/usr/bin/env python3

from datetime import date
import sys
import unittest
from pathlib import Path


sys.path.insert(0, str((Path(__file__).resolve().parents[1] / "src")))

from phenoage import (  # noqa: E402
    age_years_at_sample,
    calculate_phenoage,
    calculate_phenoage_from_lab_report,
    convert_albumin_to_g_l,
    convert_creatinine_to_umol_l,
    convert_crp_to_mg_dl,
    convert_glucose_to_mmol_l,
    convert_wbc_to_k_per_ul,
)


class TestPhenoAge(unittest.TestCase):
    def test_age_years_at_sample_uses_sample_date(self):
        age_years = age_years_at_sample(date(1980, 1, 1), date(2025, 1, 1))
        self.assertAlmostEqual(age_years, 45.0030, places=3)

    def test_unit_conversions_match_published_units(self):
        self.assertAlmostEqual(convert_albumin_to_g_l(4.3, "g/dL"), 43.0, places=6)
        self.assertAlmostEqual(convert_creatinine_to_umol_l(0.9, "mg/dL"), 79.56153, places=6)
        self.assertAlmostEqual(convert_glucose_to_mmol_l(93.0, "mg/dL"), 5.1615, places=6)
        self.assertAlmostEqual(convert_crp_to_mg_dl(1.5, "mg/L"), 0.15, places=6)
        self.assertAlmostEqual(convert_wbc_to_k_per_ul(6.0, "10^9/L"), 6.0, places=6)

    def test_calculate_phenoage_returns_expected_value(self):
        result = calculate_phenoage(
            age_years=45.0,
            albumin_g_l=43.0,
            creatinine_umol_l=80.0,
            glucose_mmol_l=5.2,
            crp_mg_dl=0.15,
            lymphocyte_percent=30.0,
            mcv_fl=90.0,
            rdw_percent=13.2,
            alkaline_phosphatase_u_l=70.0,
            white_blood_cell_k_per_ul=6.0,
        )
        self.assertAlmostEqual(result, 39.7228, places=4)

    def test_calculate_phenoage_from_lab_report_handles_common_us_units(self):
        result = calculate_phenoage_from_lab_report(
            birth_date=date(1980, 1, 1),
            sample_date=date(2025, 1, 1),
            albumin_value=4.3,
            albumin_unit="g/dL",
            creatinine_value=0.9,
            creatinine_unit="mg/dL",
            glucose_value=93.0,
            glucose_unit="mg/dL",
            crp_value=1.5,
            crp_unit="mg/L",
            lymphocyte_percent=30.0,
            mcv_fl=90.0,
            rdw_percent=13.2,
            alkaline_phosphatase_u_l=70.0,
            white_blood_cell_value=6.0,
            white_blood_cell_unit="10^3/uL",
        )

        self.assertAlmostEqual(result["chronological_age_years"], 45.0030, places=3)
        self.assertAlmostEqual(result["phenoage_years"], 39.5958, places=3)
        self.assertAlmostEqual(result["difference_years"], -5.4071, places=3)
        self.assertAlmostEqual(result["crp_mg_dl"], 0.15, places=6)

    def test_calculate_phenoage_rejects_non_positive_crp(self):
        with self.assertRaises(ValueError):
            calculate_phenoage(
                age_years=45.0,
                albumin_g_l=43.0,
                creatinine_umol_l=80.0,
                glucose_mmol_l=5.2,
                crp_mg_dl=0.0,
                lymphocyte_percent=30.0,
                mcv_fl=90.0,
                rdw_percent=13.2,
                alkaline_phosphatase_u_l=70.0,
                white_blood_cell_k_per_ul=6.0,
            )


if __name__ == "__main__":
    unittest.main()
