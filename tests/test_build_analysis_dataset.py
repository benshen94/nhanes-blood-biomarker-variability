#!/usr/bin/env python3

import sys
import unittest
from pathlib import Path

import pandas as pd

sys.path.insert(0, str((Path(__file__).resolve().parents[1] / "src")))

from build_analysis_dataset import build_pooling_map, is_comment_or_code_variable, normalize_unit, parse_terminal_unit


class TestBuildAnalysisDataset(unittest.TestCase):
    def test_parse_terminal_unit_supports_suffix_and_inner_alias_units(self):
        self.assertEqual(parse_terminal_unit("Albumin g/dL"), ("Albumin", "g/dL"))
        self.assertEqual(
            parse_terminal_unit("Sex Hormone Binding Globulin (SHBG, nmol/L)"),
            ("Sex Hormone Binding Globulin SHBG", "nmol/L"),
        )

    def test_normalize_unit_collapses_known_synonyms(self):
        self.assertEqual(normalize_unit("U/L"), "iu/l")
        self.assertEqual(normalize_unit("IU/L"), "iu/l")
        self.assertEqual(normalize_unit("mOsm/kg"), "mmol/kg")
        self.assertEqual(normalize_unit("SHBG, nmol/L"), "shbgnmol/l")

    def test_build_pooling_map_merges_aliases_and_missing_units(self):
        manifest = pd.DataFrame(
            [
                {
                    "variable_name": "SSALB",
                    "variable_desc": "Albumin",
                    "xpt_url": "https://wwwn.cdc.gov/Nchs/Data/Nhanes/Public/2001/DataFiles/SSOL_A.xpt",
                    "is_blood_candidate": True,
                },
                {
                    "variable_name": "LBXSAL",
                    "variable_desc": "Albumin (g/dL)",
                    "xpt_url": "https://wwwn.cdc.gov/Nchs/Data/Nhanes/Public/2003/DataFiles/BIOPRO_D.xpt",
                    "is_blood_candidate": True,
                },
                {
                    "variable_name": "LBDSALSI",
                    "variable_desc": "Albumin (g/L)",
                    "xpt_url": "https://wwwn.cdc.gov/Nchs/Data/Nhanes/Public/2003/DataFiles/BIOPRO_D.xpt",
                    "is_blood_candidate": True,
                },
                {
                    "variable_name": "LBXTR",
                    "variable_desc": "Triglyceride (mg/dL)",
                    "xpt_url": "https://wwwn.cdc.gov/Nchs/Data/Nhanes/Public/2005/DataFiles/TRIGLY_E.xpt",
                    "is_blood_candidate": True,
                },
                {
                    "variable_name": "LBXSTR",
                    "variable_desc": "Triglycerides (mg/dL)",
                    "xpt_url": "https://wwwn.cdc.gov/Nchs/Data/Nhanes/Public/2007/DataFiles/BIOPRO_G.xpt",
                    "is_blood_candidate": True,
                },
            ]
        )

        pooled = build_pooling_map(manifest, raw_dir=None, candidate_column="is_blood_candidate")

        albumin = pooled[pooled["pool_group_key"] == "albumin"].sort_values("variable_name")
        self.assertEqual(set(albumin["pooled_id"]), {"albumin"})
        self.assertEqual(set(albumin["pooled_unit"]), {"g/dL"})
        g_per_l_row = albumin.loc[albumin["variable_name"] == "LBDSALSI"].iloc[0]
        self.assertAlmostEqual(g_per_l_row["conversion_factor_to_pooled_unit"], 0.1)

        triglyceride = pooled[pooled["pool_group_key"] == "triglyceride"]
        self.assertEqual(set(triglyceride["pooled_id"]), {"triglyceride"})

    def test_is_comment_or_code_variable_excludes_formula_ldl_variants(self):
        self.assertTrue(
            is_comment_or_code_variable(
                "LBDLDLM",
                "LDL-Cholesterol, Martin-Hopkins equation (mg/dL). LBDLDLM = (LBXTC-(LBDHDD + LBXTR/Adjustable Factor), round to 0 decimal places)",
            )
        )
        self.assertTrue(
            is_comment_or_code_variable(
                "LBDLDLN",
                "LBDLDLN = (LBXTC/0.948 - LBDHDD/0.971 - (LBXTR/8.56 + (LBXTR * (LBXTC - LBDHDD))/2140 - LBXTR^2/16100) - 9.44)",
            )
        )
        self.assertFalse(is_comment_or_code_variable("LBDLDL", "LDL-cholesterol (mg/dL)"))


if __name__ == "__main__":
    unittest.main()
