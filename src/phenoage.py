#!/usr/bin/env python3

from __future__ import annotations

from datetime import date
import math
from typing import Mapping


DAYS_PER_YEAR = 365.2425

ALBUMIN_UNIT_FACTORS: Mapping[str, float] = {
    "g/dL": 10.0,
    "g/L": 1.0,
}

CREATININE_UNIT_FACTORS: Mapping[str, float] = {
    "mg/dL": 88.4017,
    "umol/L": 1.0,
}

GLUCOSE_UNIT_FACTORS: Mapping[str, float] = {
    "mg/dL": 0.0555,
    "mmol/L": 1.0,
}

CRP_UNIT_FACTORS: Mapping[str, float] = {
    "mg/dL": 1.0,
    "mg/L": 0.1,
}

WBC_UNIT_FACTORS: Mapping[str, float] = {
    "10^3/uL": 1.0,
    "10^9/L": 1.0,
}


def _convert_unit(value: float, unit: str, factors: Mapping[str, float], label: str) -> float:
    factor = factors.get(unit)
    if factor is None:
        raise ValueError(f"Unsupported {label} unit: {unit}")
    return value * factor


def age_years_at_sample(birth_date: date, sample_date: date) -> float:
    if sample_date <= birth_date:
        raise ValueError("Sample date must be after birth date")

    delta_days = (sample_date - birth_date).days
    return delta_days / DAYS_PER_YEAR


def convert_albumin_to_g_l(value: float, unit: str) -> float:
    return _convert_unit(value, unit, ALBUMIN_UNIT_FACTORS, "albumin")


def convert_creatinine_to_umol_l(value: float, unit: str) -> float:
    return _convert_unit(value, unit, CREATININE_UNIT_FACTORS, "creatinine")


def convert_glucose_to_mmol_l(value: float, unit: str) -> float:
    return _convert_unit(value, unit, GLUCOSE_UNIT_FACTORS, "glucose")


def convert_crp_to_mg_dl(value: float, unit: str) -> float:
    return _convert_unit(value, unit, CRP_UNIT_FACTORS, "CRP")


def convert_wbc_to_k_per_ul(value: float, unit: str) -> float:
    return _convert_unit(value, unit, WBC_UNIT_FACTORS, "white blood cell count")


def calculate_phenoage(
    *,
    age_years: float,
    albumin_g_l: float,
    creatinine_umol_l: float,
    glucose_mmol_l: float,
    crp_mg_dl: float,
    lymphocyte_percent: float,
    mcv_fl: float,
    rdw_percent: float,
    alkaline_phosphatase_u_l: float,
    white_blood_cell_k_per_ul: float,
) -> float:
    if crp_mg_dl <= 0:
        raise ValueError("CRP must be greater than 0 because the published PhenoAge model uses log(CRP)")

    xb = (
        -19.90667
        + (-0.03359355 * albumin_g_l)
        + (0.009506491 * creatinine_umol_l)
        + (0.1953192 * glucose_mmol_l)
        + (0.09536762 * math.log(crp_mg_dl))
        + (-0.01199984 * lymphocyte_percent)
        + (0.02676401 * mcv_fl)
        + (0.3306156 * rdw_percent)
        + (0.001868778 * alkaline_phosphatase_u_l)
        + (0.05542406 * white_blood_cell_k_per_ul)
        + (0.08035356 * age_years)
    )

    mortality_score = 1.0 - math.exp((-1.51714 * math.exp(xb)) / 0.007692696)
    if mortality_score <= 0 or mortality_score >= 1:
        raise ValueError("PhenoAge calculation produced an invalid mortality score")

    return (math.log(-0.0055305 * math.log(1.0 - mortality_score)) / 0.090165) + 141.50225


def calculate_phenoage_from_lab_report(
    *,
    birth_date: date,
    sample_date: date,
    albumin_value: float,
    albumin_unit: str,
    creatinine_value: float,
    creatinine_unit: str,
    glucose_value: float,
    glucose_unit: str,
    crp_value: float,
    crp_unit: str,
    lymphocyte_percent: float,
    mcv_fl: float,
    rdw_percent: float,
    alkaline_phosphatase_u_l: float,
    white_blood_cell_value: float,
    white_blood_cell_unit: str,
) -> dict[str, float]:
    age_years = age_years_at_sample(birth_date, sample_date)

    albumin_g_l = convert_albumin_to_g_l(albumin_value, albumin_unit)
    creatinine_umol_l = convert_creatinine_to_umol_l(creatinine_value, creatinine_unit)
    glucose_mmol_l = convert_glucose_to_mmol_l(glucose_value, glucose_unit)
    crp_mg_dl = convert_crp_to_mg_dl(crp_value, crp_unit)
    white_blood_cell_k_per_ul = convert_wbc_to_k_per_ul(white_blood_cell_value, white_blood_cell_unit)

    phenoage_years = calculate_phenoage(
        age_years=age_years,
        albumin_g_l=albumin_g_l,
        creatinine_umol_l=creatinine_umol_l,
        glucose_mmol_l=glucose_mmol_l,
        crp_mg_dl=crp_mg_dl,
        lymphocyte_percent=lymphocyte_percent,
        mcv_fl=mcv_fl,
        rdw_percent=rdw_percent,
        alkaline_phosphatase_u_l=alkaline_phosphatase_u_l,
        white_blood_cell_k_per_ul=white_blood_cell_k_per_ul,
    )

    return {
        "chronological_age_years": age_years,
        "phenoage_years": phenoage_years,
        "difference_years": phenoage_years - age_years,
        "albumin_g_l": albumin_g_l,
        "creatinine_umol_l": creatinine_umol_l,
        "glucose_mmol_l": glucose_mmol_l,
        "crp_mg_dl": crp_mg_dl,
        "white_blood_cell_k_per_ul": white_blood_cell_k_per_ul,
    }
