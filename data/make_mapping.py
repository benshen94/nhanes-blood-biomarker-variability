import pandas as pd
import json
import difflib
import re

# Load Clalit Test IDs
df1 = pd.read_csv('clalit/females_all_statistics.csv')
clalit_tests = df1[['test', 'short_name', 'long_name', 'test_name']].drop_duplicates().to_dict(orient='records')

# Load NHANES Biomarker Names
df2 = pd.read_parquet('processed/biomarker_catalog.parquet')
nhanes_tests = df2[['biomarker_id', 'biomarker_name', 'variable_name']].drop_duplicates().to_dict(orient='records')

mapping = {}

def clean(name):
    if not isinstance(name, str): return ""
    name = name.lower().strip()
    name = re.sub(r'[^a-z0-9]', ' ', name)
    name = re.sub(r'\s+', ' ', name)
    return name.strip()

nhanes_by_name = {clean(n['biomarker_name']): n['biomarker_id'] for n in nhanes_tests if n['biomarker_name']}
nhanes_by_var = {clean(n['variable_name']): n['biomarker_id'] for n in nhanes_tests if n['variable_name']}
nhanes_all_names = list(nhanes_by_name.keys())

# Manual mappings for tricky / very specific names where difflib or substring might fail or pick wrong target
MANUAL_MAP = {
    'hb a1c': 'lab.10.no_meds',
    'cholesterol': 'lab.58.no_meds',
    'hdl': 'lab.59.no_meds',
    'ldl': 'lab.60.no_meds',
    'creatinine': 'lab.20.no_meds',
    'ast': 'lab.41.no_meds', # aspartate aminotransferase
    'alt': 'lab.42.no_meds', # alanine aminotransferase
    'alp': 'lab.45.no_meds', # alkaline phosphatase
    'crp': 'lab.56.no_meds',
    'tsh': 'lab.75.no_meds',
    'wbc': 'lab.90.no_meds',
    'rbc': 'lab.84.no_meds',
    'platelets': 'lab.91.no_meds',
    'hemoglobin': 'lab.85.no_meds',
    'hematocrit': 'lab.86.no_meds',
    'mcv': 'lab.87.no_meds',
    'bun': 'lab.19.no_meds',
    'uric acid': 'lab.21.no_meds',
    'sodium': 'lab.24.no_meds',
    'potassium': 'lab.25.no_meds',
    'chloride': 'lab.26.no_meds',
    'calcium': 'lab.29.no_meds',
    'phosphorus': 'lab.30.no_meds',
    'total protein': 'lab.35.no_meds',
    'albumin': 'lab.36.no_meds',
    'globulin': 'lab.37.no_meds',
    'bilirubin': 'lab.39.no_meds', # total
    'ggt': 'lab.44.no_meds',
    'ldh': 'lab.48.no_meds',
    'iron': 'lab.102.no_meds',
    'triglycerides': 'lab.61.no_meds',
    'glucose': 'lab.9.no_meds'
}

for c in clalit_tests:
    names_to_try = [clean(c[k]) for k in ['test_name', 'long_name', 'short_name'] if c[k]]
    
    match_id = None
    
    # 1. Manual check
    for n in names_to_try:
        for mk, vid in MANUAL_MAP.items():
            if mk in n.split():
                 match_id = vid
                 break
        if match_id: break
    
    # 2. Exact match
    if not match_id:
        for n in names_to_try:
            if n in nhanes_by_name:
                match_id = nhanes_by_name[n]
                break
            if n in nhanes_by_var:
                match_id = nhanes_by_var[n]
                break
    
    # 3. Fuzzy match
    if not match_id:
        for n in names_to_try:
            matches = difflib.get_close_matches(n, nhanes_all_names, n=1, cutoff=0.85)
            if matches:
                match_id = nhanes_by_name[matches[0]]
                break
                
    if match_id:
        mapping[c['test']] = match_id

with open('clalit_mapping.json', 'w') as f:
    json.dump(mapping, f, indent=2)

print(f"Mapped {len(mapping)} tests to NHANES equivalent.")
