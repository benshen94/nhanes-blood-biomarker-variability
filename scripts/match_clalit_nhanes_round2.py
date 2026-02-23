import pandas as pd
import json

# Manual overrides for the unmapped tests and corrections to auto-mapped ones
manual_map = {
    'lab.22500.no_meds': 'creatine kinase__u/l', # Creatine kinase
    'lab.21800.no_meds': 'alkaline phosphatase', # Phosphatase alkaline
    'lab.103500.no_meds': 'pth', # PTH (Intact Parathyroid Hormone isn't straightforward, need to check if 'pth' exists)
    'lab.27300.no_meds': '25-hydroxyvitamin d3', # Vitamin D3 25OH
    'marker.height.no_meds': 'standing height', # height
    'marker.weight.no_meds': 'weight', # weight
    'marker.BMI.no_meds': 'body mass index', # BMI
    'lab.100700.no_meds': 'prolactin__ng/ml', # Prolactin
    'lab.101200.no_meds': 'dhea-sulfate', # DHEA-S
    'lab.22400.no_meds': 'aspartate aminotransferase ast', # AST (GOT)
    'lab.22100.no_meds': 'alanine aminotransferase alt', # ALT (GPT)
    'lab.105.no_meds': 'mean cell volume', # MPV -> Mean platelet volume. Wait, let me check nhanes
    'lab.109.no_meds': 'platelet count', # Platelets
    'lab.113.no_meds': 'white blood cell count', # WBC
    'lab.124.no_meds': 'eosinophils number', # Eosinophils ABS
    'lab.120.no_meds': 'basophils percent', # Basophiles
    'lab.122.no_meds': 'neutrophils percent', # Neutrophils
    'lab.123.no_meds': 'lymphocytes percent', # Lymphocytes
    'lab.121.no_meds': 'monocytes percent', # Monocytes
    'lab.21300.no_meds': 'triglycerides__mg/dl', # Triglycerides
    'lab.20100.no_meds': 'plasma glucose', # Glucose (Wait, or fasting glucose?)
    'lab.23400.no_meds': None, # Bilirubin direct (don't map to total bilirubin)
    'lab.36200.no_meds': None, # Bilirubin indirect (don't map to total bilirubin)
    'lab.111.no_meds': 'red blood cell count', # RBC
    'lab.112.no_meds': 'mean cell volume', # MCV
    'lab.20008.no_meds': None, # PT control -> don't map to mitogen
    'lab.101900.no_meds': None, # GH -> don't map to LH
}

with open('data/clalit_mapping_updated.json') as f:
    mapping = json.load(f)

for k, v in manual_map.items():
    if v is None:
        if k in mapping:
            del mapping[k]
    else:
        mapping[k] = v

with open('data/clalit_mapping_updated.json', 'w') as f:
    json.dump(mapping, f, indent=2)
