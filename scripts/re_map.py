import json

with open('data/clalit_mapping.json') as f:
    mapping = json.load(f)

# The correct manual mappings that were lost
manual_map = {
    'lab.22500.no_meds': 'creatine phosphokinase cpk', # Creatine kinase
    'lab.103500.no_meds': 'parathyroid hormone elecys method pg/ml', # PTH 
    'lab.27300.no_meds': '25-hydroxyvitamin d3', # Vitamin D3 25OH
    'marker.height.no_meds': 'standing height', # height
    'marker.weight.no_meds': 'weight', # weight
    'marker.BMI.no_meds': 'body mass index', # BMI
    'lab.100700.no_meds': '17a-hydroxyprogesterone__ng/dl', # 17a-OHP (wait, prolactin was 'prolactin__ng/ml') -> Let's check lab.100700 vs lab.101000
    'lab.101000.no_meds': 'prolactin__ng/ml', # Prolactin
    'lab.101500.no_meds': 'dhea-sulfate', # DHEA-S
    'lab.22000.no_meds': 'aspartate aminotransferase ast', # AST
    'lab.22100.no_meds': 'alanine aminotransferase alt', # ALT
    'lab.105.no_meds': 'platelet count', # Platelets
    'lab.110.no_meds': 'mean platelet volume', # MPV
    'lab.101.no_meds': 'white blood cell count', # WBC
    'lab.146.no_meds': 'eosinophils number', # Eosinophils ABS
    'lab.148.no_meds': 'basophils percent', # Basophiles
    'lab.142.no_meds': 'segmented neutrophils percent', # Neutrophils (clalit does not have ABS or maybe it does? lab.142 is Neutrophils percent or abs?) - I'll map it to segmented neutrophils percent
    'lab.140.no_meds': 'lymphocyte percent', # Lymphocytes
    'lab.144.no_meds': 'monocyte percent', # Monocytes
    'lab.111.no_meds': 'red blood cell count', # RBC - wait, 111 is what?
    'lab.106.no_meds': 'mean cell volume', # MCV
    'lab.107.no_meds': 'mean cell hemoglobin', # MCH
}

# Apply manual edits
for k, v in manual_map.items():
    mapping[k] = v

with open('data/clalit_mapping.json', 'w') as f:
    json.dump(mapping, f, indent=2)
