import pandas as pd
import json
import re

def normalize_name(name):
    if not isinstance(name, str):
        return ""
    name = name.lower()
    # Remove common extra words and symbols
    name = re.sub(r'[\(\[\{].*?[\)\]\}]', '', name) # Remove anything in brackets
    name = re.sub(r'[^a-z0-9]', ' ', name) # Replace non-alphanumeric with spaces
    # Standardize some common terms
    replacements = {
        'blood': '',
        'serum': '',
        'plasma': '',
        'automated': '',
        'total': '',
        'count': '',
        'level': '',
        'immunoassay': '',
        'spectrometry': '',
        'mass': '',
        'liquid': '',
        'chromatography': '',
        'concentration': '',
        'measurement': '',
        'clinical': '',
        'chemistry': '',
        'analyzer': '',
        'cholesterol': 'chol'
    }
    for k, v in replacements.items():
        name = re.sub(rf'\b{k}\b', v, name)
    
    name = re.sub(r'\s+', ' ', name).strip()
    return name

# 1. Load NHANES metadata
try:
    with open('dashboard/data/metadata.json') as f:
        nhanes_metadata = json.load(f)
except FileNotFoundError:
    print("Error: dashboard/data/metadata.json not found")
    exit(1)

nhanes_dict = {}
for m in nhanes_metadata:
    b_id = m['biomarker_id']
    name = m.get('display_name') or m.get('biomarker_name') or ''
    nhanes_dict[b_id] = {
        'id': b_id,
        'name': name,
        'norm_name': normalize_name(name),
        'norm_id': normalize_name(b_id)
    }

# 2. Load Clalit tests
df = pd.read_csv('data/clalit/females_all_statistics.csv', usecols=['test', 'test_name', 'long_name'])
clalit_tests = df.drop_duplicates().to_dict('records')

# 3. Load existing mapping
try:
    with open('data/clalit_mapping.json') as f:
        existing_mapping = json.load(f)
except FileNotFoundError:
    existing_mapping = {}

print(f"Loaded {len(nhanes_dict)} NHANES tests, {len(clalit_tests)} Clalit tests, {len(existing_mapping)} existing mappings.")

# 4. Try to find matches
new_mapping = existing_mapping.copy()
match_log = []

for c_test in clalit_tests:
    c_id = c_test['test']
    c_name1 = str(c_test.get('test_name', ''))
    c_name2 = str(c_test.get('long_name', ''))
    
    if c_id in new_mapping:
        match_log.append({
            'clalit_id': c_id,
            'clalit_name': c_name2 or c_name1,
            'nhanes_id': new_mapping[c_id],
            'nhanes_name': nhanes_dict.get(new_mapping[c_id], {}).get('name', new_mapping[c_id]),
            'status': 'Mapped (Existing)'
        })
        continue
    
    norm_c1 = normalize_name(c_name1)
    norm_c2 = normalize_name(c_name2)
    norm_cid = normalize_name(c_id.replace('lab.', '').replace('.no_meds', ''))
    
    best_match = None
    best_score = 0
    
    # Simple keyword matching heuristic
    c_keywords = set(norm_c1.split() + norm_c2.split())
    # remove very short keywords
    c_keywords = {k for k in c_keywords if len(k) > 2}
    
    if not c_keywords:
        match_log.append({'clalit_id': c_id, 'clalit_name': c_name2 or c_name1, 'nhanes_id': '', 'nhanes_name': '', 'status': 'Unmapped'})
        continue
        
    for n_id, n_data in nhanes_dict.items():
        n_keywords = set(n_data['norm_name'].split() + n_data['norm_id'].split())
        
        if not n_keywords:
            continue
            
        common = c_keywords.intersection(n_keywords)
        score = len(common) / (len(c_keywords) + len(n_keywords) - len(common)) # Jaccard
        
        # Hardcode some common aliases if needed, but lets rely on Jaccard first
        # Give bonus if all clalit keywords are in nhanes
        if len(c_keywords) > 0 and common == c_keywords:
            score += 1.0
            
        if score > best_score:
            best_score = score
            best_match = n_id
            
    if best_match and best_score > 0.3: # Threshold
        new_mapping[c_id] = best_match
        match_log.append({
            'clalit_id': c_id,
            'clalit_name': c_name2 or c_name1,
            'nhanes_id': best_match,
            'nhanes_name': nhanes_dict[best_match]['name'],
            'status': 'Auto-Mapped'
        })
    else:
        match_log.append({
            'clalit_id': c_id,
            'clalit_name': c_name2 or c_name1,
            'nhanes_id': '',
            'nhanes_name': '',
            'status': 'Unmapped'
        })

# 5. Create final Data Availability dataset
# We want all Clalit tests, AND all mapped NHANES tests, to see availability. Let's make a unified list.
rows = []
for m in match_log:
    row = {
        'Test Name (Clalit)': m['clalit_name'],
        'Test Name (NHANES)': m['nhanes_name'],
        'Clalit ID': m['clalit_id'],
        'NHANES ID': m['nhanes_id'],
        'In Clalit': 'V',
        'In NHANES': 'V' if m['nhanes_id'] else '',
        'Mapping Status': m['status']
    }
    rows.append(row)

# Now add NHANES tests that are NOT in Clalit (optional, but requested "with a V if its available in that DB")
mapped_nhanes = {m['nhanes_id'] for m in match_log if m['nhanes_id']}
for n_id, n_data in nhanes_dict.items():
    if n_id not in mapped_nhanes:
        rows.append({
            'Test Name (Clalit)': '',
            'Test Name (NHANES)': n_data['name'],
            'Clalit ID': '',
            'NHANES ID': n_id,
            'In Clalit': '',
            'In NHANES': 'V',
            'Mapping Status': 'NHANES Only'
        })

df_out = pd.DataFrame(rows)
df_out.to_csv('data_availability.csv', index=False)
print(f"Generated data_availability.csv with {len(df_out)} total tests.")

with open('data/clalit_mapping_updated.json', 'w') as f:
    json.dump(new_mapping, f, indent=2)
print(f"Saved candidate mapping to data/clalit_mapping_updated.json ({len(new_mapping)} keys).")

