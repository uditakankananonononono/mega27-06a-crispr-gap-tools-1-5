"""FnCas9 engineered variants PAM screen (Nat Commun 2024,
10.1038/s41467-024-49233-w, MOESM9 Figs 3b/3c/3d/4a): per-locus editing
% for FnCas9, en1/en15/en31, SpCas9-HF1, eSpCas9 across NGG/AGG/NGA
PAMs. Per-variant median activity by PAM."""
import json
import openpyxl
import numpy as np
from collections import defaultdict

wb = openpyxl.load_workbook('data/raw/fncas9_pam/moesm9.xlsx', read_only=True)

def parse(sheet, pam, variant_cols):
    ws = wb[sheet]
    out = defaultdict(list)
    cur = None
    for r in ws.iter_rows(min_row=5, values_only=True):
        if r[0] is not None and str(r[0]).strip():
            cur = str(r[0])
        if cur is None or '_ON' not in cur or 'Note' in cur:
            continue
        for name, idx in variant_cols.items():
            try:
                v = float(r[idx])
            except (TypeError, ValueError, IndexError):
                continue
            if v >= 0:
                out[name].append(v)
    return {pam: out}

data = {}
data.update(parse('Fig. 3b', 'NGG', {'FnCas9': 3, 'en1': 4, 'SpCas9-HF1': 5, 'eSpCas9': 6}))
data.update(parse('Fig. 3c', 'AGG', {'en1': 3, 'en15': 4, 'en31': 5}))
data.update(parse('Fig. 3d', 'NGA', {'en31': 3}))
data.update(parse('Fig. 4a', 'AGG_b', {'FnCas9': 3, 'en1': 4, 'en15': 5, 'en31': 6, 'SpCas9-HF1': 7, 'eSpCas9': 8}))

# combine AGG and AGG_b
comb = defaultdict(lambda: defaultdict(list))
for pam, vd in data.items():
    p = 'AGG' if pam == 'AGG_b' else pam
    for v, vals in vd.items():
        comb[p][v].extend(vals)

out = {}
for pam, vd in comb.items():
    out[pam] = {v: dict(n=len(x), median=round(float(np.median(x)), 2))
                for v, x in vd.items() if len(x) >= 3}
json.dump(out, open('results/fncas9_variants.json', 'w'), indent=1)
print(json.dumps(out, indent=1))
