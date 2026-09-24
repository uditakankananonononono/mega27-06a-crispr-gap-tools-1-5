"""Cas12a variant PAM-flexibility screen (Nat Commun 2025,
10.1038/s41467-025-57150-9, MOESM5): wide matrix, row1 = PAM block
labels, row2 = variant names, rows 3+ = per-target indel frequencies.
Per-variant activity at canonical TTTV vs alternative PAMs."""
import json
import openpyxl
import numpy as np
from collections import defaultdict

wb = openpyxl.load_workbook('data/raw/cas12a_variants/moesm5.xlsx', read_only=True)
ws = wb['Sheet1']
rows = list(ws.iter_rows(values_only=True))
pam_row, var_row = rows[0], rows[1]

# PAM blocks: spans between non-empty row-1 cells
blocks = []
cur = None
for i, v in enumerate(pam_row):
    if v is not None and str(v).strip():
        if cur:
            blocks.append(cur)
        cur = [str(v).strip(), i, i + 1]
    elif cur:
        cur[2] = i + 1
if cur:
    blocks.append(cur)

def norm(name):
    n = str(name).lower().replace(' ', '').replace('-', '').replace('_', '')
    return n

# map each column to (pam, normalized variant)
col_meta = {}
for pam, s, e in blocks:
    for i in range(s, min(e, len(var_row))):
        v = var_row[i]
        if v is not None and str(v).strip():
            col_meta[i] = (pam, norm(v))

vals = defaultdict(lambda: defaultdict(list))
for r in rows[2:]:
    for i, (pam, var) in col_meta.items():
        if i < len(r) and r[i] is not None:
            try:
                x = float(r[i])
            except (TypeError, ValueError):
                continue
            if 0 <= x <= 1:
                vals[var][pam].append(x)

pams = sorted({p for vv in vals.values() for p in vv})
out = dict(n_pam_blocks=len(blocks), n_variant_columns=len(col_meta),
           variants_parsed=len(vals))
# canonical TTTV reference and flexibility per variant
flex = {}
for var, d in vals.items():
    if 'TTTV' in d and len(d['TTTV']) >= 100:
        ref = float(np.mean(d['TTTV']))
        if ref > 0.05:
            alt = {}
            for p in pams:
                if p != 'TTTV' and len(d[p]) >= 50:
                    alt[p] = round(float(np.mean(d[p])) / ref, 3)
            if alt:
                flex[var] = dict(ref_tttv=round(ref, 3), alt=alt)
out['n_variants_with_tttv'] = len(flex)
# PAM-level mean across all variant columns
pam_means = {}
for pam in pams:
    allv = [x for vv in vals.values() if pam in vv for x in vv[pam]]
    pam_means[pam] = round(float(np.mean(allv)), 4) if allv else None
out['pam_mean_activity'] = pam_means
# breadth: for variants with TTTV ref, count alt PAMs retaining >10%
breadth = {v: sum(1 for a in f['alt'].values() if a > 0.1) for v, f in flex.items()}
out['breadth_gt10pct'] = dict(sorted(breadth.items(), key=lambda kv: -kv[1]))
json.dump(out, open('results/cas12a_variants_pam.json', 'w'), indent=1)
print('PAMs:', pams)
print('pam_mean_activity:', pam_means)
print('n_variants_with_tttv:', out['n_variants_with_tttv'])
print('breadth top:', list(out['breadth_gt10pct'].items())[:12])
