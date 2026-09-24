"""hpCasMINI indel spectra (Nat Commun 2025, 10.1038/s41467-025-60124-6,
MOESM4 Fig 2c): per-indel-length read fractions for CasMINI vs hpCasMINI at
KLHL29/NLRC4/SITE1, 3 replicates. Does the engineered Cas12f change the
repair-outcome spectrum, not just efficiency?"""
import json
import numpy as np
import openpyxl
from collections import defaultdict
from scipy.stats import spearmanr

wb = openpyxl.load_workbook('data/raw/hpcasmini/moesm4.xlsx', read_only=True)
ws = wb['Figure 2c']
recs = []
for r in ws.iter_rows(min_row=2, values_only=True):
    if r[0] is None:
        continue
    try:
        recs.append(dict(lib=str(r[0]), grna=str(r[1]), rep=str(r[2]),
                         indel=int(r[3]), frac=float(r[4])))
    except (TypeError, ValueError):
        continue

out = dict(n_rows=len(recs),
           libs=sorted({r['lib'] for r in recs}),
           grnas=sorted({r['grna'] for r in recs}))
# per lib: aggregate spectra over grnas and reps
per_lib = defaultdict(list)
for r in recs:
    if r['lib'] in ('CasMINI', 'hpCasMINI'):
        per_lib[r['lib']].append(r)
stats = {}
for lib, rs in per_lib.items():
    dels = [r for r in rs if r['indel'] < 0]
    ins = [r for r in rs if r['indel'] > 0]
    tot_del = sum(r['frac'] for r in dels)
    tot_ins = sum(r['frac'] for r in ins)
    # size-weighted mean deletion length
    if tot_del > 0:
        mean_del = sum(-r['indel'] * r['frac'] for r in dels) / tot_del
    else:
        mean_del = None
    stats[lib] = dict(n_rows=len(rs),
                      total_frac=round(tot_del + tot_ins, 3),
                      del_share=round(tot_del / (tot_del + tot_ins), 3) if tot_del + tot_ins else None,
                      ins1_share_of_all=round(sum(r['frac'] for r in rs if r['indel'] == 1) / (tot_del + tot_ins), 3),
                      mean_del_len=round(mean_del, 2) if mean_del else None)
out['per_library'] = stats
# spectrum correlation between libs at same grna (KLHL29, pooled reps)
for g in ['KLHL29', 'NLRC4', 'SITE1']:
    a = defaultdict(float)
    b = defaultdict(float)
    for r in recs:
        if r['grna'] == g and r['lib'] == 'CasMINI':
            a[r['indel']] += r['frac']
        if r['grna'] == g and r['lib'] == 'hpCasMINI':
            b[r['indel']] += r['frac']
    keys = sorted(set(a) & set(b))
    if len(keys) > 5:
        out[f'spectrum_spearman_{g}'] = round(float(spearmanr([a[k] for k in keys], [b[k] for k in keys]).statistic), 3)
json.dump(out, open('results/hpcasmini_indel_spectra.json', 'w'), indent=1)
print(json.dumps(out, indent=1))
