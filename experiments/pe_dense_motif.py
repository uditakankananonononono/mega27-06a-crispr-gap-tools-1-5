"""Dense RNA-motif prime editing (Nat Biomed Eng 2026, 10.1038/s41551-026-01787-4,
MOESM4 Fig 1-2): motif-variant pegRNAs at EMX1 and HEK3 under PEmax/PE6c/PE6d,
intended-edit and indel percentages in triplicate. Gap-3: how much do 3' motif
variants move PE efficiency, and is the ranking consistent across loci/editors?"""
import json
import numpy as np
import openpyxl
from collections import defaultdict

wb = openpyxl.load_workbook('data/raw/pe_dense_motif/moesm4.xlsx', read_only=True)
ws = wb['Fig.1-2']
rows = list(ws.iter_rows(values_only=True))

recs = []
locus = None
editor = None
for r in rows:
    c0 = str(r[0]).strip() if r[0] is not None else ''
    if c0 in ('EMX1', 'HEK3'):
        locus = c0
        continue
    if c0 in ('PEmax', 'PE6c', 'PE6d'):
        editor = c0
        continue
    if not c0 or c0.startswith(('Intended', 'Rep')):
        continue
    vals = [x for x in r[1:7] if isinstance(x, (int, float))]
    if len(vals) >= 6 and locus and editor:
        recs.append(dict(locus=locus, editor=editor, variant=c0,
                         edit=float(np.mean(vals[:3])), indel=float(np.mean(vals[3:6]))))

out = dict(n_records=len(recs),
           loci=sorted({r['locus'] for r in recs}),
           editors=sorted({r['editor'] for r in recs}),
           n_variants=len({r['variant'] for r in recs}))
by_var = defaultdict(list)
for r in recs:
    by_var[r['variant']].append(r)
var_stats = []
for v, rs in by_var.items():
    var_stats.append(dict(variant=v, n=len(rs),
                          mean_edit=round(float(np.mean([x['edit'] for x in rs])), 2),
                          mean_indel=round(float(np.mean([x['indel'] for x in rs])), 2)))
var_stats.sort(key=lambda x: -x['mean_edit'])
out['top_variants'] = var_stats[:6]
base = [x for x in var_stats if 'Em/Sca' in x['variant']]
if base:
    out['baseline_variant'] = base[0]
    out['best_fold_over_baseline'] = round(var_stats[0]['mean_edit'] / base[0]['mean_edit'], 2)
# ranking consistency: EMX1 vs HEK3, and PEmax vs PE6c
from scipy.stats import spearmanr
def rankcorr(a_key, b_key, dim, keydim):
    A = {(r['variant'], r[keydim]): r['edit'] for r in recs if r[dim] == a_key}
    B = {(r['variant'], r[keydim]): r['edit'] for r in recs if r[dim] == b_key}
    shared = sorted(set(A) & set(B))
    return dict(n=len(shared),
                rho=round(float(spearmanr([A[k] for k in shared], [B[k] for k in shared]).statistic), 3))
out['locus_rank_spearman'] = rankcorr('EMX1', 'HEK3', 'locus', 'editor')
out['pemax_vs_pe6c_spearman'] = rankcorr('PEmax', 'PE6c', 'editor', 'locus')
json.dump(out, open('results/pe_dense_motif.json', 'w'), indent=1)
print(json.dumps(out, indent=1)[:2400])
