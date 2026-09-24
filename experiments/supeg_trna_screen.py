"""Prime-editing-installed suppressor tRNAs (Nature 2025, 10.1038/s41586-025-09732-2,
MOESM4): 53,966 epegRNAs across TAG/TGA/TAA screens with sorted-vs-plasmid fold
enrichment, PBS length, RTT length, tRNA family, amino acid.
Gap-3: which pegRNA design features predict successful sup-tRNA installation?"""
import json
import numpy as np
import openpyxl
from collections import defaultdict
from scipy.stats import spearmanr

wb = openpyxl.load_workbook('data/raw/supeg_trna/moesm4.xlsx', read_only=True)
recs = []
for sheet in ('TAG', 'TGA', 'TAA'):
    rows = wb[sheet].iter_rows(values_only=True)
    next(rows)  # header
    for r in rows:
        if r[0] is None:
            continue
        try:
            recs.append({
                'sheet': sheet,
                'pbs': int(r[3]),
                'rtt': int(r[4]),
                'aa': str(r[17]),
                'fe': float(r[15]),
                'sorted1': float(r[11]) if r[11] is not None else np.nan,
                'sorted2': float(r[12]) if r[12] is not None else np.nan,
            })
        except (TypeError, ValueError):
            continue
n = len(recs)
assert n > 50000, n
fe = np.array([r['fe'] for r in recs])
frac_enriched = float(np.mean(fe > 1))
per_codon = {}
for codon in ('TAG', 'TGA', 'TAA'):
    v = np.array([r['fe'] for r in recs if r['sheet'] == codon])
    per_codon[codon] = {'n': int(v.size), 'median_fold_enrichment': float(np.median(v)), 'frac_enriched': float(np.mean(v > 1))}
by_pbs = defaultdict(list)
for r in recs:
    by_pbs[r['pbs']].append(r['fe'])
pbs_table = {str(k): {'n': len(v), 'median_fe': float(np.median(v))} for k, v in sorted(by_pbs.items()) if len(v) >= 200}
# replicate correlation on sorted fractions (log1p, finite only)
s1 = np.array([r['sorted1'] for r in recs]); s2 = np.array([r['sorted2'] for r in recs])
mask = np.isfinite(s1) & np.isfinite(s2)
rep_rho = float(spearmanr(np.log1p(s1[mask]), np.log1p(s2[mask])).statistic)
by_pbs_enriched = defaultdict(list)
for r in recs:
    if r['fe'] > 1:
        by_pbs_enriched[r['pbs']].append(r['fe'])
pbs_enriched_table = {str(k): {'n_enriched': len(v), 'median_fe': float(np.median(v))}
                      for k, v in sorted(by_pbs_enriched.items()) if len(v) >= 50}
out = {
    'n_epegrnas': n,
    'by_pbs_length_among_enriched': pbs_enriched_table,
    'overall_frac_enriched_gt1': frac_enriched,
    'per_codon': per_codon,
    'by_pbs_length': pbs_table,
    'sorted_replicate_spearman': rep_rho,
}
print(json.dumps(out, indent=1))
json.dump(out, open('results/supeg_trna_screen.json', 'w'), indent=1)
