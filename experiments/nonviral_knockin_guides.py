"""Nonviral gene-insertion screen (Nat Commun 2026, 10.1038/s41467-026-76350-5,
MOESM4 Fig 1d): 77,440 sgRNAs with edited vs non-edited read counts in a
nonviral knock-in screen. Per-guide edited fraction as guide-efficacy readout;
GC dependence; per-gene spread. Gap-1."""
import json
import re
import numpy as np
import openpyxl
from collections import defaultdict
from scipy.stats import spearmanr

wb = openpyxl.load_workbook('data/raw/nonviral_screen/moesm4.xlsx', read_only=True)
ws = wb['Fig 1d']
recs = []
for r in ws.iter_rows(min_row=2, values_only=True):
    if r[0] is None:
        continue
    m = re.match(r'>([A-Za-z0-9]+)_([ACGT]+)', str(r[0]))
    if not m:
        continue
    try:
        ne, e = float(r[1]), float(r[2])
    except (TypeError, ValueError):
        continue
    recs.append(dict(gene=m.group(1), seq=m.group(2), nonedited=ne, edited=e))

out = dict(n_guides=len(recs))
for r in recs:
    tot = r['edited'] + r['nonedited']
    r['tot'] = tot
    r['efrac'] = r['edited'] / tot if tot > 0 else np.nan
    r['gc'] = (r['seq'].count('G') + r['seq'].count('C')) / len(r['seq'])

usable = [r for r in recs if r['tot'] >= 100 and not np.isnan(r['efrac'])]
out['n_usable_totreads_ge100'] = len(usable)
ef = np.array([r['efrac'] for r in usable])
out['edited_frac'] = dict(median=round(float(np.median(ef)), 3),
                          p10=round(float(np.percentile(ef, 10)), 3),
                          p90=round(float(np.percentile(ef, 90)), 3),
                          frac_ge_09=round(float(np.mean(ef >= 0.9)), 3),
                          frac_le_05=round(float(np.mean(ef <= 0.5)), 3))
# GC dependence
gc = np.array([r['gc'] for r in usable])
out['gc_spearman'] = round(float(spearmanr(gc, ef).statistic), 3)
# per-gene spread (genes with >= 5 guides)
by_gene = defaultdict(list)
for r in usable:
    by_gene[r['gene']].append(r['efrac'])
gm = {g: float(np.median(v)) for g, v in by_gene.items() if len(v) >= 3}
out['n_genes_ge3_guides'] = len(gm)
out['guides_per_gene_median'] = float(np.median([len(v) for v in by_gene.values()]))
if gm:
    vals = sorted(gm.values())
    out['gene_median_efrac_p10'] = round(vals[max(len(vals) // 10, 0)], 3)
    out['gene_median_efrac_p90'] = round(vals[min(9 * len(vals) // 10, len(vals) - 1)], 3)
json.dump(out, open('results/nonviral_knockin_guides.json', 'w'), indent=1)
print(json.dumps(out, indent=1))
