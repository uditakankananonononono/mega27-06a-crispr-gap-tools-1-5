"""La/PE screen (Nature 2024, 10.1038/s41586-024-07259-6, MOESM3 Supp
Table 1): 103k pegRNAs x FACS_PE3 / MCS_PE3 / MCS_PE4 / MCS_PE5 log2
enrichment. Replicate concordance and cross-screen transfer."""
import json
import openpyxl
import numpy as np
from scipy.stats import spearmanr

wb = openpyxl.load_workbook('data/raw/pe_la/moesm3.xlsx', read_only=True)
ws = wb['Supplementary Table 1']
cond_cols = {'FACS_PE3': (1, 2, 3), 'MCS_PE3': (4, 5, 6), 'MCS_PE4': (7, 8, 9), 'MCS_PE5': (10, 11, 12)}
vals = {c: [[], [], []] for c in cond_cols}
n = 0
for r in ws.iter_rows(min_row=4, values_only=True):
    if r[0] is None:
        continue
    n += 1
    for c, (a, b, m) in cond_cols.items():
        for j, idx in enumerate((a, b, m)):
            try:
                vals[c][j].append(float(r[idx]))
            except (TypeError, ValueError):
                vals[c][j].append(np.nan)

out = dict(n_pegrnas=n, conditions={})
for c, (r1, r2, ave) in vals.items():
    r1, r2, ave = map(np.array, (r1, r2, ave))
    ok = ~np.isnan(r1) & ~np.isnan(r2)
    rep_rho = float(spearmanr(r1[ok], r2[ok]).statistic)
    out['conditions'][c] = dict(n_valid=int(ok.sum()), rep1_rep2_rho=round(rep_rho, 3),
                                median_ave=round(float(np.nanmedian(ave)), 3))
aves = {c: np.array(vals[c][2]) for c in vals}
transfer = {}
cs = list(aves)
for i in range(len(cs)):
    for j in range(i + 1, len(cs)):
        a, b = aves[cs[i]], aves[cs[j]]
        ok = ~np.isnan(a) & ~np.isnan(b)
        transfer[f'{cs[i]}_vs_{cs[j]}'] = round(float(spearmanr(a[ok], b[ok]).statistic), 3)
out['cross_screen_spearman'] = transfer
json.dump(out, open('results/pe_la_transfer.json', 'w'), indent=1)
print(json.dumps(out, indent=1))
