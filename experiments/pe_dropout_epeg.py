"""Benchmarked PE platform epegRNA library (Nat Methods 2024,
10.1038/s41592-024-02502-4, MOESM3 Supp Table 1): editing efficiency
(Freq.EditOnly) vs design features (PBS/RTT length, edit position) and
editing purity (EditOnly vs TotalErrors)."""
import json
import openpyxl
import numpy as np
from scipy.stats import spearmanr
from collections import defaultdict

wb = openpyxl.load_workbook('data/raw/pe_dropout/moesm3.xlsx', read_only=True)
ws = wb['Supplementary Table 1']
recs = {}
cells = defaultdict(set)
for r in ws.iter_rows(min_row=2, values_only=True):
    if r[0] is None or r[22] != 'yes':
        continue
    key = (r[0], r[13])
    cells[r[0]].add(r[13])
    recs[key] = dict(pbs=r[7], rtt=r[9], pos=r[2], edit=r[21], err=r[19],
                     eff=r[20])
out_rows = [(k[0], k[1], v) for k, v in recs.items()]
print('unique epegRNA x cellline:', len(out_rows))
def f(x):
    try:
        return float(x)
    except (TypeError, ValueError):
        return np.nan
pbs = np.array([f(v['pbs']) for _, _, v in out_rows])
rtt = np.array([f(v['rtt']) for _, _, v in out_rows])
pos = np.array([f(v['pos']) for _, _, v in out_rows])
eff = np.array([f(v['eff']) for _, _, v in out_rows])
err = np.array([f(v['err']) for _, _, v in out_rows])
ok = ~np.isnan(pos) & ~np.isnan(eff)
purity = eff / (eff + err + 1e-9)
out = dict(
    n_epegrna_x_cellline=len(out_rows),
    cell_lines={str(k): len(v) for k, v in
                defaultdict(list, {c: [1 for kk in cells.values() if c in kk] for c in {x for s in cells.values() for x in s}}).items()},
    eff_median=round(float(np.median(eff)), 4),
    purity_median=round(float(np.median(purity)), 3),
    n_with_position=int(ok.sum()),
    spearman_eff_vs=dict(pbs_len=round(float(spearmanr(pbs, eff, nan_policy='omit').statistic), 3),
                         rtt_len=round(float(spearmanr(rtt, eff, nan_policy='omit').statistic), 3),
                         edit_pos=round(float(spearmanr(pos[ok], eff[ok]).statistic), 3)),
    spearman_eff_vs_purity=round(float(spearmanr(eff, purity).statistic), 3),
)
json.dump(out, open('results/pe_dropout_epeg.json', 'w'), indent=1)
print(json.dumps(out, indent=1))
