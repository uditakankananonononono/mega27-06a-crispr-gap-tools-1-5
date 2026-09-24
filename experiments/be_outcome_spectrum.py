"""Multi-dataset BE outcome prediction study (Nat Commun 2025,
10.1038/s41467-025-65200-5, MOESM3): per-gRNA outcome-product spectra
for ABE7.10 (75,500 rows) and BE4 (166,309). Per-gRNA product entropy,
top-product share, WT retention; editor comparison."""
import json
import openpyxl
import numpy as np
from collections import defaultdict


def analyze(path, sheet):
    wb = openpyxl.load_workbook(path, read_only=True)
    ws = wb[sheet]
    prods = defaultdict(list)
    eff = {}
    for r in ws.iter_rows(min_row=2, values_only=True):
        if r[1] is None:
            continue
        g = str(r[1])
        try:
            fr = float(r[8])
        except (TypeError, ValueError, IndexError):
            continue
        prods[g].append(fr)
        try:
            eff[g] = float(r[6])
        except (TypeError, ValueError, IndexError):
            pass
    ent, top1, np5 = [], [], []
    for g, fr in prods.items():
        tot = sum(fr)
        if tot <= 0:
            continue
        p = np.array(fr) / tot
        ent.append(float(-(p * np.log(p + 1e-12)).sum()))
        top1.append(float(p.max()))
        np5.append(int((p > 0.05).sum()))
    e = np.array(list(eff.values()))
    return dict(n_grnas=len(prods),
                median_efficiency_pct=round(float(np.median(e)), 2),
                median_entropy=round(float(np.median(ent)), 3),
                median_top1_share=round(float(np.median(top1)), 3),
                median_n_products_gt5pct=round(float(np.median(np5)), 1))


out = dict(
    ABE7_10=analyze('data/raw/be_multidata/moesm3.xlsx', 'ABE7.10'),
    BE4=analyze('data/raw/be_multidata/moesm3.xlsx', 'BE4'))
json.dump(out, open('results/be_outcome_spectrum.json', 'w'), indent=1)
print(json.dumps(out, indent=1))
