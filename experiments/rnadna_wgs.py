"""WGS off-target validation with guide optimization (Nat Commun 2023,
10.1038/s41467-023-42695-4, MOESM9): per-site background-corrected indel
freq for WT vs optimized guides at PCSK9 and BCL11A."""
import json
import openpyxl
import numpy as np

wb = openpyxl.load_workbook('data/raw/rnadna_off/moesm9.xlsx', read_only=True)

def load(sheet):
    ws = wb[sheet]
    rows = []
    for r in ws.iter_rows(min_row=3, values_only=True):
        if r[0] is None or r[5] is None:
            continue
        try:
            rows.append(dict(mm=int(float(r[5])),
                             fixed=float(r[10]), raw=float(r[8])))
        except (TypeError, ValueError, IndexError):
            continue
    return rows

out = {}
for guide, s_wt, s_op in (('PCSK9', 'WGS_PCSK9', 'WGS_PCSK9-opti'),
                          ('BCL11A', 'WGS_BCL11A', 'WGS_BCL11A-opti')):
    res = {}
    for tag, sheet in (('wt', s_wt), ('opti', s_op)):
        rows = load(sheet)
        on = [r for r in rows if r['mm'] == 0]
        off = [r for r in rows if r['mm'] > 0]
        res[tag] = dict(n_sites=len(rows),
                        on_target_fixed=round(on[0]['fixed'], 4) if on else None,
                        off_gt5pct=sum(1 for r in off if r['fixed'] > 0.05),
                        off_burden=round(float(sum(r['fixed'] for r in off)), 3))
    out[guide] = res
json.dump(out, open('results/rnadna_wgs.json', 'w'), indent=1)
print(json.dumps(out, indent=1))
