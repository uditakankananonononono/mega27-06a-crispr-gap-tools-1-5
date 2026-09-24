"""Repair-context x outcome study (Nat Commun 2024,
10.1038/s41467-024-54566-7, MOESM4 Fig2b): Cas9 outcome-type
distribution across 18 repair-gene knockout contexts vs control.
L1 shift per knockout and largest shifted outcome class."""
import json
import openpyxl
import numpy as np

wb = openpyxl.load_workbook('data/raw/repair_context/moesm4.xlsx', read_only=True)
ws = wb['Fig2']
rows = list(ws.iter_rows(min_row=2, values_only=True))
hdr = [str(x) if x else '' for x in rows[0]]
ctrl, cond = {}, {}
for r in rows[1:]:
    if len(r) < 3 or r[1] is None:
        if ctrl:
            break
        continue
    otype = str(r[1])
    try:
        ctrl[otype] = float(r[2])
    except (TypeError, ValueError, IndexError):
        continue
    for j in range(3, min(len(hdr), len(r))):
        if not hdr[j]:
            continue
        try:
            cond.setdefault(hdr[j], {})[otype] = float(r[j])
        except (TypeError, ValueError):
            pass
otypes = list(ctrl)
cv = np.array([ctrl[o] for o in otypes])
out = {}
for name, d in cond.items():
    if len(d) < len(otypes) * 0.8:
        continue
    v = np.array([d.get(o, 0.0) for o in otypes])
    delta = v - cv
    k = int(np.argmax(np.abs(delta)))
    out[name] = dict(l1_vs_control=round(float(np.abs(delta).sum()), 2),
                     biggest_shift=dict(outcome=otypes[k], ctrl=round(float(cv[k]), 2),
                                        ko=round(float(v[k]), 2)))
res = dict(n_outcome_types=len(otypes), n_conditions=len(out),
           conditions=dict(sorted(out.items(), key=lambda kv: -kv[1]['l1_vs_control'])))
json.dump(res, open('results/repair_context.json', 'w'), indent=1)
print(list(res['conditions'])[:5], 'top:', res['conditions']['Nbn'])
