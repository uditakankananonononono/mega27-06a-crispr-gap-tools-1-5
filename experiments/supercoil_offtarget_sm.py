"""Supercoiling-induced Cas9 off-target activity (Nature 2026,
10.1038/s41586-026-10255-7, MOESM10 Ext Data Fig 3d-f): single-molecule
contour-length distributions for Cas9 bound at on-target vs two off-targets,
relaxed vs negatively supercoiled DNA. Gap-2/5: DNA topology as off-target
context."""
import json
import numpy as np
import openpyxl

wb = openpyxl.load_workbook('data/raw/supercoil_offtarget/moesm10.xlsx', read_only=True)
out = {}
for sn, tag in [('Ext Data Fig. 3d', 'relaxed'), ('Ext Data Fig. 3e', 'supercoiled'),
                ('Ext Data Fig. 3f', 'supercoiled_rep2')]:
    per = {}
    for r in wb[sn].iter_rows(min_row=2, values_only=True):
        if r[0] is None or r[1] is None:
            continue
        try:
            v = float(r[1])
        except (TypeError, ValueError):
            continue
        per.setdefault(str(r[0]).strip(), []).append(v)
    out[tag] = {k: dict(n=len(v), median=round(float(np.median(v)), 2),
                        mean=round(float(np.mean(v)), 2),
                        iqr=round(float(np.percentile(v, 75) - np.percentile(v, 25)), 2))
                for k, v in sorted(per.items())}
json.dump(out, open('results/supercoil_offtarget_sm.json', 'w'), indent=1)
print(json.dumps(out, indent=1))
