"""Evolved high-precision base editors (Nat Commun 2024, 10.1038/s41467-024-52483-3,
MOESM8 S Fig12): per-guide blocks x ABE variant columns; rows On-target/OTk.
Gap-2/4: do evolved ABEs reduce OT activity, and at what on-target cost?"""
import json
import numpy as np
import openpyxl
from collections import defaultdict

wb = openpyxl.load_workbook('data/raw/be_evolved/moesm8.xlsx', read_only=True)
rows = list(wb['S Fig12'].iter_rows(values_only=True))
blocks = []  # each: dict(site, editors=[(name,col)], measures=[(label, {editor: val})])
for r in rows:
    if r[0] and str(r[0]).strip() and not str(r[0]).startswith('Supplementary'):
        blocks.append(dict(site=str(r[0]).strip(),
                           editors=[(str(c).strip(), i) for i, c in enumerate(r)
                                    if c and str(c).strip() and i >= 2],
                           measures=[]))
        continue
    if blocks and r[1] and blocks[-1]['editors']:
        label = str(r[1]).strip()
        vals = {}
        for name, c in blocks[-1]['editors']:
            xs = []
            for k in range(c, min(c + 3, len(r))):
                try:
                    if r[k] is not None:
                        xs.append(float(r[k]))
                except (TypeError, ValueError):
                    pass
            if xs:
                vals[name] = float(np.mean(xs))
        if vals:
            blocks[-1]['measures'].append((label, vals))

per_editor_on = defaultdict(list)
per_editor_ot_ret = defaultdict(list)
n_ot = 0
for b in blocks:
    site, measures = b['site'], b['measures']
    on = next((v for l, v in measures if l == 'On-target'), None)
    if not on:
        continue
    for e, x in on.items():
        per_editor_on[e].append(x)
    for label, vals in measures:
        if not label.startswith('OT'):
            continue
        n_ot += 1
        for e, x in vals.items():
            if e in on and on[e] > 0:
                per_editor_ot_ret[e].append(x / on[e])

out = dict(n_blocks=len(blocks), n_ot_rows=n_ot,
           editors={e: dict(n_sites=len(per_editor_on[e]),
                            mean_on_target=round(float(np.mean(per_editor_on[e])), 2),
                            n_ot=len(per_editor_ot_ret[e]),
                            median_ot_retention=round(float(np.median(per_editor_ot_ret[e])), 4) if per_editor_ot_ret[e] else None,
                            frac_ot_eliminated=round(float(np.mean([r < 0.1 for r in per_editor_ot_ret[e]])), 3) if per_editor_ot_ret[e] else None)
                    for e in sorted(per_editor_on)})
json.dump(out, open('results/be_evolved_offtarget.json', 'w'), indent=1)
print(json.dumps(out, indent=1))
