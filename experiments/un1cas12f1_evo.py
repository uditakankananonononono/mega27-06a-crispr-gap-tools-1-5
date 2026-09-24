"""Engineered Un1Cas12f1 (Nat Commun 2026, 10.1038/s41467-026-69678-5,
MOESM7 Fig 2): WT vs evoCas12f1 editing across ~40 noncanonical-PAM targets
(two blocks), plus vs enAsCas12a. Fold improvement and PAM-class dependence."""
import json
import numpy as np
import openpyxl
from collections import defaultdict

wb = openpyxl.load_workbook('data/raw/un1cas12f1/moesm7.xlsx', read_only=True)
ws = wb['Fig 2']
rows = list(ws.iter_rows(values_only=True))

blocks = [(3, 44, 'fig2a', 'Un1Cas12f1-WT', 'evoCas12f'),
          (47, 88, 'fig2c', 'Un1Cas12f1-WT', 'evoCas12f'),
          (91, 103, 'fig2fg', 'enAsCas12a', 'evoCas12f')]
out = {}
all_wt, all_evo = [], []
for lo, hi, tag, ref, evo_name in blocks:
    recs = []
    for r in rows[lo:hi]:
        if r[1] is None:
            continue
        try:
            refv = [float(x) for x in r[2:5] if isinstance(x, (int, float))]
            evv = [float(x) for x in r[5:8] if isinstance(x, (int, float))]
        except (TypeError, ValueError):
            continue
        if not refv or not evv:
            continue
        recs.append(dict(target=str(r[1]), ref=round(float(np.mean(refv)), 2),
                         evo=round(float(np.mean(evv)), 2)))
    for x in recs:
        x['fold'] = round(x['evo'] / max(x['ref'], 0.01), 2)
    folds = [x['fold'] for x in recs if x['ref'] > 0.05]
    out[tag] = dict(ref_enzyme=ref, n_targets=len(recs),
                    median_fold=round(float(np.median(folds)), 1) if folds else None,
                    max_fold=max((x['fold'] for x in recs), default=None),
                    ref_median=round(float(np.median([x['ref'] for x in recs])), 2),
                    evo_median=round(float(np.median([x['evo'] for x in recs])), 2),
                    targets_ref_lt2_evo_gt10=sum(1 for x in recs if x['ref'] < 2 and x['evo'] > 10))
    if tag in ('fig2a', 'fig2c'):
        all_wt.extend([x['ref'] for x in recs])
        all_evo.extend([x['evo'] for x in recs])
# PAM class: target name prefix like AATA/AATG/CATG
pam = defaultdict(list)
for lo, hi, tag, ref, evo_name in blocks[:2]:
    for r in rows[lo:hi]:
        if r[1] is None:
            continue
        try:
            evv = [float(x) for x in r[5:8] if isinstance(x, (int, float))]
            refv = [float(x) for x in r[2:5] if isinstance(x, (int, float))]
        except (TypeError, ValueError):
            continue
        if refv and evv:
            pam[str(r[1]).split('-')[0]].append((float(np.mean(refv)), float(np.mean(evv))))
out['pam_class'] = {p: dict(n=len(v), wt_median=round(float(np.median([x[0] for x in v])), 2),
                            evo_median=round(float(np.median([x[1] for x in v])), 2))
                    for p, v in sorted(pam.items())}
json.dump(out, open('results/un1cas12f1_evo.json', 'w'), indent=1)
print(json.dumps(out, indent=1)[:2600])
