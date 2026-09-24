"""Tunable Cas12a platform (Nat Commun 2026, 10.1038/s41467-026-73955-8,
MOESM6 Fig 2b): FACS readout of CRISPRi knockdown across 4 conditions
(Cas12a, Cas12aDeg degron, +/- 1uM dTagV1) x 2 guides x 4 surface markers,
3 replicates. How well does degron+dTag control Cas12a activity?"""
import json
import numpy as np
import openpyxl
from collections import defaultdict

wb = openpyxl.load_workbook('data/raw/cas12a_perturb/moesm6.xlsx', read_only=True)
ws = wb['Fig. 2b']
markers = ['CD47', 'CD63', 'CD81', 'CD9']
data = defaultdict(lambda: defaultdict(list))
for r in ws.iter_rows(min_row=7, values_only=True):
    if r[0] is None or r[1] is None:
        continue
    cond, guide = str(r[0]).strip(), str(r[1]).strip()
    if not cond.startswith('Cas12a') or not guide.startswith('Guide'):
        continue
    # cols 2,3 CD47 MFI/%+; 4,5 CD63; 6,7 CD81; 8,9 CD9
    for mi, m in enumerate(markers):
        pct = r[3 + mi * 2]
        mfi = r[2 + mi * 2]
        if isinstance(pct, (int, float)):
            data[(cond, guide)][m].append(float(pct))

out = {}
for (cond, guide), per_m in sorted(data.items()):
    out.setdefault(cond, {})[guide] = {m: round(float(np.mean(v)), 2)
                                       for m, v in per_m.items()}
# control = Guide_025 (non-depleting), targeting = Guide_034
summary = {}
for cond in out:
    if 'Guide_025' in out[cond] and 'Guide_034' in out[cond]:
        kd = {m: round(out[cond]['Guide_034'][m] / out[cond]['Guide_025'][m], 3)
              for m in markers if m in out[cond]['Guide_025'] and m in out[cond]['Guide_034']}
        summary[cond] = dict(mean_residual_frac=round(float(np.mean(list(kd.values()))), 3),
                             per_marker_residual=kd)
result = dict(conditions=sorted(out), raw=out, knockdown_residual=summary)
# dynamic range: induced vs uninduced for the degron system
if 'Cas12aDeg' in summary and 'Cas12aDeg_1uMdTagV1' in summary:
    result['degron_dynamic_range'] = round(
        summary['Cas12aDeg']['mean_residual_frac'] /
        max(summary['Cas12aDeg_1uMdTagV1']['mean_residual_frac'], 1e-9), 2)
if 'Cas12a' in summary and 'Cas12a_1uMdTagV1' in summary:
    result['constitutive_residual'] = summary['Cas12a']['mean_residual_frac']
json.dump(result, open('results/cas12a_tunable_control.json', 'w'), indent=1)
print(json.dumps(result['knockdown_residual'], indent=1))
print('degron dynamic range:', result.get('degron_dynamic_range'))
