"""Kim et al mismatch screen (Day7 ON+OFF sheet, MOESM10 of
10.1038/s41467-023-41393-5): per-variant indel for mismatch-carrying
guides against their target contexts. Gap-4: mismatch tolerance gradient
per variant (relative to WT at the same mismatch)."""
import json
import numpy as np
import openpyxl
from collections import defaultdict

wb = openpyxl.load_workbook('data/raw/hifi_rule/moesm10.xlsx', read_only=True)
ws = wb['Kim et al MMS (Day7-ON+OFF)']
rows = list(ws.iter_rows(min_row=2, values_only=True))
VARS = ['WT', 'Sniper', 'eSpCas9', 'HF1', 'Hypa', 'evo']

recs = []
for r in rows:
    guide = str(r[1]).strip().upper() if r[1] else None
    ctx = str(r[2]).strip().upper() if r[2] else None
    if not guide or not ctx or len(guide) != 20 or len(ctx) < 24:
        continue
    proto = ctx[4:24]
    if set(guide) - set('ACGT') or set(proto) - set('ACGT'):
        continue
    vals = []
    for i in range(5, 11):
        try:
            vals.append(float(r[i]))
        except (TypeError, ValueError):
            vals.append(np.nan)
    mm_pos = [i for i in range(20) if guide[i] != proto[i]]
    recs.append((len(mm_pos), mm_pos, vals))

by_mm = defaultdict(lambda: defaultdict(list))
for n_mm, pos, vals in recs:
    if n_mm == 0 or np.isnan(vals[0]) or vals[0] <= 0:
        continue
    for vi, v in enumerate(VARS[1:], start=1):
        if not np.isnan(vals[vi]):
            by_mm[n_mm][VARS[vi]].append(vals[vi] / vals[0])

out = dict(n_rows=len(recs), n_offtarget_pairs=sum(1 for n, _, _ in recs if n > 0),
           retention_by_mm={str(mm): {v: dict(n=len(x), median_ret=round(float(np.median(x)), 4))
                                      for v, x in d.items()}
                            for mm, d in sorted(by_mm.items()) if mm <= 3})
json.dump(out, open('results/kim_mms_tolerance.json', 'w'), indent=1)
print(json.dumps(out, indent=1))
