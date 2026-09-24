"""UNCOVERseq (Nat Commun 2026, 10.1038/s41467-026-74623-7, MOESM9):
6,487 nominated off-target candidates with nomination frequency, bin
classification, and amplicon-validated indel rates. Validation rate by
bin and per-gRNA burden."""
import json
import openpyxl
import numpy as np
from collections import defaultdict

wb = openpyxl.load_workbook('data/raw/uncoverseq/moesm9.xlsx', read_only=True)
ws = wb['Sheet1']
rows = []
for r in ws.iter_rows(min_row=2, values_only=True):
    if r[0] is None:
        continue
    try:
        rows.append(dict(nom=float(r[4]), method=str(r[6]), grna=str(r[7]),
                         bin=str(r[8]), sel=str(r[9]),
                         indel=float(r[11]) if r[11] is not None else np.nan))
    except (TypeError, ValueError, IndexError):
        continue

out = dict(n_rows=len(rows),
           grnas=sorted({r['grna'] for r in rows}),
           methods=sorted({r['method'] for r in rows}))
by_bin = defaultdict(list)
for r in rows:
    if not np.isnan(r['indel']):
        by_bin[r['bin']].append(r)
val = {}
for b, rs in sorted(by_bin.items()):
    val[b] = dict(n=len(rs),
                  validated_gt1pct=sum(1 for x in rs if x['indel'] > 0.01),
                  median_nomfreq=round(float(np.median([x['nom'] for x in rs])), 1),
                  median_indel=round(float(np.median([x['indel'] for x in rs])), 4))
out['by_bin'] = val
per_grna = defaultdict(int)
for r in rows:
    if not np.isnan(r['indel']) and r['indel'] > 0.01:
        per_grna[r['grna']] += 1
out['validated_sites_per_grna'] = dict(sorted(per_grna.items(), key=lambda kv: -kv[1]))
json.dump(out, open('results/uncoverseq.json', 'w'), indent=1)
print(json.dumps(out, indent=1))
