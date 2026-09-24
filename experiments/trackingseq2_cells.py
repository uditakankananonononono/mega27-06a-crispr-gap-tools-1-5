"""Tracking-seq2 (Nat Commun 2026, 10.1038/s41467-026-76778-9, MOESM3):
Cas9 off-target site sets in 293T vs primary T-cells vs HSPCs for HEK4
and VEGFA2. Coordinate-overlap (+-50bp) cross-cell-type replication."""
import json
import re
import openpyxl
from collections import defaultdict

wb = openpyxl.load_workbook('data/raw/trackingseq2/moesm3.xlsx', read_only=True)

def sites(sheet):
    ws = wb[sheet]
    out = set()
    for r in ws.iter_rows(min_row=2, values_only=True):
        if r[0] is None:
            continue
        m = re.match(r'(chr[\w]+):(\d+)-(\d+)', str(r[0]))
        if m:
            out.add((m.group(1), (int(m.group(2)) + int(m.group(3))) // 2))
    return out

def overlap(a, b, pad=50):
    byc = defaultdict(list)
    for c, s in b:
        byc[c].append(s)
    return sum(1 for c, s in a if any(abs(s - t) <= pad for t in byc.get(c, [])))

out = {}
for guide, sheets in (('HEK4', ('Cas9_HEK4', 'Cas9_HEK4_T-cells', 'Cas9_HEK4_HSPCs')),
                      ('VEGFA2', ('Cas9_VEGFA2', 'Cas9_VEGFA2_T-cells'))):
    sets = {s: sites(s) for s in sheets}
    res = {s: len(v) for s, v in sets.items()}
    ks = list(sets)
    for i in range(len(ks)):
        for j in range(i + 1, len(ks)):
            a, b = sets[ks[i]], sets[ks[j]]
            ov = overlap(a, b)
            res[f'{ks[i]}__vs__{ks[j]}'] = dict(overlap=ov,
                                                jaccard=round(ov / (len(a) + len(b) - ov), 3))
    out[guide] = res
json.dump(out, open('results/trackingseq2_cells.json', 'w'), indent=1)
print(json.dumps(out, indent=1))
