"""Cas12a bridge-helix allostery (Nat Commun 2026, 10.1038/s41467-026-68657-0,
MOESM8 graph sheets): cleavage kinetics (1/2.5/5/15 min) of WT and bridge-helix
variants on matched (MCH) vs mismatched (MM2, MM13) substrates. Each sheet has
two stacked blocks; we use the second (full-precision) block consistently.
Gap-2/4: do bridge-helix mutations confer kinetic discrimination - suppressing
mismatch cleavage more than matched cleavage?"""
import json
import numpy as np
import openpyxl

wb = openpyxl.load_workbook('data/raw/cas12a_bridge_helix/moesm8.xlsx', read_only=True)

def block2(sheet):
    rows = list(wb[sheet].iter_rows(values_only=True))
    blocks, cur = [], {}
    for r in rows:
        if r[0] is not None and str(r[0]).strip() == 'Protein':
            if cur:
                blocks.append(cur)
            cur = {}
            continue
        if r[0] is None:
            continue
        name = str(r[0]).strip()
        try:
            cur[name] = [float(r[k]) for k in (1, 2, 3, 4)]
        except (TypeError, ValueError):
            continue
    if cur:
        blocks.append(cur)
    return blocks[-1]

mch = block2('Fig3 (e) MCH_graph')
mm2 = block2('Fig18 (b) MM2 graph')
mm13 = block2('Fig 3 (f) MM13 graph')
common = sorted(set(mch) & set(mm2) & set(mm13))
assert len(common) >= 8, common
per_variant = {}
for v in common:
    per_variant[v] = {
        'matched_15min': mch[v][3],
        'mm2_15min': mm2[v][3],
        'mm13_15min': mm13[v][3],
        'discrimination_mm13_15min': mch[v][3] - mm13[v][3],
        'matched_1min': mch[v][0],
        'mm13_1min': mm13[v][0],
    }
out = {
    'n_variants': len(common),
    'per_variant': per_variant,
    'wt_discrimination_mm13_15min': per_variant['WT']['discrimination_mm13_15min'],
    'best_discriminator': max(per_variant, key=lambda v: per_variant[v]['discrimination_mm13_15min']),
}
print(json.dumps(out, indent=1))
json.dump(out, open('results/cas12a_bridge_helix.json', 'w'), indent=1)
