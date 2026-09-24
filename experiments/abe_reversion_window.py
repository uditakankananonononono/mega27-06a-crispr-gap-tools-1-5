"""Mutation-reversion ABEs (Nat Biotechnol 2026, 10.1038/s41587-026-03045-z,
MOESM5 Source_Data11): 10,713 editing measurements - 14 editors x 28 endogenous
sites x A-positions x tri-motifs x replicates.
Gap-1/2: do reversion mutations narrow the editing window and flatten motif
dependence, and at what activity cost vs the parent editor?"""
import json
import numpy as np
import openpyxl
from collections import defaultdict

wb = openpyxl.load_workbook('data/raw/abe_reversion/moesm5.xlsx', read_only=True)
rows = list(wb['Source_Data11'].iter_rows(values_only=True))
recs = []
for r in rows:
    if r[0] is None or str(r[0]) in ('site',) or str(r[0]).startswith('Editing at'):
        continue
    try:
        pos = int(r[3]); pct = float(r[8])
    except (TypeError, ValueError):
        continue
    recs.append({'site': str(r[0]), 'pos': pos, 'motif': str(r[6]), 'editor': str(r[7]), 'pct': pct})
assert len(recs) > 10000, len(recs)

by_editor = defaultdict(list)
for r in recs:
    by_editor[r['editor']].append(r)
per_editor = {}
for ed, rs in sorted(by_editor.items()):
    if ed == 'NT' or len(rs) < 100:
        continue
    pcts = np.array([x['pct'] for x in rs])
    # window concentration: share of total editing at canonical positions 4-8
    win = sum(x['pct'] for x in rs if 4 <= x['pos'] <= 8)
    tot = sum(x['pcttargets'] if False else x['pct'] for x in rs)
    # motif spread: mean editing per tri-motif, max/min
    by_motif = defaultdict(list)
    for x in rs:
        by_motif[x['motif']].append(x['pct'])
    motif_means = [np.mean(v) for v in by_motif.values() if len(v) >= 3]
    per_editor[ed] = {
        'n': len(rs),
        'median_pct_edit': float(np.median(pcts)),
        'mean_pct_edit': float(pcts.mean()),
        'window_4to8_share': float(win / tot) if tot > 0 else None,
        'motif_max_over_min': float(max(motif_means) / max(min(motif_means), 1e-12)),
    }
parent = per_editor['ABE7.10']['mean_pct_edit']
ratios = {ed: round(v['mean_pct_edit'] / parent, 3) for ed, v in per_editor.items()}
out = {
    'n_records': len(recs),
    'n_editors': len(per_editor),
    'per_editor': per_editor,
    'mean_activity_vs_ABE7.10': ratios,
}
print(json.dumps(out, indent=1))
json.dump(out, open('results/abe_reversion_window.json', 'w'), indent=1)
