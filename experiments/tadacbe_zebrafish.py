"""TadA-derived cytosine base editors in zebrafish embryos (Nat Biomed Eng 2025,
10.1038/s41551-025-01607-1, MOESM8 fig1b/1c): C-to-T editing (3 injection
replicates) per site-position across 5 editors (zTadCBE, TCBE1.1-1.4).
Gap-1/2: which editor wins on mean activity, and do site rankings transfer
between editors?"""
import json
import numpy as np
import openpyxl
from scipy.stats import spearmanr

wb = openpyxl.load_workbook('data/raw/tadacbe_zebrafish/moesm8.xlsx', read_only=True)
BLOCKS = [  # (name, first replicate column 0-indexed)
    ('zTadCBE', 1), ('TCBE1.1', 7), ('TCBE1.2', 12), ('TCBE1.3', 17), ('TCBE1.4', 22),
]
site_rows = {}  # site -> {editor: mean of 3 reps}
for sheet in ('fig1b', 'fig1c'):
    rows = list(wb[sheet].iter_rows(values_only=True))
    for r in rows[3:]:
        if r[0] is None:
            continue
        site = str(r[0]).strip()
        entry = {}
        for name, c0 in BLOCKS:
            try:
                reps = [float(r[c0 + k]) for k in range(3)]
            except (TypeError, ValueError, IndexError):
                continue
            entry[name] = float(np.mean(reps))
        if len(entry) == len(BLOCKS):
            site_rows[f'{sheet}:{site}'] = entry
assert len(site_rows) >= 20, len(site_rows)
editors = [b[0] for b in BLOCKS]
per_editor = {}
for ed in editors:
    vals = np.array([v[ed] for v in site_rows.values()])
    per_editor[ed] = {'mean_ctot': float(vals.mean()), 'median_ctot': float(np.median(vals))}
rho = {}
for i, a in enumerate(editors):
    for b in editors[i + 1:]:
        x = [v[a] for v in site_rows.values()]
        y = [v[b] for v in site_rows.values()]
        rho[f'{a}_vs_{b}'] = float(spearmanr(x, y).statistic)
out = {
    'n_site_positions': len(site_rows),
    'per_editor': per_editor,
    'cross_editor_site_rank_spearman': rho,
    'min_cross_editor_rho': float(min(rho.values())),
    'max_cross_editor_rho': float(max(rho.values())),
    'best_editor_by_mean': max(per_editor, key=lambda e: per_editor[e]['mean_ctot']),
}
print(json.dumps(out, indent=1))
json.dump(out, open('results/tadacbe_zebrafish.json', 'w'), indent=1)
