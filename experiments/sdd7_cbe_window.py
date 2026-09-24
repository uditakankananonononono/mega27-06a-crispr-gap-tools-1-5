"""Engineered Sdd7 cytosine base editors (Nat Commun 2025,
10.1038/s41467-025-60789-z, MOESM8): Fig 3b per-site editing (60 sites x
BE4max/Sdd7/Sdd7e1/Sdd7e2) and Fig 3c positional profiles (C2..C10 per site
per editor).
Gap-1/2: does the engineered deaminase shift/narrow the editing window, and
do site rankings transfer between editor variants?"""
import json
import numpy as np
import openpyxl
from scipy.stats import spearmanr

wb = openpyxl.load_workbook('data/raw/sdd7_cbe/moesm8.xlsx', read_only=True)

# Fig 3b: per-site editor activity
rows = list(wb['Figure 3b'].iter_rows(values_only=True))
# two stacked blocks of the same 30 sites: block 1 = primary on-target editing,
# block 2 = secondary low-signal measurement (bystander class) at the same sites
blocks_b = []
cur = {}
for r in rows[2:]:
    c0 = str(r[0]).strip() if r[0] is not None else ''
    if c0 == 'BE4max' or (not c0 and r[1] == 'BE4max'):
        if cur:
            blocks_b.append(cur)
        cur = {}
        continue
    if not c0:
        continue
    try:
        cur[c0] = [float(r[k]) for k in (1, 2, 3, 4)]
    except (TypeError, ValueError):
        continue
if cur:
    blocks_b.append(cur)
assert len(blocks_b) == 2 and len(blocks_b[0]) == len(blocks_b[1]) == 30, [len(b) for b in blocks_b]
primary, secondary = blocks_b
editors = ['BE4max', 'Sdd7', 'Sdd7e1', 'Sdd7e2']
common = sorted(set(primary) & set(secondary))
assert len(common) == 30
rho = {}
for i, a in enumerate(editors):
    for b in editors[i + 1:]:
        x = [primary[s][i] for s in common]
        y = [primary[s][editors.index(b)] for s in common]
        rho[f'{a}_vs_{b}'] = float(spearmanr(x, y).statistic)
means = {ed: float(np.mean([primary[s][k] for s in common])) for k, ed in enumerate(editors)}
sec_means = {ed: float(np.mean([secondary[s][k] for s in common])) for k, ed in enumerate(editors)}
spec_ratio = {ed: sec_means[ed] / means[ed] for ed in editors}

# Fig 3c: positional profiles, blocks per editor
rows = list(wb['Figure 3c'].iter_rows(values_only=True))
blocks = {}
cur = None
for r in rows[1:]:
    c0 = str(r[0]).strip() if r[0] is not None else ''
    if c0:
        cur = c0
        blocks[cur] = []
        continue
    if cur is None:
        continue
    vals = []
    ok = True
    for k in range(1, 10):
        try:
            vals.append(float(r[k]))
        except (TypeError, ValueError):
            vals.append(np.nan)
    if any(np.isfinite(vals)):
        blocks[cur].append(vals)
profiles = {}
for ed, mat in blocks.items():
    if len(mat) < 5:
        continue
    m = np.array([row for row in mat], dtype=float)
    prof = np.nanmean(m, axis=0)
    positions = np.arange(2, 11)
    peak = int(positions[np.argmax(prof)])
    total = np.nansum(prof)
    inwindow = float(np.nansum(prof[2:6]) / total) if total > 0 else None  # C4-C7
    profiles[ed] = {'n_sites': int(len(m)), 'peak_position': peak,
                    'share_C4_to_C7': inwindow,
                    'profile': [round(float(x), 2) for x in prof]}
out = {
    'n_sites': len(common),
    'secondary_mean_activity': sec_means,
    'specificity_index_secondary_over_primary': spec_ratio,
    'note': "Fig 3b carries two stacked blocks over the same 30 sites: primary on-target editing and a secondary low-signal (bystander-class) measurement; ratio used as specificity index.",
    'n_sites_activity': len(common),
    'editor_mean_activity': means,
    'cross_editor_site_spearman': rho,
    'min_cross_editor_rho': float(min(rho.values())),
    'positional_profiles': profiles,
}
print(json.dumps(out, indent=1))
json.dump(out, open('results/sdd7_cbe_window.json', 'w'), indent=1)
