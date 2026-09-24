"""Chimeric oligonucleotide-directed editing (CODE) (Nat Commun 2026,
10.1038/s41467-026-71624-4, MOESM5 Fig 4c-4h): intended vs unintended edit
percentages (3 reps each) for PE2/PEMax/CODEMax(+/-exo+, +/- nicking) across
6 endogenous sites.
Gap-2/3: which editor class maximizes the precision ratio (intended:
unintended), and does nicking raise activity at the cost of precision?"""
import json
import numpy as np
import openpyxl

wb = openpyxl.load_workbook('data/raw/code_editing/moesm5.xlsx', read_only=True)
SITES = ['Fig. 4c', 'Fig. 4d', 'Fig. 4e', 'Fig. 4f', 'Fig. 4g', 'Fig. 4h']
per_site = {}
for sheet in SITES:
    eds = {}
    for r in wb[sheet].iter_rows(values_only=True):
        if r[0] is None:
            continue
        name = str(r[0]).replace('\xa0', ' ').strip()
        if name in ('', 'PE2') and name == '':
            continue
        try:
            intended = [float(r[k]) for k in (1, 2, 3)]
            unintended = [float(r[k]) for k in (4, 5, 6)]
        except (TypeError, ValueError, IndexError):
            continue
        eds[name] = {
            'intended_mean': float(np.mean(intended)),
            'unintended_mean': float(np.mean(unintended)),
            'precision_ratio': float(np.mean(intended) / (np.mean(unintended) + 0.01)),
        }
    per_site[sheet] = eds

# aggregate per editor across sites
editors = sorted({e for s in per_site.values() for e in s})
agg = {}
for ed in editors:
    ratios = [s[ed]['precision_ratio'] for s in per_site.values() if ed in s]
    ints = [s[ed]['intended_mean'] for s in per_site.values() if ed in s]
    agg[ed] = {
        'n_sites': len(ratios),
        'median_precision_ratio': float(np.median(ratios)),
        'mean_intended': float(np.mean(ints)),
    }
# nicking effect: paired editors
nick = {}
for base in ('PE2', 'PEMax', 'CODEMax', 'CODEMax(exo+)'):
    nicked = base + ' + nicking'
    pairs = []
    for s in per_site.values():
        if base in s and nicked in s:
            pairs.append((s[base]['intended_mean'], s[nicked]['intended_mean'],
                          s[base]['unintended_mean'], s[nicked]['unintended_mean']))
    if pairs:
        nick[base] = {
            'n_sites': len(pairs),
            'intended_fold': float(np.mean([p[1] / max(p[0], 0.01) for p in pairs])),
            'unintended_fold': float(np.mean([p[3] / max(p[2], 0.01) for p in pairs])),
        }
out = {'n_sites': len(per_site), 'per_editor': agg, 'nicking_effect': nick}
print(json.dumps(out, indent=1))
json.dump(out, open('results/code_editing_precision.json', 'w'), indent=1)
