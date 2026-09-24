"""Organ-specific in vivo editing (Nat Commun 2026, 10.1038/s41467-026-77144-5,
MOESM9): doxycycline-induced Cas9 mice, rhAmpSeq quantification of on-target
plus candidate off-target (OT) sites across 8 organs, 4 mice per organ, for two
guides (gP targeting Pcsk9, gMH targeting Msh2-locus). How organ-specific is
off-target editing in vivo?"""
import json
import numpy as np
import openpyxl
from scipy.stats import spearmanr

wb = openpyxl.load_workbook('data/raw/organ_specific/moesm9.xlsx', read_only=True)

def parse_sheet(sn):
    rows = list(wb[sn].iter_rows(values_only=True))
    hdr0, hdr1 = rows[0], rows[1]
    blocks = [(i, str(v)) for i, v in enumerate(hdr0) if v is not None and 'Dox-induced' in str(v)]
    if not blocks:
        return None, None
    bstart = blocks[0][0]
    # organ spans inside the induced block
    organ_cols = []
    cur, cols = None, []
    for i in range(bstart, len(hdr1)):
        v = hdr1[i] if i < len(hdr1) else None
        if v is not None:
            if cur is not None:
                organ_cols.append((cur, cols))
            cur, cols = str(v), [i]
        else:
            cols.append(i)
    if cur is not None:
        organ_cols.append((cur, cols))
    sites = {}
    for r in rows[2:]:
        label = r[0]
        if label is None:
            continue
        label = str(label)
        per_organ = {}
        for organ, cols in organ_cols:
            vals = [r[c] for c in cols if c < len(r) and isinstance(r[c], (int, float))]
            per_organ[organ] = float(np.mean(vals)) if vals else np.nan
        sites[label] = per_organ
    return organ_cols, sites

out = {}
for sn, tag in [('gP in vivo - editing', 'gP_Pcsk9'), ('gMH in vivo - editing', 'gMH')]:
    organ_cols, sites = parse_sheet(sn)
    organs = [o for o, _ in organ_cols]
    on = [k for k in sites if not k.startswith('OT')]
    ots = {k: v for k, v in sites.items() if k.startswith('OT')}
    rec = dict(n_ot_sites=len(ots), organs=organs,
               on_target_editing={o: round(sites[on[0]][o], 2) for o in organs} if on else None)
    # detection breadth
    for thr in (0.5, 1.0):
        breadth = [sum(1 for o in organs if v[o] >= thr) for v in ots.values()]
        rec[f'detected_ge_{thr}pct_breadth_hist'] = {str(k): sum(1 for b in breadth if b == k) for k in range(0, 9)}
    # organ-specific sites: top organ >= 1%, second organ < 30% of top
    nspec = 0
    examples = []
    for k, v in ots.items():
        vals = {o: max(v[o], 0.0) for o in organs}
        top = max(vals, key=vals.get)
        tv = vals[top]
        second = max(x for o, x in vals.items() if o != top)
        if tv >= 1.0 and second < 0.3 * tv:
            nspec += 1
            examples.append((k, top, round(tv, 2), round(second, 2)))
    rec['organ_specific_sites_n'] = nspec
    rec['organ_specific_examples'] = examples[:6]
    # cross-organ Spearman over OT site profiles
    labels = sorted(ots)
    mat = np.array([[max(ots[l][o], 0.0) for l in labels] for o in organs])
    rhos = []
    for i in range(len(organs)):
        for j in range(i + 1, len(organs)):
            rho, _ = spearmanr(mat[i], mat[j])
            rhos.append((organs[i], organs[j], round(float(rho), 3)))
    rec['cross_organ_spearman_min'] = min(rhos, key=lambda x: x[2])
    rec['cross_organ_spearman_max'] = max(rhos, key=lambda x: x[2])
    rec['cross_organ_spearman_median'] = round(float(np.median([r[2] for r in rhos])), 3)
    # per-organ detected site counts at >=1%
    rec['sites_ge1pct_per_organ'] = {o: sum(1 for l in labels if ots[l][o] >= 1.0) for o in organs}
    out[tag] = rec

json.dump(out, open('results/organ_specific_invivo.json', 'w'), indent=1)
print(json.dumps(out, indent=1)[:3500])
