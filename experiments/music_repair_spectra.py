"""Mutational-scar DSB repair map (MUSIC; Nat Commun 2026,
10.1038/s41467-026-72744-7, MOESM4/6/7): per-barcode repair-outcome
fractions across genes and pathway contexts, mouse (MOESM4: 121 genes,
186k rows; MOESM6: 28 genes) and human RPE1 (MOESM7: 29 genes).
Gap-2/5: how deterministic is the repair-outcome spectrum per locus, and
does it transfer across species/contexts?"""
import json
import re
from collections import defaultdict
import numpy as np
import openpyxl
from scipy.stats import spearmanr

def outcome_class(s):
    s = str(s)
    if 'DEL' in s:
        m = re.search(r'(\d+)', s)
        size = int(m.group(1)) if m else None
        if size is not None and size <= 3:
            return 'DEL_1-3'
        return 'DEL_4+'
    if 'INS' in s:
        return 'INS'
    if 'WT' in s.upper() or 'UNMOD' in s.upper():
        return 'WT'
    return 'OTHER'

def load(path, sheet, gene_col, outcome_col, frac_col):
    wb = openpyxl.load_workbook(path, read_only=True)
    ws = wb[sheet]
    recs = defaultdict(list)
    for r in ws.iter_rows(values_only=True, min_row=2):
        if r[0] is None:
            continue
        try:
            frac = float(r[frac_col])
        except (TypeError, ValueError, IndexError):
            continue
        gene = str(r[gene_col])
        cls = outcome_class(r[outcome_col])
        recs[gene].append((cls, frac))
    return recs

def spectra(recs):
    """per-gene class-fraction vectors"""
    classes = ['DEL_1-3', 'DEL_4+', 'INS', 'WT', 'OTHER']
    out = {}
    for g, lst in recs.items():
        tot = sum(f for _, f in lst)
        if tot <= 0:
            continue
        agg = defaultdict(float)
        for c, f in lst:
            agg[c] += f
        out[g] = {c: agg[c] / tot for c in classes}
    return out

mouse_big = load('data/raw/music_repair/moesm4.xlsx', 'SupplementaryData2_MUSIC', 3, 6, 7)
mouse = load('data/raw/music_repair/moesm6.xlsx', 'SupplementaryData4_MUSIC', 2, 5, 6)
human = load('data/raw/music_repair/moesm7.xlsx', 'SupplementaryData5_MUSIC', 2, 5, 6)

sp_big, sp_m, sp_h = spectra(mouse_big), spectra(mouse), spectra(human)

# determinism: dominant class share per gene
def dominant_share(sp):
    return {g: max(v.values()) for g, v in sp.items()}

det = {name: float(np.mean(list(dominant_share(sp).values())))
       for name, sp in (('mouse_121', sp_big), ('mouse_28', sp_m), ('human_29', sp_h))}

# class means per context
classes = ['DEL_1-3', 'DEL_4+', 'INS', 'WT', 'OTHER']
class_means = {name: {c: float(np.mean([v[c] for v in sp.values()])) for c in classes}
               for name, sp in (('mouse_121', sp_big), ('mouse_28', sp_m), ('human_29', sp_h))}

# cross-context transfer: shared genes (case-insensitive), per-class Spearman
def norm(g):
    return g.upper()
shared = {norm(g): (g, None) for g in sp_m}
transfer = {}
m_by = {norm(g): v for g, v in sp_m.items()}
h_by = {norm(g): v for g, v in sp_h.items()}
shared = sorted(set(m_by) & set(h_by))
if len(shared) >= 5:
    for c in classes:
        x = [m_by[g][c] for g in shared]
        y = [h_by[g][c] for g in shared]
        if len(set(x)) < 2 or len(set(y)) < 2:
            transfer[c] = None  # constant vector: correlation undefined
        else:
            transfer[c] = float(spearmanr(x, y).statistic)

# per-gene top-class distribution
top_class = defaultdict(int)
for g, v in sp_big.items():
    top_class[max(v, key=v.get)] += 1

out = {
    'n_genes': {'mouse_121': len(sp_big), 'mouse_28': len(sp_m), 'human_29': len(sp_h)},
    'mean_dominant_class_share': det,
    'class_means': class_means,
    'shared_genes_mouse28_human29': shared,
    'cross_context_class_spearman': transfer,
    'top_class_counts_mouse121': dict(top_class),
}
print(json.dumps(out, indent=1))
json.dump(out, open('results/music_repair_spectra.json', 'w'), indent=1)
