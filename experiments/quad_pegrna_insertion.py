"""Quadruple-pegRNA large-insertion editing (Nature 2026,
10.1038/s41586-026-10395-w, MOESM6/7/8): payload-size scaling, vector
evolution (LV4 -> cV6), cross-cell-line transfer, method comparison
(eePASSIGE/eePASTE, cV6/evoCAST), and off-target indel panel.
Gap-1/3: how does large-insertion efficiency scale with payload size, and
does the ranking transfer across cell lines and effectors?"""
import json
import re
import numpy as np
import openpyxl
from scipy.stats import spearmanr

wb6 = openpyxl.load_workbook('data/raw/quad_pegrna/moesm6.xlsx', read_only=True)
wb7 = openpyxl.load_workbook('data/raw/quad_pegrna/moesm7.xlsx', read_only=True)
wb8 = openpyxl.load_workbook('data/raw/quad_pegrna/moesm8.xlsx', read_only=True)

def size_grid(wb, sheet):
    """rows = loci, col groups of 3 under '<size>-<L|R>' headers."""
    rows = list(wb[sheet].iter_rows(values_only=True))
    hdr = rows[1]
    heads, cur = [], None
    for c in range(2, len(hdr)):
        v = hdr[c]
        if v is not None and str(v).strip():
            cur = str(v).strip()
        heads.append(cur)
    out = {}
    for r in rows[3:]:
        locus = r[1]
        if locus is None or not str(locus).strip():
            continue
        locus = str(locus).strip()
        rec = {}
        for i, c in enumerate(range(2, len(r))):
            h = heads[i]
            if h is None:
                continue
            try:
                v = float(r[c])
            except (TypeError, ValueError):
                continue
            rec.setdefault(h, []).append(v)
        out[locus] = {k: float(np.mean(v)) for k, v in rec.items()}
    return out

# --- payload scaling, HEK293T with LV4 vs cV6 ---
lv4 = size_grid(wb6, 'Fig.2f')
cv6 = size_grid(wb6, 'Fig.2g')
sizes = ['1.6 kb', '3.5 kb', '9.5 kb']
scaling = {}
for name, grid in (('LV4', lv4), ('cV6', cv6)):
    per_size = {}
    for s in sizes:
        vals = [g[f'{s}-{o}'] for g in grid.values()
                for o in ('L', 'R') if f'{s}-{o}' in g]
        per_size[s] = float(np.mean(vals)) if vals else None
    common = [l for l in grid if all(f'{s}-{o}' in grid[l] for s in ('1.6 kb', '9.5 kb') for o in ('L', 'R'))]
    folds = [np.mean([grid[l]['1.6 kb-L'], grid[l]['1.6 kb-R']]) /
             max(np.mean([grid[l]['9.5 kb-L'], grid[l]['9.5 kb-R']]), 1e-9)
             for l in common]
    scaling[name] = {'mean_by_size': per_size, 'n_loci': len(grid),
                     'fold_decay_1.6_to_9.5kb': float(np.mean(folds))}
# paired vector improvement per locus-size cell
pairs = []
for l in lv4:
    if l in cv6:
        for k, v in lv4[l].items():
            if k in cv6[l]:
                pairs.append((v, cv6[l][k]))
vec = {'n_cells': len(pairs),
       'lv4_mean': float(np.mean([a for a, _ in pairs])),
       'cv6_mean': float(np.mean([b for _, b in pairs])),
       'paired_fold': float(np.mean([b / max(a, 1e-9) for a, b in pairs])),
       'cv6_wins': int(sum(1 for a, b in pairs if b > a))}
# 15/26 kb at VEGFA (Fig 2i, 2j), cV6
big = {}
for sn, tag in (('Fig.2i', '15 kb'), ('Fig.2j', '26 kb')):
    g = size_grid(wb6, sn)
    big[tag] = {l: float(np.mean(list(rec.values()))) for l, rec in g.items()}

# --- cross-cell-line transfer (Fig 2k Huh-7, 2l Jurkat, 2m Hepa1-6, 2n mESCs) ---
lines = {'HEK293T-cV6': cv6}
for sn, name in (('Fig.2k', 'Huh-7-cV6'), ('Fig.2l', 'Jurkat-cV6'),
                 ('Fig.2m', 'Hepa1-6-cV6'), ('Fig.2n', 'mESCs-cV6')):
    lines[name] = size_grid(wb6, sn)
line_means = {n: float(np.mean([v for g in gr.values() for v in g.values()]))
              for n, gr in lines.items()}
# locus-rank transfer on shared loci at 1.6/3.5 kb (HEK293T vs Huh-7 vs Jurkat)
shared = set(cv6) & set(lines['Huh-7-cV6'])
transfer = {}
if len(shared) >= 3:
    for other in ('Huh-7-cV6', 'Jurkat-cV6'):
        sh = sorted(set(cv6) & set(lines[other]))
        if len(sh) >= 3:
            x = [np.mean(list(cv6[l].values())) for l in sh]
            y = [np.mean(list(lines[other][l].values())) for l in sh]
            transfer[f'HEK293T_vs_{other}'] = float(spearmanr(x, y).statistic)

# --- method comparisons ---
ws = wb7['Fig.3b']
rows = list(ws.iter_rows(values_only=True))
comp = {}
hdr = [str(c).strip() if c else '' for c in rows[1]]
for r in rows[3:]:
    if r[1] is None:
        continue
    locus = str(r[1]).strip()
    try:
        comp.setdefault('eePASSIGE', []).append(float(r[2]) if r[2] is not None else np.nan)
        comp.setdefault('eePASTE', []).append(float(r[5]) if r[5] is not None else np.nan)
    except (TypeError, ValueError):
        continue
comp = {k: float(np.nanmean(v)) for k, v in comp.items()}
ws = wb7['Fig.3d']
rows = list(ws.iter_rows(values_only=True))
cast = {}
for r in rows[3:]:
    if r[1] is None:
        continue
    locus = str(r[1]).strip()
    try:
        a = np.nanmean([float(x) for x in r[2:5] if x is not None])
        b = np.nanmean([float(x) for x in r[5:8] if x is not None])
        cast[locus] = {'cV6': float(a), 'evoCAST': float(b)}
    except (TypeError, ValueError):
        continue
cast_fold = float(np.mean([v['cV6'] / max(v['evoCAST'], 1e-9) for v in cast.values()]))

# --- off-target indel panel (MOESM8 Fig.4d) ---
ws = wb8['Fig.4d']
rows = list(ws.iter_rows(values_only=True))
ot = {}
cur_peg = None
for r in rows[1:]:
    if r[1] is not None and str(r[1]).strip():
        cur_peg = str(r[1]).strip()
    name = str(r[2]).strip() if r[2] else ''
    try:
        v = float(r[5])
    except (TypeError, ValueError, IndexError):
        continue
    ot.setdefault(cur_peg, {})[name or 'unnamed'] = v
ot_summary = {}
for peg, rec in ot.items():
    on = rec.get('On-target')
    ots = [v for k, v in rec.items() if k != 'On-target']
    ot_summary[peg] = {'on_target': on, 'max_ot': max(ots) if ots else None,
                       'n_ot': len(ots)}

out = {
    'payload_scaling': scaling,
    'vector_evolution': vec,
    'large_payload_per_locus': big,
    'cell_line_means': line_means,
    'cross_line_locus_rank': transfer,
    'passige_vs_paste': comp,
    'cv6_vs_evocast': {'per_locus': cast, 'mean_fold': cast_fold},
    'offtarget_panel': ot_summary,
}
print(json.dumps(out, indent=1))
json.dump(out, open('results/quad_pegrna_insertion.json', 'w'), indent=1)
