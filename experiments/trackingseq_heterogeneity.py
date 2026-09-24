"""Tracking-seq (Nat Biotechnol 2024, 10.1038/s41587-024-02307-y, MOESM4-6):
one-method off-target capture across editors and cellular contexts.
A) Same guide, different editors (VEGFA_site_2: Cas9/CBE/PE2/PE3;
   HEK293_site_4: Cas9/CBE/ABE/PE2): pairwise Jaccard of site sets.
B) Same guide (Pcsk9-gP) in 5 mouse contexts (NIH3T3/AML12/mHSPC/mESC/Liver):
   pairwise Jaccard + does track_score predict cross-context replication?"""
import json
import numpy as np
import openpyxl
from scipy.stats import spearmanr, mannwhitneyu


def load(path, sheet):
    wb = openpyxl.load_workbook(path, read_only=True)
    ws = wb[sheet]
    rows = list(ws.iter_rows(values_only=True))
    hdr = [str(h).strip() if h else '' for h in rows[0]]
    try:
        i_loc = hdr.index('target_location')
        i_score = hdr.index('track_score') if 'track_score' in hdr else None
    except ValueError:
        return {}
    out = {}
    for r in rows[1:]:
        loc = r[i_loc]
        if loc is None:
            continue
        out[str(loc).strip()] = float(r[i_score]) if i_score and r[i_score] is not None else 0.0
    return out


def jacc(a, b):
    return len(a & b) / len(a | b) if a | b else 0.0

# A) editors, same guide
editors = {}
for sheet, tag in [('Cas9_VEGFA_site_2', 'Cas9'), ('CBE_VEGFA_site_2', 'CBE'),
                   ('PE2_VEGFA_site_2', 'PE2'), ('PE3_VEGFA_site_2', 'PE3')]:
    editors[tag] = load('data/raw/trackingseq/moesm5.xlsx', sheet)
ed = {t: set(d) for t, d in editors.items()}
ed_jacc = {}
keys = sorted(ed)
for i, a in enumerate(keys):
    for b in keys[i+1:]:
        ed_jacc[f'{a}_vs_{b}'] = round(jacc(ed[a], ed[b]), 3)

# B) contexts, same guide
ctx = {}
for sheet, tag in [('Cas9_Pcsk9-gP_NIH3T3', 'NIH3T3'), ('Cas9_Pcsk9-gP_AML12', 'AML12'),
                   ('Cas9_Pcsk9-gP_mHSPC', 'mHSPC'), ('Cas9_Pcsk9-gP_mESC', 'mESC'),
                   ('Cas9_Pcsk9-gP_Liver', 'Liver')]:
    ctx[tag] = load('data/raw/trackingseq/moesm6.xlsx', sheet)
cs = {t: set(d) for t, d in ctx.items()}
ck = sorted(cs)
ctx_jacc = {}
for i, a in enumerate(ck):
    for b in ck[i+1:]:
        ctx_jacc[f'{a}_vs_{b}'] = round(jacc(cs[a], cs[b]), 3)

# replication vs score in the largest context (NIH3T3) vs union of others
others = set().union(*(cs[t] for t in ck if t != 'NIH3T3'))
nih = ctx['NIH3T3']
rep = [(nih[k], k in others) for k in nih]
rep_sorted = sorted(rep, key=lambda x: -x[0])
n_top = max(1, len(rep_sorted) // 4)
top_rep = sum(r for _, r in rep_sorted[:n_top]) / n_top
rest_rep = sum(r for _, r in rep_sorted[n_top:]) / max(1, len(rep_sorted) - n_top)
rho = float(spearmanr([s for s, _ in rep], [int(r) for _, r in rep])[0])
in_rep = [s for s, r in rep if r]
out_rep = [s for s, r in rep if not r]
mwu = float(mannwhitneyu(in_rep, out_rep)[1]) if in_rep and out_rep else None

out = dict(
    vegfa_site2_editor_sets={t: len(ed[t]) for t in keys},
    vegfa_site2_editor_jaccard=ed_jacc,
    pcsk9_context_sets={t: len(cs[t]) for t in ck},
    pcsk9_context_jaccard=ctx_jacc,
    nih3t3_rep_vs_score=dict(n=len(rep), spearman_score_replicated=round(rho, 3),
                             top_quartile_rep_rate=round(top_rep, 3),
                             rest_rep_rate=round(rest_rep, 3), mwu_p=mwu))
json.dump(out, open('results/trackingseq_heterogeneity.json', 'w'), indent=1)
print(json.dumps(out, indent=1))
