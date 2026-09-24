"""Allelic-readout tiling (Nat Commun 2026, 10.1038/s41467-026-69918-8,
MOESM7 Fig 2a): nucleotide-resolution base-editing scores for ABE8e and
EvoCDA across a tiled locus (CRISPRSURF deconvolution, negative-selection and
raw replicate blocks). Do ABE and CDA editing landscapes coincide?"""
import json
import numpy as np
import openpyxl
from scipy.stats import spearmanr

wb = openpyxl.load_workbook('data/raw/allelic_tiling/moesm7.xlsx', read_only=True)

def parse_blocks(sn):
    ws = wb[sn]
    rows = list(ws.iter_rows(values_only=True))
    hdr = rows[0]
    blocks = {}
    for c0, v in enumerate(hdr):
        if v is None or not str(v).strip():
            continue
        name = str(v).strip()
        series = {}
        for r in rows[1:]:
            if c0 + 3 < len(r) and isinstance(r[c0 + 1], (int, float)) and isinstance(r[c0 + 3], (int, float)):
                series[int(r[c0 + 1])] = float(r[c0 + 3])
        if series:
            blocks[name] = series
    return blocks

out = {}
editors = {}
for sn, tag in [('Fig2a ABE8eCRISPRSURF', 'ABE8e'), ('Fig2a EvoCDACRISPRSURF', 'EvoCDA')]:
    blocks = parse_blocks(sn)
    editors[tag] = blocks
    out[tag] = dict(blocks=list(blocks), n_positions={k: len(v) for k, v in blocks.items()})

# replicate/selection consistency within editor: correlate blocks on shared positions
for tag, blocks in editors.items():
    names = list(blocks)
    if len(names) >= 2:
        a, b = blocks[names[0]], blocks[names[1]]
        shared = sorted(set(a) & set(b))
        if len(shared) > 10:
            rho = spearmanr([a[p] for p in shared], [b[p] for p in shared]).statistic
            out[tag]['block_pair_spearman'] = dict(pair=names[:2], n_shared=len(shared),
                                                   rho=round(float(rho), 3))

# raw replicate consistency within ABE8e (Rep1/2/3)
reps = [editors['ABE8e']['Rep%d Raw Score' % i] for i in (1, 2, 3)]
shared = sorted(set(reps[0]) & set(reps[1]) & set(reps[2]))
out['ABE8e']['raw_replicate_spearman'] = dict(
    n_shared=len(shared),
    r1_r2=round(float(spearmanr([reps[0][p] for p in shared], [reps[1][p] for p in shared]).statistic), 3),
    r1_r3=round(float(spearmanr([reps[0][p] for p in shared], [reps[2][p] for p in shared]).statistic), 3))

# cross-editor landscape comparison on deconvolved scores
a = editors['ABE8e']['Deconvolved Scores']
b = editors['EvoCDA']['Deconvolved Score']
shared = sorted(set(a) & set(b))
out['cross_editor_deconvolved'] = dict(n_shared_positions=len(shared))
if len(shared) > 10:
    rho = spearmanr([a[p] for p in shared], [b[p] for p in shared]).statistic
    out['cross_editor_deconvolved']['spearman'] = round(float(rho), 3)
    thr_a = np.percentile([abs(a[p]) for p in shared], 75)
    thr_b = np.percentile([abs(b[p]) for p in shared], 75)
    hi_a = {p for p in shared if abs(a[p]) > thr_a}
    hi_b = {p for p in shared if abs(b[p]) > thr_b}
    out['cross_editor_deconvolved']['top_quartile_jaccard'] = round(len(hi_a & hi_b) / len(hi_a | hi_b), 3)

json.dump(out, open('results/allelic_tiling_editors.json', 'w'), indent=1)
print(json.dumps(out, indent=1))
