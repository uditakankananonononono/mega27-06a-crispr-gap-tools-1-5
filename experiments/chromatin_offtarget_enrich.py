"""Chromatin context of off-targets (Cell Res 2026, 10.1038/s41421-026-00889-2,
MOESM2): Table S5 = per off-target site, average epigenetic feature signal at
non-off-target vs off-target sites across 14 features and 6 editors (Cas9,
RYCas9, TadA-8e-nCas9/nRYCas9 ABEs, rAPOBEC1-nCas9/nRYCas9 CBEs).
Table S1 = per-guide genome-wide candidate-site burden by mismatch class."""
import json
import numpy as np
import openpyxl
from collections import defaultdict

wb = openpyxl.load_workbook('data/raw/chromatin_offtarget/moesm2.xlsx', read_only=True)

# S5: feature enrichment at off-targets
pairs = defaultdict(list)  # (editor, feature) -> list of (un, off)
for r in wb['Supplementary Table S5'].iter_rows(min_row=3, values_only=True):
    if r[0] is None or r[6] is None:
        continue
    try:
        un, off = float(r[2]), float(r[3])
    except (TypeError, ValueError):
        continue
    pairs[(str(r[6]).strip(), str(r[1]).strip())].append((un, off))

feat_rank = defaultdict(list)
per_editor = defaultdict(dict)
for (enz, feat), vals in pairs.items():
    mu = np.mean([v[0] for v in vals])
    mo = np.mean([v[1] for v in vals])
    if mu > 0:
        ratio = mo / mu
        per_editor[enz][feat] = dict(n=len(vals), mean_off=round(mo, 4),
                                     mean_nonoff=round(mu, 4), ratio=round(ratio, 3))
        feat_rank[feat].append((enz, ratio))

out = dict(n_s5_rows=sum(len(v) for v in pairs.values()),
           editors=sorted({k[0] for k in pairs}),
           n_features=len({k[1] for k in pairs}))
# features ranked by mean ratio across editors
feat_summary = {}
for feat, lst in feat_rank.items():
    feat_summary[feat] = dict(mean_ratio=round(float(np.mean([x[1] for x in lst])), 3),
                              max_ratio=round(max(x[1] for x in lst), 3),
                              min_ratio=round(min(x[1] for x in lst), 3))
out['feature_enrichment'] = dict(sorted(feat_summary.items(), key=lambda kv: -kv[1]['mean_ratio']))
# per-editor top feature
out['per_editor_top_feature'] = {enz: max(feats.items(), key=lambda kv: kv[1]['ratio'])[0]
                                 for enz, feats in per_editor.items()}
out['per_editor_h3k9me3_ratio'] = {enz: feats.get('H3K9me3', {}).get('ratio')
                                   for enz, feats in per_editor.items()}
out['per_editor_atac_ratio'] = {enz: feats.get('ATAC-seq', {}).get('ratio')
                                for enz, feats in per_editor.items()}

# S1: candidate burden per guide
burdens = []
m_class = defaultdict(list)
for r in wb['Supplementary Table S1'].iter_rows(min_row=3, values_only=True):
    if r[0] is None:
        continue
    try:
        m = [int(r[i]) for i in range(5, 10)]
    except (TypeError, ValueError):
        continue
    tot = sum(m)
    burdens.append(tot)
    for k, v in zip(['m1', 'm2', 'm3', 'm4', 'm5plus'], m):
        m_class[k].append(v)
burdens = np.array(burdens, dtype=float)
out['candidate_burden'] = dict(
    n_guides=len(burdens),
    median_sites_le4mm=float(np.median(burdens)),
    p90=float(np.percentile(burdens, 90)),
    max=float(burdens.max()),
    per_class_median={k: float(np.median(v)) for k, v in m_class.items()})
json.dump(out, open('results/chromatin_offtarget_enrich.json', 'w'), indent=1)
print(json.dumps(out, indent=1)[:2600])
