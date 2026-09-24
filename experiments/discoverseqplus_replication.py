"""DISCOVER-seq+ (Lazzarotto/Malinin 2023, Nat Methods 10.1038/s41592-023-01840-z).
Off-target site tables per condition for the same guides: VEGFA site 2 (Vs2) in
K562 and iPSC, each without (nD) and with (KU) DNA-PKcs inhibitor. Question:
how reproducible is the in-vivo/cellular off-target set across arms and across
cell types, and does reproducibility track mismatch count / enrichment?"""
import json
import numpy as np
import openpyxl


def load(path, sheet):
    ws = openpyxl.load_workbook(path, read_only=True)[sheet]
    rows = list(ws.iter_rows(min_row=2, values_only=True))
    out = {}
    for r in rows:
        if not r[0]:
            continue
        key = (str(r[0]).split(':')[0], int(str(r[1])), str(r[2]).strip())
        out[key] = dict(mm=int(r[6]), ctotal=float(r[7]) if r[7] is not None else 0.0)
    return out


def jacc(a, b):
    return len(a & b) / len(a | b) if a | b else 0.0

k_nd = load('data/raw/discoverseqplus/moesm6.xlsx', 'K562_Vs2_nD')
k_ku = load('data/raw/discoverseqplus/moesm6.xlsx', 'K562_Vs2_KU')
i_nd = load('data/raw/discoverseqplus/moesm11.xlsx', 'iPSC_Vs2_nD')
i_ku = load('data/raw/discoverseqplus/moesm11.xlsx', 'iPSC_Vs2_KU')
m_nd = load('data/raw/discoverseqplus/moesm11.xlsx', 'mm10_mP9_nD')
m_ku = load('data/raw/discoverseqplus/moesm11.xlsx', 'mm10_mP9_KU')

K, I = set(k_nd) | set(k_ku), set(i_nd) | set(i_ku)
M = set(m_nd) | set(m_ku)
out = dict(
    n_sites=dict(k562_nd=len(k_nd), k562_ku=len(k_ku), ipsc_nd=len(i_nd),
                 ipsc_ku=len(i_ku), mouse_nd=len(m_nd), mouse_ku=len(m_ku)),
    jaccard_k562_nd_ku=round(jacc(set(k_nd), set(k_ku)), 3),
    jaccard_ipsc_nd_ku=round(jacc(set(i_nd), set(i_ku)), 3),
    jaccard_k562_ipsc=round(jacc(K, I), 3),
)

# replication vs mismatch count and enrichment, within K562 (KU arm as discovery, nD as replicate)
both = set(k_nd) & set(k_ku)
by_mm = {}
for key, v in k_ku.items():
    by_mm.setdefault(v['mm'], []).append(key in both)
out['k562_replication_by_mm'] = {str(k): dict(n=len(v), rep_rate=round(sum(v)/len(v), 3))
                                 for k, v in sorted(by_mm.items())}
# note: nD and KU arms report identical site lists (Jaccard 1.0 within cell type);
# the informative contrast is across cell types.

# cross-cell-type: K562 union vs iPSC union for same guide
rep_ct = [(k_ku.get(k, k_nd.get(k, {})).get('mm', None), k in I) for k in K]
rep_ct = [(m, r) for m, r in rep_ct if m is not None]
by_mm2 = {}
for m, r in rep_ct:
    by_mm2.setdefault(m, []).append(r)
out['k562_sites_replicated_in_ipsc_by_mm'] = {str(k): dict(n=len(v), rep_rate=round(sum(v)/len(v), 3))
                                              for k, v in sorted(by_mm2.items())}
json.dump(out, open('results/discoverseqplus_replication.json', 'w'), indent=1)
print(json.dumps(out, indent=1))
