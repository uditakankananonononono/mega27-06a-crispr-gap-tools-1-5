"""Kim et al high-fidelity variant on-target screen (reprocessed deposit in
Nat Commun 2023 10.1038/s41467-023-41393-5 MOESM10): 6,481 guides x 6
variants (WT, Sniper, eSpCas9(1.1), SpCas9-HF1, HypaCas9, evoCas9).
30-mer = 4bp + 20nt guide (lowercase) + 6bp. Gap-1 x gap-4: within-variant
ridge CV, cross-variant transfer matrix, variant retention profile."""
import json
import numpy as np
import openpyxl
from scipy.stats import spearmanr
from sklearn.linear_model import Ridge
from sklearn.model_selection import KFold

wb = openpyxl.load_workbook('data/raw/hifi_rule/moesm10.xlsx', read_only=True)
ws = wb['Kim et al dataset (ON)']
rows = list(ws.iter_rows(min_row=2, values_only=True))
VARS = ['WT', 'Sniper', 'eSpCas9', 'HF1', 'Hypa', 'evo']
seqs, Y = [], {v: [] for v in VARS}
for r in rows:
    s = str(r[0]).strip() if r[0] else None
    if not s or len(s) != 30:
        continue
    g = s[4:24].upper()
    if len(g) != 20 or set(g) - set('ACGT'):
        continue
    vals = []
    for i in range(3, 9):
        try:
            vals.append(float(r[i]))
        except (TypeError, ValueError):
            vals.append(np.nan)
    seqs.append(g)
    for v, x in zip(VARS, vals):
        Y[v].append(x)

n = len(seqs)
print('n =', n)

def feats(seq):
    v = np.zeros(20*4 + 19*16)
    for i, ch in enumerate(seq):
        v[i*4 + 'ACGT'.index(ch)] = 1
    for i in range(19):
        v[80 + i*16 + 'ACGT'.index(seq[i])*4 + 'ACGT'.index(seq[i+1])] = 1
    return v

X = np.stack([feats(s) for s in seqs])
Ym = {v: np.array(Y[v]) for v in VARS}

kf = KFold(5, shuffle=True, random_state=0)
within = {}
for v in VARS:
    y = Ym[v]
    ok = np.isfinite(y)
    sp = []
    for tr, te in kf.split(X[ok]):
        m = Ridge(alpha=1.0).fit(X[ok][tr], y[ok][tr])
        sp.append(float(spearmanr(m.predict(X[ok][te]), y[ok][te])[0]))
    within[v] = round(float(np.mean(sp)), 3)

r_wt = Ridge(alpha=1.0).fit(X, Ym['WT'])
p_wt = r_wt.predict(X)
transfer = {f'WT_to_{v}': round(float(spearmanr(p_wt, Ym[v])[0]), 3) for v in VARS if v != 'WT'}
retention = {v: round(float(np.mean(Ym[v]) / np.mean(Ym['WT'])), 3) for v in VARS if v != 'WT'}
pair = {f'WT_{v}': round(float(spearmanr(Ym['WT'], Ym[v])[0]), 3) for v in VARS if v != 'WT'}

out = dict(n=n, variants=VARS, within_cv=within, wt_transfer=transfer,
           retention_vs_wt=retention, pairwise_wt_spearman=pair)
json.dump(out, open('results/kim_hf_variant_on.json', 'w'), indent=1)
print(json.dumps(out, indent=1))
