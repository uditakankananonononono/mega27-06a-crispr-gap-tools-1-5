"""DeepHF (Wang D 2019, Nat Commun 10.1038/s41467-019-12281-8, MOESM3):
~59.8k gRNAs with matched WT-SpCas9, eSpCas9(1.1), SpCas9-HF1 efficiencies.
Gap-1 x gap-4: cross-enzyme transfer of a position-specific mono+dinucleotide
ridge (same protocol as gap1_variant_transfer) plus pairwise rank agreement
and high-fidelity activity-retention statistics."""
import json
import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.linear_model import Ridge
from sklearn.model_selection import KFold

df = pd.read_excel('data/raw/deephf/moesm3.xlsx', sheet_name=0, skiprows=1,
                   usecols=['gRNA_Seq', 'Wt_Efficiency', 'eSpCas 9_Efficiency',
                            'SpCas9-HF1_Efficiency'])
df = df.dropna()
df.columns = ['seq', 'wt', 'esp', 'hf1']
df = df[df['seq'].str.len() == 20]
print('n =', len(df))

def feats(seq):
    seq = seq.upper()
    v = np.zeros(20*4 + 19*16)
    for i, ch in enumerate(seq):
        if ch in 'ACGT':
            v[i*4 + 'ACGT'.index(ch)] = 1
    for i in range(19):
        a, b = seq[i:i+2]
        if a in 'ACGT' and b in 'ACGT':
            v[80 + i*16 + 'ACGT'.index(a)*4 + 'ACGT'.index(b)] = 1
    return v

X = np.stack([feats(s) for s in df['seq']])
Y = {k: df[k].values for k in ('wt', 'esp', 'hf1')}

def cv_spearman(X, y, seed=0):
    kf = KFold(5, shuffle=True, random_state=seed)
    out = []
    for tr, te in kf.split(X):
        r = Ridge(alpha=1.0).fit(X[tr], y[tr])
        out.append(float(spearmanr(r.predict(X[te]), y[te])[0]))
    return float(np.mean(out))

within = {k: round(cv_spearman(X, Y[k]), 3) for k in Y}
transfer = {}
for src in Y:
    r = Ridge(alpha=1.0).fit(X, Y[src])
    p = r.predict(X)
    for dst in Y:
        if src != dst:
            transfer[f'{src}_to_{dst}'] = round(float(spearmanr(p, Y[dst])[0]), 3)

pair = {f'{a}_{b}': round(float(spearmanr(Y[a], Y[b])[0]), 3)
        for a, b in (('wt', 'esp'), ('wt', 'hf1'), ('esp', 'hf1'))}
ret = (df['hf1'] / (df['wt'] + 1e-6)).clip(0, 2)
out = dict(n_guides=len(df), within_cv=within, transfer=transfer,
           pairwise_spearman=pair,
           hf1_retention=dict(frac_gt80pct=round(float((ret >= 0.8).mean()), 3),
                              frac_lt20pct=round(float((ret < 0.2).mean()), 3),
                              median=round(float(ret.median()), 3)))
json.dump(out, open('results/deephf_transfer.json', 'w'), indent=1)
print(json.dumps(out, indent=1))
