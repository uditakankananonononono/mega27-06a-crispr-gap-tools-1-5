"""CRISPRon TRAP12K (Xiang 2021, Nat Commun 10.1038/s41467-021-23576-0, MOESM4):
~11.5k gRNAs assayed for indel efficiency in 5 conditions (Day2, Day8 +/- dox,
Day10 +/- dox). Gap-1 question: how stable is the efficacy ranking across time
and induction? Cross-condition Spearman, top-decile overlap, and a ridge
trained in one condition tested in another (cross-condition generalization)."""
import json
import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.linear_model import Ridge
from sklearn.model_selection import KFold

sheets = {'D2': 'SpCas9_eff_Day 2', 'D8minus': 'spCas9_eff_D8-dox',
          'D8plus': 'spCas9_eff_D8+dox', 'D10minus': 'spCas9_eff_D10-dox',
          'D10plus': 'spCas9_eff_D10+dox'}
dfs = {}
for name, sheet in sheets.items():
    df = pd.read_excel('data/raw/crispron/moesm4.xlsx', sheet_name=sheet,
                       usecols=['Surrogate ID', 'gRNA', 'total_indel_eff'])
    df = df.dropna(subset=['gRNA', 'total_indel_eff'])
    dfs[name] = df.set_index('Surrogate ID')[['gRNA', 'total_indel_eff']]
    print(name, len(df))

base = dfs['D2'].rename(columns={'gRNA': 'gRNA', 'total_indel_eff': 'D2'})
for name in list(sheets)[1:]:
    base = base.join(dfs[name][['total_indel_eff']].rename(columns={'total_indel_eff': name}), how='inner')
base = base.dropna()
print('joined:', len(base))

conds = list(sheets)
corr = {}
for i, a in enumerate(conds):
    for b in conds[i+1:]:
        corr[f'{a}_vs_{b}'] = round(float(spearmanr(base[a], base[b])[0]), 3)

def top_decile(col):
    thr = base[col].quantile(0.9)
    return set(base.index[base[col] >= thr])
td = {c: top_decile(c) for c in conds}
top_overlap = {f'{a}_vs_{b}': round(len(td[a] & td[b]) / len(td[a]), 3)
               for i, a in enumerate(conds) for b in conds[i+1:]}

# one-hot ridge: train on D2, test on D10minus; compare with within-D10 CV
def onehot(seq):
    seq = str(seq).upper()[:20]
    arr = np.zeros(80)
    for i, ch in enumerate(seq):
        if ch in 'ACGT':
            arr[i*4 + 'ACGT'.index(ch)] = 1
    return arr

X = np.stack([onehot(g) for g in base['gRNA']])
y2 = base['D2'].values
y10 = base['D10minus'].values
ridge = Ridge(alpha=1.0).fit(X, y2)
cross = float(spearmanr(ridge.predict(X), y10)[0])
kf = KFold(5, shuffle=True, random_state=0)
within = []
for tr, te in kf.split(X):
    r = Ridge(alpha=1.0).fit(X[tr], y10[tr])
    within.append(float(spearmanr(r.predict(X[te]), y10[te])[0]))
within_mean = round(float(np.mean(within)), 3)

out = dict(n_guides=len(base), conditions=conds,
           spearman_cross_condition=corr, top_decile_overlap=top_overlap,
           ridge_trainD2_testD10=round(cross, 3), ridge_within_D10_cv=within_mean,
           cross_condition_gap=round(within_mean - cross, 3))
json.dump(out, open('results/crispron_conditions.json', 'w'), indent=1)
print(json.dumps(out, indent=1))
