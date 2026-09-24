"""Gap-3 corpora:
A) PRIDICT2.0 (Mathis 2023, Nat Biotechnol 10.1038/s41587-022-01613-7, MOESM5):
   Library_1 92,423 pegRNAs; ridge CV on 19 design-only features (per-replicate
   outcome columns excluded after they produced CV 1.0 leakage).
B) PE across chromatin contexts (2024, Nat Biotechnol 10.1038/s41587-024-02268-2,
   MOESM3 sheet 2): 16,055 pegRNAs x HEK293T/K562/K562-MLH1dn/AdV.
Expects /tmp/pridict2_l1.parquet and /tmp/pe_chromatin.parquet (see
scripts/convert_pridict2.py)."""
import json
import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.linear_model import Ridge
from sklearn.model_selection import KFold

out = {}
l1 = pd.read_parquet('/tmp/pridict2_l1.parquet')
design = ['PBSlength', 'RToverhanglength', 'RTlength', 'Correction_Length',
          'Editing_Position', 'deepcas9', 'MFE_protospacer', 'MFE_protospacer_scaffold',
          'MFE_extension', 'MFE_extension_scaffold', 'MFE_protospacer_extension_scaffold',
          'MFE_rt', 'MFE_pbs', 'Proto_GC_content', 'Extension_GC_content',
          'RT_GC_content', 'PBS_GC_content', 'original_GC_content', 'RTmt']
y = pd.to_numeric(l1['averageedited'], errors='coerce')
ok = y.notna()
X = l1.loc[ok, design].apply(pd.to_numeric, errors='coerce').fillna(0).values
yv = y[ok].values
kf = KFold(5, shuffle=True, random_state=0)
cvs = []
for tr, te in kf.split(X):
    r = Ridge(alpha=1.0).fit(X[tr], yv[tr])
    cvs.append(float(spearmanr(r.predict(X[te]), yv[te])[0]))
out['pridict2'] = dict(n_lib1=int(ok.sum()), n_design_features=len(design),
                       ridge_cv_spearman=round(float(np.mean(cvs)), 3),
                       median_edited=round(float(np.median(yv)), 3))

pc = pd.read_parquet('/tmp/pe_chromatin.parquet')
ctx = ['HEKaverageedited', 'K562averageedited', 'K562MLH1dnaverageedited', 'AdVaverageedited']
for c in ctx:
    pc[c] = pd.to_numeric(pc[c], errors='coerce')
pc = pc.dropna(subset=ctx)
mat = {}
for i, a in enumerate(ctx):
    for b in ctx[i+1:]:
        mat[f'{a[:12]}_vs_{b[:12]}'] = round(float(spearmanr(pc[a], pc[b])[0]), 3)
mmr = pc['K562MLH1dnaverageedited'] / (pc['K562averageedited'] + 0.1)
out['pe_chromatin'] = dict(n=len(pc), context_spearman=mat,
                           mlh1dn_median_fold=round(float(mmr.median()), 3))
json.dump(out, open('results/pridict2_pechromatin.json', 'w'), indent=1)
print(json.dumps(out, indent=1))
