"""Gap 2 mechanism probe: does class balancing fix the external inversion?

The crisprSQL-trained poslog inverted on the Haeussler external set
(AUROC 0.461). Hypothesis: training at 94% prevalence produces a model
whose ranking is tuned to the wrong operating point. Test: retrain with
(a) class_weight='balanced' and (b) RandomUnderSampler (imbalanced-learn),
evaluate on the identical external pairs. If the inversion persists under
both, prevalence handling is not the mechanism.
"""
import json

import numpy as np
import pandas as pd
from imblearn.under_sampling import RandomUnderSampler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score

from crisprgap.data.datasets import load_crisprsql
from experiments.poslog_offtarget import pair_features, split_for_seed


def main():
    ds = load_crisprsql()
    tr, va, te = split_for_seed(ds, 0)
    g = np.array(ds.guides); o = np.array(ds.offtargets)
    Xtr = pair_features([x[:20] for x in g[tr]], [x[:20] for x in o[tr]])
    ytr = ds.labels[tr]

    h = pd.read_csv("data/haeussler/fig2-crisporData_withReadFraction.tab", sep="\t")
    pairs = h.otSeq.str.split(",", expand=True)
    h["g20"], h["o20"] = pairs[0].str.upper().str[:20], pairs[1].str.upper().str[:20]
    sql_pairs = set(zip([x[:20] for x in ds.guides], [x[:20] for x in ds.offtargets]))
    mask = np.array([(a, b) in sql_pairs for a, b in zip(h.g20, h.o20)])
    ext = h[~mask]
    Xe = pair_features(ext.g20.tolist(), ext.o20.tolist())
    ye = ext.wasValidated.to_numpy()

    out = {"n_external": int(len(ext)), "n_pos_external": int(ye.sum())}
    m1 = LogisticRegression(max_iter=1000, random_state=0).fit(Xtr, ytr)
    out["auroc_unweighted"] = float(roc_auc_score(ye, m1.predict_proba(Xe)[:, 1]))
    m2 = LogisticRegression(max_iter=1000, random_state=0, class_weight="balanced").fit(Xtr, ytr)
    out["auroc_class_weight_balanced"] = float(roc_auc_score(ye, m2.predict_proba(Xe)[:, 1]))
    rus = RandomUnderSampler(random_state=0)
    Xr, yr = rus.fit_resample(Xtr, ytr)
    m3 = LogisticRegression(max_iter=1000, random_state=0).fit(Xr, yr)
    out["auroc_undersampled"] = float(roc_auc_score(ye, m3.predict_proba(Xe)[:, 1]))
    out["reference"] = {"cfd_bundled": 0.9852, "mit": 0.9981}
    out["note"] = ("imbalanced-learn 0.13 RandomUnderSampler; identical external pairs; "
                   "8 external positives - still anecdote-grade, mechanism probe only")
    json.dump(out, open("results/prevalence_rebalance.json", "w"), indent=2)
    print(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()
