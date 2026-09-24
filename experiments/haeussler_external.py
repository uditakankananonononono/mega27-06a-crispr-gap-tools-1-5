"""Gap 2 external validation: Haeussler 2016 CRISPOR benchmark set.

Train the positional logistic model on crisprSQL (seed-0 train split, same
protocol as the poslog experiment), then evaluate on the Haeussler/CRISPOR
benchmark (26,052 guide--off-target pairs, wasValidated labels, bundled
CFD scores) EXCLUDING any (guide, off-target) pair present in crisprSQL.
Compares poslog vs bundled CFD vs our MIT reimplementation on identical
external pairs.
"""
import json

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score

from crisprgap.baselines import mit_score
from crisprgap.data.datasets import load_crisprsql
from experiments.poslog_offtarget import pair_features, split_for_seed


def main():
    h = pd.read_csv("data/haeussler/fig2-crisporData_withReadFraction.tab", sep="\t")
    pairs = h.otSeq.str.split(",", expand=True)
    h["guide"], h["off"] = pairs[0].str.upper(), pairs[1].str.upper()
    assert (h.guide.str.len() == 23).all() and (h["off"].str.len() == 23).all()

    ds = load_crisprsql()
    # crisprSQL stores 20-mers; Haeussler stores 23-mers (20+PAM) - compare on 20-mer core
    sql_pairs = set(zip([g[:20] for g in ds.guides], [o[:20] for o in ds.offtargets]))
    h["g20"], h["o20"] = h.guide.str[:20], h["off"].str[:20]
    in_sql = np.array([(g, o) in sql_pairs for g, o in zip(h.g20, h.o20)])
    ext = h[~in_sql].copy()
    print(f"Haeussler pairs: {len(h)}, overlapping crisprSQL: {in_sql.sum()}, external: {len(ext)}")

    tr, va, te = split_for_seed(ds, 0)
    X = pair_features([g[:20] for g in np.array(ds.guides)[tr]],
                      [o[:20] for o in np.array(ds.offtargets)[tr]])
    m = LogisticRegression(max_iter=1000, random_state=0).fit(X, ds.labels[tr])

    Xe = pair_features(ext.g20.tolist(), ext.o20.tolist())
    p_poslog = m.predict_proba(Xe)[:, 1]
    y = ext.wasValidated.to_numpy()
    mit = np.array([mit_score(g[:20], o[:20]) for g, o in zip(ext.guide, ext["off"])])
    res = {
        "dataset": "Haeussler 2016 CRISPOR benchmark via microsoft/Elevation repo",
        "n_total": int(len(h)), "n_overlap_crisprsql": int(in_sql.sum()),
        "n_external": int(len(ext)), "n_pos": int(y.sum()),
        "auroc_poslog_external": float(roc_auc_score(y, p_poslog)),
        "auroc_cfd_bundled_external": float(roc_auc_score(y, ext.cfdScore)),
        "auroc_mit_external": float(roc_auc_score(y, mit)),
        "note": "poslog trained on crisprSQL seed-0 train split; pairs present in crisprSQL excluded",
    }
    json.dump(res, open("results/haeussler_external.json", "w"), indent=2)
    print(json.dumps(res, indent=2))


if __name__ == "__main__":
    main()
