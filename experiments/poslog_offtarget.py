"""Position-specific logistic baseline for off-target activity (gap 2).

The GNN benchmark (0.609 mean AUROC) has only CFD (0.552) and MIT (0.516)
as classical comparators. A position-specific logistic model (one-hot
guide x offtarget base pairing at each position) is the canonical
strong-but-simple baseline family. Same guide-grouped splits and seeds as
the seeded GNN eval, so the comparison is apples-to-apples.
"""
import json

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score

from crisprgap.calibration import expected_calibration_error
from crisprgap.data.datasets import load_crisprsql

B = {"A": 0, "C": 1, "G": 2, "T": 3}


def pair_features(guides, offs):
    n, L = len(guides), len(guides[0])
    X = np.zeros((n, L * 16 + 1), dtype=np.float32)
    for i, (g, o) in enumerate(zip(guides, offs)):
        mm = 0
        for p, (a, b) in enumerate(zip(g, o)):
            ia, ib = B.get(a.upper(), 0), B.get(b.upper(), 0)
            X[i, p * 16 + ia * 4 + ib] = 1.0
            mm += a.upper() != b.upper()
        X[i, -1] = mm
    return X


def split_for_seed(ds, seed, max_pairs=8000):
    rng = np.random.default_rng(seed)
    pos_idx = np.where(ds.labels == 1)[0]
    neg_idx = np.where(ds.labels == 0)[0]
    n_pos = min(len(pos_idx), max_pairs - len(neg_idx))
    keep = np.concatenate([rng.choice(pos_idx, n_pos, replace=False), neg_idx])
    rng.shuffle(keep)
    guides = np.array([f"{g}|{s}" for g, s in zip(np.array(ds.guides)[keep], np.array(ds.studies)[keep])])
    uniq = np.unique(guides)
    rng.shuffle(uniq)
    test_guides = set(uniq[: max(1, len(uniq) // 5)])
    is_test = np.array([g in test_guides for g in guides])
    tr, te = keep[~is_test], keep[is_test]
    val = tr[: len(tr) // 10]
    tr = tr[len(tr) // 10:]
    return tr, val, te


def main():
    ds = load_crisprsql()
    res = {"tool": "scikit-learn LogisticRegression (positional duplex one-hot)",
           "seeds": {}}
    for seed in range(5):
        tr, val, te = split_for_seed(ds, seed)
        Xtr = pair_features([ds.guides[i] for i in tr], [ds.offtargets[i] for i in tr])
        Xte = pair_features([ds.guides[i] for i in te], [ds.offtargets[i] for i in te])
        clf = LogisticRegression(max_iter=1000, random_state=seed)
        clf.fit(Xtr, ds.labels[tr])
        p = clf.predict_proba(Xte)[:, 1]
        res["seeds"][str(seed)] = {
            "auroc": float(roc_auc_score(ds.labels[te], p)),
            "ece": float(expected_calibration_error(ds.labels[te], p)),
            "n_test": int(len(te))}
    aucs = [v["auroc"] for v in res["seeds"].values()]
    res["auroc_mean"] = float(np.mean(aucs))
    res["auroc_sd"] = float(np.std(aucs))
    res["reference"] = {"gnn_mean": 0.609, "gnn_sd": 0.084, "cfd": 0.552, "mit": 0.516}
    json.dump(res, open("results/offtarget_poslog.json", "w"), indent=2)
    print(json.dumps({k: res[k] for k in ("auroc_mean", "auroc_sd")}, indent=2))
    print({s: round(v["auroc"], 3) for s, v in res["seeds"].items()})


if __name__ == "__main__":
    main()
