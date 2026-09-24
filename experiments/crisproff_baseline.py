"""Gap-2 published-model baseline: CRISPRoff binding energy (Alkan 2018)
on crisprSQL - global and guide-grouped seed splits, same as parasail."""
import json

import numpy as np
from sklearn.metrics import average_precision_score, roc_auc_score

from crisprgap.crisproff import crisproff_score
from crisprgap.data.datasets import load_crisprsql
from experiments.poslog_offtarget import split_for_seed


def main():
    ds = load_crisprsql()
    ok = set("ACGT")
    keep = np.array([len(p) == 3 and len(q) == 3 and
                     set(g) <= ok and set(o) <= ok
                     for p, q, g, o in zip(ds.pam, ds.off_pam,
                                           ds.guides, ds.offtargets)])
    print(f"pairs with both PAMs: {keep.sum()} / {len(keep)}")
    idx = np.where(keep)[0]
    vals, ok_idx, n_fail = [], [], 0
    for i in idx:
        try:
            vals.append(crisproff_score(ds.guides[i] + ds.pam[i],
                                        ds.offtargets[i] + ds.off_pam[i]))
            ok_idx.append(i)
        except (KeyError, IndexError):
            n_fail += 1
    print(f"scorable: {len(ok_idx)}, out-of-model (loop>11): {n_fail}")
    idx = np.array(ok_idx)
    labels = ds.labels[idx]
    scores = np.array(vals, dtype=np.float64)
    out = {"tool": "CRISPRoff (Alkan 2018; vendored RTH-tools, pos+PAM+DNA-opening)",
           "global": {"auroc": float(roc_auc_score(labels, scores)),
                      "auprc": float(average_precision_score(labels, scores)),
                      "n": int(len(idx))}, "seeds": {}}
    for seed in range(3):
        _, _, te = split_for_seed(ds, seed)
        # map crisprSQL row indices -> position in the filtered score array
        pos = {row: k for k, row in enumerate(idx)}
        te2 = [pos[i] for i in te if i in pos]
        out["seeds"][seed] = {
            "auroc": float(roc_auc_score(labels[te2], scores[te2])),
            "auprc": float(average_precision_score(labels[te2], scores[te2])),
            "n_test": int(len(te2))}
    json.dump(out, open("results/crisproff_baseline.json", "w"), indent=2)
    print(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
