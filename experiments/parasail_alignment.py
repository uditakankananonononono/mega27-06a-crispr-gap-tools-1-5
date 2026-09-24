"""Gap-2 classical baseline: parasail Smith-Waterman local alignment score
between guide and off-target as the predictor, same crisprSQL corpus and
guide-grouped seed-0 test split as the GNN/poslog baselines."""
import json

import numpy as np
import parasail
from sklearn.metrics import average_precision_score, roc_auc_score

from crisprgap.data.datasets import load_crisprsql
from experiments.poslog_offtarget import split_for_seed


def sw_score(a, b):
    return parasail.sw_trace_striped_32(a.upper(), b.upper(), 2, 1,
                                        parasail.dnafull).score


def main():
    ds = load_crisprsql()
    n = len(ds.labels)
    scores = np.empty(n, dtype=np.float32)
    for i in range(n):
        scores[i] = sw_score(ds.guides[i], ds.offtargets[i])
    out = {"tool": "parasail Smith-Waterman (dnafull, match 2, gap 1)",
           "global": {"auroc": float(roc_auc_score(ds.labels, scores)),
                      "auprc": float(average_precision_score(ds.labels, scores)),
                      "n": n}}
    out["seeds"] = {}
    for seed in range(3):
        _, _, te = split_for_seed(ds, seed)
        out["seeds"][seed] = {
            "auroc": float(roc_auc_score(ds.labels[te], scores[te])),
            "auprc": float(average_precision_score(ds.labels[te], scores[te])),
            "n_test": int(len(te))}
    json.dump(out, open("results/parasail_alignment.json", "w"), indent=2)
    print(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
