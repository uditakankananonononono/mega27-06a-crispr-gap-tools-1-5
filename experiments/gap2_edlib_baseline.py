"""Gap 2: edlib edit-distance baseline on the crisprSQL corpus.
Plain Levenshtein distance (guide vs off-target) as a scoring function,
grouped-by-guide 3-seed AUROC, compared against the mismatch-count
baseline (identical by construction when no indels) and CFD where
scorable. The point: edlib handles INDELS, which the mismatch-only
baselines ignore - how many crisprSQL pairs actually contain indels?
"""
import json

import edlib
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import GroupShuffleSplit


def main():
    df = pd.read_csv("data/crisprsql/100720.csv", low_memory=False)
    df = df.dropna(subset=["grna_target_sequence", "target_sequence"]).reset_index(drop=True)
    dists, indel_rows = [], 0
    for r in df.itertuples():
        g, o = str(r.grna_target_sequence).upper(), str(r.target_sequence).upper()
        d = edlib.align(g, o)["editDistance"]
        dists.append(d)
        if len(g) != len(o):
            indel_rows += 1
    df["edlib_dist"] = dists
    y = (df.cleavage_freq > 0).astype(int).to_numpy()
    groups = df.grna_target_sequence.to_numpy()
    aurocs = []
    for seed in (0, 1, 2):
        tr, te = next(GroupShuffleSplit(1, test_size=0.25, random_state=seed).split(
            df[["edlib_dist"]], y, groups))
        aurocs.append(float(roc_auc_score(y[te], -df.edlib_dist.to_numpy()[te])))
    out = {
        "n_pairs": int(len(df)),
        "n_indel_pairs": int(indel_rows),
        "indel_frac": indel_rows / len(df),
        "auroc_grouped_3seeds": aurocs,
        "auroc_mean": float(np.mean(aurocs)),
    }
    json.dump(out, open("results/gap2_edlib_baseline.json", "w"), indent=2)
    print(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()
