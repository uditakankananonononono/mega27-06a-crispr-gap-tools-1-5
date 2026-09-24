"""Gap 3 cross-library check: PRIDICT v1 Library_1 (Mathis 2023, 92,423
pegRNAs, HEK293T) vs PRIDICT2 23k. Same minimal shared feature set
(PBSlength, RTlength, Editing_Position, Correction_Length, one-hot
Correction_Type), ridge, 3 seeds, within and cross both directions."""
import json

import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.linear_model import Ridge

FEATS = ["PBSlength", "RTlength", "Editing_Position", "Correction_Length"]


def frame(df):
    ct = pd.get_dummies(df["Correction_Type"]).astype(np.float32)
    X = np.concatenate([df[FEATS].to_numpy(dtype=np.float32), ct.to_numpy()], axis=1)
    return X


def main():
    v1 = pd.read_csv("data/pridict/library1_v1.csv.gz")
    v1 = v1.dropna(subset=FEATS + ["Correction_Type", "averageedited"])
    v2 = pd.read_csv("data/pridict/data_23k_v1.csv")
    v2 = v2.dropna(subset=FEATS + ["Correction_Type", "HEKaverageedited_clamped"])
    # align one-hot columns
    cats = sorted(set(v1["Correction_Type"]) | set(v2["Correction_Type"]))
    for df in (v1, v2):
        df["Correction_Type"] = pd.Categorical(df["Correction_Type"], categories=cats)
    X1, X2 = frame(v1), frame(v2)
    y1 = (v1["averageedited"] / 100.0).to_numpy(dtype=np.float32)
    y2 = v2["HEKaverageedited_clamped"].to_numpy(dtype=np.float32)
    out = {"features": FEATS + [f"ctype_{c}" for c in cats],
           "n_v1": len(v1), "n_v2": len(v2)}
    rng = np.random.default_rng
    within, cross = {}, {}
    for seed in (0, 1, 2):
        r = rng(seed)
        i1 = r.permutation(len(y1)); i2 = r.permutation(len(y2))
        t1, e1 = i1[: int(.7 * len(i1))], i1[int(.7 * len(i1)): int(.85 * len(i1))]
        t2, e2 = i2[: int(.7 * len(i2))], i2[int(.7 * len(i2)): int(.85 * len(i2))]
        m1 = Ridge(alpha=1.0).fit(X1[t1], y1[t1])
        m2 = Ridge(alpha=1.0).fit(X2[t2], y2[t2])
        within.setdefault("v1", []).append(float(spearmanr(m1.predict(X1[e1]), y1[e1]).statistic))
        within.setdefault("v2", []).append(float(spearmanr(m2.predict(X2[e2]), y2[e2]).statistic))
        cross.setdefault("v1_to_v2", []).append(float(spearmanr(m1.predict(X2[e2]), y2[e2]).statistic))
        cross.setdefault("v2_to_v1", []).append(float(spearmanr(m2.predict(X1[e1]), y1[e1]).statistic))
    out["within"] = {k: {"mean": float(np.mean(v)), "seeds": v} for k, v in within.items()}
    out["cross"] = {k: {"mean": float(np.mean(v)), "seeds": v} for k, v in cross.items()}
    json.dump(out, open("results/pridict_v1_transfer.json", "w"), indent=2)
    print(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
