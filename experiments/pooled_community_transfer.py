"""Gap 1: does pooling within a transfer community close the gap?
Ridge trained on pooled community data vs single datasets, tested on
held-out deephf_wt (HEK community) and dc_hela (Wang/Xu community).
3 seeds. Quantifies the community barrier as a training-data effect."""
import json

import numpy as np
from scipy.stats import spearmanr
from sklearn.linear_model import Ridge

from crisprgap.crossdata import featurize
from crisprgap.data.datasets import (load_crispron, load_deepcrispr,
                                     load_deepspcas9, load_deephf,
                                     load_doench_fcres, load_doench_v1,
                                     load_sgdesigner)

HEK = {"fcres": load_doench_fcres, "v1": load_doench_v1,
       "deepspcas9": load_deepspcas9, "sgdesigner": load_sgdesigner}
WX = {"dc_hct116": lambda: load_deepcrispr("hct116"),
      "dc_hl60": lambda: load_deepcrispr("hl60"),
      "crispron": load_crispron}
TARGETS = {"deephf_wt": lambda: load_deephf("wt"),
           "dc_hela": lambda: load_deepcrispr("hela")}


def pooled_Xy(loaders, rng, cap=8000):
    Xs, ys = [], []
    for name, fn in loaders.items():
        ds = fn()
        n = min(cap, len(ds.sequences))
        idx = rng.choice(len(ds.sequences), n, replace=False)
        seqs = [ds.sequences[i] for i in idx]
        Xs.append(featurize(seqs))
        ys.append(ds.scores[idx])
    return np.concatenate(Xs), np.concatenate(ys)


def main():
    out = {}
    for seed in (0, 1, 2):
        rng = np.random.default_rng(seed)
        res = {}
        for tname, tfn in TARGETS.items():
            tds = tfn()
            ti = rng.choice(len(tds.sequences), min(4000, len(tds.sequences)), replace=False)
            Xt = featurize([tds.sequences[i] for i in ti])
            yt = tds.scores[ti]
            row = {}
            for cname, loaders in (("HEK_pool", HEK), ("WX_pool", WX)):
                Xp, yp = pooled_Xy(loaders, rng)
                m = Ridge(alpha=1.0).fit(Xp, yp)
                row[cname] = float(spearmanr(m.predict(Xt), yt).statistic)
            res[tname] = row
        out[seed] = res
    json.dump(out, open("results/pooled_community_transfer.json", "w"), indent=2)
    for seed, res in out.items():
        for t, row in res.items():
            print(f"seed {seed} -> {t}: HEK_pool {row['HEK_pool']:.3f} | WX_pool {row['WX_pool']:.3f}")


if __name__ == "__main__":
    main()
