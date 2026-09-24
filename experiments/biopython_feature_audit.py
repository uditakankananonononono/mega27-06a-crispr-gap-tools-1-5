"""Tool audit: do biopython biophysical features (GC, nearest-neighbor
RNA-DNA Tm, length) change ridge within/cross performance on gap-1?
3 seeds, fcres and dc_hela, within + cross both ways."""
import json

import numpy as np
from Bio.SeqUtils import MeltingTemp, gc_fraction
from scipy.stats import spearmanr
from sklearn.linear_model import Ridge

from crisprgap.crossdata import featurize
from crisprgap.data.datasets import load_deepcrispr, load_doench_fcres


def bio_feats(seqs):
    out = np.zeros((len(seqs), 3), dtype=np.float32)
    for i, s in enumerate(seqs):
        out[i, 0] = gc_fraction(s)
        out[i, 1] = MeltingTemp.Tm_NN(s[:23], nn_table=MeltingTemp.R_DNA_NN1)
        out[i, 2] = len(s)
    return out


def run(seed):
    rng = np.random.default_rng(seed)
    res = {}
    data = {}
    for name, fn in (("fcres", load_doench_fcres),
                     ("dc_hela", lambda: load_deepcrispr("hela"))):
        ds = fn()
        X0 = featurize(ds.sequences)
        X1 = np.concatenate([X0, bio_feats(ds.sequences)], axis=1)
        y = ds.scores
        idx = rng.permutation(len(y))
        n_te = len(y) // 5
        data[name] = (X0, X1, y, idx[n_te:], idx[:n_te])
    for tr_name, te_name in (("fcres", "fcres"), ("dc_hela", "dc_hela"),
                             ("fcres", "dc_hela"), ("dc_hela", "fcres")):
        X0t, X1t, yt, tr, _ = data[tr_name]
        X0e, X1e, ye, _, te = data[te_name]
        for tag, Xt, Xe in (("base", X0t, X0e), ("bio", X1t, X1e)):
            m = Ridge(alpha=1.0).fit(Xt[tr], yt[tr])
            res[f"{tr_name}->{te_name}:{tag}"] = float(
                spearmanr(m.predict(Xe[te]), ye[te]).statistic)
    return res


def main():
    out = {str(s): run(s) for s in (0, 1, 2)}
    keys = list(out["0"])
    summary = {k: float(np.mean([out[s][k] for s in out])) for k in keys}
    json.dump({"per_seed": out, "mean": summary},
              open("results/biopython_feature_audit.json", "w"), indent=2)
    for k, v in summary.items():
        print(f"{k:24s} {v:.3f}")


if __name__ == "__main__":
    main()
