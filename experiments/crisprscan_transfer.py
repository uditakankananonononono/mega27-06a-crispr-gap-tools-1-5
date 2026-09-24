"""Gap 1, seventh family: CRISPRscan (zebrafish in vivo, 492 guides) joins
the cross-dataset matrix. Ridge, 3 seeds, same featurization as the
published matrix. Question: does the DeepHF-source boundary extend to a
non-mammalian, in-vivo assay, and does anything transfer INTO zebrafish?
"""
import json

import numpy as np
from scipy.stats import spearmanr

from crisprgap.crossdata import featurize, ridge_model
from crisprgap.data.datasets import (load_crispron, load_crisprscan,
                                     load_deepspcas9, load_deephf,
                                     load_doench_fcres, load_doench_v1,
                                     load_sgdesigner)


def main():
    loaders = {
        "fcres": load_doench_fcres, "v1": load_doench_v1,
        "deephf_wt": lambda: load_deephf("wt"), "deepspcas9": load_deepspcas9,
        "crispron": load_crispron, "sgdesigner": load_sgdesigner,
        "crisprscan": load_crisprscan,
    }
    ds = {k: f() for k, f in loaders.items()}
    feats = {k: featurize(d.sequences) for k, d in ds.items()}
    fit, predict = ridge_model()
    res = {"seeds": {}, "note": "ridge, 3 seeds, same featurization as crossdata matrix"}
    for seed in (0, 1, 2):
        rng = np.random.default_rng(seed)
        out = {}
        # within-crisprscan + train-crisprscan -> others
        for name in loaders:
            n = len(ds[name].sequences)
            perm = rng.permutation(n)
            n_te = int(n * 0.2)
            te, tr = perm[:n_te], perm[n_te:]
            m = fit(feats[name][tr], ds[name].scores[tr])
            within = float(spearmanr(predict(m, feats[name][te]), ds[name].scores[te]).statistic)
            cross = {}
            for other in loaders:
                if other == name:
                    continue
                cross[other] = float(spearmanr(predict(m, feats[other]), ds[other].scores).statistic)
            out[name] = {"within": within, "cross": cross}
        res["seeds"][str(seed)] = out
    json.dump(res, open("results/crisprscan_transfer.json", "w"), indent=2)
    for seed, out in res["seeds"].items():
        o = out["crisprscan"]
        print(f"seed {seed}: crisprscan within {o['within']:.3f}; into-crisprscan from:",
              {k: round(out[k]['cross']['crisprscan'], 3) for k in loaders if k != 'crisprscan'})
        print(f"          from-crisprscan to:", {k: round(v, 3) for k, v in o['cross'].items()})


if __name__ == "__main__":
    main()
