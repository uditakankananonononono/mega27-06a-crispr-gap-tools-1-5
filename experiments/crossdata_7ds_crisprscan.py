"""Gap 1, CNN: crisprscan row/column of the cross-dataset matrix.

The full 6x6 CNN matrix lives in results/crossdata_cnn_6ds*.json. This adds
only the pairs involving crisprscan (7th family, zebrafish in vivo):
train CNN on crisprscan -> test all six; train CNN on each of the six ->
test crisprscan. Same CNN config (6 epochs), one seed per invocation.
crisprscan is 20-mers; featurize pads to 30 via one_hot (zero-pad), same
as the ridge crisprscan experiment.
"""
import json
import sys

import numpy as np
import torch
from scipy.stats import spearmanr

from crisprgap.crossdata import featurize
from crisprgap.data.datasets import (load_crispron, load_crisprscan,
                                     load_deepspcas9, load_deephf,
                                     load_doench_fcres, load_doench_v1,
                                     load_sgdesigner)
sys.path.insert(0, "scripts")
from run_crossdata_cnn import cnn_model  # noqa: E402

torch.set_num_threads(2)

if __name__ == "__main__":
    seed = int(sys.argv[1]) if len(sys.argv) > 1 else 0
    torch.manual_seed(seed)
    loaders = {"fcres": load_doench_fcres, "v1": load_doench_v1,
               "deephf_wt": lambda: load_deephf("wt", max_n=12000),
               "deepspcas9": lambda: load_deepspcas9(max_n=12000),
               "crispron": lambda: load_crispron(max_n=12000),
               "sgdesigner": load_sgdesigner, "crisprscan": load_crisprscan}
    ds = {k: f() for k, f in loaders.items()}
    feats = {k: featurize(d.sequences) for k, d in ds.items()}
    fit, predict = cnn_model()
    rng = np.random.default_rng(seed)
    res = {"crisprscan_row": {}, "crisprscan_col": {}, "crisprscan_within": None}
    # crisprscan within + row
    n = len(ds["crisprscan"].sequences)
    perm = rng.permutation(n)
    n_te = int(n * 0.2)
    te, tr = perm[:n_te], perm[n_te:]
    m = fit(feats["crisprscan"][tr], ds["crisprscan"].scores[tr])
    res["crisprscan_within"] = float(spearmanr(
        predict(m, feats["crisprscan"][te]), ds["crisprscan"].scores[te]).statistic)
    for other in loaders:
        if other == "crisprscan":
            continue
        res["crisprscan_row"][other] = float(spearmanr(
            predict(m, feats[other]), ds[other].scores).statistic)
    # column: each dataset -> crisprscan
    for name in loaders:
        if name == "crisprscan":
            continue
        d = ds[name]
        nn = len(d.sequences)
        pm = rng.permutation(nn)
        tr2 = pm[int(nn * 0.2):]
        m = fit(feats[name][tr2], d.scores[tr2])
        res["crisprscan_col"][name] = float(spearmanr(
            predict(m, feats["crisprscan"]), ds["crisprscan"].scores).statistic)
    json.dump(res, open(f"results/crossdata_cnn_7ds_crisprscan_seed{seed}.json", "w"), indent=2)
    print(json.dumps(res, indent=2))
