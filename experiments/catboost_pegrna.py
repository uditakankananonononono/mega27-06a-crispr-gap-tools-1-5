"""CatBoost on pegRNA numerics (gap 3): third GBM library. Does the
tuned-GBM advantage over the sequence CNN hold across all three major
gradient-boosting implementations? Same 3 grouped splits, seeded.
CatBoost defaults are strong out of the box, so run default + a config
in the Optuna-tuned region."""
import json

import numpy as np
from catboost import CatBoostRegressor
from scipy.stats import spearmanr

from crisprgap.data.datasets import load_pridict
from crisprgap.train_pegrna import _split

TUNED_REGION = {"iterations": 700, "learning_rate": 0.03, "depth": 4,
                "subsample": 0.7, "min_data_in_leaf": 33}


def run(ds, target, params, tag, res):
    yt = ds.targets[target]
    per = []
    for seed in (0, 1, 2):
        tr, va, te = _split(ds, seed)
        m = CatBoostRegressor(random_seed=seed, verbose=0, **params)
        m.fit(ds.numerics[tr], yt[tr])
        per.append(float(spearmanr(m.predict(ds.numerics[te]), yt[te]).statistic))
    res["targets"][target][tag] = {"by_split": per, "mean": float(np.mean(per))}


def main():
    ds = load_pridict()
    res = {"tool": "catboost 1.2.10", "targets": {}}
    for target in ("HEK", "K562"):
        res["targets"][target] = {}
        run(ds, target, {"iterations": 500}, "default_catboost", res)
        run(ds, target, TUNED_REGION, "tuned_region_catboost", res)
    res["reference"] = {"tuned_lgbm_HEK": 0.7978, "tuned_xgb_HEK": 0.7985,
                        "cnn_HEK": 0.794, "tuned_lgbm_K562": 0.6161,
                        "tuned_xgb_K562": 0.6060, "cnn_K562": 0.610}
    json.dump(res, open("results/pegrna_catboost.json", "w"), indent=2)
    print(json.dumps(res, indent=2))


if __name__ == "__main__":
    main()
