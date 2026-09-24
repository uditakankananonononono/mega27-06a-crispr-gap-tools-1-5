"""XGBoost on pegRNA numerics (gap 3): is the tuned-GBM advantage
library-specific or feature-driven? Default XGB and an XGB config mapped
from the Optuna-tuned LightGBM region (same 3 grouped splits, seeded)."""
import json

import numpy as np
import xgboost as xgb
from scipy.stats import spearmanr

from crisprgap.data.datasets import load_pridict
from crisprgap.train_pegrna import _split

# mapped from the Optuna-tuned LightGBM region (results/pegrna_gbm_tuned.json)
TUNED_MAP = {"n_estimators": 704, "learning_rate": 0.03,
             "max_depth": 4,  # num_leaves 13 ~ depth 4
             "subsample": 0.70, "min_child_weight": 33}


def run(ds, target, params, tag, res):
    yt = ds.targets[target]
    per = []
    for seed in (0, 1, 2):
        tr, va, te = _split(ds, seed)
        m = xgb.XGBRegressor(random_state=seed, verbosity=0, **params)
        m.fit(ds.numerics[tr], yt[tr])
        per.append(float(spearmanr(m.predict(ds.numerics[te]), yt[te]).statistic))
    res["targets"][target][tag] = {"by_split": per, "mean": float(np.mean(per))}


def main():
    ds = load_pridict()
    res = {"tool": "xgboost 3.2.0", "targets": {}}
    for target in ("HEK", "K562"):
        res["targets"][target] = {}
        run(ds, target, {"n_estimators": 500}, "default_xgb", res)
        run(ds, target, TUNED_MAP, "tuned_region_xgb", res)
    res["reference"] = {"tuned_lgbm_HEK": 0.7978, "cnn_HEK": 0.794,
                        "tuned_lgbm_K562": 0.6161, "cnn_K562": 0.610}
    json.dump(res, open("results/pegrna_xgb.json", "w"), indent=2)
    print(json.dumps(res, indent=2))


if __name__ == "__main__":
    main()
