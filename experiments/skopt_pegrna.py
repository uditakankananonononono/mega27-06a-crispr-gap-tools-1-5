"""scikit-optimize gp_minimize cross-check (gap 3): a Gaussian-process
Bayesian search - a different surrogate family than Optuna/hyperopt TPE -
on the same space, budget, and seed-0 train/val split. If it lands on an
equivalent config and held-out gains, the tuned-GBM conclusion is robust
to the search algorithm."""
import json

import lightgbm as lgb
import numpy as np
from scipy.stats import spearmanr
from skopt import gp_minimize
from skopt.space import Integer, Real

from crisprgap.data.datasets import load_pridict
from crisprgap.train_pegrna import _split

SPACE = [Integer(200, 800, name="n_estimators"),
         Real(0.01, 0.2, prior="log-uniform", name="learning_rate"),
         Integer(8, 64, name="num_leaves"),
         Real(0.5, 1.0, name="subsample"),
         Integer(10, 100, name="min_child_samples")]


def main():
    ds = load_pridict()
    y = ds.targets["HEK"]
    tr0, va0, te0 = _split(ds, 0)

    def objective(vals):
        p = dict(zip(("n_estimators", "learning_rate", "num_leaves",
                      "subsample", "min_child_samples"), vals))
        m = lgb.LGBMRegressor(random_state=0, **p)
        m.fit(ds.numerics[tr0], y[tr0])
        return -spearmanr(m.predict(ds.numerics[va0]), y[va0]).statistic

    r = gp_minimize(objective, SPACE, n_calls=30, random_state=0, verbose=False)
    best = dict(zip(("n_estimators", "learning_rate", "num_leaves",
                     "subsample", "min_child_samples"), r.x))
    res = {"tool": "scikit-optimize 0.10.2 (GP-BO) + lightgbm 4.7.0",
           "n_trials": 30, "tuned_on": "seed-0 train/val only (HEK)",
           "best_params": {k: (int(v) if isinstance(v, (int, np.integer)) else float(v))
                           for k, v in best.items()},
           "best_val_spearman": float(-r.fun), "targets": {}}
    for target in ("HEK", "K562"):
        yt = ds.targets[target]
        per = []
        for seed in (0, 1, 2):
            tr, va, te = _split(ds, seed)
            m = lgb.LGBMRegressor(random_state=seed, **res["best_params"])
            m.fit(ds.numerics[tr], yt[tr])
            per.append(float(spearmanr(m.predict(ds.numerics[te]), yt[te]).statistic))
        res["targets"][target] = {"by_split": per, "mean": float(np.mean(per))}
    res["reference"] = {"optuna_tuned_HEK": 0.7978, "hyperopt_tuned_HEK": 0.7979,
                        "cnn_HEK": 0.794, "default_lgbm_HEK": 0.795}
    json.dump(res, open("results/pegrna_skopt.json", "w"), indent=2)
    print(json.dumps({k: v for k, v in res.items() if k != "targets"}, indent=1))
    print({t: round(v["mean"], 4) for t, v in res["targets"].items()})


if __name__ == "__main__":
    main()
