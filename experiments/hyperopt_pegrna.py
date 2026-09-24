"""Hyperopt cross-check of the Optuna tuning conclusion (gap 3).

The tuned-GBM-beats-CNN claim rests on a 30-trial Optuna TPE search. If a
different search algorithm (hyperopt TPE, same space, same budget, same
seed-0 train/val split) finds an equivalent config and the same held-out
gains, the conclusion is search-algorithm-robust, not an Optuna artifact.
"""
import json

import lightgbm as lgb
import numpy as np
from hyperopt import Trials, fmin, hp, tpe
from scipy.stats import spearmanr

from crisprgap.data.datasets import load_pridict
from crisprgap.train_pegrna import _split

SPACE = {
    "n_estimators": 200 + hp.randint("n_estimators", 601),
    "learning_rate": hp.loguniform("learning_rate", np.log(0.01), np.log(0.2)),
    "num_leaves": 8 + hp.randint("num_leaves", 57),
    "subsample": hp.uniform("subsample", 0.5, 1.0),
    "min_child_samples": 10 + hp.randint("min_child_samples", 91),
}


def main():
    ds = load_pridict()
    y = ds.targets["HEK"]
    tr0, va0, te0 = _split(ds, 0)

    def objective(p):
        m = lgb.LGBMRegressor(random_state=0, **p)
        m.fit(ds.numerics[tr0], y[tr0])
        return -spearmanr(m.predict(ds.numerics[va0]), y[va0]).statistic

    trials = Trials()
    best = fmin(objective, SPACE, algo=tpe.suggest, max_evals=30,
                rstate=np.random.default_rng(0), trials=trials, verbose=False)
    best = {k: (int(v) if k in ("n_estimators", "num_leaves", "min_child_samples") else float(v))
            for k, v in best.items()}
    best["n_estimators"] += 200
    best["num_leaves"] += 8
    best["min_child_samples"] += 10

    res = {"tool": "hyperopt 0.3.0 (TPE) + lightgbm 4.7.0", "n_trials": 30,
           "tuned_on": "seed-0 train/val only (HEK)", "best_params": best,
           "best_val_spearman": float(-trials.best_trial["result"]["loss"]),
           "targets": {}}
    for target in ("HEK", "K562"):
        yt = ds.targets[target]
        per = []
        for seed in (0, 1, 2):
            tr, va, te = _split(ds, seed)
            m = lgb.LGBMRegressor(random_state=seed, **best)
            m.fit(ds.numerics[tr], yt[tr])
            per.append(float(spearmanr(m.predict(ds.numerics[te]), yt[te]).statistic))
        res["targets"][target] = {"by_split": per, "mean": float(np.mean(per))}
    res["reference"] = {"optuna_tuned_HEK": 0.7978, "optuna_tuned_K562": 0.6161,
                        "cnn_HEK": 0.794, "cnn_K562": 0.610, "default_lgbm_HEK": 0.795}
    json.dump(res, open("results/pegrna_hyperopt.json", "w"), indent=2)
    print(json.dumps({k: v for k, v in res.items() if k != "targets"}, indent=1))
    print({t: round(v["mean"], 4) for t, v in res["targets"].items()})


if __name__ == "__main__":
    main()
