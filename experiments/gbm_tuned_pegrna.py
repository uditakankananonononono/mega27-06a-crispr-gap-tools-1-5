"""Optuna-tuned LightGBM on pegRNA numerics (gap 3): does tuning flip the
GBM-vs-CNN comparison? Tuning on the seed-0 train/val split only (HEK),
then the frozen config is evaluated on all three grouped test splits and on
K562. No test-set leakage into tuning."""
import json

import lightgbm as lgb
import numpy as np
import optuna
from scipy.stats import spearmanr

from crisprgap.data.datasets import load_pridict
from crisprgap.train_pegrna import _split

optuna.logging.set_verbosity(optuna.logging.WARNING)


def main(n_trials=30):
    ds = load_pridict()
    y = ds.targets["HEK"]
    tr0, va0, te0 = _split(ds, 0)

    def objective(trial):
        p = {
            "n_estimators": trial.suggest_int("n_estimators", 200, 800),
            "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.2, log=True),
            "num_leaves": trial.suggest_int("num_leaves", 8, 64),
            "subsample": trial.suggest_float("subsample", 0.5, 1.0),
            "min_child_samples": trial.suggest_int("min_child_samples", 10, 100),
        }
        m = lgb.LGBMRegressor(random_state=0, **p)
        m.fit(ds.numerics[tr0], y[tr0])
        return spearmanr(m.predict(ds.numerics[va0]), y[va0]).statistic

    study = optuna.create_study(direction="maximize",
                                sampler=optuna.samplers.TPESampler(seed=0))
    study.optimize(objective, n_trials=n_trials)
    best = study.best_params

    res = {"tool": "optuna 5.0.0 (TPE) + lightgbm 4.7.0", "n_trials": n_trials,
           "tuned_on": "seed-0 train/val only (HEK)", "best_params": best,
           "best_val_spearman": float(study.best_value), "targets": {}}
    for target in ("HEK", "K562"):
        yt = ds.targets[target]
        per = []
        for seed in (0, 1, 2):
            tr, va, te = _split(ds, seed)
            m = lgb.LGBMRegressor(random_state=seed, **best)
            m.fit(ds.numerics[tr], yt[tr])
            per.append(float(spearmanr(m.predict(ds.numerics[te]), yt[te]).statistic))
        res["targets"][target] = {"tuned_gbm_by_split": per,
                                  "tuned_gbm_mean": float(np.mean(per))}
    res["reference"] = {"untuned_gbm_HEK": 0.795, "cnn_HEK": 0.794,
                        "untuned_gbm_K562": 0.610, "cnn_K562": 0.610}
    json.dump(res, open("results/pegrna_gbm_tuned.json", "w"), indent=2)
    return res


if __name__ == "__main__":
    r = main()
    print("best val:", round(r["best_val_spearman"], 4))
    for t, d in r["targets"].items():
        print(t, "tuned GBM:", [round(x, 4) for x in d["tuned_gbm_by_split"]],
              "mean", round(d["tuned_gbm_mean"], 4))
