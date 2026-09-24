"""LightGBM tabular baseline on pegRNA numerics (gap 3) + SHAP feature attribution.

Published-leader-style comparator: gradient-boosted trees are the strong tabular
baseline the ridge/CNN comparison lacked. Same grouped splits (seeds 0,1,2) and
targets (HEK, K562) as the ridge/CNN benchmark. SHAP importances identify which
engineered features actually drive predictions.
"""
import json

import lightgbm as lgb
import numpy as np
from scipy.stats import spearmanr

from crisprgap.data.datasets import load_pridict
from crisprgap.train_pegrna import _split

OUT = "results/pegrna_gbm.json"


def main():
    ds = load_pridict()
    res = {"tool": "lightgbm 4.7.0", "splits": [0, 1, 2], "targets": {}}
    for target in ("HEK", "K562"):
        y = ds.targets[target]
        per_split, models = [], []
        for seed in (0, 1, 2):
            tr, va, te = _split(ds, seed)
            m = lgb.LGBMRegressor(n_estimators=400, learning_rate=0.05,
                                  num_leaves=31, subsample=0.8, random_state=seed)
            m.fit(ds.numerics[tr], y[tr])
            per_split.append(float(spearmanr(m.predict(ds.numerics[te]), y[te]).statistic))
            models.append(m)
        res["targets"][target] = {"gbm_by_split": per_split,
                                  "gbm_mean": float(np.mean(per_split))}
    # SHAP on the last HEK model (full train split of seed 0)
    import shap
    tr, va, te = _split(ds, 0)
    m = lgb.LGBMRegressor(n_estimators=400, learning_rate=0.05, num_leaves=31,
                          subsample=0.8, random_state=0)
    y = ds.targets["HEK"]
    m.fit(ds.numerics[tr], y[tr])
    ex = shap.TreeExplainer(m)
    sv = np.abs(ex.shap_values(ds.numerics[te])).mean(axis=0)
    order = np.argsort(-sv)
    res["shap_top_features_HEK"] = [
        {"feature": ds.numeric_col_names[i], "mean_abs_shap": float(sv[i])}
        for i in order[:10]]
    res["reference_means"] = {
        "ridge_numeric_HEK": 0.7366, "cnn_scratch_HEK": 0.794,
        "ridge_numeric_K562": 0.554, "cnn_scratch_K562": 0.610,
        "note": "ridge means from results/pegrna_rna_features.json (same split recipe); "
                "CNN numbers from the gap-3 benchmark (seeds 0,1)"}
    json.dump(res, open(OUT, "w"), indent=2)
    return res


if __name__ == "__main__":
    r = main()
    for t, d in r["targets"].items():
        print(t, "GBM:", [round(x, 4) for x in d["gbm_by_split"]], "mean", round(d["gbm_mean"], 4))
    print("top SHAP:", [(f["feature"], round(f["mean_abs_shap"], 4)) for f in r["shap_top_features_HEK"][:6]])
