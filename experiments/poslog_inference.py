"""Gap 2 inferential layer: per-position mismatch log-odds with Wald CIs.

The predictive poslog model (0.692 AUROC) uses 320 one-hot pair features.
For interpretable inference we refit a reduced model on the same seed-0
train split: one binary mismatch indicator per protospacer position (20
features) + intercept, fit with statsmodels GLM (Binomial, logit link).
Yields per-position log-odds coefficients, standard errors, 95% Wald CIs
and a likelihood-ratio test against the intercept-only null - the
quantitative "seed region" profile for this corpus.
"""
import json

import numpy as np
import statsmodels.api as sm
from scipy.stats import chi2

from crisprgap.data.datasets import load_crisprsql
from experiments.poslog_offtarget import split_for_seed


def mismatch_matrix(guides, offs):
    L = len(guides[0])
    X = np.zeros((len(guides), L), dtype=np.float64)
    for i, (g, o) in enumerate(zip(guides, offs)):
        for p, (a, b) in enumerate(zip(g, o)):
            X[i, p] = 1.0 if a.upper() != b.upper() else 0.0
    return X


def main():
    ds = load_crisprsql()
    tr, va, te = split_for_seed(ds, 0)
    g = np.array(ds.guides); o = np.array(ds.offtargets)
    X = mismatch_matrix(g[tr], o[tr])
    y = ds.labels[tr].astype(np.float64)
    groups = np.array(ds.guides)[tr]
    Xd = sm.add_constant(X)
    m = sm.GLM(y, Xd, family=sm.families.Binomial()).fit(
        cov_type="cluster", cov_kwds={"groups": groups})
    null = sm.GLM(y, Xd[:, :1], family=sm.families.Binomial()).fit()
    lrt = 2 * (m.llf - null.llf)
    res = {
        "tool": "statsmodels 0.15.0 GLM Binomial, guide-cluster-robust SEs",
        "n_guides_train": int(len(np.unique(groups))),
        "n_train": int(len(y)), "n_pos": int(y.sum()),
        "lrt_chi2": float(lrt), "lrt_df": 20,
        "lrt_p": float(chi2.sf(lrt, 20)),
        "intercept": {"coef": float(m.params[0]), "se": float(m.bse[0])},
        "per_position": [
            {"pos": p + 1, "coef": float(m.params[p + 1]), "se": float(m.bse[p + 1]),
             "ci_lo": float(m.params[p + 1] - 1.96 * m.bse[p + 1]),
             "ci_hi": float(m.params[p + 1] + 1.96 * m.bse[p + 1]),
             "z": float(m.tvalues[p + 1]), "p": float(m.pvalues[p + 1])}
            for p in range(20)],
    }
    json.dump(res, open("results/poslog_inference.json", "w"), indent=2)
    print(f"LRT chi2={lrt:.1f} df=20 p={res['lrt_p']:.2e}")
    for r in res["per_position"]:
        print(f"pos{r['pos']:2d} beta={r['coef']:+.3f} +- {1.96*r['se']:.3f}  p={r['p']:.1e}")


if __name__ == "__main__":
    main()
