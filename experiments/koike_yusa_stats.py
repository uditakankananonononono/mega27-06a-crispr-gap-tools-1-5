"""statsmodels logistic regression on the Koike-Yusa MiSeq set:
cleavage (>0.1%) ~ mismatch_count + NAG_PAM. Odds ratios with 95% CIs.
"""
import json

import numpy as np
import pandas as pd
import statsmodels.formula.api as smf

from crisprgap.data.datasets import load_koike_yusa_miseq


def main():
    ds = load_koike_yusa_miseq()
    y = ds.cleavage_freq
    ok = ~np.isnan(y)
    mm = np.array([sum(a != b for a, b in zip(g, o))
                   for g, o in zip(ds.guides, ds.offtargets)])
    nag = np.array([p.upper().endswith("AG") and not p.upper().endswith("GG")
                    for p in ds.off_pam])
    df = pd.DataFrame({
        "hit": (y[ok] > 0.1).astype(int),
        "mm": mm[ok],
        "nag": nag[ok].astype(int),
    })
    m = smf.logit("hit ~ mm + nag", data=df).fit(disp=0)
    out = {
        "n": int(len(df)),
        "n_hit": int(df["hit"].sum()),
        "threshold_pct": 0.1,
        "coef": {k: float(v) for k, v in m.params.items()},
        "odds_ratio": {k: float(np.exp(v)) for k, v in m.params.items()},
        "ci95_odds_ratio": {k: [float(np.exp(lo)), float(np.exp(hi))]
                            for k, (lo, hi) in m.conf_int().iterrows()},
        "pvalues": {k: float(v) for k, v in m.pvalues.items()},
    }
    with open("results/koike_yusa_stats.json", "w") as fh:
        json.dump(out, fh, indent=2)
    print(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()
