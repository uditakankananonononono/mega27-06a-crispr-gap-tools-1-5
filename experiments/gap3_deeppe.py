"""Gap 3 external validation: DeepPE Library 1 (Kim 2021 Nat Biotechnol,
10.1038/s41587-020-0677-y MOESM4; 43,150 pegRNAs, HEK293T PE2).
Readouts: (a) numeric-feature ridge 5-fold CV Spearman (compare with
PRIDICT2 HEK ridge 0.74), (b) DeepSpCas9-score-only transfer (the Cas9
prior feature), (c) univariate feature effects vs PRIDICT2 directions."""
import json

import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.linear_model import Ridge
from sklearn.model_selection import KFold

FEATS = ["PBS length", "RT length", "PBS-RT length", "Tm1\n(PBS)",
         "Tm 2\n(target DNA region corresponding to RT template)",
         "Tm 3\n(reverse transcribed cDNA and PAM-opposite DNA strand)",
         "Tm 4\n(RT template region and reverse transcribed cDNA)",
         "deltaTm\n(Tm3-Tm2)", "GC count_1\n(PBS)", "GC count_2\n(RT)",
         "GC count_3\n(PBS-RT)", "GC contents_1\n(PBS)", "GC contents_2\n(RT)",
         "GC contents_3\n(PBS-RT)", "MFE_1\n(pegRNA)", "MFE_2\n(-spacer)",
         "MFE_3\n(RT-PBS-PolyT)", "MFE_4\n(spacer only)", "MFE_5\n(Spacer+Scaffold)",
         "DeepSpCas9 score"]


def main():
    df = pd.read_excel("data/raw/deeppe/moesm4.xlsx",
                       sheet_name="Library 1 (HT-training, test)", skiprows=1)
    y = pd.to_numeric(df["Measured PE efficiency"], errors="coerce")
    X = df[FEATS].apply(pd.to_numeric, errors="coerce")
    ok = y.notna() & X.notna().all(axis=1)
    X, y = X[ok].to_numpy(), y[ok].to_numpy()
    out = {"n": int(len(y)), "y_mean": float(y.mean())}
    # ridge 5-fold CV
    kf = KFold(5, shuffle=True, random_state=0)
    sp = []
    for tr, te in kf.split(X):
        m = Ridge(alpha=10).fit(X[tr], y[tr])
        sp.append(float(spearmanr(m.predict(X[te]), y[te]).statistic))
    out["ridge_cv_spearman"] = [round(s, 3) for s in sp]
    out["ridge_cv_mean"] = round(float(np.mean(sp)), 3)
    # Cas9-score-only transfer
    i_ds9 = FEATS.index("DeepSpCas9 score")
    out["deepspcas9_only_spearman"] = round(float(spearmanr(X[:, i_ds9], y).statistic), 3)
    # univariate effects
    out["univariate"] = {f.split("\n")[0]: round(float(spearmanr(X[:, i], y).statistic), 3)
                         for i, f in enumerate(FEATS)}
    json.dump(out, open("results/gap3_deeppe.json", "w"), indent=1)
    print(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
