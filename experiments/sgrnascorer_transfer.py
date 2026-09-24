"""Chari sgRNAScorer 2.0 high/low lists as a gap-1 external discrimination
test: train the standard ridge on each existing efficacy dataset, score the
430 Chari guides (first 20 nt, per the author's own model code), report
AUROC separating high from low. Binary endpoint -> AUROC, not Spearman.
"""
import json

import numpy as np
from sklearn.metrics import roc_auc_score

from crisprgap.crossdata import featurize, ridge_model
from crisprgap.data import datasets as D

TRAIN_SETS = {
    "doench2016_fcres": D.load_doench_fcres,
    "doench2016_v1": D.load_doench_v1,
    "crisprscan": D.load_crisprscan,
    "sgdesigner_plasmid": D.load_sgdesigner,
}


def main():
    chari = D.load_sgrnascorer()
    Xc = featurize(chari.sequences)
    yc = chari.scores
    fit, predict = ridge_model()
    out = {"n_high": int((yc == 1).sum()), "n_low": int((yc == 0).sum()),
           "auroc_by_train_source": {}}
    for name, loader in TRAIN_SETS.items():
        ds = loader()
        m = fit(featurize(ds.sequences), ds.scores)
        out["auroc_by_train_source"][name] = float(
            roc_auc_score(yc, predict(m, Xc)))
    # positive control: 5-fold CV AUROC within Chari (ridge can learn the split)
    from sklearn.model_selection import StratifiedKFold
    aucs = []
    for tr, te in StratifiedKFold(5, shuffle=True, random_state=0).split(Xc, yc):
        m = fit(Xc[tr], yc[tr])
        aucs.append(float(roc_auc_score(yc[te], predict(m, Xc[te]))))
    out["chari_within_cv_auroc"] = {"mean": float(np.mean(aucs)), "folds": aucs}
    with open("results/sgrnascorer_transfer.json", "w") as fh:
        json.dump(out, fh, indent=2)
    print(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()
