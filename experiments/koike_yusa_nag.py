"""Koike-Yusa 2014 MiSeq off-target set (190 sites, 1 guide, mESC):
gap-4 PAM-variant analysis + CFD baseline.

Questions:
1. Mismatch gradient by PAM class (NGG vs NAG): cleavage (exp1_tf) by
   mismatch count.
2. Published baseline: CFD score vs measured cleavage (Spearman, AUROC),
   all sites and NGG-only.
"""
import json

import numpy as np
from scipy.stats import spearmanr
from sklearn.metrics import roc_auc_score

from crisprgap.data.datasets import load_koike_yusa_miseq
from crisprgap.cfd import cfd_score, cfd_applicable


def main():
    ds = load_koike_yusa_miseq()
    y = ds.cleavage_freq
    n = len(y)
    guide23 = ds.guides[0] + ds.pam[0]
    pams = np.array([p.upper() for p in ds.off_pam])
    is_ngg = np.array([p.endswith("GG") for p in pams])
    is_nag = np.array([p.endswith("AG") and not p.endswith("GG") for p in pams])
    mm = np.array([sum(1 for a, b in zip(g, o) if a != b)
                   for g, o in zip(ds.guides, ds.offtargets)])

    gradient = {}
    for grp, sel in (("ngg", is_ngg), ("nag", is_nag)):
        per_mm = {}
        for k in sorted(set(mm)):
            vals = y[sel & (mm == k)]
            vals = vals[~np.isnan(vals)]
            if len(vals):
                per_mm[int(k)] = {
                    "n": int(len(vals)),
                    "mean_cleavage": float(np.mean(vals)),
                    "median_cleavage": float(np.median(vals)),
                    "frac_ge_1pct": float(np.mean(vals >= 1.0)),
                }
        gradient[grp] = per_mm

    cfd = np.array([cfd_score(guide23, o + p) if cfd_applicable(guide23, o + p)
                    else np.nan for o, p in zip(ds.offtargets, ds.off_pam)])
    valid = ~np.isnan(cfd) & ~np.isnan(y)
    out = {
        "n_sites": int(n),
        "n_ngg": int(is_ngg.sum()),
        "n_nag": int(is_nag.sum()),
        "pam_counts": {p: int((pams == p).sum()) for p in sorted(set(pams))},
        "gradient": gradient,
        "cfd": {
            "n_scored": int(valid.sum()),
            "spearman_all": float(spearmanr(cfd[valid], y[valid]).statistic),
            "spearman_ngg": float(spearmanr(cfd[valid & is_ngg], y[valid & is_ngg]).statistic)
                            if (valid & is_ngg).sum() > 2 else None,
            "auroc_ge1pct_all": float(roc_auc_score(y[valid] >= 1.0, cfd[valid]))
                                if len(set(y[valid] >= 1.0)) > 1 else None,
            "auroc_gt0_all": float(roc_auc_score(y[valid] > 0, cfd[valid]))
                              if len(set(y[valid] > 0)) > 1 else None,
        },
        "subclone_corr": float(np.corrcoef(
            ds.energies["subclone_mean"][~np.isnan(ds.energies["subclone_mean"]) & ~np.isnan(y)],
            y[~np.isnan(ds.energies["subclone_mean"]) & ~np.isnan(y)])[0, 1]),
    }
    with open("results/koike_yusa_nag.json", "w") as fh:
        json.dump(out, fh, indent=2)
    print(json.dumps(out, indent=2)[:1500])


if __name__ == "__main__":
    main()
