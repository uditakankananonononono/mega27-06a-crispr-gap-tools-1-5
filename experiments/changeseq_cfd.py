"""Gap 2 external corpus: CHANGE-seq (Lazzarotto 2020, Nat Biotechnol
10.1038/s41587-020-0555-7 MOESM3 Table 3): 202,043 candidate off-target
sites across 110 guides with read counts. NOT in crisprSQL (2020-10
snapshot check: no Lazzarotto/CHANGE study). CFD scored where
applicable; AUROC for active-vs-zero at several read thresholds,
Spearman CFD vs log(reads) on positives."""
import json

import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.metrics import roc_auc_score

from crisprgap.cfd import cfd_score, cfd_applicable


def main():
    t3 = pd.read_excel("data/raw/changeseq/moesm3.xlsx",
                       sheet_name="CHANGE-seq_Supp_Table_3")
    t4 = pd.read_excel("data/raw/changeseq/moesm3.xlsx",
                       sheet_name="CHANGE-seq_Supp_Table_4")
    guide_map = dict(zip(t4["Sample"], t4["Protospacer (target without PAM)"]))
    t3["guide"] = t3["name"].map(guide_map)
    t3 = t3.dropna(subset=["guide", "offtarget_sequence"])
    off = t3["offtarget_sequence"].astype(str).str.upper()
    n_indel = int(((off.str.len() != 23) | off.str.contains("-")).sum())
    # CFD applies only to substitution-only 23-mers over ACGT
    keep = (off.str.len() == 23) & ~off.str.contains("-") & off.str.fullmatch("[ACGT]{23}")
    t3 = t3[keep]
    off = off[keep]
    t3["off20"] = off.str[:20]
    t3["offpam"] = off.str[20:23]
    t3["guide"] = t3["guide"].astype(str).str.upper().str[:20]
    ok_len = (t3.guide.str.len() == 20)
    t3 = t3[ok_len]
    scores = [cfd_score(g + "NGG", o + p) for g, o, p in
              zip(t3.guide, t3.off20, t3.offpam)]
    t3["cfd"] = scores
    reads = t3["CHANGEseq_reads"].to_numpy(dtype=float)
    out = {"n_rows_total": 202043, "n_indel_excluded": n_indel, "n_scored": int(len(t3)),
           "n_guides": int(t3.guide.nunique()),
           "frac_scorable": round(len(t3) / 202043, 3)}
    for thr in [1, 5, 20, 100]:
        y = (reads >= thr).astype(int)
        if 10 < y.sum() < len(y) - 10:
            out[f"auroc_ge{thr}"] = round(float(roc_auc_score(y, t3.cfd)), 3)
            out[f"n_pos_ge{thr}"] = int(y.sum())
    pos = reads > 0
    out["spearman_cfd_logreads_pos"] = round(
        float(spearmanr(t3.cfd[pos], np.log10(reads[pos])).statistic), 3)
    out["n_pos"] = int(pos.sum())
    json.dump(out, open("results/changeseq_cfd.json", "w"), indent=1)
    print(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
