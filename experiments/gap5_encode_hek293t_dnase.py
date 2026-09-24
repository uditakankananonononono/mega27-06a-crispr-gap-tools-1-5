"""Gap 5, best-powered cell line: ENCODE HEK293T DNase-seq (ENCSR000EJR,
file ENCFF529BOG, GRCh38 read-depth normalized signal) queried at all
1,630 crisprSQL HEK293 hg38 rows (Cameron SITE-Seq) - 52 negatives, the
best-powered chromatin question in the corpus. Signal = mean over
locus +/- 500 bp from a locally downloaded bigWig. Seeded, hermetic.
"""
import json

import numpy as np
import pandas as pd
import pyBigWig
from scipy.stats import mannwhitneyu

BW = "data/encff529bog_hek293t_dnase_grch38.bigWig"


def auroc(pos, neg):
    vals = np.concatenate([pos, neg])
    order = np.argsort(vals, kind="stable")
    ranks = np.empty(len(vals)); ranks[order] = np.arange(1, len(vals) + 1)
    _, inv, cnt = np.unique(vals, return_inverse=True, return_counts=True)
    sums = np.zeros(len(cnt)); np.add.at(sums, inv, ranks)
    ranks = sums[inv] / cnt[inv]
    return float((ranks[:len(pos)].sum() - len(pos) * (len(pos) + 1) / 2) / (len(pos) * len(neg)))


def main():
    df = pd.read_csv("data/crisprsql/100720.csv", low_memory=False)
    h = df[df.cell_line.str.contains("HEK", na=False) & (df.genome == "hg38")].reset_index(drop=True)
    bw = pyBigWig.open(BW)
    sig = []
    for r in h.itertuples():
        s, e = int(r.target_start), int(r.target_end)
        v = bw.stats(r.target_chr, max(0, s - 500), e + 500, type="mean")
        sig.append(v[0] if v and v[0] is not None else np.nan)
    bw.close()
    sig = np.array(sig)
    labels = (h.cleavage_freq > 0).astype(int).to_numpy()
    ok = ~np.isnan(sig)
    y, s = labels[ok], sig[ok]
    pos, neg = s[y == 1], s[y == 0]
    res = {
        "dataset": "ENCODE ENCSR000EJR / ENCFF529BOG (HEK293T DNase-seq, GRCh38, read-depth normalized)",
        "rows": "crisprSQL HEK hg38 (Cameron SITE-Seq), all 1630 loci, +/-500bp mean",
        "n_ok": int(ok.sum()), "n_pos": int(len(pos)), "n_neg": int(len(neg)),
        "auroc": auroc(pos, neg),
        "median_pos": float(np.median(pos)), "median_neg": float(np.median(neg)),
        "mean_pos": float(pos.mean()), "mean_neg": float(neg.mean()),
        "mannwhitney_p": float(mannwhitneyu(pos, neg, alternative="two-sided").pvalue),
    }
    json.dump(res, open("results/gap5_hek293t_dnase.json", "w"), indent=2)
    print(json.dumps(res, indent=2))


if __name__ == "__main__":
    main()
