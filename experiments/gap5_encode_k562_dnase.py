"""Gap 5, second cell line: ENCODE K562 DNase-seq (ENCSR000EKN, file
ENCFF526LYS, hg19 read-depth normalized signal) queried at all 1,299
crisprSQL K562 rows (hg19). Signal = mean over locus +/- 500 bp.
Only 35/1299 rows are cleavage-negative, so the primary readout is the
Spearman rank correlation between DNase signal and cleavage frequency on
the 1,253 positives; the pos/neg AUROC is reported with its n. Seeded,
hermetic.
"""
import json

import numpy as np
import pandas as pd
import pyBigWig
from scipy.stats import mannwhitneyu, spearmanr

BW = "data/encff526lys_k562_dnase_hg19.bigWig"


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
    h = df[df.cell_line.str.contains("K562", na=False)].reset_index(drop=True)
    bw = pyBigWig.open(BW)
    sig = []
    for r in h.itertuples():
        try:
            s, e = int(r.target_start), int(r.target_end)
            chrom = str(r.target_chr)
            v = bw.stats(chrom, max(0, s - 500), e + 500, type="mean")
            sig.append(v[0] if v and v[0] is not None else np.nan)
        except (ValueError, TypeError, RuntimeError):
            sig.append(np.nan)
    bw.close()
    sig = np.array(sig)
    labels = (h.cleavage_freq > 0).astype(int).to_numpy()
    ok = ~np.isnan(sig)
    y, s = labels[ok], sig[ok]
    cleav = h.cleavage_freq.to_numpy()[ok]
    pos, neg = s[y == 1], s[y == 0]
    res = {
        "dataset": "ENCODE ENCSR000EKN / ENCFF526LYS (K562 DNase-seq, hg19, read-depth normalized)",
        "rows": "crisprSQL K562 hg19, all loci, +/-500bp mean",
        "n_ok": int(ok.sum()), "n_pos": int(len(pos)), "n_neg": int(len(neg)),
        "auroc": auroc(pos, neg) if len(neg) else None,
        "median_pos": float(np.median(pos)), "median_neg": float(np.median(neg)),
        "mannwhitney_p": float(mannwhitneyu(pos, neg, alternative="two-sided").pvalue),
        "spearman_dnase_vs_cleavage_positives": float(
            spearmanr(s[y == 1], cleav[y == 1]).statistic),
        "spearman_p": float(spearmanr(s[y == 1], cleav[y == 1]).pvalue),
    }
    json.dump(res, open("results/gap5_k562_dnase.json", "w"), indent=2)
    print(json.dumps(res, indent=2))


if __name__ == "__main__":
    main()
