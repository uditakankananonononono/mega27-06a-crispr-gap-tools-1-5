"""Gap 5, K562: does a crisprSQL K562 locus falling INSIDE a called DNase
peak (ENCFF795GKD narrowPeak, hg19) separate cleaved from non-cleaved
better than the continuous read-depth signal? intervaltree overlap query.
"""
import gzip
import json

import numpy as np
import pandas as pd
from intervaltree import IntervalTree
from scipy.stats import fisher_exact, mannwhitneyu

BED = "data/encff795gkd_k562_dnase_narrowpeak.bed.gz"


def main():
    trees = {}
    with gzip.open(BED, "rt") as fh:
        for line in fh:
            f = line.split("\t")
            chrom, s, e = f[0], int(f[1]), int(f[2])
            trees.setdefault(chrom, IntervalTree()).addi(s, e)
    df = pd.read_csv("data/crisprsql/100720.csv", low_memory=False)
    h = df[df.cell_line.str.contains("K562", na=False)].reset_index(drop=True)
    in_peak, ok = [], []
    for r in h.itertuples():
        try:
            s, e = int(r.target_start), int(r.target_end)
            chrom = str(r.target_chr)
            in_peak.append(bool(trees.get(chrom) and trees[chrom].overlaps(s, e)))
            ok.append(True)
        except (ValueError, TypeError):
            in_peak.append(False); ok.append(False)
    h = h[np.array(ok)]
    in_peak = np.array(in_peak)[np.array(ok)]
    y = (h.cleavage_freq > 0).to_numpy()
    # 2x2: in_peak x cleaved
    a = int((in_peak & (y == 1)).sum()); b = int((in_peak & (y == 0)).sum())
    c = int((~in_peak & (y == 1)).sum()); d = int((~in_peak & (y == 0)).sum())
    or_, p = fisher_exact([[a, b], [c, d]])
    res = {
        "dataset": "ENCODE ENCSR000EKN / ENCFF795GKD (K562 DNase narrowPeak, hg19)",
        "n": int(len(y)), "n_pos": int((y == 1).sum()), "n_neg": int((y == 0).sum()),
        "table": {"in_peak_cleaved": a, "in_peak_not": b,
                  "out_peak_cleaved": c, "out_peak_not": d},
        "odds_ratio": float(or_), "fisher_p": float(p),
        "frac_cleaved_in_peak": a / (a + b) if a + b else None,
        "frac_cleaved_out_peak": c / (c + d) if c + d else None,
    }
    json.dump(res, open("results/gap5_k562_peaks.json", "w"), indent=2)
    print(json.dumps(res, indent=2))


if __name__ == "__main__":
    main()
