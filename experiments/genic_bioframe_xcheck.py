"""bioframe cross-check of the gap-5 genic stratification: same crisprSQL
hg38 loci vs GENCODE v47 basic exons, using bioframe.overlap instead of
intervaltree. Agreement of per-locus exonic calls is the readout."""
import gzip
import json

import bioframe as bf
import numpy as np
import pandas as pd
from scipy.stats import fisher_exact


def main():
    exons = []
    with gzip.open("data/gencode_v47_basic.gtf.gz", "rt") as fh:
        for line in fh:
            if line.startswith("#"):
                continue
            f = line.rstrip("\n").split("\t")
            if len(f) >= 9 and f[2] == "exon":
                exons.append((f[0], int(f[3]) - 1, int(f[4])))  # 0-based
    ex = pd.DataFrame(exons, columns=["chrom", "start", "end"])

    df = pd.read_csv("data/crisprsql/100720.csv", low_memory=False)
    h = df[df.genome == "hg38"].dropna(subset=["target_chr", "target_start", "target_end"]).reset_index(drop=True)
    loci = pd.DataFrame({"chrom": h.target_chr.astype(str),
                         "start": h.target_start.astype(int),
                         "end": h.target_end.astype(int)})
    ov = bf.overlap(loci, ex, how="left", return_index=True, suffixes=("", "_exon"))
    locus_hit = ov.groupby("index")["start_exon"].apply(lambda s: s.notna().any())
    exonic = np.zeros(len(h), dtype=bool)
    exonic[locus_hit.index.to_numpy()] = locus_hit.to_numpy()
    y = (h.cleavage_freq.to_numpy() > 0)
    a = int((exonic & y).sum()); b = int((exonic & ~y).sum())
    c = int((~exonic & y).sum()); d = int((~exonic & ~y).sum())
    orr, p = fisher_exact([[a, b], [c, d]])
    out = {
        "n": int(len(h)),
        "exonic_frac_cleaved": float(y[exonic].mean()),
        "nonexonic_frac_cleaved": float(y[~exonic].mean()),
        "fisher_or": float(orr), "fisher_p": float(p),
        "cross_check": "intervaltree result: exonic 53.8% vs non-exonic 54.5%, OR 0.96",
    }
    json.dump(out, open("results/genic_bioframe_xcheck.json", "w"), indent=1)
    print(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
