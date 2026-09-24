"""Gap 5 genic context: are crisprSQL hg38 loci in exons, introns, or
intergenic regions, and does genic context relate to cleavage?
GENCODE v47 basic GTF parsed by streaming; intervaltree lookups.
"""
import gzip
import json

import numpy as np
import pandas as pd
from intervaltree import IntervalTree
from scipy.stats import fisher_exact

GTF = "data/gencode_v47_basic.gtf.gz"


def build_trees():
    exon_trees, gene_trees = {}, {}
    with gzip.open(GTF, "rt") as fh:
        for line in fh:
            if line.startswith("#"):
                continue
            f = line.rstrip("\n").split("\t")
            if len(f) < 9:
                continue
            feat, chrom, s, e = f[2], f[0], int(f[3]), int(f[4])
            if feat == "exon":
                exon_trees.setdefault(chrom, IntervalTree()).addi(s, e + 1)
            elif feat == "gene":
                gene_trees.setdefault(chrom, IntervalTree()).addi(s, e + 1)
    return exon_trees, gene_trees


def main():
    exon_trees, gene_trees = build_trees()
    df = pd.read_csv("data/crisprsql/100720.csv", low_memory=False)
    h = df[df.genome == "hg38"].reset_index(drop=True)
    ctx, ok = [], []
    for r in h.itertuples():
        try:
            s, e = int(r.target_start), int(r.target_end)
            chrom = str(r.target_chr)
            if exon_trees.get(chrom) and exon_trees[chrom].overlaps(s, e):
                ctx.append("exonic")
            elif gene_trees.get(chrom) and gene_trees[chrom].overlaps(s, e):
                ctx.append("intronic_or_other_genic")
            else:
                ctx.append("intergenic")
            ok.append(True)
        except (ValueError, TypeError):
            ctx.append(None); ok.append(False)
    h = h[np.array(ok)]
    ctx = np.array(ctx)[np.array(ok)]
    y = (h.cleavage_freq > 0).to_numpy()
    table = {}
    for c in ("exonic", "intronic_or_other_genic", "intergenic"):
        sel = ctx == c
        table[c] = {"n": int(sel.sum()), "frac_cleaved": float(y[sel].mean())}
    ex = ctx == "exonic"; nex = ctx != "exonic"
    a, b = int((ex & (y == 1)).sum()), int((ex & (y == 0)).sum())
    c2, d2 = int((nex & (y == 1)).sum()), int((nex & (y == 0)).sum())
    orr, p = fisher_exact([[a, b], [c2, d2]])
    out = {"annotation": "GENCODE v47 basic (GRCh38)", "n": int(len(y)),
           "contexts": table, "exonic_vs_rest_or": float(orr),
           "fisher_p": float(p)}
    json.dump(out, open("results/gap5_genic_context.json", "w"), indent=2)
    print(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()
