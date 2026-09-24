"""Karyotype figure: crisprSQL hg38 off-target sites on chromosome ideograms,
per-chromosome site counts and cleavage rates (pycirclize, GRC ideogram bands)."""
import json
import numpy as np
import pandas as pd
from scipy import stats
from pycirclize import Circos

def main():
    df = pd.read_csv("data/crisprsql/100720.csv", low_memory=False)
    h = df[df.genome == "hg38"].dropna(subset=["target_chr", "target_start", "cleavage_freq"])
    # 685 rows (all Cameron whole-genome) carry tiny negative cleavage values
    # (mean -1.7e-4): background-subtraction artifacts; clip to zero.
    h = h.assign(cleavage_freq=h.cleavage_freq.clip(lower=0))
    # pycirclize ships UCSC hg38 cytobands
    canonical = [f"chr{i}" for i in range(1, 23)] + ["chrX", "chrY"]
    sizes = {}
    for line in open("/tmp/hg38.chrom.sizes"):
        c, s = line.split()
        if c in canonical:
            sizes[c] = int(s)
    circos = Circos(sizes, space=1)
    for sector in circos.sectors:
        sector.text(sector.name.replace("chr", ""), r=104, size=7, adjust_rotation=False)
    circos.add_cytoband_tracks((93, 100), "/tmp/cytoband_hg38.bed")
    # scatter cleavage sites on each chromosome track
    for sector in circos.sectors:
        track = sector.add_track((80, 92))
        chrom = sector.name
        sub = h[h.target_chr == chrom]
        if len(sub):
            track.scatter(sub.target_start.to_numpy(),
                          np.full(len(sub), 0.5),
                          c=np.clip(sub.cleavage_freq.to_numpy(), 0, 1),
                          s=1.5, vmin=0, vmax=1, cmap="viridis")
        track.axis(fc="none", ec="lightgrey", lw=0.4)
    fig = circos.plotfig()
    fig.savefig("papers/figs/karyotype_crisprsql.pdf", bbox_inches="tight")
    fig.savefig("papers/figs/karyotype_crisprsql.png", dpi=150, bbox_inches="tight")
    # per-chromosome stats: does chromosome identity predict cleavage?
    per = h.groupby("target_chr").cleavage_freq.agg(["count", "mean"])
    per = per[per["count"] >= 20]
    kw = stats.kruskal(*[h[h.target_chr == c].cleavage_freq for c in per.index])
    out = {
        "n_sites": int(len(h)),
        "n_negative_clipped": 685,
        "n_chromosomes_ge20": int(len(per)),
        "per_chrom_mean_min": float(per["mean"].min()),
        "per_chrom_mean_max": float(per["mean"].max()),
        "kruskal_H": float(kw.statistic),
        "kruskal_p": float(kw.pvalue),
        "top3": per["mean"].nlargest(3).round(4).to_dict(),
    }
    json.dump(out, open("results/karyotype.json", "w"), indent=1)
    print(json.dumps(out, indent=1))

if __name__ == "__main__":
    main()
