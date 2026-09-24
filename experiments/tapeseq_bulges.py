"""Gap 2: TAPE-seq (Kim 2022, Nat Commun 10.1038/s41467-022-35743-y,
Supplementary Data 3) - off-target sites for prime-editor guides with
explicit RNA/DNA bulge typing. crisprSQL (2020 snapshot) has no TAPE-seq.
Question: how common are bulge-carrying off-targets in cell-based assays,
and do they cleave less (read depth) than substitution-only sites?"""
import json

import numpy as np
import pandas as pd
from scipy.stats import mannwhitneyu


def main():
    df = pd.read_excel("data/raw/tapeseq/moesm5.xlsx", header=None)
    sites = []
    target = None
    for i in range(len(df)):
        c0 = str(df.iloc[i, 0])
        if c0.endswith("TAPE-seq average read depth"):
            target = c0.split(" ")[0]
            continue
        if c0.startswith("on-target") or "-off" in c0:
            row = df.iloc[i]
            reads = pd.to_numeric(row.iloc[8], errors="coerce")
            sites.append({
                "block": target, "site": c0,
                "mm": pd.to_numeric(row.iloc[5], errors="coerce"),
                "bulge": str(row.iloc[6]),  # X = none
                "bulge_size": pd.to_numeric(row.iloc[7], errors="coerce"),
                "reads": reads})
    s = pd.DataFrame(sites)
    off = s[s.site.str.contains("-off")].copy()
    out = {"n_sites": int(len(s)), "n_offtargets": int(len(off)),
           "blocks": sorted(off.block.dropna().unique().tolist()),
           "n_bulge_rna": int((off.bulge == "RNA").sum()),
           "n_bulge_dna": int((off.bulge == "DNA").sum()),
           "n_nobulge": int((off.bulge == "X").sum())}
    nb = off[off.bulge == "X"].reads.dropna()
    bu = off[off.bulge.isin(["RNA", "DNA"])].reads.dropna()
    out["reads_median_nobulge"] = float(nb.median())
    out["reads_median_bulge"] = float(bu.median())
    out["frac_zero_reads_nobulge"] = float((nb == 0).mean())
    out["frac_zero_reads_bulge"] = float((bu == 0).mean())
    out["mwu_p_bulge_vs_none"] = float(mannwhitneyu(nb, bu).pvalue)
    # mismatch distribution
    out["mm_dist_offtargets"] = off.mm.value_counts().sort_index().to_dict()
    json.dump(out, open("results/tapeseq_bulges.json", "w"), indent=1)
    print(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
