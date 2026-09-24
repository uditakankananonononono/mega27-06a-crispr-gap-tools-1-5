"""pingouin effect sizes for the key distributional comparisons:
(1) Koike-Yusa NAG vs NGG cleavage (gap 4),
(2) K562 DNase signal, cleaved vs not (gap 5),
(3) HEK293T DNase signal, cleaved vs not (gap 5, SITE-Seq slice).
Rank-biserial r (Mann-Whitney effect size) + Cohen's d, with CIs.
"""
import json

import numpy as np
import pandas as pd
import pingouin as pg
import pyBigWig

from crisprgap.data.datasets import load_koike_yusa_miseq


def effect(x, y):
    m = pg.mwu(x, y)
    return {
        "rbc": float(m["RBC"].iloc[0]),
        "p": float(m["p_val"].iloc[0]),
        "cohens_d": float(pg.compute_effsize(x, y, eftype="cohen")),
    }


def main():
    out = {}
    ds = load_koike_yusa_miseq()
    y = ds.cleavage_freq
    ok = ~np.isnan(y)
    nag = np.array([p.upper().endswith("AG") and not p.upper().endswith("GG")
                    for p in ds.off_pam])[ok]
    yy = y[ok]
    out["koike_yusa_nag_vs_ngg"] = effect(yy[~nag], yy[nag])
    out["koike_yusa_nag_vs_ngg"]["n_ngg"] = int((~nag).sum())
    out["koike_yusa_nag_vs_ngg"]["n_nag"] = int(nag.sum())

    df = pd.read_csv("data/crisprsql/100720.csv", low_memory=False)
    for tag, cell, bw_path in (("k562", "K562", "data/encff526lys_k562_dnase_hg19.bigWig"),
                               ("hek293t", "HEK", "data/encff529bog_hek293t_dnase_grch38.bigWig")):
        h = df[df.cell_line.str.contains(cell, na=False)]
        if tag == "hek293t":
            h = h[h.genome == "hg38"]
        h = h.reset_index(drop=True)
        bw = pyBigWig.open(bw_path)
        sig = []
        for r in h.itertuples():
            try:
                s, e = int(r.target_start), int(r.target_end)
                v = bw.stats(str(r.target_chr), max(0, s - 500), e + 500, type="mean")
                sig.append(v[0] if v and v[0] is not None else np.nan)
            except (ValueError, TypeError, RuntimeError):
                sig.append(np.nan)
        bw.close()
        sig = np.array(sig)
        lab = (h.cleavage_freq > 0).to_numpy()
        m = ~np.isnan(sig)
        pos, neg = sig[m & (lab == 1)], sig[m & (lab == 0)]
        out[f"{tag}_dnase_cleaved_vs_not"] = effect(pos, neg)
        out[f"{tag}_dnase_cleaved_vs_not"]["n_pos"] = int(len(pos))
        out[f"{tag}_dnase_cleaved_vs_not"]["n_neg"] = int(len(neg))

    json.dump(out, open("results/effect_sizes_pingouin.json", "w"), indent=2)
    print(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()
