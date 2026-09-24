"""Gap 4: PAM-variant activity structure from Kim 2020 Nat Biotechnol
(DeepSpCas9variants, 10.1038/s41587-020-0537-9, MOESM3 PAM determination
sheet): 8,130 guide x PAM rows with background-subtracted indel
frequencies for 13 Cas9 variants.
Questions: (a) PAM-class activity per variant (NGG vs NAG/NGA vs non-NG),
(b) variant activity correlation structure: do high-fidelity variants
cluster separately from PAM-expansion variants?"""
import json

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

VARIANTS = ["SpCas9", "xCas9", "SpCas9-NG", "eSpCas9(1.1)", "SpCas9-HF1",
            "HypaCas9", "evoCas9", "Sniper-Cas9", "VRQR-HF1", "VQR",
            "VRQR", "VRER", "QQR1"]


def pam_class(pam):
    # library PAMs are 6-mers; the canonical PAM dinucleotide sits at
    # positions 2-3 (e.g. AGGTTT -> GG). Classify on that dinucleotide.
    p = str(pam).upper()
    if len(p) >= 3:
        di = p[1:3]
        if di == "GG":
            return "NGG"
        if di[0] == "G":
            return "NG" + di[1]
        if di == "AG":
            return "NAG"
        if di[0] == "A":
            return "NA" + di[1]
        return "N" + di
    return "other"


def main():
    df = pd.read_excel("data/raw/deepspcas9variants/moesm3.xlsx",
                       sheet_name="PAM determination")
    df["pam_class"] = df["PAM sequence"].map(pam_class)
    out = {"n_rows": int(len(df)), "variants": {}, "pam_classes": {}}
    # per-variant PAM-class means
    freq_cols = {v: f"Background subtracted indel frequencies\n(%, {v}, Day 4)" for v in VARIANTS}
    for v, col in freq_cols.items():
        x = pd.to_numeric(df[col], errors="coerce")
        ok = x.notna()
        by = df[ok].groupby("pam_class")[col].mean()
        out["variants"][v] = {k: round(float(val), 2) for k, val in by.items()}
    # PAM classes present
    out["pam_classes"] = df.pam_class.value_counts().to_dict()
    # variant-variant Spearman on complete rows
    mat = df[[c for c in freq_cols.values()]].apply(pd.to_numeric, errors="coerce")
    comp = mat.dropna()
    out["n_complete"] = int(len(comp))
    corr = comp.corr(method="spearman")
    out["spearman_matrix"] = {v: {w: round(float(corr.loc[freq_cols[v], freq_cols[w]]), 3)
                                  for w in VARIANTS} for v in VARIANTS}
    # WT vs each variant
    wt = freq_cols["SpCas9"]
    out["wt_correlations"] = {v: round(float(corr.loc[wt, freq_cols[v]]), 3)
                              for v in VARIANTS}
    json.dump(out, open("results/gap4_pam_variants.json", "w"), indent=1)
    print(json.dumps({k: out[k] for k in ["n_rows", "n_complete", "pam_classes", "wt_correlations"]}, indent=1))
    print("NGG means:", {v: out["variants"][v].get("NGG") for v in VARIANTS})


if __name__ == "__main__":
    main()


def figure():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    df = pd.read_excel("data/raw/deepspcas9variants/moesm3.xlsx",
                       sheet_name="PAM determination")
    df["pam_class"] = df["PAM sequence"].map(pam_class)
    order = ["NGG", "NGA", "NGC", "NGT", "NAG", "NAA", "NAC", "NAT",
             "NCA", "NCC", "NCG", "NCT", "NTA", "NTC", "NTG", "NTT"]
    mat = np.full((len(VARIANTS), len(order)), np.nan)
    for i, v in enumerate(VARIANTS):
        col = f"Background subtracted indel frequencies\n(%, {v}, Day 4)"
        x = pd.to_numeric(df[col], errors="coerce")
        by = df[x.notna()].groupby("pam_class")[col].mean()
        for j, pc in enumerate(order):
            if pc in by:
                mat[i, j] = by[pc]
    fig, ax = plt.subplots(figsize=(7.5, 4.5))
    im = ax.imshow(np.log10(np.clip(mat, 0.01, None)), aspect="auto", cmap="viridis")
    ax.set_xticks(range(len(order))); ax.set_xticklabels(order, rotation=90, fontsize=7)
    ax.set_yticks(range(len(VARIANTS))); ax.set_yticklabels(VARIANTS, fontsize=7)
    cb = fig.colorbar(im, ax=ax, ticks=[-2, 0, 1, 2])
    cb.ax.set_yticklabels(["0.01", "1", "10", "100"])
    cb.set_label("mean indel % (log10)", fontsize=8)
    ax.set_title("PAM-variant activity (Kim 2020 MOESM3)", fontsize=9)
    fig.tight_layout()
    fig.savefig("papers/figs/fig_pam_variant_heatmap.png", dpi=200)
    fig.savefig("papers/fig_pam_variant_heatmap.pdf")


if __name__ == "__main__" or True:
    pass
