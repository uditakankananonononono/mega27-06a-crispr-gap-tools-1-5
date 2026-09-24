"""Sequence logos of the Chari high- vs low-activity guides (20-mers).
Gap-1 figure: which positions carry the high/low signal."""
import json

import logomaker
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from crisprgap.data.datasets import load_sgrnascorer


def prob_df(seqs):
    arr = np.array([list(s) for s in seqs])
    return pd.DataFrame({p: {b: float((arr[:, p] == b).mean())
                             for b in "ACGT"} for p in range(arr.shape[1])}).T


def main():
    ds = load_sgrnascorer()
    hi = [s for s, y in zip(ds.sequences, ds.scores) if y == 1]
    lo = [s for s, y in zip(ds.sequences, ds.scores) if y == 0]
    fig, axes = plt.subplots(2, 1, figsize=(8.2, 3.4), sharex=True)
    for ax, seqs, title in ((axes[0], hi, "Chari high-activity (n=215)"),
                            (axes[1], lo, "Chari low-activity (n=215)")):
        logomaker.Logo(prob_df(seqs), ax=ax, color_scheme="classic")
        ax.set_ylabel("freq")
        ax.set_title(title, fontsize=9)
        ax.set_ylim(0, 0.62)
    axes[1].set_xlabel("guide position (5' to 3')")
    fig.tight_layout()
    fig.savefig("papers/fig_chari_logo.pdf")
    # positional max-frequency summary for the results file
    hi_p, lo_p = prob_df(hi), prob_df(lo)
    diff = (hi_p - lo_p).abs().max(axis=1)
    top = diff.sort_values(ascending=False).head(5)
    out = {"top_divergent_positions": [
        {"pos0": int(i), "max_abs_freq_diff": float(v),
         "hi_top_base": str(hi_p.loc[i].idxmax()), "lo_top_base": str(lo_p.loc[i].idxmax())}
        for i, v in top.items()]}
    with open("results/chari_logo.json", "w") as fh:
        json.dump(out, fh, indent=2)
    print(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()
