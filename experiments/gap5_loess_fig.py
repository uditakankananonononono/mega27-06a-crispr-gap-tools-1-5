"""Gap 5 dose-response figure: cleavage frequency vs DNase signal with a
LOESS trend (scikit-misc), K562 positives - is there ANY monotone
dose-response under the near-zero rank correlation?
"""
import json

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pyBigWig
from skmisc.loess import loess

BW = "data/encff526lys_k562_dnase_hg19.bigWig"


def main():
    df = pd.read_csv("data/crisprsql/100720.csv", low_memory=False)
    h = df[df.cell_line.str.contains("K562", na=False)].reset_index(drop=True)
    bw = pyBigWig.open(BW)
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
    y = h.cleavage_freq.to_numpy()
    m = ~np.isnan(sig) & ~np.isnan(y) & (y > 0)
    x, yy = sig[m], y[m]
    lo = loess(x, np.log10(yy + 1e-4), span=0.75)
    lo.fit()
    grid = np.linspace(np.percentile(x, 1), np.percentile(x, 99), 200)
    pred = lo.predict(grid).values

    fig, ax = plt.subplots(figsize=(5.5, 3.4))
    ax.scatter(x, np.log10(yy + 1e-4), s=3, alpha=0.25, color="grey", rasterized=True)
    ax.plot(grid, pred, color="firebrick", lw=2, label="LOESS (span 0.75)")
    ax.set_xlabel("K562 DNase signal (+/-500 bp mean)")
    ax.set_ylabel("log10(cleavage + 1e-4)")
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig("papers/fig_k562_loess.pdf")
    out = {
        "n_pos_plotted": int(len(x)),
        "loess_endpoint_slope": float((pred[-1] - pred[0]) / (grid[-1] - grid[0])),
        "loess_range": [float(pred.min()), float(pred.max())],
    }
    json.dump(out, open("results/gap5_loess.json", "w"), indent=2)
    print(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()
