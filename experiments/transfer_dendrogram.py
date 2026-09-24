"""Hierarchical clustering (scipy, average linkage) on the 12x12 transfer
matrix's symmetrized positive-part similarity - dendrogram figure
complementing the Louvain community analysis."""
import json

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy.cluster.hierarchy import dendrogram, linkage
from scipy.spatial.distance import squareform

raw = json.load(open("results/crossdata_ridge_12ds.json"))["raw"]
names = sorted(raw["0"]["ridge"].keys())
M = np.zeros((len(names), len(names))); cnt = np.zeros_like(M)
for seed, models in raw.items():
    for tr, res in models["ridge"].items():
        for te, v in res["cross"].items():
            i, j = names.index(tr), names.index(te)
            M[i, j] += v; cnt[i, j] += 1
S = np.maximum(0.0, (np.divide(M, cnt, where=cnt > 0) + np.divide(M, cnt, where=cnt > 0).T) / 2.0)
np.fill_diagonal(S, 1.0)
D = 1.0 - S
np.fill_diagonal(D, 0.0)
Z = linkage(squareform(D, checks=False), method="average")
fig, ax = plt.subplots(figsize=(7, 3.2))
dendrogram(Z, labels=names, ax=ax, leaf_rotation=45, leaf_font_size=8,
           color_threshold=0.85)
ax.set_ylabel("1 - symmetrized transfer (positive part)")
ax.set_title("Assay-family transfer hierarchy (12 datasets, ridge, 3 seeds)")
fig.tight_layout()
fig.savefig("papers/fig_transfer_dendrogram.pdf")
json.dump({"linkage": Z.tolist(), "names": names},
          open("results/transfer_dendrogram.json", "w"), indent=2)
print("saved papers/fig_transfer_dendrogram.pdf")
