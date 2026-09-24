"""Heatmap figure of the 14x14 ridge transfer matrix (seaborn)."""
import json

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

raw = json.load(open("results/crossdata_ridge_14ds.json"))["raw"]
names = sorted(raw["0"]["ridge"].keys())
M = np.zeros((len(names), len(names))); cnt = np.zeros_like(M)
for seed, models in raw.items():
    for tr, res in models["ridge"].items():
        M[names.index(tr), names.index(tr)] += res["within"]
        for te, v in res["cross"].items():
            M[names.index(tr), names.index(te)] += v
            cnt[names.index(tr), names.index(te)] += 1
M = np.divide(M, np.maximum(cnt, 1), where=cnt > 0)
for i, n in enumerate(names):
    M[i, i] = np.mean([raw[s]["ridge"][n]["within"] for s in raw])
fig, ax = plt.subplots(figsize=(6.2, 5.4))
sns.heatmap(M, xticklabels=names, yticklabels=names, cmap="RdBu_r",
            center=0, vmin=-0.35, vmax=0.75, square=True,
            cbar_kws={"label": "Spearman (train row -> test col)"}, ax=ax,
            annot=True, fmt=".2f", annot_kws={"size": 5})
ax.set_title("Cross-dataset transfer matrix (ridge, 3 seeds)")
fig.tight_layout()
fig.savefig("papers/fig_transfer_heatmap.pdf")
print("saved")
