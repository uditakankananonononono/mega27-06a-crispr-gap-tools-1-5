"""Regenerate papers/figs/fig_ablation_bars.png from the 5-seed multiseed artifact."""
import json

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

d = json.load(open("results/offtarget_ablation_multiseed.json"))
conds = ["seq_only", "pam", "epigen", "pam_epigen"]
labels = ["seq only", "+PAM", "+chromatin", "+PAM+chromatin"]
overall = [(d["conditions"][c]["auroc_mean"], d["conditions"][c]["auroc_sd"]) for c in conds]
nonngg = [(d["conditions"][c]["auroc_non_ngg_mean"], d["conditions"][c]["auroc_non_ngg_sd"]) for c in conds]

x = np.arange(len(conds))
w = 0.38
fig, ax = plt.subplots(figsize=(5.5, 3.2))
ax.bar(x - w/2, [m for m, _ in overall], w, yerr=[s for _, s in overall],
       capsize=3, label="overall AUROC", color="#4C72B0")
ax.bar(x + w/2, [m for m, _ in nonngg], w, yerr=[s for _, s in nonngg],
       capsize=3, label="non-NGG AUROC", color="#DD8452")
ax.axhline(0.5, ls="--", lw=0.8, color="grey")
ax.set_xticks(x); ax.set_xticklabels(labels, fontsize=9)
ax.set_ylabel("AUROC (mean +/- sd, 5 seeded splits)")
ax.set_ylim(0.35, 0.75)
ax.legend(fontsize=8)
fig.tight_layout()
fig.savefig("papers/figs/fig_ablation_bars.png", dpi=200)
print("figure written; seeds:", d["seeds"])
