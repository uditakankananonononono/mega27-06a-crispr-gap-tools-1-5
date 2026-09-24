"""UMAP embedding of guide sequences (k-mer features) colored by source
dataset: is the transfer failure a visible covariate-shift artifact?
Subsamples each of the 12 gap-1 datasets, embeds with UMAP, saves figure."""
import json

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import umap

from crisprgap.data.datasets import (load_crispron, load_crisprscan,
                                     load_deepcrispr, load_deepspcas9,
                                     load_deephf, load_depmap_efficacy,
                                     load_doench_fcres, load_doench_v1,
                                     load_sgdesigner)
from crisprgap.sequence import one_hot

LOADERS = {
    "fcres": load_doench_fcres, "v1": load_doench_v1,
    "deephf_wt": lambda: load_deephf("wt"),
    "deepspcas9": load_deepspcas9, "crispron": load_crispron,
    "sgdesigner": load_sgdesigner, "crisprscan": load_crisprscan,
    "dc_hct116": lambda: load_deepcrispr("hct116"),
    "dc_hek293t": lambda: load_deepcrispr("hek293t"),
    "dc_hela": lambda: load_deepcrispr("hela"),
    "dc_hl60": lambda: load_deepcrispr("hl60"),
    "depmap": load_depmap_efficacy,
}


def main():
    rng = np.random.default_rng(0)
    per = 600
    feats, labels = [], []
    for name, fn in LOADERS.items():
        ds = fn()
        n = min(per, len(ds.sequences))
        idx = rng.choice(len(ds.sequences), n, replace=False)
        seqs = [ds.sequences[i] for i in idx]
        feats.append(np.stack([one_hot(s, 30).reshape(-1) for s in seqs]).astype(np.float32))
        labels += [name] * n
    X = np.concatenate(feats)
    labels = np.array(labels)
    emb = umap.UMAP(n_neighbors=30, min_dist=0.3, random_state=0).fit_transform(X)
    fig, ax = plt.subplots(figsize=(6.5, 5))
    for name in LOADERS:
        m = labels == name
        ax.scatter(emb[m, 0], emb[m, 1], s=3, alpha=0.5, label=name)
    ax.legend(markerscale=3, fontsize=6, loc="best", framealpha=0.8)
    ax.set_xticks([]); ax.set_yticks([])
    ax.set_title("UMAP of guide sequence features (600/dataset)")
    fig.tight_layout()
    fig.savefig("papers/fig_umap_covariate.pdf")
    json.dump({"n_per_dataset": per, "umap_params": {"n_neighbors": 30,
               "min_dist": 0.3, "random_state": 0}},
              open("results/umap_covariate_meta.json", "w"), indent=2)
    print("saved papers/fig_umap_covariate.pdf")


if __name__ == "__main__":
    main()
