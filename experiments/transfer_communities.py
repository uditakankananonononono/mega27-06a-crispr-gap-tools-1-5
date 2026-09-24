"""Formalize the two-community claim from the 11x11 ridge transfer matrix:
Louvain community detection (networkx) on the symmetrized positive-part
transfer graph, plus modularity, 3 seeds (Louvain is stochastic)."""
import json

import networkx as nx
import numpy as np
from networkx.algorithms.community import louvain_communities, modularity


def main():
    raw = json.load(open("results/crossdata_ridge_11ds.json"))["raw"]
    names = sorted(raw["0"]["ridge"].keys())
    # mean transfer over seeds, symmetrized, positive part
    M = np.zeros((len(names), len(names)))
    cnt = np.zeros_like(M)
    for seed, models in raw.items():
        for tr, res in models["ridge"].items():
            for te, v in res["cross"].items():
                i, j = names.index(tr), names.index(te)
                if i != j:
                    M[i, j] += v
                    cnt[i, j] += 1
    M = np.divide(M, cnt, where=cnt > 0)
    S = np.maximum(0.0, (M + M.T) / 2.0)
    np.fill_diagonal(S, 0.0)
    G = nx.from_numpy_array(S)
    G = nx.relabel_nodes(G, {i: names[i] for i in range(len(names))})
    out = {"names": names, "sym_matrix": S.round(4).tolist(), "seeds": {}}
    for seed in (0, 1, 2):
        comms = louvain_communities(G, weight="weight", seed=seed, resolution=1.0)
        comms = sorted((sorted(c) for c in comms), key=len, reverse=True)
        out["seeds"][seed] = {
            "communities": comms,
            "modularity": round(modularity(G, comms, weight="weight"), 4),
        }
        print(f"seed {seed}: Q={out['seeds'][seed]['modularity']}")
        for c in comms:
            print("  ", c)
    json.dump(out, open("results/transfer_communities.json", "w"), indent=2)


if __name__ == "__main__":
    main()
