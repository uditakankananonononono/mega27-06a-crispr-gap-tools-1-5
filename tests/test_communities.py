import json

import networkx as nx
import numpy as np
import pytest
from networkx.algorithms.community import louvain_communities, modularity


def _sym_matrix():
    raw = json.load(open("results/crossdata_ridge_11ds.json"))["raw"]
    names = sorted(raw["0"]["ridge"].keys())
    M = np.zeros((len(names), len(names)))
    cnt = np.zeros_like(M)
    for seed, models in raw.items():
        for tr, res in models["ridge"].items():
            for te, v in res["cross"].items():
                i, j = names.index(tr), names.index(te)
                M[i, j] += v
                cnt[i, j] += 1
    S = np.maximum(0.0, (np.divide(M, cnt, where=cnt > 0) +
                         np.divide(M, cnt, where=cnt > 0).T) / 2.0)
    np.fill_diagonal(S, 0.0)
    return names, S


@pytest.mark.skipif(not __import__("os").path.exists("results/crossdata_ridge_11ds.json"),
                    reason="matrix results not present")
def test_two_communities_stable():
    names, S = _sym_matrix()
    for seed in (0, 1, 2):
        G = nx.from_numpy_array(S)
        G = nx.relabel_nodes(G, {i: names[i] for i in range(len(names))})
        comms = louvain_communities(G, weight="weight", seed=seed, resolution=1.0)
        assert len(comms) == 2
        assert modularity(G, comms, weight="weight") > 0.2
