"""Guide-guide off-target sharing network (networkx): nodes are crisprSQL
guides, an edge means two guides were measured against at least one common
off-target locus. Quantifies how interconnected the empirical off-target
evidence base is (gap 2)."""
import json

import networkx as nx
import pandas as pd


def main():
    df = pd.read_csv("data/crisprsql/100720.csv", low_memory=False)
    df = df.dropna(subset=["grna_target_sequence", "target_chr", "target_start"])
    # locus key per measured off-target site
    df["locus"] = (df.target_chr.astype(str) + ":" +
                   df.target_start.astype("Int64").astype(str))
    g = nx.Graph()
    for guide, sub in df.groupby("grna_target_sequence"):
        g.add_node(guide, n_loci=sub.locus.nunique())
    # edge between guides sharing a locus
    for locus, sub in df.groupby("locus"):
        guides = sub.grna_target_sequence.unique()
        for i in range(len(guides)):
            for j in range(i + 1, len(guides)):
                if g.has_edge(guides[i], guides[j]):
                    g[guides[i]][guides[j]]["n_shared"] += 1
                else:
                    g.add_edge(guides[i], guides[j], n_shared=1)
    comps = sorted((len(c) for c in nx.connected_components(g)), reverse=True)
    degs = [d for _, d in g.degree()]
    out = {
        "n_guides": g.number_of_nodes(),
        "n_edges": g.number_of_edges(),
        "n_components": len(comps),
        "largest_component": comps[0] if comps else 0,
        "isolated_guides": sum(1 for d in degs if d == 0),
        "max_degree": max(degs) if degs else 0,
        "mean_degree": float(sum(degs) / len(degs)) if degs else 0.0,
        "density": float(nx.density(g)),
    }
    json.dump(out, open("results/offtarget_network.json", "w"), indent=1)
    print(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
