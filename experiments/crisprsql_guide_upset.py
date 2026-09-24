"""UpSet plot of guide-sharing across crisprSQL studies: how many guides
are measured by multiple studies (the dependence structure behind the
guide-clustering corrections in the gap-2 analysis)."""
import json

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
from upsetplot import UpSet, from_contents


def main():
    df = pd.read_csv("data/crisprsql/100720.csv", low_memory=False)
    sets = {s: set(g) for s, g in df.groupby("study_name").grna_target_sequence}
    sets = {s: g for s, g in sets.items() if len(g) >= 3}
    data = from_contents(sets)
    fig = plt.figure(figsize=(7.5, 4.2))
    UpSet(data, subset_size="count", sort_by="cardinality",
          min_subset_size=1, show_counts=True).plot(fig)
    fig.savefig("papers/fig_guide_upset.pdf", bbox_inches="tight")
    multi = df.groupby("grna_target_sequence").study_name.nunique()
    out = {
        "studies_plotted": len(sets),
        "n_unique_guides": int(multi.size),
        "guides_in_2plus_studies": int((multi >= 2).sum()),
        "frac_shared": float((multi >= 2).mean()),
    }
    json.dump(out, open("results/guide_upset.json", "w"), indent=2)
    print(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()
