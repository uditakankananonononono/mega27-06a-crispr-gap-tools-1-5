"""Gap-1 leakage audit: do different efficacy datasets share identical or
near-identical (edit distance <= 1) 20-mer guides? If so, cross-dataset
transfer scores could be inflated by leakage. rapidfuzz cdist on unique
20-mers per dataset pair; counts + the worst offenders.
"""
import json

import numpy as np
from rapidfuzz.distance import Levenshtein
from rapidfuzz.process import cdist

from crisprgap.data import datasets as D

LOADERS = {
    "fcres": D.load_doench_fcres,
    "v1": D.load_doench_v1,
    "crisprscan": D.load_crisprscan,
    "sgdesigner": D.load_sgdesigner,
    "sgrnascorer": D.load_sgrnascorer,
}


def guides20(ds):
    out = []
    for s in ds.sequences:
        s = s.upper()
        # take the 20-mer guide region: for 30-mers (Azimuth convention)
        # positions 4:24; shorter sequences used as-is from the left
        out.append(s[4:24] if len(s) >= 30 else s[:20])
    return sorted(set(out))


def main():
    seqs = {name: guides20(loader()) for name, loader in LOADERS.items()}
    names = list(seqs)
    out = {"dataset_sizes": {n: len(s) for n, s in seqs.items()}, "pairs": {}}
    for i, a in enumerate(names):
        for b in names[i + 1:]:
            D1 = cdist(seqs[a], seqs[b], scorer=Levenshtein.distance,
                       score_cutoff=1, workers=4)
            exact = int((D1 == 0).sum())
            near = int((D1 == 1).sum())
            out["pairs"][f"{a}|{b}"] = {"identical_20mers": exact,
                                        "edit1_pairs": near}
    json.dump(out, open("results/leakage_audit.json", "w"), indent=2)
    print(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()
