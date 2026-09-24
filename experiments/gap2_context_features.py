"""Gap 2: does genomic context around the off-target locus add signal over
guide/off-target sequence features alone? pyfaidx extracts +/-30/100 bp
context from local hg38 (UCSC bigZips) for the Cameron SITE-Seq hg38
slice; grouped-by-guide 3-seed logistic AUROC, seq-only vs seq+context.
"""
import json

import numpy as np
import pandas as pd
from pyfaidx import Fasta
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import GroupShuffleSplit

FA = "data/hg38.fa"


def gc(s):
    s = s.upper()
    n = sum(c in "ACGT" for c in s)
    return (s.count("G") + s.count("C")) / n if n else np.nan


def homopoly(s):
    best, run = 1, 1
    for a, b in zip(s, s[1:]):
        run = run + 1 if a == b else 1
        best = max(best, run)
    return best


def main():
    df = pd.read_csv("data/crisprsql/100720.csv", low_memory=False)
    h = df[df.cell_line.str.contains("HEK", na=False) & (df.genome == "hg38")].reset_index(drop=True)
    fa = Fasta(FA)
    rows = []
    for r in h.itertuples():
        try:
            s, e = int(r.target_start), int(r.target_end)
            chrom = str(r.target_chr)
            if chrom not in fa.keys():
                raise ValueError
            c30 = fa[chrom][max(0, s - 30):e + 30].seq
            c100 = fa[chrom][max(0, s - 100):e + 100].seq
            site = fa[chrom][s:e].seq
        except (ValueError, KeyError):
            rows.append((np.nan,) * 6)
            continue
        g, o = str(r.grna_target_sequence).upper(), str(r.target_sequence).upper()
        mm = sum(a != b for a, b in zip(g, o)) if len(g) == len(o) else np.nan
        rows.append((mm, gc(c30), gc(c100), homopoly(c30), gc(site), len(o)))
    fa.close()
    F = np.array(rows, dtype=float)
    ok = ~np.isnan(F).any(axis=1)
    F, h = F[ok], h[ok]
    y = (h.cleavage_freq > 0).astype(int).to_numpy()
    groups = h.grna_target_sequence.to_numpy()

    X_seq = F[:, [0]]                      # mismatch count only
    X_ctx = F[:, :]                        # + context features
    out = {"n": int(len(y)), "n_pos": int(y.sum()),
           "aurocs": {"mm_only": [], "mm_plus_context": []}}
    for seed in (0, 1, 2):
        tr, te = next(GroupShuffleSplit(1, test_size=0.25, random_state=seed).split(X_seq, y, groups))
        for name, X in (("mm_only", X_seq), ("mm_plus_context", X_ctx)):
            m = LogisticRegression(max_iter=1000).fit(X[tr], y[tr])
            out["aurocs"][name].append(float(roc_auc_score(y[te], m.predict_proba(X[te])[:, 1])))
    out["mean"] = {k: float(np.mean(v)) for k, v in out["aurocs"].items()}
    json.dump(out, open("results/gap2_context_features.json", "w"), indent=2)
    print(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()
