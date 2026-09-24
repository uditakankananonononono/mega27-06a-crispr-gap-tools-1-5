"""Gap 5, K562 multi-track: CTCF (ENCFF014YLT) and H3K27ac (ENCFF010PHG)
fold-change bigWigs (hg19) queried at all crisprSQL K562 loci alongside
the DNase track. Same readouts as the DNase analysis: pos/neg AUROC +
MWU, Spearman on positives.
"""
import json

import numpy as np
import pandas as pd
import pyBigWig
from scipy.stats import mannwhitneyu, spearmanr

TRACKS = {
    "ctcf_ENCFF014YLT": "data/encff014ylt_k562_ctcf_hg19.bigWig",
    "h3k27ac_ENCFF010PHG": "data/encff010phg_k562_h3k27ac_hg19.bigWig",
}


def auroc(pos, neg):
    vals = np.concatenate([pos, neg])
    order = np.argsort(vals, kind="stable")
    ranks = np.empty(len(vals)); ranks[order] = np.arange(1, len(vals) + 1)
    _, inv, cnt = np.unique(vals, return_inverse=True, return_counts=True)
    sums = np.zeros(len(cnt)); np.add.at(sums, inv, ranks)
    ranks = sums[inv] / cnt[inv]
    return float((ranks[:len(pos)].sum() - len(pos) * (len(pos) + 1) / 2) / (len(pos) * len(neg)))


def main():
    df = pd.read_csv("data/crisprsql/100720.csv", low_memory=False)
    h = df[df.cell_line.str.contains("K562", na=False)].reset_index(drop=True)
    y = (h.cleavage_freq > 0).astype(int).to_numpy()
    cleav = h.cleavage_freq.to_numpy()
    out = {"n": int(len(h)), "n_pos": int(y.sum()), "n_neg": int((1 - y).sum()),
           "tracks": {}}
    for name, path in TRACKS.items():
        bw = pyBigWig.open(path)
        sig = []
        for r in h.itertuples():
            try:
                s, e = int(r.target_start), int(r.target_end)
                v = bw.stats(str(r.target_chr), max(0, s - 500), e + 500, type="mean")
                sig.append(v[0] if v and v[0] is not None else np.nan)
            except (ValueError, TypeError, RuntimeError):
                sig.append(np.nan)
        bw.close()
        sig = np.array(sig)
        ok = ~np.isnan(sig)
        pos, neg = sig[ok & (y == 1)], sig[ok & (y == 0)]
        out["tracks"][name] = {
            "n_ok": int(ok.sum()),
            "auroc": auroc(pos, neg),
            "mannwhitney_p": float(mannwhitneyu(pos, neg).pvalue),
            "spearman_pos": float(spearmanr(sig[ok & (y == 1)], cleav[ok & (y == 1)]).statistic),
            "spearman_pos_p": float(spearmanr(sig[ok & (y == 1)], cleav[ok & (y == 1)]).pvalue),
        }
    json.dump(out, open("results/gap5_k562_multitrack.json", "w"), indent=2)
    print(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()
