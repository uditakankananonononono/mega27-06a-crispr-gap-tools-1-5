"""Gap 5 ENCODE sweep part 3: 5 further K562 histone marks + 4 HeLa-S3
histone marks (hg19 fold-change bigWigs) at matching crisprSQL loci."""
import json

import numpy as np
import pandas as pd
import pyBigWig
from scipy.stats import mannwhitneyu, spearmanr

K562 = {
    "H3K4me2_ENCFF118MMT": "data/encode_sweep/ENCFF118MMT.bigWig",
    "H3K9ac_ENCFF866KTJ": "data/encode_sweep/ENCFF866KTJ.bigWig",
    "H3K79me2_ENCFF003CLZ": "data/encode_sweep/ENCFF003CLZ.bigWig",
    "H4K20me1_ENCFF143CUR": "data/encode_sweep/ENCFF143CUR.bigWig",
    "H2AFZ_ENCFF191EXE": "data/encode_sweep/ENCFF191EXE.bigWig",
}
HELA = {
    "HeLa_H3K4me3_ENCFF699TXY": "data/encode_sweep/ENCFF699TXY.bigWig",
    "HeLa_H3K9me3_ENCFF761QZP": "data/encode_sweep/ENCFF761QZP.bigWig",
    "HeLa_H3K27me3_ENCFF484EAZ": "data/encode_sweep/ENCFF484EAZ.bigWig",
    "HeLa_H3K36me3_ENCFF559FSM": "data/encode_sweep/ENCFF559FSM.bigWig",
}


def auroc(pos, neg):
    vals = np.concatenate([pos, neg])
    order = np.argsort(vals, kind="stable")
    ranks = np.empty(len(vals)); ranks[order] = np.arange(1, len(vals) + 1)
    _, inv, cnt = np.unique(vals, return_inverse=True, return_counts=True)
    sums = np.zeros(len(cnt)); np.add.at(sums, inv, ranks)
    ranks = sums[inv] / cnt[inv]
    return float((ranks[:len(pos)].sum() - len(pos) * (len(pos) + 1) / 2) / (len(pos) * len(neg)))


def score(h, path):
    y = (h.cleavage_freq > 0).astype(int).to_numpy()
    cleav = h.cleavage_freq.to_numpy()
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
    if len(pos) < 5 or len(neg) < 5:
        return {"n_ok": int(ok.sum()), "note": "too few for test"}
    sp = spearmanr(sig[ok & (y == 1)], cleav[ok & (y == 1)])
    return {"n_ok": int(ok.sum()), "n_pos": int(len(pos)), "n_neg": int(len(neg)),
            "auroc": auroc(pos, neg), "mwu_p": float(mannwhitneyu(pos, neg).pvalue),
            "spearman_pos": float(sp.statistic), "spearman_pos_p": float(sp.pvalue)}


def main():
    df = pd.read_csv("data/crisprsql/100720.csv", low_memory=False)
    k = df[df.cell_line.str.contains("K562", na=False)].reset_index(drop=True)
    hela = df[df.cell_line.str.contains("HeLa", na=False)].reset_index(drop=True)
    out = {"tracks": {}}
    for name, path in {**K562, **HELA}.items():
        sub = hela if name.startswith("HeLa") else k
        out["tracks"][name] = score(sub, path)
        print(name, out["tracks"][name])
    json.dump(out, open("results/gap5_encode_sweep3.json", "w"), indent=1)


if __name__ == "__main__":
    main()
