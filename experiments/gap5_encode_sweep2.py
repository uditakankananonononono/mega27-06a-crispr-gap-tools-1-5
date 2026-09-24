"""Gap 5 ENCODE sweep part 2: HeLa-S3 (4 tracks) and HAP1 (2 tracks)
hg19 bigWigs scored at crisprSQL loci of the matching cell line.
U2OS (5,920 hg19 loci, the largest slice) has NO ENCODE chromatin track
of any assay/assembly - recorded as a coverage gap."""
import json

import numpy as np
import pandas as pd
import pyBigWig
from scipy.stats import mannwhitneyu, spearmanr

HELA = {
    "HeLa_DNase_ENCFF426GCA": "data/encode_sweep/ENCFF426GCA.bigWig",
    "HeLa_ATAC_ENCFF051PGV": "data/encode_sweep/ENCFF051PGV.bigWig",
    "HeLa_H3K27ac_ENCFF388WMD": "data/encode_sweep/ENCFF388WMD.bigWig",
    "HeLa_CTCF_ENCFF799KLZ": "data/encode_sweep/ENCFF799KLZ.bigWig",
}
HAP1 = {
    "HAP1_DNase_ENCFF095QLZ": "data/encode_sweep/ENCFF095QLZ.bigWig",
    "HAP1_ATAC_ENCFF559JGD": "data/encode_sweep/ENCFF559JGD.bigWig",
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
        return {"n_ok": int(ok.sum()), "n_pos": int((ok & (y == 1)).sum()),
                "note": "too few for test"}
    sp = spearmanr(sig[ok & (y == 1)], cleav[ok & (y == 1)])
    return {"n_ok": int(ok.sum()), "n_pos": int(len(pos)), "n_neg": int(len(neg)),
            "auroc": auroc(pos, neg), "mwu_p": float(mannwhitneyu(pos, neg).pvalue),
            "spearman_pos": float(sp.statistic), "spearman_pos_p": float(sp.pvalue)}


def main():
    df = pd.read_csv("data/crisprsql/100720.csv", low_memory=False)
    hela = df[df.cell_line.str.contains("HeLa", na=False)].reset_index(drop=True)
    hap1 = df[df.cell_line.str.contains("HAP1", na=False)].reset_index(drop=True)
    out = {"hela_n": int(len(hela)), "hap1_n": int(len(hap1)),
           "u2os_encode_coverage": "none (no ENCODE bigWig for U2OS, any assay/assembly)",
           "tracks": {}}
    for name, path in {**HELA, **HAP1}.items():
        sub = hela if name.startswith("HeLa") else hap1
        out["tracks"][name] = score(sub, path)
        print(name, out["tracks"][name])
    json.dump(out, open("results/gap5_encode_sweep2.json", "w"), indent=1)


if __name__ == "__main__":
    main()
