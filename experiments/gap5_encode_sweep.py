"""Gap 5 ENCODE per-modality sweep: 7 K562 hg19 tracks (H3K4me3, H3K4me1,
H3K36me3, H3K27me3, H3K9me3, ATAC-seq, DNase-seq-pval) and 1 HEK293T GRCh38
ATAC-seq track, scored at crisprSQL loci for the matching cell line.
Readouts per track: pos/neg AUROC + MWU p, Spearman on positives."""
import json

import numpy as np
import pandas as pd
import pyBigWig
from scipy.stats import mannwhitneyu, spearmanr

K562 = {
    "H3K4me3_ENCFF291SWG": "data/encode_sweep/ENCFF291SWG.bigWig",
    "H3K4me1_ENCFF526QTS": "data/encode_sweep/ENCFF526QTS.bigWig",
    "H3K36me3_ENCFF745HXR": "data/encode_sweep/ENCFF745HXR.bigWig",
    "H3K27me3_ENCFF312LYO": "data/encode_sweep/ENCFF312LYO.bigWig",
    "H3K9me3_ENCFF834YLI": "data/encode_sweep/ENCFF834YLI.bigWig",
    "ATAC_ENCFF915UDE": "data/encode_sweep/ENCFF915UDE.bigWig",
    "DNase_pval_ENCFF643IZX": "data/encode_sweep/ENCFF643IZX.bigWig",
}
HEK = {"ATAC_ENCFF128DKF": "data/encode_sweep/ENCFF128DKF.bigWig"}


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
    return {"n_ok": int(ok.sum()), "auroc": auroc(pos, neg),
            "mwu_p": float(mannwhitneyu(pos, neg).pvalue),
            "spearman_pos": float(sp.statistic), "spearman_pos_p": float(sp.pvalue)}


def main():
    df = pd.read_csv("data/crisprsql/100720.csv", low_memory=False)
    k = df[df.cell_line.str.contains("K562", na=False)].reset_index(drop=True)
    h = df[df.cell_line.str.contains("HEK", na=False) & (df.genome == "hg38")].reset_index(drop=True)
    out = {"k562_n": int(len(k)), "hek293t_n": int(len(h)), "tracks": {}}
    for name, path in {**K562, **HEK}.items():
        sub = k if name in K562 else h
        out["tracks"][name] = score(sub, path)
        print(name, out["tracks"][name])
    json.dump(out, open("results/gap5_encode_sweep.json", "w"), indent=1)


if __name__ == "__main__":
    main()
