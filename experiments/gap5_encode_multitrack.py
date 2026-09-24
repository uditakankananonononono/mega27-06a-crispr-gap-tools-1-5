"""Gap 5 multi-track ENCODE chromatin screen over crisprSQL K562 hg19 loci.

Queries four ENCODE signal tracks via the UCSC REST API at a seeded
400-locus subsample of the 1,271 K562 hg19 crisprSQL rows, and measures
whether local chromatin signal separates active from inactive off-targets
(per-track AUROC). Tracks: H3K4me3 (active promoter), H3K27ac (active
enhancer), H3K9me3 (heterochromatin), DNase (open chromatin).
Self-contained: re-queries H3K4me3 so all four tracks share the identical
locus set. Signal = mean value over locus +/- 500 bp.
Run: python3 experiments/gap5_encode_multitrack.py
"""
import json
import os
import subprocess

import numpy as np
import pandas as pd
from scipy.stats import mannwhitneyu

TRACKS = {
    "h3k4me3": "wgEncodeBroadHistoneK562H3k4me3StdSig",
    "h3k27ac": "wgEncodeBroadHistoneK562H3k27acStdSig",
    "h3k9me3": "wgEncodeBroadHistoneK562H3k9me3StdSig",
    "dnase": "wgEncodeOpenChromDnaseK562SigV2",
}
API = "https://api.genome.ucsc.edu/getData/track?genome=hg19;track={tr};chrom={c};start={s};end={e}"
RAW = "/tmp/gap5_raw"


def fetch_all(cmds):
    os.makedirs(RAW, exist_ok=True)
    with open("/tmp/gap5_jobs.txt", "w") as fh:
        for tag, url in cmds:
            fh.write(f"{tag} {url}\n")
    # 8-way parallel curl; output to RAW/<tag>.json
    subprocess.run(
        f"xargs -P8 -n2 bash -c 'curl -s --max-time 25 \"$1\" -o {RAW}/$0.json' < /tmp/gap5_jobs.txt",
        shell=True, check=True)


def parse_one(path):
    try:
        d = json.load(open(path))
        for k, v in d.items():
            if isinstance(v, list):
                vals = [r["value"] for r in v]
                return float(np.mean(vals)) if vals else np.nan
    except Exception:
        pass
    return np.nan


def main():
    df = pd.read_csv("data/crisprsql/100720.csv", low_memory=False)
    k = df[(df.cell_line == "K562") & (df.genome == "hg19")].reset_index(drop=True)
    rng = np.random.default_rng(0)
    sub = k.iloc[np.sort(rng.choice(len(k), 400, replace=False))]
    loci = [(r.target_chr, int(r.target_start), int(r.target_end)) for r in sub.itertuples()]
    labels = (sub.cleavage_freq > 0).astype(int).to_numpy()

    cmds = []
    for tkey, track in TRACKS.items():
        for i, (c, s, e) in enumerate(loci):
            cmds.append((f"{tkey}__{i}", API.format(tr=track, c=c, s=max(0, s - 500), e=e + 500)))
    fetch_all(cmds)

    res = {"tracks": TRACKS, "rows": "crisprSQL K562 hg19, seed-0 400-locus subsample",
           "labels_positive": int(labels.sum()), "per_track": {}}
    for tkey in TRACKS:
        sig = np.array([parse_one(f"{RAW}/{tkey}__{i}.json") for i in range(len(loci))])
        ok = ~np.isnan(sig)
        y, s = labels[ok], sig[ok]
        pos, neg = s[y == 1], s[y == 0]
        # AUROC via rank statistic (identical to sklearn roc_auc_score)
        order = np.argsort(np.concatenate([pos, neg]))
        ranks = np.empty(len(order)); ranks[order] = np.arange(1, len(order) + 1)
        # average ties
        vals = np.concatenate([pos, neg])
        _, inv, cnt = np.unique(vals, return_inverse=True, return_counts=True)
        sums = np.zeros(len(cnt)); np.add.at(sums, inv, ranks)
        ranks = sums[inv] / cnt[inv]
        auroc = (ranks[:len(pos)].sum() - len(pos) * (len(pos) + 1) / 2) / (len(pos) * len(neg))
        mw = mannwhitneyu(pos, neg, alternative="two-sided").pvalue if len(neg) >= 3 else None
        res["per_track"][tkey] = {
            "n_ok": int(ok.sum()), "n_pos": int(len(pos)), "n_neg": int(len(neg)),
            "auroc": float(auroc), "median_pos": float(np.median(pos)),
            "median_neg": float(np.median(neg)) if len(neg) else None,
            "mannwhitney_p": float(mw) if mw is not None else None}
    json.dump(res, open("results/gap5_encode_multitrack.json", "w"), indent=2)
    for t, r in res["per_track"].items():
        print(t, "AUROC", round(r["auroc"], 3), "n", r["n_ok"], "pos/neg", r["n_pos"], r["n_neg"],
              "med", round(r["median_pos"], 3), "/", round(r["median_neg"] or -1, 3),
              "p", r["mannwhitney_p"])


if __name__ == "__main__":
    main()
