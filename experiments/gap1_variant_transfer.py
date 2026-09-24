"""Gap 1 across enzymes: does a WT-SpCas9-trained sequence model transfer
to engineered Cas9 variants? Kim 2020 (DeepSpCas9variants) main libraries:
MOESM4 (23,999 guides, (G/g)N19) and MOESM5 (7,999 guides, N20), each with
background-subtracted indel frequencies for 13 variants.
Ridge on mono+dinucleotide one-hot; WT->variant and variant->WT transfer."""
import json

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

VARIANTS = ["SpCas9", "xCas9", "SpCas9-NG", "eSpCas9(1.1)", "SpCas9-HF1",
            "HypaCas9", "evoCas9", "Sniper-Cas9", "VRQR-HF1", "VQR",
            "VRQR", "VRER", "QQR1"]
FCOL = {v: f"Background subtracted indel frequencies\n(%, {v}, Day 4)" for v in VARIANTS}

BASES = "ACGT"


def encode(seqs):
    n = len(seqs)
    L = len(seqs.iloc[0])
    mono = np.zeros((n, L * 4))
    di = np.zeros((n, (L - 1) * 16))
    bidx = {b: i for i, b in enumerate(BASES)}
    didx = {a + b: i for i, (a, b) in enumerate((x + y for x in BASES for y in BASES))}
    for i, s in enumerate(seqs):
        s = str(s).upper()
        for p, ch in enumerate(s):
            if ch in bidx:
                mono[i, p * 4 + bidx[ch]] = 1
        for p in range(L - 1):
            d = s[p:p + 2]
            if d in didx:
                di[i, (L - 1 - 0) * 0 + p * 16 + didx[d]] = 1
    return np.hstack([mono, di])


def ridge_fit(X, y, lam=10.0):
    Xm = X.mean(0); Xs = X.std(0); Xs[Xs == 0] = 1
    Xc = (X - Xm) / Xs
    yc = y - y.mean()
    w = np.linalg.solve(Xc.T @ Xc + lam * np.eye(X.shape[1]), Xc.T @ yc)
    return w, Xm, Xs, y.mean()


def ridge_pred(model, X):
    w, Xm, Xs, ym = model
    return ((X - Xm) / Xs) @ w + ym


def run(file, sheet_rows):
    df = pd.read_excel(file)
    variants = [c.split("%, ")[1].split(",")[0] for c in df.columns
                if "Background subtracted" in str(c)]
    df = df.dropna(subset=[[c for c in df.columns if "Guide RNA sequence" in c][0]])
    gcol = [c for c in df.columns if "Guide RNA sequence" in c][0]
    seqs = df[gcol].astype(str).str.upper()
    seqs = seqs.str.replace("G", "G", regex=False)
    X = encode(seqs)
    ywt = pd.to_numeric(df[FCOL["SpCas9"]], errors="coerce")
    out = {"n": int(len(df)), "wt_to_variant": {}, "variant_to_wt": {}}
    ok0 = ywt.notna().to_numpy()
    rng = np.random.RandomState(0)
    idx = rng.permutation(ok0.sum())
    cut = int(0.8 * len(idx))
    Xall = X[ok0]
    for v in variants:
        yv = pd.to_numeric(df[FCOL[v]], errors="coerce").to_numpy()[ok0]
        ok = ~np.isnan(yv)
        tr = idx[:cut]; te = idx[cut:]
        m_wt = ridge_fit(Xall[tr], ywt.to_numpy()[ok0][tr])
        p = ridge_pred(m_wt, Xall[te])
        ok_te = ~np.isnan(yv[te])
        out["wt_to_variant"][v] = round(float(spearmanr(p[ok_te], yv[te][ok_te]).statistic), 3)
        # variant -> WT
        m_v = ridge_fit(Xall[tr][ok[tr]], yv[tr][ok[tr]])
        pv = ridge_pred(m_v, Xall[te])
        ywt_te = ywt.to_numpy()[ok0][te]
        ok2 = ~np.isnan(ywt_te)
        out["variant_to_wt"][v] = round(float(spearmanr(pv[ok2], ywt_te[ok2]).statistic), 3)
    return out


def main():
    res = {}
    for name, f in [("moesm4", "data/raw/deepspcas9variants/moesm4.xlsx"),
                    ("moesm5", "data/raw/deepspcas9variants/moesm5.xlsx")]:
        res[name] = run(f, None)
        print(name, "wt->variant:", res[name]["wt_to_variant"])
    json.dump(res, open("results/gap1_variant_transfer.json", "w"), indent=1)


if __name__ == "__main__":
    main()
