"""Sequence logos of extreme-portability guides (DeepHF WT screen).

Reconstructs the PORTA pipeline end-to-end: FC+RES ridge base model ->
per-guide rank residuals on the full WT screen -> PortabilityModel on the
11 DeepHF biofeatures -> per-guide portability scores. Logos show the 500
most portable vs 500 least portable guides (20-mer protospacer, positions
4:24 of the 30-mer context).
"""
import pickle

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import logomaker
from scipy.stats import rankdata, spearmanr
from sklearn.linear_model import Ridge

from crisprgap.data.datasets import load_doench_fcres
from crisprgap.porta import fit_portability
from crisprgap.sequence import one_hot

rng = np.random.default_rng(0)

# base model: FC+RES ridge on one-hot 30-mers (gap-1 recipe)
fc = load_doench_fcres()
Xf = np.stack([one_hot(s, 30) for s in fc.sequences]).reshape(len(fc.sequences), -1)
base = Ridge(alpha=1.0).fit(Xf, fc.scores)

with open("data/raw/public_data_crisprCas9/data/deepHF/wt_seq_data_array.pkl", "rb") as fh:
    seq_int, biofeat, indel = pickle.load(fh)
_INT2BASE = {1: "A", 2: "C", 3: "G", 4: "T", 0: "N", 5: "N"}
seqs = ["".join(_INT2BASE[b] for b in row) for row in seq_int]
biofeat = np.asarray(biofeat, dtype=np.float32)
measured = np.clip(np.asarray(indel, dtype=np.float32), 0, 1)

Xw = np.stack([one_hot(s, 30) for s in seqs]).reshape(len(seqs), -1)
pred = base.predict(Xw)
resid = rankdata(pred) - rankdata(measured)

# portability features: 11 biofeatures + one-hot sequence context (paper recipe)
FEAT = np.hstack([biofeat, Xw])
port = fit_portability(FEAT, pred, measured)
score = port.score(FEAT)

# held-out honesty check: split halves, fit on A, score-vs-residual Spearman on B
half = len(seqs) // 2
perm = rng.permutation(len(seqs))
A, B = perm[:half], perm[half:]
portA = fit_portability(FEAT[A], pred[A], measured[A])
rhoB = spearmanr(portA.score(FEAT[B]), resid[B]).statistic

valid = np.array([len(s) >= 20 and all(b in "ACGT" for b in s[:20]) for s in seqs])
order_hi = np.argsort(-score)[valid[np.argsort(-score)]]
order_lo = np.argsort(score)[valid[np.argsort(score)]]
hi = order_hi[:500]
lo = order_lo[:500]


def counts(idxs):
    mat = np.zeros((20, 4))
    for i in idxs:
        for p, b in enumerate(seqs[i][:20]):
            if b in "ACGT":
                mat[p, "ACGT".index(b)] += 1
    rs = mat.sum(axis=1, keepdims=True)
    assert (rs > 0).all(), "position with zero ACGT counts"
    return mat / rs


fig, axes = plt.subplots(2, 1, figsize=(8.5, 3.6))
for ax, idxs, title in zip(axes, [hi, lo],
                           ["500 most portable guides (lowest predicted transfer loss)",
                            "500 least portable guides (highest predicted transfer loss)"]):
    import pandas as pd
    df = pd.DataFrame(counts(idxs), columns=list("ACGT"))
    logomaker.Logo(df, ax=ax, shade_below=0.5, fade_below=0.5)
    ax.set_title(title, fontsize=9)
    ax.set_ylabel("bits" if False else "frequency")
fig.tight_layout()
fig.savefig("papers/figs/fig_porta_logo.png", dpi=200)
print("held-out score-vs-residual Spearman:", round(float(rhoB), 4))
print("figure saved")
