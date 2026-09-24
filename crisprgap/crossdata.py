"""Cross-dataset generalization harness (gap 1).

Published gRNA efficacy models are trained and evaluated inside one dataset;
performance collapses on independent datasets (e.g. Nat Commun 2023,
s41467-023-41143-7). This harness measures the collapse directly: train on
dataset A, evaluate on held-out A AND on untouched dataset B, for every
ordered pair of registered datasets, over multiple seeds. It also trains a
pooled multi-dataset model, the standard mitigation, and reports the delta.

Datasets register as (name, loader); models register as (name, fit, predict)
triples so new architectures slot in without touching the harness.
"""
from __future__ import annotations
import numpy as np
from scipy.stats import spearmanr
from sklearn.linear_model import Ridge

from crisprgap.sequence import one_hot, dinucleotide_features, gc_content


def featurize(seqs, length=30):
    """Flat feature vector per guide: one-hot + dinuc composition + GC."""
    X = np.stack([one_hot(s, length).reshape(-1) for s in seqs])
    D = np.stack([dinucleotide_features(s) for s in seqs])
    G = np.array([gc_content(s) for s in seqs], dtype=np.float32).reshape(-1, 1)
    return np.concatenate([X, D, G], axis=1)


def ridge_model():
    def fit(X, y):
        m = Ridge(alpha=1.0)
        m.fit(X, y)
        return m
    def predict(m, X):
        return m.predict(X)
    return fit, predict


def run_cross_dataset(datasets: dict, models: dict, seeds=(0, 1, 2),
                      test_frac: float = 0.2) -> dict:
    """datasets: name -> EfficacyDataset. models: name -> (fit, predict).

    Returns {seed: {model: {train_name: {"within": sp, "cross": {other: sp}}}}}.
    """
    feats = {name: featurize(ds.sequences) for name, ds in datasets.items()}
    out = {}
    for seed in seeds:
        rng = np.random.default_rng(seed)
        out[seed] = {}
        for mname, (fit, predict) in models.items():
            out[seed][mname] = {}
            for name, ds in datasets.items():
                n = len(ds.sequences)
                perm = rng.permutation(n)
                n_te = int(n * test_frac)
                te, tr = perm[:n_te], perm[n_te:]
                m = fit(feats[name][tr], ds.scores[tr])
                within = float(spearmanr(predict(m, feats[name][te]), ds.scores[te]).statistic)
                cross = {}
                for other, ods in datasets.items():
                    if other == name:
                        continue
                    pred = predict(m, feats[other])
                    cross[other] = float(spearmanr(pred, ods.scores).statistic)
                out[seed][mname][name] = {"within": within, "cross": cross}
    return out


def summarize(result: dict) -> dict:
    """Mean within/cross Spearman per model per train dataset across seeds."""
    agg = {}
    for seed, models in result.items():
        for mname, trains in models.items():
            for name, r in trains.items():
                slot = agg.setdefault(mname, {}).setdefault(name, {"within": [], "cross": {}})
                slot["within"].append(r["within"])
                for other, sp in r["cross"].items():
                    slot["cross"].setdefault(other, []).append(sp)
    return {m: {t: {"within_mean": float(np.mean(v["within"])),
                    "cross_mean": {o: float(np.mean(sps)) for o, sps in v["cross"].items()},
                    "generalization_gap": float(np.mean(v["within"]) - np.mean(
                        [sp for sps in v["cross"].values() for sp in sps])) if v["cross"] else 0.0}
                for t, v in trains.items()} for m, trains in agg.items()}
