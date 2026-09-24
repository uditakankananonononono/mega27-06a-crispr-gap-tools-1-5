import numpy as np
from crisprgap.crossdata import run_cross_dataset, summarize, ridge_model
from crisprgap.data.datasets import EfficacyDataset


def _toy(n, shift, seed):
    rng = np.random.default_rng(seed)
    seqs = []
    scores = []
    bases = "ACGT"
    for _ in range(n):
        s = "".join(rng.choice(list(bases), 30, replace=True))
        seqs.append(s)
        gc = (s.count("G") + s.count("C")) / 30.0
        scores.append((s[0] == "G") * 1.0 + 0.5 * gc + shift + rng.normal(0, 0.01))
    scores = np.array(scores, dtype=np.float32)
    scores = (scores - scores.min()) / (scores.max() - scores.min())
    return EfficacyDataset(sequences=seqs, scores=scores, source=f"toy{shift}")


def test_harness_ranks_learnable_signal_within_dataset():
    ds = {"a": _toy(300, 0.0, 1), "b": _toy(300, 0.0, 2)}
    res = run_cross_dataset(ds, {"ridge": ridge_model()}, seeds=(0,))
    r = res[0]["ridge"]["a"]
    assert r["within"] > 0.9
    assert "b" in r["cross"]


def test_cross_dataset_gap_detected_when_distribution_shifts():
    # dataset b has the sign of the signal flipped: a model trained on a must fail on b
    a = _toy(400, 0.0, 3)
    b = _toy(400, 0.0, 4)
    b.scores = (1.0 - b.scores).astype(np.float32)
    res = run_cross_dataset({"a": a, "b": b}, {"ridge": ridge_model()}, seeds=(0,))
    r = res[0]["ridge"]["a"]
    assert r["within"] > 0.9
    assert r["cross"]["b"] < 0.0


def test_summarize_shape():
    ds = {"a": _toy(200, 0.0, 5), "b": _toy(200, 0.1, 6)}
    res = run_cross_dataset(ds, {"ridge": ridge_model()}, seeds=(0, 1))
    s = summarize(res)
    assert set(s["ridge"]["a"]) == {"within_mean", "cross_mean", "generalization_gap"}
    assert "b" in s["ridge"]["a"]["cross_mean"]
