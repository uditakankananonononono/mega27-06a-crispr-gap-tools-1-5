import json

from crisprgap.data.datasets import load_sgrnascorer


def test_loader_counts_and_labels():
    ds = load_sgrnascorer()
    assert len(ds.sequences) == 430
    assert all(len(s) == 20 for s in ds.sequences)
    assert int((ds.scores == 1).sum()) == 215
    assert int((ds.scores == 0).sum()) == 215


def test_transfer_results_file():
    d = json.load(open("results/sgrnascorer_transfer.json"))
    assert d["n_high"] == 215 and d["n_low"] == 215
    a = d["auroc_by_train_source"]
    # Doench-trained ridges anti-predict the Chari split
    assert a["doench2016_fcres"] < 0.5
    assert a["doench2016_v1"] < 0.5
    # within-Chari ridge learns the split well above chance
    assert d["chari_within_cv_auroc"]["mean"] > 0.7
