import json


def test_poslog_artifact():
    d = json.load(open("results/offtarget_poslog.json"))
    assert len(d["seeds"]) == 5
    for v in d["seeds"].values():
        assert 0.3 < v["auroc"] < 1.0
        assert 0.0 <= v["ece"] <= 1.0
        assert v["n_test"] > 400
    # headline: positional logistic beats the GNN mean on identical splits
    assert d["auroc_mean"] > d["reference"]["gnn_mean"]
    # split variance dominates: within-model range exceeds the GNN-CFD gap
    aucs = [v["auroc"] for v in d["seeds"].values()]
    assert max(aucs) - min(aucs) > 0.4
