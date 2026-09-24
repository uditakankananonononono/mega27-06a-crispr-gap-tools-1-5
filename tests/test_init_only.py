import json


def test_init_only_artifact():
    d = json.load(open("results/offtarget_init_only.json"))
    assert len(d["runs"]) == 5
    assert all(r["n_test"] == 526 for r in d["runs"])  # identical seed-0 split
    assert d["init_only"]["auroc_sd"] < d["split_and_init_reference"]["gnn_sd"]
    # init 0 on split 0 must reproduce the canonical seed-0 run exactly
    assert abs(d["runs"][0]["gnn_auroc"] - 0.598422762938892) < 1e-9
