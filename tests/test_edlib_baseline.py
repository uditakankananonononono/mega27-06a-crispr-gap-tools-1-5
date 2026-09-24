import json


def test_edlib_audit():
    d = json.load(open("results/gap2_edlib_baseline.json"))
    assert d["n_pairs"] == 25632
    assert d["n_indel_pairs"] == 8949
    assert 0.45 < d["auroc_mean"] < 0.60
