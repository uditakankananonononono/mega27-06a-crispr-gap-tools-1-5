import json


def test_multimodal():
    d = json.load(open("results/multimodal_scan.json"))
    assert d["activation"]["n_common"] == 2005
    assert 0.6 < d["activation"]["spearman"] < 0.8
    assert d["gefitinib"]["spearman"] < d["activation"]["spearman"] + 0.05
