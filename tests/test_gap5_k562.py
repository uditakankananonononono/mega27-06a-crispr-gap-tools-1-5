import json


def test_k562_dnase_results():
    d = json.load(open("results/gap5_k562_dnase.json"))
    assert d["n_ok"] == 1271
    assert d["n_neg"] == 35
    # no pos/neg separation; tiny positive rank correlation on positives
    assert d["auroc"] < 0.5
    assert 0 < d["spearman_dnase_vs_cleavage_positives"] < 0.15
