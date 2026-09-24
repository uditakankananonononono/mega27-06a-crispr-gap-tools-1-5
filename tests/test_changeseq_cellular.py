import json


def test_changeseq_cellular():
    d = json.load(open("results/changeseq_cellular.json"))
    assert d["n_matched"] == 151
    assert 0.4 < d["spearman_change_vs_cellular"] < 0.75
    assert d["frac_cell_positive"] < 0.25
