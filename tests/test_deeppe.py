import json


def test_deeppe():
    d = json.load(open("results/gap3_deeppe.json"))
    assert d["n"] == 43149
    assert 0.6 < d["ridge_cv_mean"] < 0.75
    assert 0.4 < d["deepspcas9_only_spearman"] < 0.6
    assert d["univariate"]["GC contents_1"] > 0.3
    assert d["univariate"]["MFE_1"] < 0
