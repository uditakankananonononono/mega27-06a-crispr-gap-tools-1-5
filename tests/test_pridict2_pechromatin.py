import json


def test_pridict2_pechromatin():
    d = json.load(open("results/pridict2_pechromatin.json"))
    assert d["pridict2"]["n_lib1"] == 92423
    assert 0.4 < d["pridict2"]["ridge_cv_spearman"] < 0.7
    assert d["pe_chromatin"]["n"] == 16055
    assert d["pe_chromatin"]["mlh1dn_median_fold"] > 1.5
