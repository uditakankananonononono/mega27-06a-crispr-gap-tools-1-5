import json


def test_pe_transfer():
    d = json.load(open("results/pe_cross_library_transfer.json"))
    assert d["n_deeppe"] == 43149
    assert d["n_pridict2"] == 92423
    assert d["deeppe_to_pridict2"] < d["deeppe_within_cv"]
    assert d["pridict2_to_deeppe"] < d["deeppe_within_cv"]
