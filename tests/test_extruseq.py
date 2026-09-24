import json


def test_extruseq():
    d = json.load(open("results/extruseq_validation.json"))
    assert d["n_sites"] == 309
    assert d["n_targets"] == 7
    assert d["mismatch_curve"]["2"]["n"] == 74
    assert d["spearman_mm_indel"]["rho"] < -0.5
