import json


def test_cas12a_variants():
    d = json.load(open("results/cas12a_variants_pam.json"))
    assert d["pam_mean_activity"]["TTTV"] > 0.4
    assert d["pam_mean_activity"]["TGCV"] < 0.08
    assert d["n_variants_with_tttv"] == 13
    assert d["breadth_gt10pct"]["lbrvrr"] >= 8
