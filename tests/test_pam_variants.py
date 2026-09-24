import json


def test_pam_variants():
    d = json.load(open("results/gap4_pam_variants.json"))
    assert d["n_rows"] == 8130
    assert d["n_complete"] == 7718
    # WT ranking does not transfer to PAM-expansion variants
    assert d["wt_correlations"]["QQR1"] < 0.35
    assert d["wt_correlations"]["Sniper-Cas9"] > 0.8
    # NGG dominates WT, SpCas9-NG is flat across NG*
    assert d["variants"]["SpCas9"]["NGG"] > 40
    assert d["variants"]["SpCas9-NG"]["NGT"] > 20
