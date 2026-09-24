import json


def test_pe_dropout():
    d = json.load(open("results/pe_dropout_epeg.json"))
    assert d["pemax_vs_pemaxKO"]["n"] == 1974
    assert d["pemax_vs_pemaxKO"]["median_ratio_ko_over_wt"] > 5
    assert d["pemax_vs_pemaxKO"]["spearman"] > 0.7
