import json


def test_kim_hf():
    d = json.load(open("results/kim_hf_variant_on.json"))
    assert d["n"] == 6481
    assert d["retention_vs_wt"]["Sniper"] > d["retention_vs_wt"]["evo"]
    assert d["wt_transfer"]["WT_to_evo"] < d["within_cv"]["evo"]
