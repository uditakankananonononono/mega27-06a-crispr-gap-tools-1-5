import json


def test_bioframe_xcheck():
    d = json.load(open("results/genic_bioframe_xcheck.json"))
    assert d["n"] == 1630
    assert abs(d["exonic_frac_cleaved"] - d["nonexonic_frac_cleaved"]) < 0.05
    assert d["fisher_p"] > 0.05  # confirms the intervaltree null
