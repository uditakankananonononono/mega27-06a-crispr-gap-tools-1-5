import json


def test_genic_context_results():
    d = json.load(open("results/gap5_genic_context.json"))
    assert d["n"] == 1630
    fr = d["contexts"]
    assert abs(fr["exonic"]["frac_cleaved"] - fr["intronic_or_other_genic"]["frac_cleaved"]) < 0.05
    assert d["fisher_p"] > 0.5
