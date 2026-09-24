import json


def test_crispron():
    d = json.load(open("results/crispron_conditions.json"))
    assert d["n_guides"] == 11446
    assert len(d["spearman_cross_condition"]) == 10
    assert d["spearman_cross_condition"]["D8minus_vs_D10minus"] > 0.8
    assert d["spearman_cross_condition"]["D2_vs_D10minus"] < 0.4
