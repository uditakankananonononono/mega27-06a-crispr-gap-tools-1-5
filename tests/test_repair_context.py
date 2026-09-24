import json


def test_repair_context():
    d = json.load(open("results/repair_context.json"))
    c = d["conditions"]
    assert c["Nbn"]["l1_vs_control"] > 90
    assert c["Lig4"]["biggest_shift"]["ko"] < 1
    assert d["n_conditions"] == 18
