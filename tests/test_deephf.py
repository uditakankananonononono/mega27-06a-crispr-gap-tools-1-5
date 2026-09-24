import json


def test_deephf():
    d = json.load(open("results/deephf_transfer.json"))
    assert d["n_guides"] == 53937
    assert all(0.7 <= v <= 0.8 for v in d["within_cv"].values())
    assert d["transfer"]["wt_to_hf1"] < d["within_cv"]["hf1"]
