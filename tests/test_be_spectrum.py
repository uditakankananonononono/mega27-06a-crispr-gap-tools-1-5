import json


def test_be_spectrum():
    d = json.load(open("results/be_outcome_spectrum.json"))
    assert d["ABE7_10"]["median_top1_share"] > d["BE4"]["median_top1_share"]
    assert d["BE4"]["median_entropy"] > d["ABE7_10"]["median_entropy"]
    assert d["ABE7_10"]["n_grnas"] == 11484
