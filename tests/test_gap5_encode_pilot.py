import json


def test_gap5_encode_pilot_artifact():
    d = json.load(open("results/gap5_encode_h3k4me3_pilot.json"))
    assert d["n_queried"] >= 300
    assert d["bundled_nonzero_frac"] < 0.05  # the coverage gap
    assert 0.4 < d["auroc_h3k4me3_alone"] < 0.9
    assert d["mannwhitney_p"] > 0.01  # underpowered, not a claimed effect
