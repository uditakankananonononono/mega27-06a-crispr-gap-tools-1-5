import json


def test_context_feature_results():
    d = json.load(open("results/gap2_context_features.json"))
    assert d["n"] == 1630 and d["n_pos"] == 893
    assert len(d["aurocs"]["mm_only"]) == 3
    # context features add nothing on the cell-free slice
    assert abs(d["mean"]["mm_only"] - d["mean"]["mm_plus_context"]) < 0.05
