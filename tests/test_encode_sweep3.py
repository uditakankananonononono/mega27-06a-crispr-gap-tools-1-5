import json


def test_encode_sweep3():
    d = json.load(open("results/gap5_encode_sweep3.json"))
    assert len(d["tracks"]) == 9
    for name, r in d["tracks"].items():
        assert r["n_ok"] > 300
        assert 0.4 <= r["auroc"] <= 0.6
        assert abs(r["spearman_pos"]) < 0.1
