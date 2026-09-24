import json


def test_encode_sweep2():
    d = json.load(open("results/gap5_encode_sweep2.json"))
    assert len(d["tracks"]) == 6
    assert "U2OS" in d["u2os_encode_coverage"]
    for name, r in d["tracks"].items():
        assert 0.4 <= r["auroc"] <= 0.6
        assert abs(r["spearman_pos"]) < 0.2
