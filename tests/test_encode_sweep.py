import json


def test_encode_sweep_results():
    d = json.load(open("results/gap5_encode_sweep.json"))
    assert d["k562_n"] > 0 and d["hek293t_n"] > 0
    assert len(d["tracks"]) == 8
    for name, r in d["tracks"].items():
        assert r["n_ok"] > 1000
        assert 0.35 <= r["auroc"] <= 0.60  # all null/anti-predictive
        assert abs(r["spearman_pos"]) < 0.1  # no real effect anywhere
