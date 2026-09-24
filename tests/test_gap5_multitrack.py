import json


def test_multitrack_results():
    d = json.load(open("results/gap5_k562_multitrack.json"))
    assert d["n"] == 1299
    for tr in d["tracks"].values():
        assert tr["n_ok"] == 1271
        assert tr["auroc"] < 0.55  # no chromatin track separates cleaved/not
