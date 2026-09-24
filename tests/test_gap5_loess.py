import json
import os


def test_loess_results():
    d = json.load(open("results/gap5_loess.json"))
    assert d["n_pos_plotted"] == 1236
    assert d["loess_endpoint_slope"] < 0  # no positive dose-response
    assert os.path.getsize("papers/fig_k562_loess.pdf") > 5000
