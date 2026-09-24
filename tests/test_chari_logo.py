import json
import os


def test_logo_figure_and_results():
    assert os.path.getsize("papers/fig_chari_logo.pdf") > 10000
    d = json.load(open("results/chari_logo.json"))
    top = d["top_divergent_positions"][0]
    # the largest high/low divergence is at the PAM-adjacent position
    assert top["pos0"] == 19
    assert top["max_abs_freq_diff"] > 0.3
