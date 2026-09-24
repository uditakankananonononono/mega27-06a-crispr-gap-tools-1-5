import json
import os


def test_guide_upset_results():
    d = json.load(open("results/guide_upset.json"))
    assert d["n_unique_guides"] == 87
    assert d["guides_in_2plus_studies"] == 13
    assert os.path.getsize("papers/fig_guide_upset.pdf") > 10000
