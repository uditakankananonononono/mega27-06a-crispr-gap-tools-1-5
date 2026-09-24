import json


def test_karyotype_results():
    d = json.load(open("results/karyotype.json"))
    assert d["n_sites"] == 1630
    assert d["n_negative_clipped"] == 685
    assert d["kruskal_p"] > 0.05  # no chromosome-level effect


def test_karyotype_figure_exists():
    import os
    assert os.path.getsize("papers/figs/karyotype_crisprsql.png") > 100_000
