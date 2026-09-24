import json


def test_akcakaya():
    d = json.load(open("results/akcakaya_circleseq.json"))
    assert d["large"]["decay_by_mm"]["4"]["n"] == 2386
    assert d["large"]["wt_ki_spearman"] > 0.7
    assert d["large"]["decay_by_mm"]["4"]["median_rel_wt"] < 0.05
