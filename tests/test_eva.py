import json


def test_eva():
    d = json.load(open("results/eva_benchmark.json"))
    assert d["This study"]["spearman"] > 0.4
    assert d["Doench2016"]["spearman"] < 0.2
    assert d["Leenay2019"]["n"] == 1557
