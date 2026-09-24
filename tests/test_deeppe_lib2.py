import json


def test_deeppe_lib2():
    d = json.load(open("results/gap3_deeppe_lib2.json"))
    assert d["n"] == 5431
    assert -0.3 < d["spearman_position"] < 0
    assert d["by_nedit"]["10"] < 3  # 10-base edits collapse
    assert d["by_nedit"]["1"] > 10
