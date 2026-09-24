import json


def test_rnadna_wgs():
    d = json.load(open("results/rnadna_wgs.json"))
    assert d["PCSK9"]["on_target_change"] == 0.962
    assert d["PCSK9"]["burden_reduction"] == 2.67
    assert d["BCL11A"]["on_target_change"] < 0.4
