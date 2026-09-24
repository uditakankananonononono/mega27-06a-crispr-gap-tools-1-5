import json


def test_cas12abe():
    d = json.load(open("results/cas12abe_window.json"))
    w = d["windows"]["BEACON2"]
    assert w["window_width_gt5pct"] == 6
    assert w["peak"] == "c10"
    assert w["positions"]["c3"] < 5
