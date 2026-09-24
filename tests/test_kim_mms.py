import json


def test_kim_mms():
    d = json.load(open("results/kim_mms_tolerance.json"))
    mm1 = d["retention_by_mm"]["1"]
    assert mm1["Sniper"]["median_ret"] > mm1["HF1"]["median_ret"] > mm1["evo"]["median_ret"]
    assert mm1["Sniper"]["n"] > 1000
