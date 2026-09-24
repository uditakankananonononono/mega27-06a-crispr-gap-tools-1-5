import json


def test_leakage_audit():
    d = json.load(open("results/leakage_audit.json"))
    # fcres/v1 overlap is known and disclosed
    assert d["pairs"]["fcres|v1"]["identical_20mers"] == 1830
    # everything else is leakage-free
    for k, v in d["pairs"].items():
        if k != "fcres|v1":
            assert v["identical_20mers"] == 0 and v["edit1_pairs"] == 0
