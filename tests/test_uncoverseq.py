import json


def test_uncoverseq():
    d = json.load(open("results/uncoverseq.json"))
    assert d["by_bin"]["1"]["validated_gt1pct"] == 40
    assert d["by_bin"]["5"]["validated_gt1pct"] == 5
    assert len(d["methods"]) == 9
