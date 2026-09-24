import json


def test_endovseq():
    d = json.load(open("results/endovseq_tolerance.json"))
    assert d["retention_by_mismatch_count"]["1"]["abe_retention"] > 0.5
    assert d["retention_by_mismatch_count"]["3"]["abe_retention"] < 0.15
    assert d["n_variants"] == 102
