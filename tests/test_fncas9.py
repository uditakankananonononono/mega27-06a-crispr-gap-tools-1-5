import json


def test_fncas9():
    d = json.load(open("results/fncas9_variants.json"))
    assert d["AGG"]["FnCas9"]["median"] < 3
    assert d["AGG"]["en1"]["median"] > 40
    assert d["NGG"]["en1"]["median"] > d["NGG"]["FnCas9"]["median"]
