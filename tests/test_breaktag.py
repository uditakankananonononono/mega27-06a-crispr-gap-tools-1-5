import json


def test_breaktag():
    d = json.load(open("results/breaktag.json"))
    assert d["vegfa1"]["guideseq_rediscovered"] == 0.583
    assert d["vegfa1"]["n_breaktag"] == 772
    assert 0.4 < d["variant_blunt"]["spcas9_wt"]["frac_blunt"] < 0.6
