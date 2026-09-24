import json


def test_peacseq():
    d = json.load(open("results/peacseq_crossmethod.json"))
    assert d["VEGFA_TS2"]["guide"] == "GACCCCCTCCACCCCGCCTC"
    assert d["VEGFA_TS2"]["n_sites"] == 648
    assert d["VEGFA_TS3"]["jaccard"] < 0.1
    assert d["VEGFA_TS2"]["enrich_mwu_crisprsql_vs_not_p"] < 0.001
