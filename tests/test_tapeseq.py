import json


def test_tapeseq():
    d = json.load(open("results/tapeseq_bulges.json"))
    assert d["n_offtargets"] == 74
    assert d["n_bulge_rna"] + d["n_bulge_dna"] == 14
    assert d["n_bulge_rna"] + d["n_bulge_dna"] + d["n_nobulge"] == 74
