import json


def test_pe_la():
    d = json.load(open("results/pe_la_transfer.json"))
    assert d["n_pegrnas"] == 103069
    assert d["conditions"]["MCS_PE4"]["rep1_rep2_rho"] < 0.1
    assert d["top1pct_overlap"]["MCS_PE3_vs_MCS_PE4_top1pct"]["overlap"] == 0.11
