import json


def test_discoverseqplus():
    d = json.load(open("results/discoverseqplus_replication.json"))
    assert d["n_sites"]["k562_nd"] == 185
    assert d["jaccard_k562_ipsc"] < 0.5
    bymm = d["k562_sites_replicated_in_ipsc_by_mm"]
    assert bymm["3"]["rep_rate"] > bymm["6"]["rep_rate"]
