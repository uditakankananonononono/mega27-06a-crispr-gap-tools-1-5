import json


def test_offtarget_network():
    d = json.load(open("results/offtarget_network.json"))
    assert d["n_guides"] == 83
    assert d["n_edges"] == 23
    assert d["largest_component"] == 10
    assert d["isolated_guides"] == 65
    assert d["density"] < 0.01
