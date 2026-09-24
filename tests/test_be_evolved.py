import json


def test_be_evolved():
    d = json.load(open("results/be_evolved_offtarget.json"))
    assert d["editors"]["ABE8e"]["median_ot_retention"] > d["editors"]["M1"]["median_ot_retention"]
    assert d["editors"]["ABE8e"]["mean_on_target"] > d["editors"]["M1"]["mean_on_target"]
    assert d["n_blocks"] == 20
