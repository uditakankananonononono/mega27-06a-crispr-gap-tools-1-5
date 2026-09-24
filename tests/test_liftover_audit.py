import json


def test_liftover_audit():
    d = json.load(open("results/liftover_audit.json"))
    assert d["n_hg19_rows"] == 9346
    assert d["lift_rate"] > 0.99
    assert d["seq_concordance_of_lifted"] > 0.95
