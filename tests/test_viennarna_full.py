import json


def test_viennarna_full_artifact():
    d = json.load(open("results/porta_viennarna_full.json"))
    assert d["n"] == 55604
    assert abs(d["mfe_vs_resid_rho"]) < 0.1  # folding energy is not the driver
