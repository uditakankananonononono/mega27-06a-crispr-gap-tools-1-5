import json


def test_viennarna_sub5k_artifact():
    d = json.load(open("results/porta_viennarna_sub5k.json"))
    assert d["n"] == 5000
    # folding energy is not the residual driver (post-encoding-fix)
    assert abs(d["mfe_vs_resid_rho"]) < 0.1
