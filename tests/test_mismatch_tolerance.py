import json


def test_mismatch_tolerance():
    d = json.load(open("results/gap4_mismatch_tolerance.json"))
    assert d["n_rows"] == 3120
    assert d["mm_counts"] == {"0": 210, "1": 1800, "2": 570, "3": 540}
    # WT tolerates 1 mismatch; HF variants 3-16x lower; Sniper near WT
    assert d["wt_mm1"] > 30
    assert all(v < 10 for v in [d["hf_mm1"]["eSpCas9(1.1)"], d["hf_mm1"]["SpCas9-HF1"],
                                d["hf_mm1"]["HypaCas9"], d["hf_mm1"]["evoCas9"]])
    assert d["hf_mm1"]["Sniper-Cas9"] > 20
