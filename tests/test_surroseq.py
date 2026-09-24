import json


def test_surroseq():
    d = json.load(open("results/surroseq.json"))
    assert d["libb"]["n_sites"] == 7250
    assert d["libb"]["n_guides"] == 110
    assert 0.5 < d["libb"]["auroc_cfd_sig"] < 1.0
    assert d["libc"]["n_single"] == 284
    assert d["libc"]["spearman_pos_if_vs_cfd_penalty"] > 0.7
