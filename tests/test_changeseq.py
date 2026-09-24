import json


def test_changeseq():
    d = json.load(open("results/changeseq_cfd.json"))
    assert d["n_rows_total"] == 202043
    assert d["n_scored"] == 190704
    assert d["n_indel_excluded"] == 10420
    assert 0.7 < d["auroc_ge20"] < 0.8
    assert 0.75 < d["auroc_ge100"] < 0.85
    assert 0.4 < d["spearman_cfd_logreads_pos"] < 0.6
