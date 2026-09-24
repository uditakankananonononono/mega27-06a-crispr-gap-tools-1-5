import json


def test_detectseq():
    d = json.load(open("results/detectseq_overlap.json"))
    assert d["detectseq_vs_guideseq_VEGFA2"]["jaccard"] < 0.01
    assert d["hek4_hek293t_vs_mcf7"]["jaccard"] > 0.5
    assert d["detectseq_counts"]["RUNX1_Cpf1BE"] == 950
