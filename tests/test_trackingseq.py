import json


def test_trackingseq():
    d = json.load(open("results/trackingseq_heterogeneity.json"))
    assert d["vegfa_site2_editor_sets"]["Cas9"] == 239
    assert d["vegfa_site2_editor_jaccard"]["CBE_vs_Cas9"] == 0.0
    assert d["pcsk9_context_jaccard"]["Liver_vs_NIH3T3"] == 0.0
    assert d["nih3t3_rep_vs_score"]["mwu_p"] < 1e-6
