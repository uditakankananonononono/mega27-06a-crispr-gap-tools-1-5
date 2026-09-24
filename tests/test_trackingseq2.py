import json


def test_trackingseq2():
    d = json.load(open("results/trackingseq2_cells.json"))
    h = d["HEK4"]
    assert h["Cas9_HEK4__vs__Cas9_HEK4_T-cells"]["jaccard"] == 0.276
    assert h["Cas9_HEK4_T-cells__vs__Cas9_HEK4_HSPCs"]["jaccard"] > 0.98
    assert d["VEGFA2"]["Cas9_VEGFA2__vs__Cas9_VEGFA2_T-cells"]["overlap"] == 832
