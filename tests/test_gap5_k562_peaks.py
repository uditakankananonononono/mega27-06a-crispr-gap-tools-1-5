import json


def test_peak_overlap_results():
    d = json.load(open("results/gap5_k562_peaks.json"))
    assert d["n"] == 1271
    t = d["table"]
    assert t["in_peak_cleaved"] + t["in_peak_not"] + \
        t["out_peak_cleaved"] + t["out_peak_not"] == 1271
    # no enrichment of cleavage inside called peaks
    assert abs(d["frac_cleaved_in_peak"] - d["frac_cleaved_out_peak"]) < 0.05
