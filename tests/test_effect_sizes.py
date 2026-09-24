import json


def test_effect_sizes_file():
    d = json.load(open("results/effect_sizes_pingouin.json"))
    ky = d["koike_yusa_nag_vs_ngg"]
    assert ky["n_ngg"] == 95 and ky["n_nag"] == 95
    assert 0 < ky["rbc"] < 0.3  # small positive rank effect
    hek = d["hek293t_dnase_cleaved_vs_not"]
    assert abs(hek["rbc"]) < 0.05  # SITE-Seq chromatin null
