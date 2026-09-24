import json


def test_hifi_variants():
    d = json.load(open("results/hifi_variants_offtarget.json"))
    pv = d["per_variant"]
    assert pv["WT SpCas9"]["frac_active_ge_0p1"] > 0.6
    assert pv["SpCas9-HF1"]["frac_active_ge_0p1"] < 0.2
    assert pv["B-HeFSpCas9"]["on_target_mean"] < 0.1
    assert d["guideseq_offtarget_counts"]["FANCF site 2"]["WT SpCas9"] == 85
