import json


def test_gbm_artifact():
    d = json.load(open("results/pegrna_gbm.json"))
    for tgt in ("HEK", "K562"):
        t = d["targets"][tgt]
        assert len(t["gbm_by_split"]) == 3
        assert all(0.3 < x < 0.95 for x in t["gbm_by_split"])
    # the finding: numerics-only GBM matches the sequence CNN
    assert abs(d["targets"]["HEK"]["gbm_mean"] - 0.794) < 0.05
    assert len(d["shap_top_features_HEK"]) == 10
