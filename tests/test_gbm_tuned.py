import json


def test_gbm_tuned_artifact():
    d = json.load(open("results/pegrna_gbm_tuned.json"))
    assert d["n_trials"] == 30
    assert "seed-0" in d["tuned_on"]  # tuning confined to one split, no leakage
    for tgt in ("HEK", "K562"):
        t = d["targets"][tgt]
        assert len(t["tuned_gbm_by_split"]) == 3
        assert all(0.3 < x < 0.95 for x in t["tuned_gbm_by_split"])
        ref_key = "untuned_gbm_" + tgt
        # tuned must not underperform the untuned baseline on the mean
        assert t["tuned_gbm_mean"] >= d["reference"][ref_key] - 0.005
    # the headline: tuned GBM edges out the sequence CNN on both cell lines
    assert d["targets"]["HEK"]["tuned_gbm_mean"] > d["reference"]["cnn_HEK"]
    assert d["targets"]["K562"]["tuned_gbm_mean"] > d["reference"]["cnn_K562"]


def test_gbm_tuned_params_plausible():
    d = json.load(open("results/pegrna_gbm_tuned.json"))
    p = d["best_params"]
    assert 10 < p["n_estimators"] < 5000
    assert 0.001 < p["learning_rate"] < 1.0
    assert 2 <= p["num_leaves"] < 512
    assert 0.0 < d["best_val_spearman"] < 1.0
