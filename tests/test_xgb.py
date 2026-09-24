import json


def test_xgb_artifact():
    d = json.load(open("results/pegrna_xgb.json"))
    for tgt in ("HEK", "K562"):
        for tag in ("default_xgb", "tuned_region_xgb"):
            r = d["targets"][tgt][tag]
            assert len(r["by_split"]) == 3
            assert all(0.2 < x < 0.95 for x in r["by_split"])
    # library-consistency finding: tuned-region XGB also beats the CNN on HEK
    assert d["targets"]["HEK"]["tuned_region_xgb"]["mean"] > d["reference"]["cnn_HEK"]
    # default XGB on K562 is far below its tuned region (defaults underfit)
    assert (d["targets"]["K562"]["tuned_region_xgb"]["mean"]
            - d["targets"]["K562"]["default_xgb"]["mean"]) > 0.05
