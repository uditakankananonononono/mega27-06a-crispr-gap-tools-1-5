import json


def test_crossdata_6ds_artifact():
    d = json.load(open("results/crossdata_cnn_6ds.json"))
    assert set(d["raw"]) == {"0", "1"}
    cnn = d["summary"]["cnn"]
    assert len(cnn) == 6
    # DeepHF-as-SOURCE transfers weakly negative to all three HEK293T-cluster datasets
    for other in ("deepspcas9", "crispron", "sgdesigner"):
        assert cnn["deephf_wt"]["cross_mean"][other] < 0
    # into-DeepHF is mixed: negative from deepspcas9/sgdesigner, positive from crispron
    assert cnn["deepspcas9"]["cross_mean"]["deephf_wt"] < 0
    assert cnn["sgdesigner"]["cross_mean"]["deephf_wt"] < 0
    assert cnn["crispron"]["cross_mean"]["deephf_wt"] > 0
    # HEK293T cluster transfers positively among itself
    assert cnn["deepspcas9"]["cross_mean"]["sgdesigner"] > 0.3
    # cross-exceeds-within inversions documented
    assert cnn["crispron"]["cross_mean"]["deepspcas9"] > cnn["crispron"]["within_mean"]


def test_sgdesigner_loader():
    from crisprgap.data.datasets import load_sgdesigner
    d = load_sgdesigner()
    assert 1200 < len(d.sequences) < 1400
    assert all(len(s) == 30 for s in d.sequences[:500])
    assert all(set(s) <= set("ACGT") for s in d.sequences[:500])
