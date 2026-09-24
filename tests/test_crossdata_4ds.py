import json


def test_crossdata_4ds_artifact():
    d = json.load(open("results/crossdata_cnn_4ds.json"))
    assert set(d["raw"]) == {"0", "1"}  # two seeds
    cnn = d["summary"]["cnn"]
    assert set(cnn) == {"fcres", "v1", "deephf_wt", "deepspcas9"}
    for name, r in cnn.items():
        assert -1.0 <= r["within_mean"] <= 1.0
        assert len(r["cross_mean"]) == 3
    # the finding: DeepHF <-> DeepSpCas9 transfer is negative in both directions
    assert cnn["deephf_wt"]["cross_mean"]["deepspcas9"] < 0
    assert cnn["deepspcas9"]["cross_mean"]["deephf_wt"] < 0
    # while DeepSpCas9 within-dataset performance is strong
    assert cnn["deepspcas9"]["within_mean"] > 0.45


def test_deepspcas9_loader():
    from crisprgap.data.datasets import load_deepspcas9
    d = load_deepspcas9()
    assert len(d.sequences) > 12000
    assert all(len(s) == 30 for s in d.sequences[:200])
    assert 0.0 <= float(d.scores.min()) <= float(d.scores.max()) <= 1.0
    assert d.source.startswith("deepspcas9")
