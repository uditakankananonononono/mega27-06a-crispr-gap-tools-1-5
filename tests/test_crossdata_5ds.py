import json


def test_crossdata_5ds_artifact():
    d = json.load(open("results/crossdata_cnn_5ds.json"))
    assert set(d["raw"]) == {"0", "1"}
    cnn = d["summary"]["cnn"]
    assert set(cnn) == {"fcres", "v1", "deephf_wt", "deepspcas9", "crispron"}
    for name, r in cnn.items():
        assert -1.0 <= r["within_mean"] <= 1.0
        assert len(r["cross_mean"]) == 4
    # post-encoding-fix findings
    assert cnn["fcres"]["cross_mean"]["deephf_wt"] > 0.1   # was -0.25 pre-fix
    assert cnn["deephf_wt"]["within_mean"] > 0.6           # corruption cost ~0.15
    # the residual negative pair: DeepHF <-> DeepSpCas9
    assert cnn["deephf_wt"]["cross_mean"]["deepspcas9"] < 0
    assert cnn["deepspcas9"]["cross_mean"]["deephf_wt"] < 0
    # CRISPRon anomaly: cross transfer exceeds within-dataset
    assert cnn["crispron"]["cross_mean"]["deepspcas9"] > cnn["crispron"]["within_mean"]


def test_crispron_loader():
    from crisprgap.data.datasets import load_crispron
    d = load_crispron()
    assert len(d.sequences) > 11000
    assert all(len(s) == 20 for s in d.sequences[:500])
    assert 0.0 <= float(d.scores.min()) <= float(d.scores.max()) <= 1.0
