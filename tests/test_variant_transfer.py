import json


def test_variant_transfer():
    d = json.load(open("results/gap1_variant_transfer.json"))
    assert set(d) == {"moesm4", "moesm5"}
    for k, v in d.items():
        assert v["n"] > 5000
        for var, r in v["wt_to_variant"].items():
            assert 0.1 < r < 0.7  # all transfers weak
