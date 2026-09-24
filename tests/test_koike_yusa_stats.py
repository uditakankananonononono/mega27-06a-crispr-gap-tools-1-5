import json


def test_stats_file():
    d = json.load(open("results/koike_yusa_stats.json"))
    assert d["n"] == 190
    # NAG odds ratio below 1 (protective), CI spans 1 (underpowered) - honest
    assert 0.1 < d["odds_ratio"]["nag"] < 1.0
    assert d["ci95_odds_ratio"]["nag"][0] < 1.0 < d["ci95_odds_ratio"]["nag"][1]
