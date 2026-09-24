import json

def test_cas12a_bridge_helix():
    d = json.load(open('results/cas12a_bridge_helix.json'))
    assert d['n_variants'] >= 8
    pv = d['per_variant']
    # WT barely discriminates the double mismatch at 15 min
    assert d['wt_discrimination_mm13_15min'] < 0.1
    # bridge-helix variants open a wide discrimination window
    assert pv['KD2P-KA']['discrimination_mm13_15min'] > 0.8
    assert pv['KD2P-KA']['matched_15min'] > 0.8
    # WT-RA keeps matched activity while suppressing MM13
    assert pv['WT-RA']['matched_15min'] > 0.85
    assert pv['WT-RA']['mm13_15min'] < 0.2
