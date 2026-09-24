import json

def test_abe_reversion_window():
    d = json.load(open('results/abe_reversion_window.json'))
    assert d['n_records'] > 10000
    pe = d['per_editor']
    assert len(pe) >= 12
    # ABE7.10 family concentrates editing in window 4-8; ABE8e family is broader
    assert pe['ABE7.10']['window_4to8_share'] > 0.9
    assert pe['ABE8e']['window_4to8_share'] < 0.7
    # evolved editors are motif-general; ABE7.10 is motif-picky
    assert pe['ABE7.10']['motif_max_over_min'] > 5 * pe['ABE8e']['motif_max_over_min']
    # 7.10-background reversions halve motif dependence at ~unchanged activity
    r = d['mean_activity_vs_ABE7.10']
    assert pe['ABE7.10-HRHSK']['motif_max_over_min'] < 0.5 * pe['ABE7.10']['motif_max_over_min']
    assert 0.9 < r['ABE7.10-HRHSK'] < 1.05
    # 8e-background reversions raise activity with window share unchanged
    assert r['ABE8e-K'] > r['ABE8e']
    assert abs(pe['ABE8e-K']['window_4to8_share'] - pe['ABE8e']['window_4to8_share']) < 0.03
