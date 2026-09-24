import json

def test_tadacbe_zebrafish():
    d = json.load(open('results/tadacbe_zebrafish.json'))
    assert d['n_site_positions'] >= 20
    pe = d['per_editor']
    assert d['best_editor_by_mean'] == 'TCBE1.2'
    # TCBE1.2 nearly doubles zTadCBE mean activity
    assert pe['TCBE1.2']['mean_ctot'] > 1.5 * pe['zTadCBE']['mean_ctot']
    # site rankings transfer only moderately between editors
    assert d['min_cross_editor_rho'] < 0.6
    assert d['max_cross_editor_rho'] > 0.9
