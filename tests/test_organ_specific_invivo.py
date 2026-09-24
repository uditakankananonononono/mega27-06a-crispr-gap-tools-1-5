import json

def test_organ_specific_invivo():
    d = json.load(open('results/organ_specific_invivo.json'))
    assert set(d) == {'gP_Pcsk9', 'gMH'}
    gp = d['gP_Pcsk9']
    assert gp['n_ot_sites'] == 104
    assert d['gMH']['n_ot_sites'] == 75
    # breadth histograms partition the sites
    assert sum(gp['detected_ge_1.0pct_breadth_hist'].values()) == 104
    assert gp['detected_ge_1.0pct_breadth_hist']['0'] == 86
    assert gp['detected_ge_1.0pct_breadth_hist']['1'] == 7
    # on-target organ span
    ot = gp['on_target_editing']
    assert ot['Brain'] < 7.0 and ot['Pancreas'] > 60.0
    gmh = d['gMH']['on_target_editing']
    assert gmh['Brain'] < 2.0 and gmh['Colon'] > 40.0
    # gMH: nothing off-target reaches 0.5%
    assert sum(d['gMH']['detected_ge_0.5pct_breadth_hist'].values()) == 75
    assert d['gMH']['detected_ge_0.5pct_breadth_hist']['0'] == 75
    # cross-organ correlation summary sane
    assert -1 <= gp['cross_organ_spearman_min'][2] <= gp['cross_organ_spearman_max'][2] <= 1
    assert 0.4 <= gp['cross_organ_spearman_median'] <= 0.75
    assert gp['organ_specific_sites_n'] == 2
