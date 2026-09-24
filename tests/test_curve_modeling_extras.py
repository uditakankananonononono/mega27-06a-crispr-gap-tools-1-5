import json

d = json.load(open('results/curve_modeling_extras.json'))


def test_hill_fits():
    e = d['lnp_hill_editing']
    i = d['lnp_hill_indels']
    assert e['rsquared'] > 0.99 and i['rsquared'] > 0.99
    # editing saturates at a lower dose than indels accumulate
    assert e['ec50'] < i['ec50']
    assert 0.8 < e['ec50'] < 1.5
    assert 2.0 < i['ec50'] < 3.2
    assert e['n'] > i['n']


def test_knee_null_reported():
    # 4-point curve: kneed finds no knee; keep the honest null
    assert d['lnp_knee_log2dose'] is None


def test_spacer_inverse_length():
    f = d['spacer_inverse_length_fit']
    assert f['slope'] > 100
    assert f['r2'] > 0.8


def test_window_changepoints():
    cp = d['window_changepoints']
    # engineered Sdd7 variants show a window-edge change point at index 5
    for ed in ('Sdd7', 'Sdd7e1', 'Sdd7e2'):
        assert 5 in cp[ed]['changepoints']
    assert 5 not in cp['BE4max']['changepoints']


def test_distance_correlation():
    dc = d['editor_distance_correlation']
    assert len(dc) == 6
    assert d['editor_distance_correlation_min'] < 0.7
    assert dc['Sdd7e1_vs_Sdd7e2'] > 0.85
