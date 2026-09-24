import json

d = json.load(open('results/sdd7_cbe_window.json'))


def test_two_blocks_30_sites():
    assert d['n_sites'] == 30
    assert d['n_sites_activity'] == 30


def test_cross_editor_ranks_transfer():
    for k, v in d['cross_editor_site_spearman'].items():
        assert 0.4 < v < 0.95, (k, v)
    assert d['min_cross_editor_rho'] < 0.6


def test_specificity_tradeoff():
    spec = d['specificity_index_secondary_over_primary']
    assert spec['BE4max'] < spec['Sdd7e2']
    assert spec['BE4max'] < 0.02
    assert 0.03 < spec['Sdd7e2'] < 0.08


def test_window_profiles():
    p = d['positional_profiles']
    assert len(p) == 4
    for ed, prof in p.items():
        assert prof['peak_position'] == 6
        assert 0.4 < prof['share_C4_to_C7'] < 0.75
    assert p['Sdd7']['share_C4_to_C7'] < p['Sdd7e1']['share_C4_to_C7']


def test_note_preserved():
    assert 'secondary' in d['note']
