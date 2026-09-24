import json

d = json.load(open('results/rloop_transcleavage.json'))


def test_spacer_length_tunes_transcleavage():
    s = d['fig3_spacer_signal']
    assert s['NT'] < 1.1
    assert s['49'] < s['20']
    assert d['spacer_length_spearman'] < -0.9
    assert 1.8 < d['spacer_max_over_min'] < 2.4


def test_mismatch_position_panelA():
    p = d['fig2_panelA_by_mismatch']
    # central mismatches (8/13) activate most with short dsDNA
    assert p['mm13']['short'] > p['mm19']['short']
    assert p['mm5']['short'] > 14
    # triple mismatch suppresses
    assert p['mm4-5-6']['short'] < 3
    assert d['triple_mismatch_suppression'] < 0.2
    # random DNA never activates
    for v in p.values():
        assert v['random'] < 2


def test_panel_context_dependence():
    a = d['fig2_panelA_by_mismatch']
    b = d['fig2_panelB_by_mismatch']
    # same mismatches, much weaker activation in panel B context
    assert a['mm13']['short'] > 5 * b['mm13']['short']
    assert b['mm4-5-6']['short'] < 1.5
