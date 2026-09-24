import json

d = json.load(open('results/quad_pegrna_insertion.json'))


def test_payload_scaling_vector_dependent():
    s = d['payload_scaling']
    # LV4 loses most activity by 9.5 kb; cV6 is flat
    assert 2.5 < s['LV4']['fold_decay_1.6_to_9.5kb'] < 4.5
    assert s['cV6']['fold_decay_1.6_to_9.5kb'] < 1.4
    assert s['LV4']['mean_by_size']['9.5 kb'] < s['LV4']['mean_by_size']['1.6 kb'] / 2
    assert abs(s['cV6']['mean_by_size']['1.6 kb'] - s['cV6']['mean_by_size']['9.5 kb']) < 3


def test_vector_evolution_uniform_gain():
    v = d['vector_evolution']
    assert v['n_cells'] == 36
    assert v['cv6_wins'] == 36
    assert 2.5 < v['paired_fold'] < 4.5


def test_large_payloads_retained():
    big = d['large_payload_per_locus']
    assert big['15 kb']['RAB11A'] > 35
    assert all(v > 8 for v in big['26 kb'].values())


def test_cross_line_variation():
    m = d['cell_line_means']
    assert m['HEK293T-cV6'] > 3 * m['mESCs-cV6']
    assert m['Hepa1-6-cV6'] > m['Huh-7-cV6']


def test_method_comparisons():
    c = d['passige_vs_paste']
    assert c['eePASSIGE'] > 10 * c['eePASTE']
    assert 5 < d['cv6_vs_evocast']['mean_fold'] < 10


def test_offtarget_panel():
    ot = d['offtarget_panel']
    assert len(ot) == 5
    for peg, rec in ot.items():
        assert rec['max_ot'] < 0.5
        if rec['on_target'] is not None:
            assert rec['on_target'] < 0.6
