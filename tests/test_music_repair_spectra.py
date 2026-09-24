import json

d = json.load(open('results/music_repair_spectra.json'))


def test_scale():
    assert d['n_genes'] == {'mouse_121': 121, 'mouse_28': 28, 'human_29': 29}
    assert len(d['shared_genes_mouse28_human29']) >= 20


def test_determinism():
    m = d['mean_dominant_class_share']
    assert 0.4 < m['mouse_121'] < 0.5
    assert m['mouse_28'] > 0.6
    assert m['human_29'] > 0.6


def test_context_specific_classes():
    cm = d['class_means']
    # human RPE1 is insertion-dominated, mouse_121 deletion-dominated
    assert cm['human_29']['INS'] > 0.6
    assert cm['mouse_121']['DEL_4+'] > 0.4
    assert cm['mouse_28']['DEL_4+'] == 0.0


def test_cross_context_transfer_limited():
    t = d['cross_context_class_spearman']
    assert t['DEL_4+'] is None  # constant in mouse_28, honestly undefined
    assert t['DEL_1-3'] < 0
    assert 0.3 < t['INS'] < 0.7


def test_top_class():
    tc = d['top_class_counts_mouse121']
    assert tc['DEL_4+'] > 100
