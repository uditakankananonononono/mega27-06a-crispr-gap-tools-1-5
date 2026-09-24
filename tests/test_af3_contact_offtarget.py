import json

d = json.load(open('results/af3_contact_offtarget.json'))


def test_panels_parsed():
    assert d['n_measured_regions'] >= 40
    for sn, p in d['panels'].items():
        assert p['n_sites'] >= 2000
        assert p['n_guides'] == 4
        assert sum(p['label_distribution'].values()) >= p['n_sites']


def test_label_distribution():
    for sn, p in d['panels'].items():
        ld = p['label_distribution']
        assert ld['Both changed'] > ld['Only contact prob changed'] > ld['Neither changed']


def test_scores_do_not_rank_measured_editing():
    # honest negative: AF3-derived scores sit at or below chance on the
    # measured subset
    for sn, p in d['panels'].items():
        assert p['joined_measured'] >= 30
        assert p['auroc_struc_ma_diff'] < 0.55
        assert p['auroc_cp_ma_diff'] < 0.55


def test_neither_class_not_clean():
    # 'Neither changed' sites can still be edited
    p = d['panels']['Fig. 3a']
    assert p['class_measured']['Neither changed']['frac_edited_ge_0.5'] > 0
