import json

d = json.load(open('results/abe9_spry_offtarget.json'))


def test_nomination_space():
    nom = d['aceofbase_nominated_per_locus']
    assert len(nom) == 4
    assert all(1500 < n < 5000 for n in nom.values())


def test_panel_shape():
    assert d['n_loci'] == 4
    for locus, rec in d['per_locus'].items():
        assert set(rec) == {'ABE8e-SpRY', 'ABE9-SpRY'}
        for ed, x in rec.items():
            assert len(x['ots']) == 5


def test_abe9_offtarget_cleanliness_with_exception():
    r = d['reduction_8e_vs_9']
    # ABE9 clean at 3 of 4 loci, Tpc2-L249 the honest exception
    clean = sum(1 for v in r.values() if v['n_ot_above_3x_ctrl_9'] == 0)
    assert clean == 3
    assert r['Tpc2-L249']['n_ot_above_3x_ctrl_9'] == 2
    assert r['Tpc2-L249']['max_ot_fold_8e_over_9'] < 3
    assert r['Trpm4-L903']['max_ot_fold_8e_over_9'] > 500


def test_abe8e_offtarget_cost():
    for locus, rec in d['per_locus'].items():
        assert rec['ABE8e-SpRY']['n_ot_above_3x_control'] >= 2


def test_abe9_variable_ontarget():
    # ABE9-SpRY on-target is locus-dependent: near-dead at Tpc2-K188
    p = d['per_locus']
    assert p['Tpc2-K188']['ABE9-SpRY']['on_target'] < 1
    assert p['Trpm4-L903']['ABE9-SpRY']['on_target'] > 50
