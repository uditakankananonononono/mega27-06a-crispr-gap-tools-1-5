import json

def test_acg_baseeditor_specificity():
    d = json.load(open('results/acg_baseeditor_specificity.json'))
    assert d['n_sites'] == 12
    pe = d['per_editor']
    assert set(pe) == {'A&C-BE ΔUGI', 'smACG32', 'smACG34'}
    ac = pe['A&C-BE ΔUGI']
    assert ac['n_sites'] == 12
    assert ac['ac_only_share'] > 0.95
    assert ac['g_editing_share'] < 0.05
    for sm in ('smACG32', 'smACG34'):
        assert 0.5 < pe[sm]['ac_only_share'] < 0.65
        assert 0.35 < pe[sm]['g_editing_share'] < 0.5
        assert pe[sm]['total_conversions_mean'] > ac['total_conversions_mean']
    cr = d['codon_reach']
    assert cr['A&C-BE ΔUGI']['mean_unique_aa'] < 3.0
    assert cr['smACGs']['n_codons'] == 268
    assert cr['smACs']['mean_unique_aa'] > 5.0
