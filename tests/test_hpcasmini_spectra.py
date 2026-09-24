import json

def test_hpcasmini_indel_spectra():
    d = json.load(open('results/hpcasmini_indel_spectra.json'))
    assert d['n_rows'] == 585
    assert set(d['libs']) == {'CasMINI', 'Ctrl', 'hpCasMINI'}
    cas, hp = d['per_library']['CasMINI'], d['per_library']['hpCasMINI']
    # engineered variant boosts total indel signal ~6x
    assert 5 < hp['total_frac'] / cas['total_frac'] < 8
    # and shifts to longer deletions
    assert hp['mean_del_len'] > cas['mean_del_len'] + 2
    # deletion-dominated spectra for both
    assert cas['del_share'] > 0.95 and hp['del_share'] > 0.98
    # spectra remain correlated per guide
    for g in ('KLHL29', 'NLRC4', 'SITE1'):
        assert d[f'spectrum_spearman_{g}'] > 0.5
