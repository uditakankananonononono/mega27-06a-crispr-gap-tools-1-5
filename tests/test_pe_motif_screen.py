import json

def test_pe_motif_screen():
    d = json.load(open('results/pe_motif_screen.json'))
    assert d['n_elements'] == 19000
    assert d['n_usable_reads_ge100'] == 11490
    bt = d['by_variant_type']
    assert set(bt) == {'COMBO', 'HPEXTEND', 'NEG', 'PARENT'}
    # primary-screen medians are close across types (small motif effect at this stage)
    assert abs(bt['PARENT']['pe7_median'] - bt['NEG']['pe7_median']) < 0.3
    assert bt['PARENT']['n'] == 4396
    # strong cross-editor transfer
    assert d['pe7_vs_pemax_spearman'] > 0.8
    assert d['pe7_vs_pe6c_spearman'] > 0.8
    # low hit rate at this screening depth
    assert bt['PARENT']['pe7_frac_gt10'] < 0.05
