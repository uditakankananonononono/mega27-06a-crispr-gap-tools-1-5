import json

def test_pe_dense_motif():
    d = json.load(open('results/pe_dense_motif.json'))
    assert d['n_records'] == 84
    assert d['n_variants'] == 14
    assert set(d['editors']) == {'PEmax', 'PE6c', 'PE6d'}
    top = d['top_variants'][0]
    assert top['variant'] == 'evopreQ1-M2'
    assert 30 < top['mean_edit'] < 40
    assert 2.0 < d['best_fold_over_baseline'] < 2.6
    assert d['locus_rank_spearman']['rho'] > 0.6
    assert d['pemax_vs_pe6c_spearman']['rho'] > 0.75
