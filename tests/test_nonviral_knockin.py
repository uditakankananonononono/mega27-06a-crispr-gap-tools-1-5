import json

def test_nonviral_knockin_guides():
    d = json.load(open('results/nonviral_knockin_guides.json'))
    assert d['n_guides'] == 76597
    assert d['n_usable_totreads_ge100'] == 60764
    ef = d['edited_frac']
    assert 0.4 < ef['median'] < 0.6
    assert ef['p90'] > 0.7
    assert ef['frac_le_05'] > 0.45
    # GC does not drive edited fraction in this readout
    assert abs(d['gc_spearman']) < 0.05
    assert d['n_genes_ge3_guides'] > 10000
    assert d['gene_median_efrac_p90'] > d['gene_median_efrac_p10'] + 0.1
