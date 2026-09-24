import json

def test_h3_lysine_pe():
    d = json.load(open('results/h3_lysine_pe.json'))
    assert d['n_sites'] == 6
    assert d['n_pegrnas_total'] == 65
    sites = d['per_site']
    # site medians span ~2.8x
    assert 2.5 < d['best_vs_worst_site_median_ratio'] < 3.5
    assert sites['K18R']['edit_median'] < sites['K14K']['edit_median']
    # no purity tradeoff: editing vs indels negatively correlated
    assert d['edit_vs_indel_spearman_all_pegrnas'] < -0.2
    assert sites['K18R']['spread_max_over_min'] > 4
