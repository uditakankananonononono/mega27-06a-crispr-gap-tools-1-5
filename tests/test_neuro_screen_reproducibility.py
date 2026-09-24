import json

d = json.load(open('results/neuro_screen_reproducibility.json'))


def test_scale():
    assert d['n_sgrna'] >= 1139
    assert d['n_genes_genelevel'] >= 380
    assert len(d['sgrna_conditions']) == 12


def test_replicate_conditions_transfer():
    pw = d['sgrna_condition_transfer']
    assert pw['primary_vs_primary.A']['spearman'] > 0.8
    gp = d['genelevel_condition_transfer']
    assert gp['glut.A_vs_glut.primary']['spearman'] > 0.8
    assert gp['glut.primary_vs_glut.secondary']['spearman'] > 0.4


def test_cross_condition_transfer_fails():
    pw = d['sgrna_condition_transfer']
    # glutamate readout does not transfer to unrelated conditions
    assert abs(pw['exp139.glut_vs_exp139.ttx']['spearman']) < 0.1
    assert pw['exp139.glut_vs_ipsc.glut']['spearman'] < 0.1
    gp = d['genelevel_condition_transfer']
    assert gp['glut.primary_vs_ipsc.glut']['spearman'] < 0.1
    # a large share of cross-condition pairs are near zero or negative
    neg = sum(1 for v in pw.values() if v['spearman'] < 0)
    assert neg >= 25


def test_within_gene_concordance():
    w = d['within_gene_concordance']
    assert w['n_genes_multi_sgRNA'] >= 370
    assert 0.65 < w['fraction'] < 0.8
    assert w['mean_abs_spread'] > 0.5
