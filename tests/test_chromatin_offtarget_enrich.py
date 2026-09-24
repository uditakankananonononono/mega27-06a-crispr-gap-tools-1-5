import json

def test_chromatin_offtarget_enrich():
    d = json.load(open('results/chromatin_offtarget_enrich.json'))
    assert d['n_s5_rows'] == 5844
    assert len(d['editors']) == 6
    assert d['n_features'] == 13
    fe = d['feature_enrichment']
    # TRIM28 top discriminator across editors
    assert fe['TRIM28']['mean_ratio'] == max(f['mean_ratio'] for f in fe.values())
    assert fe['TRIM28']['min_ratio'] > 1.5
    # active marks enrich, DNA methylation depletes
    assert fe['H3K27ac']['mean_ratio'] > 1.3
    assert fe['Methylation-of-DNA']['mean_ratio'] < 1.0
    # all six editors share the same top feature
    assert set(d['per_editor_top_feature'].values()) == {'TRIM28'}
    b = d['candidate_burden']
    assert b['n_guides'] == 3000
    assert b['median_sites_le4mm'] > 150000
    assert b['per_class_median']['m1'] < b['per_class_median']['m3']
