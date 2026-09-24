import json

def test_allelic_tiling_editors():
    d = json.load(open('results/allelic_tiling_editors.json'))
    abe = d['ABE8e']
    assert abe['n_positions']['Deconvolved Scores'] == 335
    rep = abe['raw_replicate_spearman']
    assert rep['n_shared'] == 151
    # honest negative: raw per-position scores barely replicate
    assert rep['r1_r2'] < 0.3
    assert rep['r1_r3'] < 0.3
    ce = d['cross_editor_deconvolved']
    assert ce['n_shared_positions'] == 127
    # editor-specific landscapes
    assert ce['spearman'] < 0.4
    assert ce['top_quartile_jaccard'] < 0.4
