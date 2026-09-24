import json

def test_cas12f_ortholog_eng():
    d = json.load(open('results/cas12f_ortholog_eng.json'))
    ret = d['guide_truncation_retention']
    # a 3'-truncated guide improves interference 10x; core deletion kills it
    assert ret['block1:d106-125 nt'] > 8
    assert ret['block2:d50-60 nt'] < 0.1
    assert d['n_mutants'] == 19
    assert d['mutants_below_half'] == 10
    assert 'Y60A' in d['most_sensitive']
    assert d['mutant_fold_median'] < 0.6
