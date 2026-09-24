import json

def test_code_editing_precision():
    d = json.load(open('results/code_editing_precision.json'))
    assert d['n_sites'] == 6
    pe = d['per_editor']
    # CODEMax is the precision leader
    assert pe['CODEMax']['median_precision_ratio'] > pe['PE2']['median_precision_ratio']
    assert pe['CODEMax']['median_precision_ratio'] > pe['PEMax']['median_precision_ratio']
    # nicking helps CODE precision but hurts PE precision
    assert pe['CODEMax + nicking']['median_precision_ratio'] > pe['CODEMax']['median_precision_ratio']
    assert pe['PEMax + nicking']['median_precision_ratio'] < pe['PEMax']['median_precision_ratio']
    nk = d['nicking_effect']
    assert nk['PEMax']['unintended_fold'] > 10
    assert nk['CODEMax']['unintended_fold'] < 2
