import json

def test_spufz1_dl_prediction():
    d = json.load(open('results/spufz1_dl_prediction.json'))
    assert d['n_mutants'] > 7000
    # 5 independently trained models agree on the ranking
    assert d['ensemble_pairwise_spearman_min'] > 0.8
    # most mutations predicted deleterious (median below zero)
    assert d['predicted_mean_median'] < 0
    assert d['frac_predicted_above_0'] < 0.15
    # top candidates carry substantial ensemble spread - honest uncertainty
    names = [t[0] for t in d['top10_candidates_(mutation,mean,std)']]
    assert 'C536V' in names
    assert d['top10_mean_std'] > 0.4
    assert 'predictions only' in d['note']
