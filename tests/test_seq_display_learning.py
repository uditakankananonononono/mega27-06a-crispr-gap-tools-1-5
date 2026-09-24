import json

def test_seq_display_learning():
    d = json.load(open('results/seq_display_learning.json'))
    m = d['r2_by_training_size_mean_over_pams_seeds']
    # small-data models are worse than predicting the mean
    assert m['100'] < -1.0
    assert m['1000'] < 0
    # only at 16k does R2 go solidly positive
    assert m['16000'] > 0.35
    assert d['r2_at_100_fraction_negative'] > 0.8
    b = d['model_benchmark_NNGA_R2']
    assert b['ESM-2'] > b['RF'] > b['MLP'] > b['CNN']
    assert len(b) == 6
