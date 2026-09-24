import json

def test_be_pance_evolution():
    d = json.load(open('results/be_pance_evolution.json'))
    assert d['n_positions'] == 167
    assert d['wt_residue_mean_pct_in_T0'] > 80
    # selection concentrates on a contiguous 67-75 window
    top = [name for _, name in d['top_enriched_substitutions_(fc,name)'][:8]]
    positions = sorted(int(''.join(ch for ch in n if ch.isdigit())) for n in top)
    assert positions[0] >= 67 and positions[-1] <= 75
    assert d['top_substitution_round10_mean_pct'] > 25
    assert d['n_substitutions_enriched_gt3x'] > 300
    for v in d['replicate_mutant_landscape_spearman'].values():
        assert v > 0.5
