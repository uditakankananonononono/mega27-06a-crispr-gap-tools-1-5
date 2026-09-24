import json

def test_un1cas12f1_evo():
    d = json.load(open('results/un1cas12f1_evo.json'))
    a, c, fg = d['fig2a'], d['fig2c'], d['fig2fg']
    assert a['n_targets'] == 40 and c['n_targets'] == 40
    assert a['median_fold'] > 5 and c['median_fold'] > 3
    assert a['targets_ref_lt2_evo_gt10'] >= 10
    # at canonical-PAM targets evoCas12f matches enAsCas12a
    assert 0.7 < fg['median_fold'] < 1.2
    # WT is near-dead at ACTG, evo rescues
    assert d['pam_class']['ACTG']['wt_median'] < 0.5
    assert d['pam_class']['ACTG']['evo_median'] > 4
