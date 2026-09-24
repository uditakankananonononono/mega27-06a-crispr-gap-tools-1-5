import json

def test_cas12a_tunable_control():
    d = json.load(open('results/cas12a_tunable_control.json'))
    kd = d['knockdown_residual']
    assert set(kd) == {'Cas12a', 'Cas12aDeg', 'Cas12a_1uMdTagV1', 'Cas12aDeg_1uMdTagV1'}
    # constitutive Cas12a knocks down strongly
    assert kd['Cas12a']['mean_residual_frac'] < 0.35
    # dTag has no effect on non-degron Cas12a
    assert abs(kd['Cas12a']['mean_residual_frac'] - kd['Cas12a_1uMdTagV1']['mean_residual_frac']) < 0.1
    # dTag relieves knockdown only in the degron line
    assert kd['Cas12aDeg_1uMdTagV1']['mean_residual_frac'] > 0.7
    assert kd['Cas12aDeg']['mean_residual_frac'] < 0.45
    assert 0.3 < d['degron_dynamic_range'] < 0.55
