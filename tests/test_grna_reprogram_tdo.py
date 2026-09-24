import json

def test_grna_reprogram_tdo():
    d = json.load(open('results/grna_reprogram_tdo.json'))
    assert d['n_tdo'] == 65
    assert d['n_canonical'] == 236
    assert len(d['tdo_samples']) == 12
    assert d['total_tdo_overlapping_canonical'] == 7
    assert abs(d['overall_frac_tdo_in_canonical'] - 0.108) < 0.01
    assert d['frac_canonical_with_tdo'] < 0.05
    assert d['tdo_reads_median'] == 9.0
    assert d['tdo_reads_max'] == 1365.0
    per = d['per_sample']
    assert per['SRR6012051']['tdo_overlapping_canonical'] == 3
    assert per['SRR6012052']['frac_tdo_in_canonical'] == 0.5
