import json

def test_pam_diversity_ml():
    d = json.load(open('results/pam_diversity_ml.json'))
    cal = d['confidence_threshold_calibration']
    # accuracy rises monotonically with confidence threshold
    accs = [c['median_accuracy'] for c in cal]
    assert all(a <= b for a, b in zip(accs, accs[1:]))
    assert accs[-1] > 0.9
    # but absolute calibration is poor even in the top bin
    top = d['accuracy_by_confidence_bin']['0.9-1.0']
    assert top['n'] > 1000
    assert 0.7 < top['mean_accuracy'] < 0.85
    assert d['calibration_gap_at_top_bin'] > 0.2
    # mined PAM diversity ~doubles ClinVar targetability
    g = d['clinvar_targetability']['G_to_A']
    assert g['Targetable by SpCas9 (NGG PAM)']['count'] * 2 < g['Targetable by CRISPR-PAMdb/CICERO Cas9 Variants']['count']
