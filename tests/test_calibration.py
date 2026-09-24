import numpy as np
from crispr_gap_tools.calibration import PlattCalibrator, IsotonicCalibrator
from crispr_gap_tools.metrics import expected_calibration_error


def _skewed_data(n=4000, seed=1):
    rng = np.random.default_rng(seed)
    score = rng.normal(0, 1, n)
    p_true = 1 / (1 + np.exp(-2 * score))
    y = (rng.uniform(0, 1, n) < p_true).astype(int)
    return score, y


def test_platt_improves_ece_over_raw_sigmoid_misuse():
    # raw scores pushed through a badly-tempered sigmoid are miscalibrated;
    # Platt fit on train split must cut ECE on the test split.
    s, y = _skewed_data()
    tr, te = slice(0, 2000), slice(2000, None)
    bad = 1 / (1 + np.exp(-8 * s[te]))  # overconfident squashing of raw scores
    cal = PlattCalibrator().fit(s[tr], y[tr])
    p_cal = cal.predict_proba(s[te])
    assert expected_calibration_error(y[te], p_cal) < expected_calibration_error(y[te], bad)


def test_platt_monotone_and_bounded():
    s, y = _skewed_data()
    p = PlattCalibrator().fit(s, y).predict_proba(np.array([-5.0, 0.0, 5.0]))
    assert (p >= 0).all() and (p <= 1).all()
    assert p[0] < p[1] < p[2]


def test_isotonic_monotone():
    s, y = _skewed_data()
    p = IsotonicCalibrator().fit(s, y).predict_proba(np.linspace(-3, 3, 50))
    assert np.all(np.diff(p) >= -1e-9)


def test_isotonic_well_calibrated():
    s, y = _skewed_data(n=8000, seed=2)
    tr, te = slice(0, 4000), slice(4000, None)
    p = IsotonicCalibrator().fit(s[tr], y[tr]).predict_proba(s[te])
    assert expected_calibration_error(y[te], p) < 0.03
