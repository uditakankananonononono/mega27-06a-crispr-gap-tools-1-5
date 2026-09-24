import numpy as np
from crisprgap.calibration import fit_isotonic, brier_score, expected_calibration_error


def test_isotonic_monotone_and_bounded():
    rng = np.random.default_rng(3)
    s = rng.normal(0, 1, 3000)
    y = (rng.uniform(0, 1, 3000) < 1 / (1 + np.exp(-2 * s))).astype(float)
    iso = fit_isotonic(s, y)
    p = iso.predict(np.linspace(-3, 3, 50))
    assert np.all(np.diff(p) >= -1e-9)
    assert (p >= 0).all() and (p <= 1).all()


def test_isotonic_beats_uncalibrated_on_ece_and_brier():
    rng = np.random.default_rng(4)
    s = rng.normal(0, 1, 6000)
    p_true = 1 / (1 + np.exp(-2 * s))
    y = (rng.uniform(0, 1, 6000) < p_true).astype(float)
    tr, te = slice(0, 3000), slice(3000, None)
    iso = fit_isotonic(s[tr], y[tr])
    p_cal = iso.predict(s[te])
    bad = 1 / (1 + np.exp(-8 * s[te]))  # overconfident squashing of raw scores
    assert expected_calibration_error(p_cal, y[te]) < expected_calibration_error(bad, y[te])
    assert brier_score(y[te], p_cal) < brier_score(y[te], bad)
