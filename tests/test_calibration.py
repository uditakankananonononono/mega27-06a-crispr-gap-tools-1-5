import numpy as np

from crisprgap.calibration import expected_calibration_error, fit_platt, apply_platt


def test_ece_perfect_calibration_near_zero():
    rng = np.random.default_rng(0)
    probs = rng.random(5000)
    labels = (rng.random(5000) < probs).astype(float)
    assert expected_calibration_error(probs, labels) < 0.05


def test_ece_miscalibrated_high():
    probs = np.repeat([0.9], 1000)
    labels = np.zeros(1000)
    assert expected_calibration_error(probs, labels) > 0.5


def test_platt_recovers_monotone_mapping():
    rng = np.random.default_rng(1)
    logits = rng.normal(0, 2, 4000)
    true_p = 1 / (1 + np.exp(-(1.5 * logits + 0.5)))
    labels = (rng.random(4000) < true_p).astype(float)
    a, b = fit_platt(logits, labels)
    assert a > 1.0 and abs(b - 0.5) < 0.3
    calibrated = apply_platt(logits, a, b)
    assert expected_calibration_error(calibrated, labels) < 0.05
