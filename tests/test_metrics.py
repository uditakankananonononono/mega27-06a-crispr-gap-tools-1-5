import numpy as np
import pytest
from crispr_gap_tools.metrics import (brier_score, expected_calibration_error,
                                      regression_metrics, classification_metrics)


def test_perfect_regression():
    m = regression_metrics([1, 2, 3, 4], [1, 2, 3, 4])
    assert m["spearman"] == pytest.approx(1.0)
    assert m["pearson"] == pytest.approx(1.0)
    assert m["mse"] == pytest.approx(0.0)


def test_auroc_extremes():
    m = classification_metrics([0, 0, 1, 1], [0.1, 0.2, 0.8, 0.9])
    assert m["auroc"] == 1.0
    m2 = classification_metrics([0, 0, 1, 1], [0.9, 0.8, 0.2, 0.1])
    assert m2["auroc"] == 0.0


def test_ece_perfect_calibration():
    rng = np.random.default_rng(0)
    p = rng.uniform(0, 1, 20000)
    y = (rng.uniform(0, 1, 20000) < p).astype(float)
    assert expected_calibration_error(y, p) < 0.02


def test_ece_worst_case():
    y = np.array([0, 0, 1, 1])
    p = np.array([0.99, 0.99, 0.01, 0.01])
    assert expected_calibration_error(y, p, n_bins=2) > 0.9


def test_brier():
    assert brier_score([1, 0], [1.0, 0.0]) == 0.0
    assert brier_score([1, 0], [0.0, 1.0]) == 1.0
