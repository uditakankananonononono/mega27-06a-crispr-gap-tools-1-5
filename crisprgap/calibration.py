"""Probability calibration for off-target cleavage scores (gap 2)."""
from __future__ import annotations

import numpy as np


def expected_calibration_error(probs: np.ndarray, labels: np.ndarray, n_bins: int = 10) -> float:
    probs = np.asarray(probs, dtype=np.float64)
    labels = np.asarray(labels, dtype=np.float64)
    edges = np.linspace(0.0, 1.0, n_bins + 1)
    ece = 0.0
    for lo, hi in zip(edges[:-1], edges[1:]):
        m = (probs >= lo) & (probs < hi)
        if m.any():
            ece += m.mean() * abs(probs[m].mean() - labels[m].mean())
    return float(ece)


def fit_platt(logits: np.ndarray, labels: np.ndarray) -> tuple[float, float]:
    """Fit Platt scaling (a, b): p = sigmoid(a * logit + b) by gradient descent."""
    from scipy.optimize import minimize

    logits = np.asarray(logits, dtype=np.float64)
    labels = np.asarray(labels, dtype=np.float64)

    def nll(params):
        a, b = params
        z = np.clip(a * logits + b, -40, 40)
        return float(np.mean(np.logaddexp(0, z) - labels * z)) + 1e-3 * (a * a + b * b)

    res = minimize(nll, x0=np.array([1.0, 0.0]), method="Nelder-Mead")
    return float(res.x[0]), float(res.x[1])


def apply_platt(logits: np.ndarray, a: float, b: float) -> np.ndarray:
    z = np.clip(a * np.asarray(logits, dtype=np.float64) + b, -40, 40)
    return 1.0 / (1.0 + np.exp(-z))


def fit_isotonic(scores, labels):
    """Non-parametric isotonic calibration (Zadrozny & Elkan 2002)."""
    from sklearn.isotonic import IsotonicRegression
    iso = IsotonicRegression(out_of_bounds="clip", y_min=0.0, y_max=1.0)
    iso.fit(np.asarray(scores, dtype=np.float64), np.asarray(labels, dtype=np.float64))
    return iso


def brier_score(labels: np.ndarray, probs: np.ndarray) -> float:
    labels = np.asarray(labels, dtype=np.float64)
    probs = np.asarray(probs, dtype=np.float64)
    return float(np.mean((probs - labels) ** 2))
