"""Probability calibration for off-target scores (gap 2 core).

Published off-target scores (MIT/Hsu 2013, CFD/Doench 2016) are uncalibrated
ranking scores, not probabilities of cleavage. These calibrators map any score
to a calibrated cleavage probability fit on held-out measured data.
"""
from __future__ import annotations
import numpy as np
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression


class PlattCalibrator:
    """Platt scaling: logistic regression on the raw score."""

    def __init__(self):
        self._lr = LogisticRegression(max_iter=1000)

    def fit(self, scores, y):
        self._lr.fit(np.asarray(scores, float).reshape(-1, 1), np.asarray(y, int))
        return self

    def predict_proba(self, scores) -> np.ndarray:
        return self._lr.predict_proba(np.asarray(scores, float).reshape(-1, 1))[:, 1]


class IsotonicCalibrator:
    """Non-parametric isotonic calibration (Zadrozny & Elkan 2002)."""

    def __init__(self):
        self._iso = IsotonicRegression(out_of_bounds="clip", y_min=0.0, y_max=1.0)

    def fit(self, scores, y):
        self._iso.fit(np.asarray(scores, float), np.asarray(y, int))
        return self

    def predict_proba(self, scores) -> np.ndarray:
        return self._iso.predict(np.asarray(scores, float))
