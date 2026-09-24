"""PORTA: portability scoring and residual-corrected cross-dataset transfer.

Implements the correction validated on the DeepHF screens: a ridge
"portability model" is fit on a calibration screen's per-guide rank
residuals and applied zero-shot to a destination screen, with no
destination labels. See paper Section 8.1 / Table 9 and
results/portability_cross_screen.json, portability_correction.json.
"""
from __future__ import annotations

import numpy as np
from scipy.stats import rankdata
from sklearn.linear_model import Ridge


class PortabilityModel:
    def __init__(self, alpha: float = 1.0):
        self.alpha = alpha
        self._ridge: Ridge | None = None

    def fit(self, features: np.ndarray, source_pred: np.ndarray,
            calib_measured: np.ndarray) -> "PortabilityModel":
        """Fit on a calibration screen: features per guide, the source
        model's predictions on that screen, and the screen's measured
        activities. Target = per-guide normalized-rank residual."""
        pred_rank = rankdata(np.asarray(source_pred, dtype=float))
        true_rank = rankdata(np.asarray(calib_measured, dtype=float))
        resid = pred_rank - true_rank
        self._ridge = Ridge(alpha=self.alpha).fit(np.asarray(features), resid)
        return self

    def score(self, features: np.ndarray) -> np.ndarray:
        """Per-guide predicted portability residual (rank units)."""
        assert self._ridge is not None, "fit first"
        return self._ridge.predict(np.asarray(features))

    def correct(self, source_pred_dest: np.ndarray,
                features_dest: np.ndarray) -> np.ndarray:
        """Corrected destination ranking: source rank minus predicted
        residual. Higher is still better."""
        base_rank = rankdata(np.asarray(source_pred_dest, dtype=float))
        return base_rank - self.score(features_dest)


def fit_portability(features, source_pred, calib_measured,
                    alpha: float = 1.0) -> PortabilityModel:
    return PortabilityModel(alpha=alpha).fit(features, source_pred, calib_measured)


def correct(model: PortabilityModel, source_pred_dest, features_dest) -> np.ndarray:
    return model.correct(source_pred_dest, features_dest)
