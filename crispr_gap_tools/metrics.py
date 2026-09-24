"""Evaluation metrics shared by all gap tools."""
from __future__ import annotations
import numpy as np
from scipy.stats import pearsonr, spearmanr
from sklearn.metrics import average_precision_score, roc_auc_score


def regression_metrics(y_true, y_pred) -> dict:
    y_true = np.asarray(y_true, float)
    y_pred = np.asarray(y_pred, float)
    sp, sp_p = spearmanr(y_true, y_pred)
    pe, pe_p = pearsonr(y_true, y_pred)
    return {"spearman": float(sp), "spearman_p": float(sp_p),
            "pearson": float(pe), "pearson_p": float(pe_p),
            "mse": float(np.mean((y_true - y_pred) ** 2))}


def classification_metrics(y_true, y_score) -> dict:
    y_true = np.asarray(y_true, int)
    y_score = np.asarray(y_score, float)
    return {"auroc": float(roc_auc_score(y_true, y_score)),
            "auprc": float(average_precision_score(y_true, y_score))}


def expected_calibration_error(y_true, p, n_bins: int = 15) -> float:
    """ECE with equal-width bins over [0,1]. Guo et al. 2017 definition."""
    y_true = np.asarray(y_true, float)
    p = np.clip(np.asarray(p, float), 0.0, 1.0)
    edges = np.linspace(0.0, 1.0, n_bins + 1)
    ece = 0.0
    for lo, hi in zip(edges[:-1], edges[1:]):
        mask = (p >= lo) & (p < hi if hi < 1.0 else p <= hi)
        if not mask.any():
            continue
        conf = p[mask].mean()
        acc = y_true[mask].mean()
        ece += mask.mean() * abs(acc - conf)
    return float(ece)


def brier_score(y_true, p) -> float:
    y_true = np.asarray(y_true, float)
    p = np.asarray(p, float)
    return float(np.mean((p - y_true) ** 2))
