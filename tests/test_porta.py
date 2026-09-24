import numpy as np
from scipy.stats import spearmanr

from crisprgap.porta import PortabilityModel, fit_portability, correct


def _toy(n=600, seed=0):
    rng = np.random.default_rng(seed)
    X = rng.normal(size=(n, 6))
    # guide-inherent residual replicates across screens (the discovery's ideal case)
    resid = X @ np.array([1.0, -0.8, 0.5, 0.0, 0.3, -0.2])
    calib_y = 0.3 * X[:, 0] + rng.normal(scale=1.0, size=n)          # calibration screen truth
    dest_y = 0.3 * X[:, 0] + rng.normal(scale=1.0, size=n) + 0.0     # destination screen truth
    # source model: misses the residual structure on both screens
    source_pred_calib = calib_y + resid
    source_pred_dest = dest_y + resid
    return X, source_pred_calib, calib_y, source_pred_dest, dest_y


def test_porta_recovers_rank_order_when_residual_replicates():
    X, pc, yc, pdst, ydst = _toy()
    m = fit_portability(X, pc, yc)
    corrected = correct(m, pdst, X)
    raw = spearmanr(pdst, ydst).statistic
    fixed = spearmanr(corrected, ydst).statistic
    assert fixed > 0.85          # strong recovery in the replication regime
    assert fixed > raw + 0.2     # and a large lift over raw transfer


def test_porta_rank_space_replication_near_perfect():
    # truer to the discovery: the residual acts in rank space
    from scipy.stats import rankdata
    rng = np.random.default_rng(0)
    n = 600
    X = rng.normal(size=(n, 6))
    resid = X @ np.array([1.0, -0.8, 0.5, 0.0, 0.3, -0.2])
    yc = 0.3 * X[:, 0] + rng.normal(size=n)
    yd = 0.3 * X[:, 0] + rng.normal(size=n)
    pc = rankdata(yc) + 40 * resid
    pd2 = rankdata(yd) + 40 * resid
    m = fit_portability(X, pc, yc)
    c = correct(m, pd2, X)
    assert spearmanr(c, yd).statistic > 0.95


def test_portability_score_shape_and_fit_required():
    m = PortabilityModel()
    X = np.zeros((5, 3))
    try:
        m.score(X)
        assert False, "should raise before fit"
    except AssertionError:
        pass
    rng = np.random.default_rng(1)
    y = rng.normal(size=5)
    m.fit(X, y, y)               # zero residual: model predicts ~0
    assert np.allclose(m.score(X), 0.0, atol=1e-6)
