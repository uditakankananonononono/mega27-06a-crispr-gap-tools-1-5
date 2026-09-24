import numpy as np
import torch
from crispr_gap_tools.models.cnn import GuideCNNRegressor, PairCNNClassifier, train_regressor, predict


def test_regressor_shapes():
    m = GuideCNNRegressor(seq_len=30)
    out = m(torch.zeros(3, 4, 30))
    assert out.shape == (3,)


def test_classifier_shapes():
    m = PairCNNClassifier(seq_len=23)
    out = m(torch.zeros(5, 8, 23))
    assert out.shape == (5,)


def test_training_reduces_loss_on_learnable_signal():
    rng = np.random.default_rng(0)
    n = 400
    X = (rng.uniform(0, 1, (n, 30, 1)) > 0.5).astype(np.float32) * np.eye(4, dtype=np.float32)[0]
    X = np.eye(4, dtype=np.float32)[rng.integers(0, 4, (n, 30))]  # random one-hot (n,30,4)
    y = X[:, 0, 0] * 2 + X[:, 1, 1]  # learnable linear function of first two bases
    m = GuideCNNRegressor(seq_len=30)
    h = train_regressor(m, X.transpose(0, 2, 1), y, epochs=15, batch=64)
    assert h["train_loss"][-1] < h["train_loss"][0] * 0.5
    assert predict(m, X.transpose(0, 2, 1)).shape == (n,)
