import numpy as np
import torch
from crisprgap.models.transfer_cnn import TransferCNN, train_model, predict_model


def test_length_agnostic():
    m = TransferCNN(n_numeric=3)
    assert m(torch.zeros(2, 4, 30), torch.zeros(2, 3)).shape == (2,)
    assert m(torch.zeros(2, 4, 99), torch.zeros(2, 3)).shape == (2,)


def test_trunk_transfer():
    a = TransferCNN(n_numeric=0)
    b = TransferCNN(n_numeric=5)
    b.load_trunk(a.trunk_state())
    x = torch.zeros(1, 4, 40)
    assert torch.equal(a.trunk(x), b.trunk(x))


def test_training_learns():
    rng = np.random.default_rng(0)
    n = 300
    X = np.eye(4, dtype=np.float32)[rng.integers(0, 4, (n, 30))].transpose(0, 2, 1)
    y = X[:, 2, 0] + 0.1 * rng.normal(size=n)  # G at position 0
    m = TransferCNN(n_numeric=0)
    h = train_model(m, X, None, y.astype(np.float32), epochs=12, batch=64)
    assert h["train_loss"][-1] < 0.2
