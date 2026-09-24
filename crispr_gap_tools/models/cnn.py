"""CNN models for guide-RNA tasks. Small-but-real architectures sized for a
2-CPU sandbox; conv kernels over one-hot sequence channels, as in
CRISPRon/DeepCRISPR-class models."""
from __future__ import annotations
import torch
import torch.nn as nn


class GuideCNNRegressor(nn.Module):
    """1D-CNN regressor over (4, L) one-hot guide sequence -> efficacy score."""

    def __init__(self, seq_len: int = 30, in_ch: int = 4, n_filters: int = 40,
                 kernel: int = 5, hidden: int = 40, dropout: float = 0.3):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv1d(in_ch, n_filters, kernel, padding=kernel // 2), nn.ReLU(),
            nn.AvgPool1d(2),
        )
        feat = n_filters * (seq_len // 2)
        self.head = nn.Sequential(
            nn.Flatten(), nn.Dropout(dropout),
            nn.Linear(feat, hidden), nn.ReLU(), nn.Dropout(dropout),
            nn.Linear(hidden, 1),
        )

    def forward(self, x):  # x: (B, in_ch, L)
        return self.head(self.conv(x)).squeeze(-1)


class PairCNNClassifier(nn.Module):
    """1D-CNN classifier over (8, L) stacked guide/off-target channels ->
    cleavage logit (off-target classification)."""

    def __init__(self, seq_len: int = 23, in_ch: int = 8, n_filters: int = 40,
                 kernel: int = 5, hidden: int = 40, dropout: float = 0.3):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv1d(in_ch, n_filters, kernel, padding=kernel // 2), nn.ReLU(),
            nn.AvgPool1d(2),
        )
        feat = n_filters * (seq_len // 2)
        self.head = nn.Sequential(
            nn.Flatten(), nn.Dropout(dropout),
            nn.Linear(feat, hidden), nn.ReLU(), nn.Dropout(dropout),
            nn.Linear(hidden, 1),
        )

    def forward(self, x):  # x: (B, in_ch, L)
        return self.head(self.conv(x)).squeeze(-1)


def train_regressor(model, X, y, epochs: int = 10, batch: int = 64, lr: float = 1e-3,
                    seed: int = 0, device: str = "cpu", val=None, verbose: bool = False):
    """MSE training loop with Adam; returns per-epoch train losses (and val loss if val given)."""
    torch.manual_seed(seed)
    model.to(device)
    X = torch.as_tensor(X, dtype=torch.float32, device=device)
    y = torch.as_tensor(y, dtype=torch.float32, device=device)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    loss_fn = nn.MSELoss()
    history = {"train_loss": [], "val_loss": []}
    n = len(X)
    for ep in range(epochs):
        model.train()
        perm = torch.randperm(n)
        tot = 0.0
        for i in range(0, n, batch):
            idx = perm[i:i + batch]
            opt.zero_grad()
            loss = loss_fn(model(X[idx]), y[idx])
            loss.backward()
            opt.step()
            tot += loss.item() * len(idx)
        history["train_loss"].append(tot / n)
        if val is not None:
            model.eval()
            with torch.no_grad():
                Xv = torch.as_tensor(val[0], dtype=torch.float32, device=device)
                yv = torch.as_tensor(val[1], dtype=torch.float32, device=device)
                history["val_loss"].append(loss_fn(model(Xv), yv).item())
        if verbose:
            print(f"epoch {ep + 1}: {history['train_loss'][-1]:.4f}")
    return history


def predict(model, X, batch: int = 512, device: str = "cpu"):
    model.eval()
    X = torch.as_tensor(X, dtype=torch.float32)
    outs = []
    with torch.no_grad():
        for i in range(0, len(X), batch):
            outs.append(model(X[i:i + batch].to(device)).cpu().numpy())
    import numpy as np
    return np.concatenate(outs)
