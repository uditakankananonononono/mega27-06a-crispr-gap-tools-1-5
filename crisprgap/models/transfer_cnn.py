"""Fully-convolutional CNN for sequence->activity regression with numeric side
features, built for transfer learning (gap 3).

The conv trunk is length-agnostic (global average + max pooling), so weights
pretrained on 30-mer Cas9 guide contexts (Doench FC+RES) transfer directly to
99-mer pegRNA target contexts. Pretrain on the abundant Cas9 efficacy data,
fine-tune on the scarce pegRNA data - the transfer-learning attack on the
pegRNA data-scarcity gap.
"""
from __future__ import annotations
import torch
from torch import nn


class TransferCNN(nn.Module):
    def __init__(self, n_numeric: int = 0, n_filters: int = 48, kernel: int = 5,
                 hidden: int = 64, dropout: float = 0.2):
        super().__init__()
        self.trunk = nn.Sequential(
            nn.Conv1d(4, n_filters, kernel, padding=kernel // 2), nn.ReLU(),
            nn.Conv1d(n_filters, n_filters, kernel, padding=kernel // 2), nn.ReLU(),
        )
        feat = n_filters * 2 + n_numeric  # avg-pool + max-pool + numerics
        self.head = nn.Sequential(
            nn.Dropout(dropout), nn.Linear(feat, hidden), nn.ReLU(),
            nn.Dropout(dropout), nn.Linear(hidden, 1),
        )

    def forward(self, x, num=None):  # x: (B, 4, L) any L; num: (B, n_numeric)
        h = self.trunk(x)
        pooled = torch.cat([h.mean(dim=2), h.amax(dim=2)], dim=1)
        if num is not None:
            pooled = torch.cat([pooled, num], dim=1)
        return self.head(pooled).squeeze(-1)

    def trunk_state(self):
        return {k: v.clone() for k, v in self.trunk.state_dict().items()}

    def load_trunk(self, state):
        self.trunk.load_state_dict(state)


def train_model(model, X, NUM, y, epochs=10, batch=128, lr=1e-3, seed=0,
                val=None, patience=None):
    """MSE/Adam loop. val=(X,NUM,y) enables early stopping on val MSE with
    `patience` epochs. Returns history dict."""
    torch.manual_seed(seed)
    X = torch.as_tensor(X, dtype=torch.float32)
    y = torch.as_tensor(y, dtype=torch.float32)
    NUM = torch.as_tensor(NUM, dtype=torch.float32) if NUM is not None else None
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    lossf = nn.MSELoss()
    hist = {"train_loss": [], "val_loss": []}
    best, best_state, stale = float("inf"), None, 0
    n = len(X)
    for ep in range(epochs):
        model.train()
        perm = torch.randperm(n)
        tot = 0.0
        for i in range(0, n, batch):
            idx = perm[i:i + batch]
            opt.zero_grad()
            loss = lossf(model(X[idx], NUM[idx] if NUM is not None else None), y[idx])
            loss.backward()
            opt.step()
            tot += loss.item() * len(idx)
        hist["train_loss"].append(tot / n)
        if val is not None:
            model.eval()
            with torch.no_grad():
                Xv = torch.as_tensor(val[0], dtype=torch.float32)
                Nv = torch.as_tensor(val[1], dtype=torch.float32) if val[1] is not None else None
                yv = torch.as_tensor(val[2], dtype=torch.float32)
                vl = lossf(model(Xv, Nv), yv).item()
            hist["val_loss"].append(vl)
            if vl < best - 1e-5:
                best, best_state, stale = vl, {k: v.clone() for k, v in model.state_dict().items()}, 0
            else:
                stale += 1
                if patience and stale >= patience:
                    break
    if best_state is not None:
        model.load_state_dict(best_state)
    return hist


@torch.no_grad()
def predict_model(model, X, NUM=None, batch=512):
    import numpy as np
    model.eval()
    X = torch.as_tensor(X, dtype=torch.float32)
    outs = []
    for i in range(0, len(X), batch):
        Nv = torch.as_tensor(NUM[i:i + batch], dtype=torch.float32) if NUM is not None else None
        outs.append(model(X[i:i + batch], Nv).numpy())
    return np.concatenate(outs)
