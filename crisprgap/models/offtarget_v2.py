"""Extended off-target GNN with global side features (gaps 4 and 5).

Lane C's duplex-message-passing trunk, plus a global-feature channel
concatenated after pooling: PAM one-hot (gap 4: PAM-variant awareness) and/or
epigenetic track values (gap 5: chromatin context). Ablation flags control
which channels the model may use.
"""
from __future__ import annotations
import numpy as np
import torch
from torch import nn

from crisprgap.models.offtarget_gnn import MessagePassingLayer, NODE_DIM, _scatter_pool
from crisprgap.sequence import BASE_TO_IDX

EPIGEN_COLS = ("epigen_ctcf", "epigen_dnase", "epigen_rrbs", "epigen_h3k4me3", "epigen_drip")


def pam_one_hot(pam: str) -> np.ndarray:
    """(12,) one-hot of a 3-nt PAM; unknown/missing -> all zeros."""
    out = np.zeros(12, dtype=np.float32)
    for i, b in enumerate(pam.upper()[:3]):
        j = BASE_TO_IDX.get(b)
        if j is not None:
            out[i * 4 + j] = 1.0
    return out


def global_features(i, ds, use_pam: bool, use_epigen: bool) -> np.ndarray:
    parts = []
    if use_pam:
        parts.append(pam_one_hot(ds.off_pam[i]))
    if use_epigen:
        parts.append(np.array([ds.epigen[c][i] for c in EPIGEN_COLS], dtype=np.float32))
    return np.concatenate(parts) if parts else np.zeros(0, dtype=np.float32)


class OffTargetGNNv2(nn.Module):
    def __init__(self, n_global: int = 0, node_dim: int = NODE_DIM, hidden: int = 48, steps: int = 3):
        super().__init__()
        self.inp = nn.Linear(node_dim, hidden)
        self.layers = nn.ModuleList([MessagePassingLayer(hidden) for _ in range(steps)])
        self.head = nn.Sequential(nn.Linear(hidden * 2 + n_global, 64), nn.ReLU(), nn.Linear(64, 1))

    def forward(self, x, edge_index, batch, glob=None):
        h = torch.relu(self.inp(x))
        for layer in self.layers:
            h = layer(h, edge_index)
        mean, mx = _scatter_pool(h, batch, int(batch.max().item()) + 1)
        pooled = torch.cat([mean, mx], dim=1)
        if glob is not None and glob.numel() > 0:
            pooled = torch.cat([pooled, glob], dim=1)
        return self.head(pooled).squeeze(-1)
