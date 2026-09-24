"""CNN for CRISPR guide-RNA on-target efficacy prediction.

Architecture: one-hot (4, L) -> Conv1d blocks -> global pooling -> MLP head.
Trained on public CRISPR screen datasets (e.g. Doench 2016 rule-set-2 style 30-mers).
"""
from __future__ import annotations

import torch
from torch import nn


class GuideEfficacyCNN(nn.Module):
    def __init__(self, seq_len: int = 30, channels: tuple[int, ...] = (32, 64, 64),
                 kernel: int = 5, hidden: int = 128, n_aux: int = 17):
        super().__init__()
        self.seq_len = seq_len
        layers: list[nn.Module] = []
        in_ch = 4
        for ch in channels:
            layers += [
                nn.Conv1d(in_ch, ch, kernel_size=kernel, padding=kernel // 2),
                nn.BatchNorm1d(ch),
                nn.ReLU(),
            ]
            in_ch = ch
        self.conv = nn.Sequential(*layers)
        self.pool = nn.AdaptiveAvgPool1d(1)
        # aux features: GC content + 16 dinucleotide frequencies
        self.head = nn.Sequential(
            nn.Linear(in_ch + n_aux, hidden),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(hidden, 1),
        )

    def forward(self, x: torch.Tensor, aux: torch.Tensor) -> torch.Tensor:
        """x: (B, 4, L) one-hot; aux: (B, n_aux). Returns (B,) efficacy logits."""
        h = self.conv(x)
        h = self.pool(h).squeeze(-1)
        h = torch.cat([h, aux], dim=1)
        return self.head(h).squeeze(-1)


def make_aux_features(seq: str):
    """Auxiliary scalar features for one sequence: GC + dinucleotide composition."""
    from crisprgap.sequence import gc_content, dinucleotide_features
    import numpy as np

    return np.concatenate([[gc_content(seq)], dinucleotide_features(seq)]).astype(np.float32)
