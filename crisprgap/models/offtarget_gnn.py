"""GNN scorer for guide/off-target duplexes (batched over disjoint-union graphs).

Each duplex is a graph: positions are nodes (one-hot base-pair identity + mismatch
flag + position), edges connect adjacent positions and all mismatch positions to
each other (mismatch interaction graph). Message passing over a disjoint-union
batch scores many duplexes per forward pass. Attacks the mismatch-context gap (5)
and the calibration gap (2) when paired with crisprgap.calibration.
"""
from __future__ import annotations

import torch
from torch import nn

from crisprgap.sequence import BASE_TO_IDX

# node feature dim: 16 (guide-target base pair one-hot, 4x4) + 1 mismatch + 1 position frac
NODE_DIM = 18


def duplex_to_graph(guide: str, offtarget: str) -> tuple[torch.Tensor, torch.Tensor]:
    """Return (node_features (L, NODE_DIM), edge_index (2, E)) for a duplex."""
    g, o = guide.upper(), offtarget.upper()
    assert len(g) == len(o)
    L = len(g)
    x = torch.zeros(L, NODE_DIM)
    for i, (a, b) in enumerate(zip(g, o)):
        ai, bi = BASE_TO_IDX.get(a), BASE_TO_IDX.get(b)
        if ai is not None and bi is not None:
            x[i, ai * 4 + bi] = 1.0
        x[i, 16] = 0.0 if (ai is not None and ai == bi) else 1.0
        x[i, 17] = i / max(L - 1, 1)
    edges: list[tuple[int, int]] = []
    for i in range(L - 1):
        edges += [(i, i + 1), (i + 1, i)]
    mm = [i for i in range(L) if x[i, 16] > 0.5]
    for i in mm:
        for j in mm:
            if i != j:
                edges.append((i, j))
    if not edges:
        edges = [(0, 0)]
    edge_index = torch.tensor(edges, dtype=torch.long).t().contiguous()
    return x, edge_index


def collate_graphs(graphs: list[tuple[torch.Tensor, torch.Tensor]]) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Merge per-duplex graphs into one disjoint-union batch.

    Returns (x (N_total, NODE_DIM), edge_index (2, E_total), batch (N_total,) graph ids).
    """
    xs, eis, batch = [], [], []
    offset = 0
    for gi, (x, ei) in enumerate(graphs):
        n = x.size(0)
        xs.append(x)
        eis.append(ei + offset)
        batch.append(torch.full((n,), gi, dtype=torch.long))
        offset += n
    return torch.cat(xs, 0), torch.cat(eis, 1), torch.cat(batch, 0)


class MessagePassingLayer(nn.Module):
    def __init__(self, dim: int):
        super().__init__()
        self.msg = nn.Linear(dim * 2, dim)
        self.upd = nn.GRUCell(dim, dim)

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
        src, dst = edge_index[0], edge_index[1]
        m = torch.relu(self.msg(torch.cat([x[src], x[dst]], dim=1)))
        agg = torch.zeros_like(x).index_add_(0, dst, m)
        deg = torch.zeros(x.size(0), device=x.device).index_add_(
            0, dst, torch.ones_like(dst, dtype=torch.float)).clamp(min=1).unsqueeze(1)
        return self.upd(agg / deg, x)


def _scatter_pool(h: torch.Tensor, batch: torch.Tensor, n_graphs: int) -> tuple[torch.Tensor, torch.Tensor]:
    mean = torch.zeros(n_graphs, h.size(1), device=h.device).index_add_(0, batch, h)
    cnt = torch.zeros(n_graphs, device=h.device).index_add_(0, batch, torch.ones_like(batch, dtype=torch.float)).clamp(min=1).unsqueeze(1)
    mean = mean / cnt
    mx = torch.full((n_graphs, h.size(1)), -1e9, device=h.device)
    mx.scatter_reduce_(0, batch.unsqueeze(1).expand_as(h), h, reduce="amax", include_self=True)
    return mean, mx


class OffTargetGNN(nn.Module):
    def __init__(self, node_dim: int = NODE_DIM, hidden: int = 48, steps: int = 3):
        super().__init__()
        self.inp = nn.Linear(node_dim, hidden)
        self.layers = nn.ModuleList([MessagePassingLayer(hidden) for _ in range(steps)])
        self.head = nn.Sequential(nn.Linear(hidden * 2, 64), nn.ReLU(), nn.Linear(64, 1))

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor, batch: torch.Tensor | None = None) -> torch.Tensor:
        """Single graph when batch is None; disjoint-union batch otherwise. Returns (n_graphs,) logits."""
        h = torch.relu(self.inp(x))
        for layer in self.layers:
            h = layer(h, edge_index)
        if batch is None:
            pooled = torch.cat([h.mean(dim=0), h.max(dim=0).values], dim=0).unsqueeze(0)
        else:
            mean, mx = _scatter_pool(h, batch, int(batch.max().item()) + 1)
            pooled = torch.cat([mean, mx], dim=1)
        return self.head(pooled).squeeze(-1)
