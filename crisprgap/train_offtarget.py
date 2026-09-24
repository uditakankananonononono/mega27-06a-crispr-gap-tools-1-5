"""Train + benchmark the off-target GNN on crisprSQL vs the Hsu-2013 MIT baseline.

Group split by guide so the held-out set measures generalization to unseen guides.
Platt scaling is fit on a validation slice; ECE reported before/after (gap 2).
"""
from __future__ import annotations

import json
import os

import numpy as np
import torch
from sklearn.metrics import roc_auc_score, average_precision_score
from torch import nn

from crisprgap.baselines import mit_score
from crisprgap.calibration import expected_calibration_error, fit_platt, apply_platt
from crisprgap.data.datasets import load_crisprsql
from crisprgap.models.offtarget_gnn import OffTargetGNN, duplex_to_graph, collate_graphs

torch.set_num_threads(2)


def train_offtarget(out_dir: str = "results", max_pairs: int = 8000, epochs: int = 8, seed: int = 0) -> dict:
    rng = np.random.default_rng(seed)
    ds = load_crisprsql()
    pos_idx = np.where(ds.labels == 1)[0]
    neg_idx = np.where(ds.labels == 0)[0]
    n_pos = min(len(pos_idx), max_pairs // 2)
    keep = np.concatenate([rng.choice(pos_idx, n_pos, replace=False),
                           rng.choice(neg_idx, n_pos, replace=False)])
    rng.shuffle(keep)

    # group split by guide: no guide appears in both train and test
    guides = np.array([f"{g}|{s}" for g, s in zip(np.array(ds.guides)[keep], np.array(ds.studies)[keep])])
    uniq = np.unique(guides)
    rng.shuffle(uniq)
    test_guides = set(uniq[: max(1, len(uniq) // 5)])
    is_test = np.array([g in test_guides for g in guides])
    tr, te = keep[~is_test], keep[is_test]
    val = tr[: len(tr) // 10]  # calibration slice
    tr = tr[len(tr) // 10:]

    def graphs_for(idx):
        return [duplex_to_graph(ds.guides[i], ds.offtargets[i]) for i in idx]

    model = OffTargetGNN()
    opt = torch.optim.Adam(model.parameters(), lr=2e-3)
    lossf = nn.BCEWithLogitsLoss()
    tr_graphs = graphs_for(tr)
    y_tr = torch.from_numpy(ds.labels[tr])
    for epoch in range(epochs):
        model.train()
        ep = rng.permutation(len(tr_graphs))
        for i in range(0, len(ep), 256):
            sel = ep[i:i + 256]
            xb, eib, batch = collate_graphs([tr_graphs[j] for j in sel])
            opt.zero_grad()
            loss = lossf(model(xb, eib, batch), y_tr[sel])
            loss.backward()
            opt.step()

    model.eval()
    def logits_for(idx):
        out = []
        gs = graphs_for(idx)
        for i in range(0, len(gs), 512):
            xb, eib, batch = collate_graphs(gs[i:i + 512])
            with torch.no_grad():
                out.append(model(xb, eib, batch))
        return torch.cat(out).numpy()

    te_logits = logits_for(te)
    val_logits = logits_for(val)
    y_te, y_val = ds.labels[te], ds.labels[val]
    mit_te = np.array([mit_score(ds.guides[i], ds.offtargets[i]) for i in te])

    a, b = fit_platt(val_logits, y_val)
    p_uncal = apply_platt(te_logits, 1.0, 0.0)
    p_cal = apply_platt(te_logits, a, b)

    result = {
        "dataset": "crisprsql_100720",
        "n_train": int(len(tr)), "n_val": int(len(val)), "n_test": int(len(te)),
        "n_test_positive": int(y_te.sum()),
        "gnn_test_auroc": float(roc_auc_score(y_te, te_logits)),
        "gnn_test_auprc": float(average_precision_score(y_te, te_logits)),
        "mit_test_auroc": float(roc_auc_score(y_te, mit_te)),
        "mit_test_auprc": float(average_precision_score(y_te, mit_te)),
        "ece_uncalibrated": expected_calibration_error(p_uncal, y_te),
        "ece_platt_calibrated": expected_calibration_error(p_cal, y_te),
        "platt_a": a, "platt_b": b,
        "epochs": epochs, "seed": seed,
    }
    os.makedirs(out_dir, exist_ok=True)
    torch.save(model.state_dict(), os.path.join(out_dir, "offtarget_gnn.pt"))
    np.savez_compressed(os.path.join(out_dir, "offtarget_preds.npz"),
                        logits=te_logits, y_test=y_te, mit=mit_te, p_cal=p_cal)
    with open(os.path.join(out_dir, "offtarget_metrics.json"), "w") as f:
        json.dump(result, f, indent=2)
    return result
