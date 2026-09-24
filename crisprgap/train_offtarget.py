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
from crisprgap.cfd import cfd_score, cfd_applicable
from crisprgap.calibration import (expected_calibration_error, fit_platt, apply_platt,
                                   fit_isotonic, brier_score)
from crisprgap.data.datasets import load_crisprsql
from crisprgap.models.offtarget_gnn import OffTargetGNN, duplex_to_graph, collate_graphs

torch.set_num_threads(2)


def train_offtarget(out_dir: str = "results", max_pairs: int = 8000, epochs: int = 8, seed: int = 0) -> dict:
    rng = np.random.default_rng(seed)
    torch.manual_seed(seed)  # deterministic model init: reproducible artifacts
    ds = load_crisprsql()
    pos_idx = np.where(ds.labels == 1)[0]
    neg_idx = np.where(ds.labels == 0)[0]
    # keep every negative (crisprSQL is ~94% positive); fill with positives so the
    # test split keeps the natural prevalence - required for honest ECE (gap 2).
    n_pos = min(len(pos_idx), max_pairs - len(neg_idx))
    keep = np.concatenate([rng.choice(pos_idx, n_pos, replace=False), neg_idx])
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
    n_tr_pos = int(ds.labels[tr].sum()) if 'tr' in dir() else None
    lossf = nn.BCEWithLogitsLoss()
    tr_graphs = graphs_for(tr)
    y_tr = torch.from_numpy(ds.labels[tr]).float()
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
    # CFD needs both 23-mers (guide+PAM, off-target+PAM); subset where both are reported
    cfd_mask = np.array([bool(ds.pam[i]) and bool(ds.off_pam[i]) and cfd_applicable(ds.guides[i] + ds.pam[i], ds.offtargets[i] + ds.off_pam[i]) for i in te])
    cfd_idx = np.where(cfd_mask)[0]
    te_l = list(te)
    cfd_te = np.array([cfd_score(ds.guides[te_l[j]] + ds.pam[te_l[j]],
                                 ds.offtargets[te_l[j]] + ds.off_pam[te_l[j]]) for j in cfd_idx])

    a, b = fit_platt(val_logits, y_val)
    p_uncal = apply_platt(te_logits, 1.0, 0.0)
    p_cal = apply_platt(te_logits, a, b)
    iso = fit_isotonic(val_logits, y_val)
    p_iso = iso.predict(te_logits)

    result = {
        "dataset": "crisprsql_100720",
        "n_train": int(len(tr)), "n_val": int(len(val)), "n_test": int(len(te)),
        "n_test_positive": int(y_te.sum()),
        "test_prevalence": float(y_te.mean()),
        "gnn_test_auroc": float(roc_auc_score(y_te, te_logits)),
        "gnn_test_auprc": float(average_precision_score(y_te, te_logits)),
        "mit_test_auroc": float(roc_auc_score(y_te, mit_te)),
        "mit_test_auprc": float(average_precision_score(y_te, mit_te)),
        "cfd_subset_auroc": float(roc_auc_score(y_te[cfd_idx], cfd_te)) if len(cfd_idx) > 10 else None,
        "cfd_subset_auprc": float(average_precision_score(y_te[cfd_idx], cfd_te)) if len(cfd_idx) > 10 else None,
        "gnn_auroc_on_cfd_subset": float(roc_auc_score(y_te[cfd_idx], te_logits[cfd_idx])) if len(cfd_idx) > 10 else None,
        "n_cfd_subset": int(len(cfd_idx)),
        "ece_uncalibrated": expected_calibration_error(p_uncal, y_te),
        "ece_platt_calibrated": expected_calibration_error(p_cal, y_te),
        "ece_isotonic_calibrated": expected_calibration_error(p_iso, y_te),
        "brier_uncalibrated": brier_score(y_te, p_uncal),
        "brier_platt_calibrated": brier_score(y_te, p_cal),
        "brier_isotonic_calibrated": brier_score(y_te, p_iso),
        "platt_a": a, "platt_b": b,
        "epochs": epochs, "seed": seed,
    }
    os.makedirs(out_dir, exist_ok=True)
    torch.save(model.state_dict(), os.path.join(out_dir, "offtarget_gnn.pt"))
    studies_te = np.array(ds.studies)[te]
    np.savez_compressed(os.path.join(out_dir, "offtarget_preds.npz"),
                        logits=te_logits, y_test=y_te, mit=mit_te, p_cal=p_cal,
                        studies=studies_te)
    with open(os.path.join(out_dir, "offtarget_metrics.json"), "w") as f:
        json.dump(result, f, indent=2)
    return result
