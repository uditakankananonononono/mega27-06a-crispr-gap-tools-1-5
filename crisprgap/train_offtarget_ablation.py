"""Ablation benchmark for gaps 4 (PAM awareness) and 5 (chromatin context).

Same honest protocol as train_offtarget.py: guide-grouped split, natural
prevalence, class-weighted BCE, isotonic calibration on a val slice.
Conditions: seq-only, +PAM, +epigen, +PAM+epigen. Reports AUROC/AUPRC/ECE per
condition, plus AUROC restricted to non-NGG off-targets (the PAM-variant
subset where gap 4 lives).
"""
from __future__ import annotations
import json
import os
import numpy as np
import torch
from scipy.stats import spearmanr
from sklearn.metrics import roc_auc_score, average_precision_score
from torch import nn

from crisprgap.calibration import expected_calibration_error, fit_isotonic, brier_score
from crisprgap.data.datasets import load_crisprsql
from crisprgap.models.offtarget_gnn import duplex_to_graph, collate_graphs
from crisprgap.models.offtarget_v2 import OffTargetGNNv2, global_features

torch.set_num_threads(2)

CONDITIONS = {
    "seq_only": dict(use_pam=False, use_epigen=False),
    "pam": dict(use_pam=True, use_epigen=False),
    "epigen": dict(use_pam=False, use_epigen=True),
    "pam_epigen": dict(use_pam=True, use_epigen=True),
}


def run_ablation(out_dir="results", max_pairs=12000, epochs=8, seed=0) -> dict:
    rng = np.random.default_rng(seed)
    ds = load_crisprsql()
    pos_idx = np.where(ds.labels == 1)[0]
    neg_idx = np.where(ds.labels == 0)[0]
    n_pos = min(len(pos_idx), max_pairs - len(neg_idx))
    keep = np.concatenate([rng.choice(pos_idx, n_pos, replace=False), neg_idx])
    rng.shuffle(keep)

    guides = np.array([f"{g}|{s}" for g, s in zip(np.array(ds.guides)[keep], np.array(ds.studies)[keep])])
    uniq = np.unique(guides)
    rng.shuffle(uniq)
    test_guides = set(uniq[: max(1, len(uniq) // 5)])
    is_test = np.array([g in test_guides for g in guides])
    tr, te = keep[~is_test], keep[is_test]
    val = tr[: len(tr) // 10]
    tr = tr[len(tr) // 10:]

    tr_graphs = [duplex_to_graph(ds.guides[i], ds.offtargets[i]) for i in tr]
    y_tr = torch.from_numpy(ds.labels[tr]).float()
    y_te, y_val = ds.labels[te], ds.labels[val]

    def logits_for(model, idx, flags):
        out = []
        gs = [duplex_to_graph(ds.guides[i], ds.offtargets[i]) for i in idx]
        gl = np.stack([global_features(i, ds, **flags) for i in idx])
        for s in range(0, len(gs), 512):
            xb, eib, batch = collate_graphs(gs[s:s + 512])
            gb = torch.from_numpy(gl[s:s + 512])
            with torch.no_grad():
                out.append(model(xb, eib, batch, gb))
        return torch.cat(out).numpy()

    result = {"dataset": "crisprsql_100720", "n_train": int(len(tr)),
              "n_val": int(len(val)), "n_test": int(len(te)), "seed": seed,
              "conditions": {}}
    non_ngg_te = np.array([not ds.off_pam[i].upper().endswith("GG") if ds.off_pam[i] else False for i in te])

    for name, flags in CONDITIONS.items():
        torch.manual_seed(seed)
        n_glob = len(global_features(int(tr[0]), ds, **flags))
        model = OffTargetGNNv2(n_global=n_glob)
        opt = torch.optim.Adam(model.parameters(), lr=2e-3)
        n_tr_pos = float(y_tr.sum())
        lossf = nn.BCEWithLogitsLoss(pos_weight=torch.tensor((len(y_tr) - n_tr_pos) / max(n_tr_pos, 1.0)))
        G_tr = np.stack([global_features(int(i), ds, **flags) for i in tr])
        for ep in range(epochs):
            model.train()
            perm = rng.permutation(len(tr_graphs))
            for s in range(0, len(perm), 256):
                sel = perm[s:s + 256]
                xb, eib, batch = collate_graphs([tr_graphs[j] for j in sel])
                gb = torch.from_numpy(G_tr[sel])
                opt.zero_grad()
                loss = lossf(model(xb, eib, batch, gb), y_tr[sel])
                loss.backward()
                opt.step()
        model.eval()
        te_logits = logits_for(model, te, flags)
        val_logits = logits_for(model, val, flags)
        iso = fit_isotonic(val_logits, y_val)
        p_cal = iso.predict(te_logits)
        cond = {
            "auroc": float(roc_auc_score(y_te, te_logits)),
            "auprc": float(average_precision_score(y_te, te_logits)),
            "ece_isotonic": expected_calibration_error(p_cal, y_te),
            "brier_isotonic": brier_score(y_te, p_cal),
        }
        if non_ngg_te.sum() > 20:
            cond["auroc_non_ngg_pam"] = float(roc_auc_score(y_te[non_ngg_te], te_logits[non_ngg_te]))
            cond["n_non_ngg_te"] = int(non_ngg_te.sum())
        result["conditions"][name] = cond
        print(f"[ablation] {name}: {cond}", flush=True)
    os.makedirs(out_dir, exist_ok=True)
    with open(os.path.join(out_dir, "offtarget_ablation.json"), "w") as f:
        json.dump(result, f, indent=2)
    return result
