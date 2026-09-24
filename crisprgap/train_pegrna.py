"""Gap 3: pegRNA efficiency prediction via transfer learning from Cas9 data.

Protocol: group split by grp_id (no pegRNA group shared between train/val/test).
Three conditions, 2 seeds:
  A. ridge on engineered numeric features only (PRIDICT1-style baseline)
  B. TransferCNN trained from scratch on sequence+numerics
  C. TransferCNN with trunk pretrained on Doench FC+RES Cas9 efficacy, fine-tuned
Report: Spearman on held-out pegRNAs (HEK primary, K562 secondary).
"""
from __future__ import annotations
import json
import os
import numpy as np
import torch
from scipy.stats import spearmanr
from sklearn.linear_model import Ridge

from crisprgap.data.datasets import load_pridict, load_doench_fcres
from crisprgap.models.transfer_cnn import TransferCNN, train_model, predict_model
from crisprgap.sequence import one_hot

torch.set_num_threads(2)


def _split(ds, seed, val_frac=0.1, test_frac=0.2):
    rng = np.random.default_rng(seed)
    grps = np.unique(ds.grp_ids)
    rng.shuffle(grps)
    n_te = int(len(grps) * test_frac)
    n_va = int(len(grps) * val_frac)
    te_g, va_g = set(grps[:n_te]), set(grps[n_te:n_te + n_va])
    gid = np.array(ds.grp_ids)
    return (~gid.isin(te_g | va_g)), gid.isin(va_g), gid.isin(te_g)


def _encode(seqs):
    return np.stack([one_hot(s, 99) for s in seqs])


def run_pegrna(out_dir="results", target="HEK", seeds=(0, 1), pre_epochs=6,
               ft_epochs=25, scratch_epochs=25) -> dict:
    ds = load_pridict()
    y_all = ds.targets[target]
    X = _encode(ds.sequences)
    NUMraw = ds.numerics

    # pretrain trunk on Cas9 efficacy (Doench FC+RES 30-mers)
    fc = load_doench_fcres()
    Xpre = np.stack([one_hot(s, 30) for s in fc.sequences])
    pre = TransferCNN(n_numeric=0)
    train_model(pre, Xpre, None, fc.scores, epochs=pre_epochs, batch=256)
    trunk_state = pre.trunk_state()
    del Xpre

    out = {"target": target, "n": len(ds.sequences), "conditions": {}}
    for seed in seeds:
        tr, va, te = _split(ds, seed)
        mu, sd = NUMraw[tr].mean(0), NUMraw[tr].std(0) + 1e-8
        NUM = (NUMraw - mu) / sd
        res = {}
        # A: ridge on numerics
        ridge = Ridge(alpha=1.0).fit(NUM[tr], y_all[tr])
        res["ridge_numeric_spearman"] = float(spearmanr(ridge.predict(NUM[te]), y_all[te]).statistic)
        # B: scratch
        m_scratch = TransferCNN(n_numeric=NUM.shape[1])
        train_model(m_scratch, X[tr], NUM[tr], y_all[tr], epochs=scratch_epochs,
                    val=(X[va], NUM[va], y_all[va]), patience=4, seed=seed)
        res["scratch_spearman"] = float(spearmanr(
            predict_model(m_scratch, X[te], NUM[te]), y_all[te]).statistic)
        # C: transfer
        m_tr = TransferCNN(n_numeric=NUM.shape[1])
        m_tr.load_trunk(trunk_state)
        train_model(m_tr, X[tr], NUM[tr], y_all[tr], epochs=ft_epochs,
                    val=(X[va], NUM[va], y_all[va]), patience=4, seed=seed)
        res["transfer_spearman"] = float(spearmanr(
            predict_model(m_tr, X[te], NUM[te]), y_all[te]).statistic)
        out["conditions"][seed] = res
        if seed == seeds[0]:
            torch.save(m_tr.state_dict(), os.path.join(out_dir, "pegrna_transfer_cnn.pt"))
    # means
    keys = ["ridge_numeric_spearman", "scratch_spearman", "transfer_spearman"]
    out["means"] = {k: float(np.mean([out["conditions"][s][k] for s in seeds])) for k in keys}
    os.makedirs(out_dir, exist_ok=True)
    with open(os.path.join(out_dir, f"pegrna_metrics_{target}.json"), "w") as f:
        json.dump(out, f, indent=2)
    return out
