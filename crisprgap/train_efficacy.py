"""Train + benchmark the efficacy CNN on Doench 2016 FC+RES; cross-test on V1 (gap 1)."""
from __future__ import annotations

import json
import os

import numpy as np
import torch
from scipy.stats import spearmanr
from sklearn.linear_model import Ridge
from torch import nn

from crisprgap.data.datasets import load_doench_fcres, load_doench_v1
from crisprgap.models.efficacy_cnn import GuideEfficacyCNN, make_aux_features
from crisprgap.sequence import one_hot

torch.set_num_threads(2)


def featurize(seqs):
    X = np.stack([one_hot(s, 30) for s in seqs])
    A = np.stack([make_aux_features(s) for s in seqs])
    return torch.from_numpy(X), torch.from_numpy(A)


def train_efficacy(out_dir: str = "results", epochs: int = 15, seed: int = 0) -> dict:
    rng = np.random.default_rng(seed)
    ds = load_doench_fcres()
    X, A = featurize(ds.sequences)
    y = torch.from_numpy(ds.scores)
    n = len(ds.sequences)
    perm = rng.permutation(n)
    n_test = n // 5
    te, tr = perm[:n_test], perm[n_test:]

    # baseline: ridge regression on aux features + one-hot position means
    Xf = X.reshape(n, -1).numpy()
    base = Ridge(alpha=1.0).fit(Xf[tr], y[tr].numpy())
    base_pred = base.predict(Xf[te])
    base_spearman = float(spearmanr(base_pred, y[te].numpy()).statistic)

    model = GuideEfficacyCNN()
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)
    lossf = nn.MSELoss()
    n_tr = len(tr)
    for epoch in range(epochs):
        model.train()
        ep = rng.permutation(n_tr)
        for i in range(0, n_tr, 128):
            idx = tr[ep[i:i + 128]]
            opt.zero_grad()
            loss = lossf(model(X[idx], A[idx]), y[idx])
            loss.backward()
            opt.step()
    model.eval()
    with torch.no_grad():
        cnn_pred = model(X[te], A[te]).numpy()

    # cross-dataset generalization (gap 1): evaluate both on Doench V1 untouched
    v1 = load_doench_v1()
    Xv, Av = featurize(v1.sequences)
    yv = v1.scores
    with torch.no_grad():
        cnn_v1 = model(Xv, Av).numpy()
    base_v1 = base.predict(Xv.reshape(len(v1.sequences), -1).numpy())

    result = {
        "dataset_train": ds.source,
        "n_train": int(n_tr), "n_test": int(n_test),
        "cnn_test_spearman": float(spearmanr(cnn_pred, y[te].numpy()).statistic),
        "ridge_test_spearman": base_spearman,
        "cnn_crossdataset_v1_spearman": float(spearmanr(cnn_v1, yv).statistic),
        "ridge_crossdataset_v1_spearman": float(spearmanr(base_v1, yv).statistic),
        "v1_n": len(v1.sequences),
        "epochs": epochs, "seed": seed,
    }
    os.makedirs(out_dir, exist_ok=True)
    torch.save(model.state_dict(), os.path.join(out_dir, "efficacy_cnn.pt"))
    np.savez_compressed(os.path.join(out_dir, "efficacy_preds.npz"),
                        cnn_pred=cnn_pred, y_test=y[te].numpy(), cnn_v1=cnn_v1, y_v1=yv)
    with open(os.path.join(out_dir, "efficacy_metrics.json"), "w") as f:
        json.dump(result, f, indent=2)
    return result
