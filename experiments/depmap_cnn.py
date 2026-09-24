"""Does a CNN beat ridge on the largest efficacy label? DepMap 23Q4 guide
efficacy (193k guides), within-dataset only, 2 seeds, same CNN as the
gap-1 work. Ridge reference from the committed 12-ds matrix: 0.328."""
import json

import numpy as np
import torch
from scipy.stats import spearmanr
from sklearn.linear_model import Ridge
from torch import nn

from crisprgap.data.datasets import load_depmap_efficacy
from crisprgap.models.efficacy_cnn import GuideEfficacyCNN, make_aux_features
from crisprgap.sequence import one_hot

torch.set_num_threads(4)


def main():
    ds = load_depmap_efficacy()
    rng0 = np.random.default_rng(0)
    sub = np.sort(rng0.choice(len(ds.sequences), 60000, replace=False))
    seqs = [ds.sequences[i] for i in sub]
    ds.scores = ds.scores[sub]
    X = np.stack([one_hot(s, 30) for s in seqs])
    A = np.stack([make_aux_features(s) for s in seqs])
    X = torch.from_numpy(X); A = torch.from_numpy(A)
    y = torch.from_numpy(ds.scores)
    n = len(seqs)
    out = {"dataset": "depmap_23q4_60k_subsample", "n": n, "seeds": {}}
    for seed in (0,):
        rng = np.random.default_rng(seed)
        perm = rng.permutation(n)
        te, tr = perm[: n // 5], perm[n // 5:]
        Xf = X.reshape(n, -1).numpy()
        ridge = Ridge(alpha=1.0).fit(Xf[tr], y[tr].numpy())
        r_sp = float(spearmanr(ridge.predict(Xf[te]), y[te].numpy()).statistic)
        model = GuideEfficacyCNN()
        opt = torch.optim.Adam(model.parameters(), lr=1e-3)
        lossf = nn.MSELoss()
        for epoch in range(6):
            model.train()
            ep = rng.permutation(len(tr))
            for i in range(0, len(tr), 256):
                idx = tr[ep[i:i + 256]]
                opt.zero_grad()
                loss = lossf(model(X[idx], A[idx]), y[idx])
                loss.backward()
                opt.step()
        model.eval()
        with torch.no_grad():
            pred = model(X[te], A[te]).numpy()
        c_sp = float(spearmanr(pred, y[te].numpy()).statistic)
        out["seeds"][seed] = {"ridge": r_sp, "cnn": c_sp, "epochs": 6}
        print(f"seed {seed}: ridge {r_sp:.3f} cnn {c_sp:.3f}")
    json.dump(out, open("results/depmap_cnn_within.json", "w"), indent=2)


if __name__ == "__main__":
    main()
