"""Gap 1 extension: CNN cross-dataset matrix (vs ridge) + pooled-training mitigation."""
import json
import numpy as np
import torch
from scipy.stats import spearmanr

from crisprgap.crossdata import run_cross_dataset, summarize, ridge_model
from crisprgap.data.datasets import load_doench_fcres, load_doench_v1, load_deephf
from crisprgap.models.transfer_cnn import TransferCNN, train_model, predict_model
from crisprgap.sequence import one_hot

torch.set_num_threads(2)
L = 30


def cnn_model(epochs=6):
    def fit(X, y):
        # X: flat features from crossdata.featurize; first 4*L entries are the one-hot
        n = len(X)
        Xs = X[:, : 4 * L].reshape(n, 4, L)
        m = TransferCNN(n_numeric=0)
        train_model(m, Xs, None, np.asarray(y, dtype=np.float32), epochs=epochs, batch=256)
        return m
    def predict(m, X):
        Xs = X[:, : 4 * L].reshape(len(X), 4, L)
        return predict_model(m, Xs)
    return fit, predict


if __name__ == "__main__":
    ds = {"fcres": load_doench_fcres(), "v1": load_doench_v1(),
          "deephf_wt": load_deephf("wt", max_n=12000)}
    res = run_cross_dataset(ds, {"cnn": cnn_model()}, seeds=(0, 1))
    out = {"raw": {str(k): v for k, v in res.items()}, "summary": summarize(res)}
    json.dump(out, open("results/crossdata_cnn_3ds.json", "w"), indent=2)
    print(json.dumps(out["summary"], indent=2))
