import json
import os

import numpy as np


def test_seeded_split_reproduces_train_sizes():
    from experiments.study_routing import seeded_split
    from crisprgap.data.datasets import load_crisprsql
    ds = load_crisprsql()
    expected = {0: 526, 1: 1019, 2: 739, 3: 1079, 4: 487}  # verified vs train_offtarget runs
    for seed, n in expected.items():
        assert len(seeded_split(ds, seed)) == n, seed


def test_routing_artifact_structure():
    from experiments.study_routing import main
    out = main()
    assert os.path.exists("results/study_routing.json")
    for key in ("rows", "n_rows", "mean_auroc", "bootstrap_routed_minus_best_fixed",
                "thin_rows_excluded"):
        assert key in out, key
    assert out["n_rows"] == len(out["rows"]) > 0
    for r in out["rows"]:
        assert r["n_pos"] >= out["min_pos_per_row"]
        assert r["n_neg"] >= out["min_neg_per_row"]
        for k in ("auroc_gnn", "auroc_mit"):
            assert 0.0 <= r[k] <= 1.0
    m = out["mean_auroc"]
    assert m["oracle"] >= m["always_mit"] - 1e-9
    assert m["oracle"] >= m["always_gnn"] - 1e-9
