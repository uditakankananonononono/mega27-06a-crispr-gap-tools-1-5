"""Study-routing predictor: learn which studies favor the GNN over the MIT baseline.

Gap-2 meta-finding follow-up. The per-study canonical table showed the GNN winning
Anderson-style studies and losing Kim. This script makes that falsifiable:

  1. Re-derives the exact seeded test split used by train_offtarget (same rng call
     order; verified against saved per-seed predictions).
  2. Computes per-(seed, study) AUROC for GNN and MIT on studies with >=30 test
     pairs and both classes present.
  3. Builds study features from the test portion (n, prevalence, mismatch profile,
     PAM mix, guide GC) - the quantities a practitioner can measure before choosing.
  4. Leave-one-out routing: logistic regression predicts P(GNN wins) from features;
     routed AUROC is the predicted winner's AUROC. Compared against always-GNN,
     always-MIT, and the oracle, with a bootstrap CI over rows.

Honesty notes: the sample is small (one row per study per seed where the study is
measurable); we report the exact row count and do not smooth over it.
"""
from __future__ import annotations

import json
import os

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score

from crisprgap.baselines import mit_score
from crisprgap.data.datasets import load_crisprsql

MIN_N = 30
MIN_POS = 5   # rows with fewer positives have near-degenerate AUROC (rank of 1-4 positives)
MIN_NEG = 5
SEEDS = [0, 1, 2, 3, 4]


def seeded_split(ds, seed: int, max_pairs: int = 8000):
    """Replicates crisprgap.train_offtarget's split (same rng call order)."""
    rng = np.random.default_rng(seed)
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
    return keep[is_test]


def study_features(ds, idx) -> dict:
    guides = [ds.guides[i] for i in idx]
    offs = [ds.offtargets[i] for i in idx]
    pams = [ds.pam[i] for i in idx]
    y = ds.labels[idx]
    mm = [sum(a != b for a, b in zip(g, o)) for g, o in zip(guides, offs)]
    gc = [ (g.count("G") + g.count("C")) / max(1, sum(g.count(b) for b in "ACGT")) for g in guides]
    return {
        "n_test": int(len(idx)),
        "prevalence": float(y.mean()),
        "mean_mismatches": float(np.mean(mm)),
        "frac_non_ngg": float(np.mean([p.upper()[-3:] != p.upper()[-3] + "GG" if len(p) >= 3 else True for p in pams])) if any(pams) else 0.0,
        "gc_guide": float(np.mean(gc)),
        "mean_mit": float(np.mean([mit_score(g, o) for g, o in zip(guides, offs)])),
    }


def main(out_path: str = "results/study_routing.json") -> dict:
    ds = load_crisprsql()
    rows = []
    for seed in SEEDS:
        te = seeded_split(ds, seed)
        z = np.load(f"results/routing/seed{seed}/offtarget_preds.npz", allow_pickle=True)
        # verify the split replication: saved logits must score the same AUROC
        assert len(z["y_test"]) == len(te), f"seed {seed}: split mismatch"
        assert int(z["y_test"].sum()) == int(ds.labels[te].sum()), f"seed {seed}: label mismatch"
        auroc_check = roc_auc_score(ds.labels[te], z["logits"])
        studies = np.array(ds.studies)[te]
        for study in sorted(set(studies)):
            m = studies == study
            if m.sum() < MIN_N:
                continue
            y, lg, mt = ds.labels[te][m], z["logits"][m], z["mit"][m]
            if len(np.unique(y)) < 2:
                continue
            feats = study_features(ds, te[m])
            rows.append({
                "seed": seed, "study": study, **feats,
                "n_pos": int(y.sum()), "n_neg": int((1 - y).sum()),
                "auroc_gnn": float(roc_auc_score(y, lg)),
                "auroc_mit": float(roc_auc_score(y, mt)),
            })

    all_rows = rows
    thin_rows = [r for r in all_rows if r["n_pos"] < MIN_POS or r["n_neg"] < MIN_NEG]
    rows = [r for r in all_rows if r["n_pos"] >= MIN_POS and r["n_neg"] >= MIN_NEG]

    FEATS = ["n_test", "prevalence", "mean_mismatches", "frac_non_ngg", "gc_guide", "mean_mit"]
    X = np.array([[r[f] for f in FEATS] for r in rows])
    y_win = np.array([r["auroc_gnn"] > r["auroc_mit"] for r in rows], dtype=int)
    gnn = np.array([r["auroc_gnn"] for r in rows])
    mit = np.array([r["auroc_mit"] for r in rows])

    # leave-one-out routing
    routed = np.zeros(len(rows))
    p_gnn = np.zeros(len(rows))
    for i in range(len(rows)):
        mask = np.ones(len(rows), dtype=bool)
        mask[i] = False
        if len(np.unique(y_win[mask])) < 2:
            p = y_win[mask].mean()  # degenerate: fall back to base rate
        else:
            clf = LogisticRegression(C=0.5, max_iter=2000)
            clf.fit((X[mask] - X[mask].mean(0)) / (X[mask].std(0) + 1e-9), y_win[mask])
            p = clf.predict_proba(((X[i] - X[mask].mean(0)) / (X[mask].std(0) + 1e-9)).reshape(1, -1))[0, 1]
        p_gnn[i] = p
        routed[i] = gnn[i] if p >= 0.5 else mit[i]

    oracle = np.maximum(gnn, mit)
    # explicit falsifiable version of the hypothesized rule: GNN for
    # Anderson-style (very low test prevalence) studies, MIT otherwise
    rule_routed = np.array([g if r["prevalence"] < 0.10 else m
                            for r, g, m in zip(rows, gnn, mit)])

    res = {
        "rows": rows,
        "thin_rows_excluded": [
            {"seed": r["seed"], "study": r["study"], "n_test": r["n_test"],
             "n_pos": r["n_pos"], "n_neg": r["n_neg"],
             "auroc_gnn": r["auroc_gnn"], "auroc_mit": r["auroc_mit"]}
            for r in thin_rows
        ],
        "n_rows": len(rows),
        "min_n_per_row": MIN_N,
        "min_pos_per_row": MIN_POS,
        "min_neg_per_row": MIN_NEG,
        "features": FEATS,
        "mean_auroc": {
            "always_gnn": float(gnn.mean()),
            "always_mit": float(mit.mean()),
            "routed_loo": float(routed.mean()),
            "routed_prevalence_rule": float(rule_routed.mean()),
            "oracle": float(oracle.mean()),
        },
        "routing_gap_closed": float((routed.mean() - max(gnn.mean(), mit.mean())) /
                                    (oracle.mean() - max(gnn.mean(), mit.mean()))) if oracle.mean() > max(gnn.mean(), mit.mean()) else None,
    }

    # bootstrap CI over rows for routed - best-fixed
    rng = np.random.default_rng(0)
    diffs = []
    best_fixed = max(gnn.mean(), mit.mean())
    for _ in range(2000):
        b = rng.choice(len(rows), len(rows), replace=True)
        diffs.append(routed[b].mean() - (gnn[b].mean() if gnn.mean() >= mit.mean() else mit[b].mean()))
    diffs = np.array(diffs)
    res["bootstrap_routed_minus_best_fixed"] = {
        "mean": float(diffs.mean()),
        "ci95": [float(np.percentile(diffs, 2.5)), float(np.percentile(diffs, 97.5))],
        "p_gt_0": float((diffs > 0).mean()),
        "best_fixed_model": "gnn" if gnn.mean() >= mit.mean() else "mit",
    }
    # fit on all rows to report feature weights (interpretability only)
    if len(np.unique(y_win)) == 2:
        clf = LogisticRegression(C=0.5, max_iter=2000)
        Xs = (X - X.mean(0)) / (X.std(0) + 1e-9)
        clf.fit(Xs, y_win)
        res["full_fit_weights"] = {f: float(w) for f, w in zip(FEATS, clf.coef_[0])}
        res["full_fit_base_rate_gnn_wins"] = float(y_win.mean())

    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(res, f, indent=2)
    return res


if __name__ == "__main__":
    out = main()
    print(json.dumps({k: v for k, v in out.items() if k not in ("rows", "thin_rows_excluded")}, indent=2))
    print("THIN ROWS EXCLUDED (n_pos<%d or n_neg<%d):" % (MIN_POS, MIN_NEG))
    for r in out["thin_rows_excluded"]:
        print(f"  seed{r['seed']} {r['study']:14s} n={r['n_test']} pos={r['n_pos']} neg={r['n_neg']} gnn={r['auroc_gnn']:.3f} mit={r['auroc_mit']:.3f}")
    for r in out["rows"]:
        print(f"seed{r['seed']} {r['study']:14s} n={r['n_test']:4d} gnn={r['auroc_gnn']:.3f} mit={r['auroc_mit']:.3f} "
              f"prev={r['prevalence']:.3f} mm={r['mean_mismatches']:.2f} nonNGG={r['frac_non_ngg']:.2f} gc={r['gc_guide']:.2f}")
