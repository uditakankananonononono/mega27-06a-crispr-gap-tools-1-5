"""Do RNA secondary-structure / thermodynamic features help pegRNA efficacy prediction?

Extends the ViennaRNA null (results/pegrna_rnafold.json: MFE of the 40nt
edit-centered window adds nothing, +0.0002 Spearman) with two more external tools:

  - seqfold: minimum-ensemble dG of the SAME 40nt edit-centered window (tool comparison:
    does a different folding engine see what ViennaRNA's MFE missed?)
  - primer3: Wallace/NN melting temperature of the protospacer sliced from
    wide_initial_target via protospacerlocation_only_initial

Evaluation: grouped-by-pegRNA held-out ridge (seeds 0,1,2; same split recipe as
crisprgap.train_pegrna._split), HEK and K562. Numeric-only baseline vs numerics+RNA
features. Honest either way: positive = new signal; null = third folding/thermo
negative, reported.

Chunked so no single run exceeds the sandbox time cap:
  python experiments/rna_features_pegrna.py compute <start> <end>   # writes partial npz
  python experiments/rna_features_pegrna.py finalize                # merges + evaluates
"""
from __future__ import annotations

import glob
import json
import os
import sys

import numpy as np
import pandas as pd

CSV = "data/pridict/data_23k_v1.csv"
PART_GLOB = "results/rnafeat_part_*.npz"
OUT = "results/pegrna_rna_features.json"


def _window(seq: str, center: int, half: int = 20) -> str:
    lo = max(0, center - half)
    return seq[lo:lo + 2 * half]


def compute(start: int, end: int) -> None:
    import primer3
    from seqfold import fold

    df = pd.read_csv(CSV)
    mut = df["wide_mutated_target"].tolist()
    init = df["wide_initial_target"].tolist()
    deeppos = df["deepeditposition"].to_numpy()
    import ast
    psloc = df["protospacerlocation_only_initial"].map(
        lambda s: tuple(int(x) for x in ast.literal_eval(s))).tolist()

    dg, tm = [], []
    for i in range(start, min(end, len(df))):
        w = _window(mut[i], int(round(deeppos[i])))
        try:
            structs = fold(w)
            e = min((s.e for s in structs), default=0.0)
            dg.append(e if np.isfinite(e) else 0.0)  # seqfold: all-unpaired => -inf
        except Exception:
            dg.append(float("nan"))
        a, b = psloc[i]
        prot = init[i][a:b] if 0 <= a < b <= len(init[i]) else ""
        try:
            tm.append(float(primer3.calc_tm(prot)) if len(prot) >= 8 else float("nan"))
        except Exception:
            tm.append(float("nan"))
    os.makedirs("results", exist_ok=True)
    np.savez_compressed(f"results/rnafeat_part_{start}_{end}.npz",
                        start=start, dg=np.array(dg, dtype=np.float32),
                        tm=np.array(tm, dtype=np.float32))
    print(f"chunk {start}:{end} done ({len(dg)} rows)")


def finalize() -> dict:
    from sklearn.linear_model import Ridge
    from scipy.stats import spearmanr

    from crisprgap.data.datasets import load_pridict
    from crisprgap.train_pegrna import _split

    parts = sorted(glob.glob(PART_GLOB))
    assert parts, "no partial feature files"
    dg = np.concatenate([np.load(p)["dg"] for p in parts])
    tm = np.concatenate([np.load(p)["tm"] for p in parts])

    ds = load_pridict()
    n = len(ds.grp_ids)
    assert len(dg) >= n, f"features cover {len(dg)} of {n} rows"
    dg, tm = dg[:n], tm[:n]
    n_nan_dg = int(np.isnan(dg).sum())
    n_nan_tm = int(np.isnan(tm).sum())
    dg = np.nan_to_num(dg, nan=float(np.nanmean(dg)))
    tm = np.nan_to_num(tm, nan=float(np.nanmean(tm)))
    rna = np.stack([dg, tm], axis=1).astype(np.float32)

    res = {"n": int(n), "n_nan_dg_imputed": n_nan_dg, "n_nan_tm_imputed": n_nan_tm,
           "external_tools": ["seqfold (min-ensemble dG, 40nt edit-centered window)",
                              "primer3 2.3.1 (protospacer Tm, nearest-neighbor)"],
           "targets": {}}
    for target in ("HEK", "K562"):
        y = ds.targets[target]
        per_split_num, per_split_rna = [], []
        for seed in (0, 1, 2):
            tr, va, te = _split(ds, seed)
            Xn_tr, Xn_te = ds.numerics[tr], ds.numerics[te]
            Xr_tr = np.hstack([ds.numerics[tr], rna[tr]])
            Xr_te = np.hstack([ds.numerics[te], rna[te]])
            m1 = Ridge(alpha=1.0).fit(Xn_tr, y[tr])
            m2 = Ridge(alpha=1.0).fit(Xr_tr, y[tr])
            per_split_num.append(float(spearmanr(m1.predict(Xn_te), y[te]).statistic))
            per_split_rna.append(float(spearmanr(m2.predict(Xr_te), y[te]).statistic))
        res["targets"][target] = {
            "numeric_only_by_split": per_split_num,
            "plus_rna_by_split": per_split_rna,
            "mean_numeric": float(np.mean(per_split_num)),
            "mean_plus_rna": float(np.mean(per_split_rna)),
            "delta": float(np.mean(per_split_rna) - np.mean(per_split_num)),
        }
    json.dump(res, open(OUT, "w"), indent=2)
    return res


if __name__ == "__main__":
    if sys.argv[1] == "compute":
        compute(int(sys.argv[2]), int(sys.argv[3]))
    else:
        print(json.dumps(finalize(), indent=2))
