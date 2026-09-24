"""Aggregate gaps 4-5 ablation across seeded runs into mean/sd per condition.

Reads results/ablation/seed*/offtarget_ablation.json (fully seeded: torch.manual_seed
per condition in train_offtarget_ablation.run_ablation). Honest partial aggregation:
uses every seed directory with a complete JSON and records exactly which.
"""
import glob
import json

import numpy as np

CONDITIONS = ["seq_only", "pam", "epigen", "pam_epigen"]


def main(out_path="results/offtarget_ablation_multiseed.json"):
    per_seed = []
    for f in sorted(glob.glob("results/ablation/seed*/offtarget_ablation.json")):
        r = json.load(open(f))
        if set(r.get("conditions", {})) == set(CONDITIONS):
            per_seed.append(r)
    agg = {"seeds": [r["seed"] for r in per_seed], "n_seeds": len(per_seed),
           "fully_seeded": True, "conditions": {}}
    for cond in CONDITIONS:
        aus = [r["conditions"][cond]["auroc"] for r in per_seed]
        non_ngg = [r["conditions"][cond]["auroc_non_ngg_pam"] for r in per_seed
                   if "auroc_non_ngg_pam" in r["conditions"][cond]]
        agg["conditions"][cond] = {
            "auroc_mean": float(np.mean(aus)), "auroc_sd": float(np.std(aus)),
            "auroc_per_seed": aus,
            "auroc_non_ngg_mean": float(np.mean(non_ngg)) if non_ngg else None,
            "auroc_non_ngg_sd": float(np.std(non_ngg)) if non_ngg else None,
            "auroc_non_ngg_per_seed": non_ngg,
        }
    # per-seed winner + condition deltas
    agg["winner_per_seed"] = {
        str(r["seed"]): max(CONDITIONS, key=lambda c: r["conditions"][c]["auroc"])
        for r in per_seed}
    json.dump(agg, open(out_path, "w"), indent=2)
    return agg


if __name__ == "__main__":
    a = main()
    print("seeds:", a["seeds"], "winners:", a["winner_per_seed"])
    for c, d in a["conditions"].items():
        nn = f"{d['auroc_non_ngg_mean']:.3f}±{d['auroc_non_ngg_sd']:.3f}" if d['auroc_non_ngg_mean'] is not None else "n/a"
        print(f"{c:11s} auroc {d['auroc_mean']:.3f}±{d['auroc_sd']:.3f}  non-NGG {nn}")
