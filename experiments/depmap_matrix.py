"""Gap 1, 11-dataset ridge matrix: adds the four DeepCRISPR cell-line screens
(hct116, hek293t, hela, hl60) to the existing seven. Ridge, 3 seeds, same
featurization/harness as the published Table-1 matrix."""
import json

from crisprgap.crossdata import ridge_model, run_cross_dataset, summarize
from crisprgap.data.datasets import (load_crispron, load_crisprscan,
                                     load_deepcrispr, load_deepspcas9,
                                     load_deephf, load_doench_fcres,
                                     load_doench_v1, load_sgdesigner, load_depmap_efficacy)

LOADERS = {
    "fcres": load_doench_fcres, "v1": load_doench_v1,
    "deephf_wt": lambda: load_deephf("wt"),
    "deepspcas9": load_deepspcas9, "crispron": load_crispron,
    "sgdesigner": load_sgdesigner, "crisprscan": load_crisprscan,
    "dc_hct116": lambda: load_deepcrispr("hct116"),
    "dc_hek293t": lambda: load_deepcrispr("hek293t"),
    "dc_hela": lambda: load_deepcrispr("hela"),
    "dc_hl60": lambda: load_deepcrispr("hl60"),
    "depmap": load_depmap_efficacy,
}


def main():
    ds = {k: f() for k, f in LOADERS.items()}
    res = run_cross_dataset(ds, {"ridge": ridge_model()}, seeds=(0, 1, 2))
    out = {"raw": {str(k): v for k, v in res.items()}, "summary": summarize(res)}
    json.dump(out, open("results/crossdata_ridge_12ds.json", "w"), indent=2)
    s = out["summary"]["ridge"]
    for name in LOADERS:
        print(f"{name:12s} within {s[name]['within_mean']:.3f} | "
              + " ".join(f"{k}:{v:.2f}" for k, v in sorted(s[name]["cross_mean"].items())))


if __name__ == "__main__":
    main()
